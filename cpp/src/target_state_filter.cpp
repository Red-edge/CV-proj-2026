#include "cvproj/target_state_filter.hpp"

#include <algorithm>
#include <cmath>

namespace cvproj {
namespace {

cv::Point2f center_of(const cv::Rect& box) {
    return cv::Point2f(static_cast<float>(box.x) + 0.5F * static_cast<float>(box.width),
                       static_cast<float>(box.y) + 0.5F * static_cast<float>(box.height));
}

bool finite_point(const cv::Point2f& point) {
    return std::isfinite(point.x) && std::isfinite(point.y);
}

cv::Point2f clamp_to_frame(const cv::Point2f& point, const cv::Size& frame_size) {
    if (frame_size.width <= 0 || frame_size.height <= 0) {
        return point;
    }
    return cv::Point2f(std::clamp(point.x, 0.0F, static_cast<float>(frame_size.width - 1)),
                       std::clamp(point.y, 0.0F, static_cast<float>(frame_size.height - 1)));
}

}  // namespace

TargetStateFilter::TargetStateFilter(TargetStateFilterConfig config) : config_(std::move(config)) {
    config_.process_noise = std::max(1e-6, config_.process_noise);
    config_.measurement_noise = std::max(1e-6, config_.measurement_noise);
    config_.max_predict_frames = std::max(0, config_.max_predict_frames);
    config_.timeout_ms = std::max(0.0, config_.timeout_ms);
    config_.motion_roi_gate = std::max(0.0, config_.motion_roi_gate);
    config_.min_motion_confidence = std::clamp(config_.min_motion_confidence, 0.0, 1.0);
}

const TargetStateFilterConfig& TargetStateFilter::config() const {
    return config_;
}

void TargetStateFilter::initialize(const cv::Point2f& center, int track_id, double dt_seconds) {
    const float dt = static_cast<float>(dt_seconds > 0.0 ? std::clamp(dt_seconds, 0.001, 0.25) : 1.0 / 30.0);
    kalman_ = cv::KalmanFilter(4, 2, 0, CV_32F);
    kalman_.transitionMatrix = (cv::Mat_<float>(4, 4) << 1, 0, dt, 0,
                                                          0, 1, 0, dt,
                                                          0, 0, 1, 0,
                                                          0, 0, 0, 1);
    kalman_.measurementMatrix = cv::Mat::zeros(2, 4, CV_32F);
    kalman_.measurementMatrix.at<float>(0, 0) = 1.0F;
    kalman_.measurementMatrix.at<float>(1, 1) = 1.0F;
    cv::setIdentity(kalman_.processNoiseCov, cv::Scalar(config_.process_noise));
    cv::setIdentity(kalman_.measurementNoiseCov, cv::Scalar(config_.measurement_noise));
    cv::setIdentity(kalman_.errorCovPost, cv::Scalar(1.0));
    kalman_.statePost = (cv::Mat_<float>(4, 1) << center.x, center.y, 0.0F, 0.0F);

    initialized_ = true;
    active_track_id_ = track_id;
    predicted_frames_ = 0;
    lost_age_ms_ = 0.0;
    last_state_ = {};
    last_state_.has_target = true;
    last_state_.track_id = track_id;
    last_state_.raw_center = center;
    last_state_.filtered_center = center;
    last_state_.predicted_center = center;
    last_state_.source = "track";
}

cv::Point2f TargetStateFilter::predict(double dt_seconds) {
    const float dt = static_cast<float>(dt_seconds > 0.0 ? std::clamp(dt_seconds, 0.001, 0.25) : 1.0 / 30.0);
    kalman_.transitionMatrix.at<float>(0, 2) = dt;
    kalman_.transitionMatrix.at<float>(1, 3) = dt;
    const cv::Mat predicted = kalman_.predict();
    return cv::Point2f(predicted.at<float>(0), predicted.at<float>(1));
}

cv::Point2f TargetStateFilter::correct(const cv::Point2f& measured) {
    const cv::Mat measurement = (cv::Mat_<float>(2, 1) << measured.x, measured.y);
    const cv::Mat corrected = kalman_.correct(measurement);
    return cv::Point2f(corrected.at<float>(0), corrected.at<float>(1));
}

cv::Point2f TargetStateFilter::clamp_prediction_to_motion_roi(const cv::Point2f& point,
                                                              const MotionPipelineResult& motion) const {
    if (!motion.motion_roi.has_value() || motion.motion_confidence < config_.min_motion_confidence ||
        config_.motion_roi_gate <= 0.0) {
        return point;
    }

    const cv::Rect roi = *motion.motion_roi;
    const float expand_x = static_cast<float>(roi.width) * static_cast<float>(config_.motion_roi_gate - 1.0) * 0.5F;
    const float expand_y = static_cast<float>(roi.height) * static_cast<float>(config_.motion_roi_gate - 1.0) * 0.5F;
    const float min_x = static_cast<float>(roi.x) - expand_x;
    const float max_x = static_cast<float>(roi.x + roi.width) + expand_x;
    const float min_y = static_cast<float>(roi.y) - expand_y;
    const float max_y = static_cast<float>(roi.y + roi.height) + expand_y;
    return cv::Point2f(std::clamp(point.x, min_x, max_x), std::clamp(point.y, min_y, max_y));
}

TargetState TargetStateFilter::update(const std::optional<TrackedTarget>& primary,
                                      const MotionPipelineResult& motion,
                                      const cv::Size& frame_size,
                                      double dt_seconds) {
    TargetState state;
    if (config_.mode == "off") {
        if (!primary.has_value()) {
            state.source = "hold";
            last_state_ = state;
            return state;
        }
        const cv::Point2f raw = clamp_to_frame(center_of(primary->box), frame_size);
        state.has_target = true;
        state.track_id = primary->track_id;
        state.raw_center = raw;
        state.filtered_center = raw;
        state.predicted_center = raw;
        state.source = "track";
        state.age_ms = 0.0;
        last_state_ = state;
        return state;
    }

    const double dt = dt_seconds > 0.0 ? std::clamp(dt_seconds, 0.001, 0.25) : 1.0 / 30.0;
    if (primary.has_value()) {
        const cv::Point2f raw = clamp_to_frame(center_of(primary->box), frame_size);
        if (!initialized_ || primary->track_id != active_track_id_) {
            initialize(raw, primary->track_id, dt);
        }
        const cv::Point2f predicted = clamp_to_frame(predict(dt), frame_size);
        const cv::Point2f filtered = clamp_to_frame(correct(raw), frame_size);
        state.has_target = true;
        state.track_id = primary->track_id;
        state.raw_center = raw;
        state.filtered_center = filtered;
        state.predicted_center = predicted;
        state.velocity_px_s = cv::Point2f(kalman_.statePost.at<float>(2), kalman_.statePost.at<float>(3));
        state.source = "track";
        state.age_ms = 0.0;
        predicted_frames_ = 0;
        lost_age_ms_ = 0.0;
        last_state_ = state;
        return state;
    }

    if (!initialized_) {
        state.source = "hold";
        last_state_ = state;
        return state;
    }

    ++predicted_frames_;
    lost_age_ms_ += dt * 1000.0;
    if (predicted_frames_ > config_.max_predict_frames || lost_age_ms_ > config_.timeout_ms) {
        initialized_ = false;
        active_track_id_ = -1;
        state.source = "hold";
        state.track_id = last_state_.track_id;
        state.age_ms = lost_age_ms_;
        last_state_ = state;
        return state;
    }

    cv::Point2f predicted = predict(dt);
    predicted = clamp_prediction_to_motion_roi(predicted, motion);
    predicted = clamp_to_frame(predicted, frame_size);
    state.has_target = true;
    state.track_id = active_track_id_;
    state.raw_center = last_state_.raw_center;
    state.filtered_center = predicted;
    state.predicted_center = predicted;
    state.velocity_px_s = cv::Point2f(kalman_.statePre.at<float>(2), kalman_.statePre.at<float>(3));
    state.source = "prediction";
    state.age_ms = lost_age_ms_;
    last_state_ = state;
    return state;
}

}  // namespace cvproj
