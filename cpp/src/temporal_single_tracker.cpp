#include "cvproj/temporal_single_tracker.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <utility>

namespace cvproj {
namespace {

cv::Point2f center_of(const cv::Rect& box) {
    return cv::Point2f(static_cast<float>(box.x) + 0.5F * static_cast<float>(box.width),
                       static_cast<float>(box.y) + 0.5F * static_cast<float>(box.height));
}

double distance_sq(const cv::Point2f& a, const cv::Point2f& b) {
    const double dx = static_cast<double>(a.x - b.x);
    const double dy = static_cast<double>(a.y - b.y);
    return dx * dx + dy * dy;
}

bool valid_point(const cv::Point2f& p) {
    return std::isfinite(p.x) && std::isfinite(p.y) && p.x >= 0.0F && p.y >= 0.0F;
}

cv::Point2f add_limited(const cv::Point2f& a, const cv::Point2f& b) {
    return cv::Point2f(a.x + b.x, a.y + b.y);
}

}  // namespace

TemporalSingleTracker::TemporalSingleTracker(TemporalSingleTrackerConfig config) : config_(std::move(config)) {
    config_.history_size = std::max(1, config_.history_size);
    config_.coordinate_output_rate_hz = std::max(1.0, config_.coordinate_output_rate_hz);
    config_.future_queue_size = std::max(1, config_.future_queue_size);
    config_.timeout_ms = std::max(0.0, config_.timeout_ms);
    config_.max_speed_px_s = std::max(1.0, config_.max_speed_px_s);
    config_.max_accel_px_s2 = std::max(1.0, config_.max_accel_px_s2);
    config_.jump_gate_norm = std::max(0.0, config_.jump_gate_norm);
    config_.select_conf_thresh = std::clamp(config_.select_conf_thresh, 0.0F, 1.0F);
    config_.select_min_area = std::max(0.0, config_.select_min_area);
    config_.select_area_tie_ratio = std::clamp(config_.select_area_tie_ratio, 0.0, 1.0);
    if (config_.output_mode != "realtime-sample" && config_.output_mode != "future-queue") {
        config_.output_mode = "future-queue";
    }
}

const TemporalSingleTrackerConfig& TemporalSingleTracker::config() const {
    return config_;
}

std::optional<TemporalSingleTracker::Observation> TemporalSingleTracker::select_target(
    const std::vector<Detection>& detections,
    const cv::Size& frame_size) {
    last_candidate_count_ = 0;
    std::vector<Observation> candidates;
    for (const auto& detection : detections) {
        const double area = static_cast<double>(detection.box.area());
        if (detection.confidence < config_.select_conf_thresh || area < config_.select_min_area) {
            continue;
        }
        cv::Rect box = detection.box & cv::Rect(0, 0, std::max(0, frame_size.width), std::max(0, frame_size.height));
        if (box.area() <= 0) {
            continue;
        }
        candidates.push_back({box, center_of(box), detection.confidence, static_cast<double>(box.area()), -1, 0.0});
    }
    last_candidate_count_ = static_cast<int>(candidates.size());
    if (candidates.empty()) {
        return std::nullopt;
    }

    std::sort(candidates.begin(), candidates.end(), [](const Observation& a, const Observation& b) {
        if (a.area != b.area) {
            return a.area > b.area;
        }
        return a.confidence > b.confidence;
    });

    if (candidates.size() < 2 || !valid_point(smoothed_center_)) {
        return candidates.front();
    }

    const double best_area = candidates.front().area;
    const double tie_floor = best_area * (1.0 - config_.select_area_tie_ratio);
    auto best = candidates.begin();
    double best_distance = distance_sq(best->center, smoothed_center_);
    for (auto it = candidates.begin(); it != candidates.end() && it->area >= tie_floor; ++it) {
        const double d = distance_sq(it->center, smoothed_center_);
        if (d < best_distance) {
            best = it;
            best_distance = d;
        }
    }
    return *best;
}

bool TemporalSingleTracker::update_measurement(const std::vector<Detection>& detections,
                                               const cv::Size& frame_size,
                                               std::int64_t frame_id,
                                               double timestamp_s) {
    auto selected = select_target(detections, frame_size);
    if (!selected.has_value()) {
        return false;
    }
    selected->frame_id = frame_id;
    selected->timestamp_s = timestamp_s;
    history_.push_back(*selected);
    while (static_cast<int>(history_.size()) > config_.history_size) {
        history_.pop_front();
    }
    future_queue_.clear();
    ++queue_rebuild_count_;
    return true;
}

cv::Point2f TemporalSingleTracker::clamp_to_frame(const cv::Point2f& point, const cv::Size& frame_size) const {
    if (frame_size.width <= 0 || frame_size.height <= 0) {
        return point;
    }
    return cv::Point2f(std::clamp(point.x, 0.0F, static_cast<float>(frame_size.width - 1)),
                       std::clamp(point.y, 0.0F, static_cast<float>(frame_size.height - 1)));
}

cv::Point2f TemporalSingleTracker::clamp_step(const cv::Point2f& desired,
                                              double dt_s,
                                              const cv::Size& frame_size,
                                              bool* clamped) {
    *clamped = false;
    if (!valid_point(smoothed_center_)) {
        return clamp_to_frame(desired, frame_size);
    }

    const double dt = std::clamp(dt_s, 0.001, 0.25);
    cv::Point2f delta = desired - smoothed_center_;
    const double distance = std::sqrt(static_cast<double>(delta.x * delta.x + delta.y * delta.y));
    const double max_step = config_.max_speed_px_s * dt;
    if (distance > max_step && distance > 1e-6) {
        const float scale = static_cast<float>(max_step / distance);
        delta.x *= scale;
        delta.y *= scale;
        *clamped = true;
    }

    cv::Point2f desired_velocity(delta.x / static_cast<float>(dt), delta.y / static_cast<float>(dt));
    cv::Point2f accel = desired_velocity - velocity_px_s_;
    const double accel_norm = std::sqrt(static_cast<double>(accel.x * accel.x + accel.y * accel.y));
    const double max_accel_step = config_.max_accel_px_s2 * dt;
    if (accel_norm > max_accel_step && accel_norm > 1e-6) {
        const float scale = static_cast<float>(max_accel_step / accel_norm);
        accel.x *= scale;
        accel.y *= scale;
        desired_velocity = velocity_px_s_ + accel;
        delta = cv::Point2f(desired_velocity.x * static_cast<float>(dt),
                            desired_velocity.y * static_cast<float>(dt));
        *clamped = true;
    }

    return clamp_to_frame(smoothed_center_ + delta, frame_size);
}

TemporalSingleTrackerSample TemporalSingleTracker::compute_sample(double timestamp_s,
                                                                  const cv::Size& frame_size,
                                                                  cv::Point2f imu_flow_px,
                                                                  const std::string& source,
                                                                  bool update_output_rate) {
    TemporalSingleTrackerSample sample;
    sample.coordinate_output_rate_hz = config_.coordinate_output_rate_hz;
    sample.history_count = static_cast<int>(history_.size());
    sample.candidate_count = last_candidate_count_;
    sample.queue_rebuild_count = queue_rebuild_count_;
    sample.queue_underrun_count = queue_underrun_count_;

    if (history_.empty()) {
        sample.source = "hold";
        if (update_output_rate) {
            update_actual_rate(timestamp_s, sample);
        }
        last_sample_ = sample;
        return sample;
    }

    const auto& latest = history_.back();
    sample.source_frame_id = latest.frame_id;
    sample.selected_center = latest.center;
    sample.selected_confidence = latest.confidence;
    sample.selected_area = latest.area;
    sample.raw_center = latest.center;
    sample.lost_ms = std::max(0.0, (timestamp_s - latest.timestamp_s) * 1000.0);

    if (sample.lost_ms > config_.timeout_ms) {
        sample.source = "timeout-hold";
        if (update_output_rate) {
            update_actual_rate(timestamp_s, sample);
        }
        last_sample_ = sample;
        return sample;
    }

    double dt = last_sample_ts_s_ > 0.0 ? timestamp_s - last_sample_ts_s_ : 1.0 / config_.coordinate_output_rate_hz;
    dt = std::clamp(dt, 0.001, 0.25);

    cv::Point2f desired = latest.center;
    if (source != "measurement-smoothed" && valid_point(smoothed_center_)) {
        desired = smoothed_center_ + velocity_px_s_ * static_cast<float>(dt);
        if (config_.use_imu_prediction) {
            desired = add_limited(desired, imu_flow_px);
        }
    }

    bool clamped = false;
    const cv::Point2f previous_smoothed = smoothed_center_;
    const cv::Point2f previous_velocity = velocity_px_s_;
    cv::Point2f smoothed = clamp_step(desired, dt, frame_size, &clamped);
    if (!valid_point(previous_smoothed)) {
        velocity_px_s_ = cv::Point2f(0.0F, 0.0F);
        acceleration_px_s2_ = cv::Point2f(0.0F, 0.0F);
    } else {
        velocity_px_s_ = (smoothed - previous_smoothed) * static_cast<float>(1.0 / dt);
        acceleration_px_s2_ = (velocity_px_s_ - previous_velocity) * static_cast<float>(1.0 / dt);
    }
    smoothed_center_ = smoothed;

    const double horizon_s = 1.0 / config_.coordinate_output_rate_hz;
    sample.valid = true;
    sample.smoothed_center = smoothed_center_;
    sample.predicted_center = clamp_to_frame(smoothed_center_ + velocity_px_s_ * static_cast<float>(horizon_s),
                                             frame_size);
    sample.velocity_px_s = velocity_px_s_;
    sample.acceleration_px_s2 = acceleration_px_s2_;
    sample.clamped = clamped;
    sample.source = source;
    if (update_output_rate) {
        update_actual_rate(timestamp_s, sample);
    } else {
        sample.output_dt_ms = 1000.0 / config_.coordinate_output_rate_hz;
        sample.coordinate_output_hz_actual = config_.coordinate_output_rate_hz;
    }
    last_sample_ = sample;
    return sample;
}

void TemporalSingleTracker::rebuild_future_queue(double timestamp_s,
                                                 const cv::Size& frame_size,
                                                 cv::Point2f imu_flow_px) {
    const cv::Point2f saved_smoothed = smoothed_center_;
    const cv::Point2f saved_velocity = velocity_px_s_;
    const cv::Point2f saved_acceleration = acceleration_px_s2_;
    const double saved_last_sample_ts = last_sample_ts_s_;
    const double saved_first_output_ts = first_output_ts_s_;
    const std::uint64_t saved_output_count = output_count_;
    const TemporalSingleTrackerSample saved_last_sample = last_sample_;

    const double dt = 1.0 / config_.coordinate_output_rate_hz;
    while (static_cast<int>(future_queue_.size()) < config_.future_queue_size) {
        const double sample_ts = timestamp_s + dt * static_cast<double>(future_queue_.size() + 1);
        auto sample = compute_sample(sample_ts,
                                     frame_size,
                                     imu_flow_px,
                                     future_queue_.empty() ? "interpolated" : "prediction",
                                     false);
        sample.future_queue_depth = static_cast<int>(future_queue_.size()) + 1;
        future_queue_.push_back(sample);
    }

    smoothed_center_ = saved_smoothed;
    velocity_px_s_ = saved_velocity;
    acceleration_px_s2_ = saved_acceleration;
    last_sample_ts_s_ = saved_last_sample_ts;
    first_output_ts_s_ = saved_first_output_ts;
    output_count_ = saved_output_count;
    last_sample_ = saved_last_sample;
}

void TemporalSingleTracker::update_actual_rate(double timestamp_s, TemporalSingleTrackerSample& sample) {
    sample.output_dt_ms = last_sample_ts_s_ > 0.0 ? (timestamp_s - last_sample_ts_s_) * 1000.0 : 0.0;
    last_sample_ts_s_ = timestamp_s;
    if (first_output_ts_s_ <= 0.0) {
        first_output_ts_s_ = timestamp_s;
    }
    ++output_count_;
    const double elapsed_s = timestamp_s - first_output_ts_s_;
    sample.coordinate_output_hz_actual = elapsed_s > 0.0 ? static_cast<double>(output_count_ - 1) / elapsed_s
                                                         : 0.0;
}

TemporalSingleTrackerSample TemporalSingleTracker::sample(double timestamp_s,
                                                          const cv::Size& frame_size,
                                                          cv::Point2f imu_flow_px) {
    if (config_.output_mode == "realtime-sample") {
        auto sample = compute_sample(timestamp_s, frame_size, imu_flow_px, history_.empty() ? "hold" : "interpolated");
        sample.future_queue_depth = 0;
        return sample;
    }

    if (future_queue_.empty()) {
        rebuild_future_queue(timestamp_s, frame_size, imu_flow_px);
    }
    if (future_queue_.empty()) {
        ++queue_underrun_count_;
        return compute_sample(timestamp_s, frame_size, imu_flow_px, "hold");
    }
    auto sample = future_queue_.front();
    future_queue_.pop_front();
    smoothed_center_ = sample.smoothed_center;
    velocity_px_s_ = sample.velocity_px_s;
    acceleration_px_s2_ = sample.acceleration_px_s2;
    last_sample_ts_s_ = timestamp_s;
    sample.future_queue_depth = static_cast<int>(future_queue_.size());
    sample.queue_rebuild_count = queue_rebuild_count_;
    sample.queue_underrun_count = queue_underrun_count_;
    last_sample_ = sample;
    return sample;
}

TemporalSingleTrackerSample TemporalSingleTracker::latest_sample() const {
    return last_sample_;
}

}  // namespace cvproj
