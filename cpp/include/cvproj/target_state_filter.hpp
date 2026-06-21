#pragma once

#include <optional>
#include <string>

#include <opencv2/core.hpp>
#include <opencv2/video/tracking.hpp>

#include "cvproj/bytetrack_tracker.hpp"
#include "cvproj/motion_pipeline.hpp"

namespace cvproj {

struct TargetStateFilterConfig {
    std::string mode = "kalman";
    double process_noise = 25.0;
    double measurement_noise = 16.0;
    int max_predict_frames = 10;
    double timeout_ms = 350.0;
    double motion_roi_gate = 1.5;
    double min_motion_confidence = 0.15;
};

struct TargetState {
    bool has_target = false;
    int track_id = -1;
    cv::Point2f raw_center{-1.0F, -1.0F};
    cv::Point2f filtered_center{-1.0F, -1.0F};
    cv::Point2f predicted_center{-1.0F, -1.0F};
    cv::Point2f velocity_px_s{0.0F, 0.0F};
    std::string source = "hold";
    double age_ms = 0.0;
};

class TargetStateFilter {
public:
    explicit TargetStateFilter(TargetStateFilterConfig config = {});

    TargetState update(const std::optional<TrackedTarget>& primary,
                       const MotionPipelineResult& motion,
                       const cv::Size& frame_size,
                       double dt_seconds);
    const TargetStateFilterConfig& config() const;

private:
    void initialize(const cv::Point2f& center, int track_id, double dt_seconds);
    cv::Point2f predict(double dt_seconds);
    cv::Point2f correct(const cv::Point2f& measured);
    cv::Point2f clamp_prediction_to_motion_roi(const cv::Point2f& point,
                                               const MotionPipelineResult& motion) const;

    TargetStateFilterConfig config_;
    cv::KalmanFilter kalman_;
    bool initialized_ = false;
    int active_track_id_ = -1;
    int predicted_frames_ = 0;
    double lost_age_ms_ = 0.0;
    TargetState last_state_;
};

}  // namespace cvproj
