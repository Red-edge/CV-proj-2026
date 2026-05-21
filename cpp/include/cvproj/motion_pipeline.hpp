#pragma once

#include <optional>
#include <vector>

#include <opencv2/core.hpp>

#include "cvproj/frame_source.hpp"

namespace cvproj {

struct MotionPipelineConfig {
    int num_motion_points = 128;
    double motion_threshold = 2.0;
    double roi_width_ratio = 1.0 / 3.0;
    int blur_kernel = 15;
    int overlay_radius = 4;
    double track_smooth_alpha = 0.3;
    int min_blob_area = 900;
};

struct MotionPipelineResult {
    cv::Mat annotated_frame;
    std::optional<cv::Rect> roi;
    std::optional<cv::Rect> target_box;
    int motion_count = 0;
    double fps = 0.0;
    double processing_ms = 0.0;
    double source_delta_seconds = 0.0;
    double source_fps = 0.0;
    double source_timestamp_seconds = 0.0;
};

class MotionPipeline {
public:
    explicit MotionPipeline(MotionPipelineConfig config);

    MotionPipelineResult process(const FramePacket& packet);
    void reset();

private:
    std::vector<cv::Point2f> generate_fixed_motion_points(const cv::Size& size) const;
    std::optional<cv::Rect> get_roi_from_motion_points(const std::vector<cv::Point2f>& points,
                                                       const std::vector<float>& magnitudes,
                                                       const cv::Size& size) const;
    std::optional<cv::Rect> detect_target_blob(const std::vector<cv::Point2f>& points,
                                               const std::vector<float>& magnitudes,
                                               const cv::Size& size) const;
    cv::Rect smooth_track(const cv::Rect& current, double delta_seconds);
    void draw_overlay(cv::Mat& vis,
                      const std::vector<cv::Point2f>& points,
                      const std::vector<float>& magnitudes,
                      const MotionPipelineResult& result) const;

    MotionPipelineConfig config_;
    std::vector<cv::Point2f> fixed_points_;
    cv::Mat prev_gray_;
    double fps_ema_ = 0.0;
    double last_frame_timestamp_seconds_ = 0.0;
    std::optional<cv::Rect2f> smoothed_track_;
};

}  // namespace cvproj
