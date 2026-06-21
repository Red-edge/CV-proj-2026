#pragma once

#include <cstdint>
#include <deque>
#include <optional>
#include <string>
#include <vector>

#include <opencv2/core.hpp>

#include "cvproj/yolo_detector.hpp"

namespace cvproj {

struct TemporalSingleTrackerConfig {
    int history_size = 10;
    double coordinate_output_rate_hz = 500.0;
    std::string output_mode = "future-queue";
    int future_queue_size = 3;
    double timeout_ms = 250.0;
    double max_speed_px_s = 1200.0;
    double max_accel_px_s2 = 6000.0;
    double jump_gate_norm = 0.25;
    float select_conf_thresh = 0.30F;
    double select_min_area = 0.0;
    double select_area_tie_ratio = 0.10;
    bool use_imu_prediction = true;
    bool use_optical_flow = false;
};

struct TemporalSingleTrackerSample {
    bool valid = false;
    std::int64_t source_frame_id = -1;
    cv::Point2f selected_center{-1.0F, -1.0F};
    float selected_confidence = 0.0F;
    double selected_area = 0.0;
    int candidate_count = 0;
    cv::Point2f raw_center{-1.0F, -1.0F};
    cv::Point2f smoothed_center{-1.0F, -1.0F};
    cv::Point2f predicted_center{-1.0F, -1.0F};
    cv::Point2f velocity_px_s{0.0F, 0.0F};
    cv::Point2f acceleration_px_s2{0.0F, 0.0F};
    int history_count = 0;
    double lost_ms = 0.0;
    bool clamped = false;
    std::string source = "hold";
    int future_queue_depth = 0;
    std::uint64_t queue_rebuild_count = 0;
    std::uint64_t queue_underrun_count = 0;
    double coordinate_output_rate_hz = 0.0;
    double coordinate_output_hz_actual = 0.0;
    double output_dt_ms = 0.0;
};

class TemporalSingleTracker {
public:
    explicit TemporalSingleTracker(TemporalSingleTrackerConfig config = {});

    const TemporalSingleTrackerConfig& config() const;

    bool update_measurement(const std::vector<Detection>& detections,
                            const cv::Size& frame_size,
                            std::int64_t frame_id,
                            double timestamp_s);

    TemporalSingleTrackerSample sample(double timestamp_s,
                                       const cv::Size& frame_size,
                                       cv::Point2f imu_flow_px = cv::Point2f(0.0F, 0.0F));

    TemporalSingleTrackerSample latest_sample() const;

private:
    struct Observation {
        cv::Rect box;
        cv::Point2f center;
        float confidence = 0.0F;
        double area = 0.0;
        std::int64_t frame_id = -1;
        double timestamp_s = 0.0;
    };

    std::optional<Observation> select_target(const std::vector<Detection>& detections,
                                             const cv::Size& frame_size);
    TemporalSingleTrackerSample compute_sample(double timestamp_s,
                                               const cv::Size& frame_size,
                                               cv::Point2f imu_flow_px,
                                               const std::string& source,
                                               bool update_output_rate = true);
    void rebuild_future_queue(double timestamp_s, const cv::Size& frame_size, cv::Point2f imu_flow_px);
    cv::Point2f clamp_to_frame(const cv::Point2f& point, const cv::Size& frame_size) const;
    cv::Point2f clamp_step(const cv::Point2f& desired,
                           double dt_s,
                           const cv::Size& frame_size,
                           bool* clamped);
    void update_actual_rate(double timestamp_s, TemporalSingleTrackerSample& sample);

    TemporalSingleTrackerConfig config_;
    std::deque<Observation> history_;
    std::deque<TemporalSingleTrackerSample> future_queue_;
    TemporalSingleTrackerSample last_sample_;
    cv::Point2f smoothed_center_{-1.0F, -1.0F};
    cv::Point2f velocity_px_s_{0.0F, 0.0F};
    cv::Point2f acceleration_px_s2_{0.0F, 0.0F};
    double last_sample_ts_s_ = 0.0;
    double first_output_ts_s_ = 0.0;
    std::uint64_t output_count_ = 0;
    std::uint64_t queue_rebuild_count_ = 0;
    std::uint64_t queue_underrun_count_ = 0;
    int last_candidate_count_ = 0;
};

}  // namespace cvproj
