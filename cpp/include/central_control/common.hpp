#pragma once

#include <chrono>
#include <cstdint>
#include <deque>
#include <filesystem>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include <opencv2/core.hpp>

namespace central_control {

using Clock = std::chrono::steady_clock;

struct AppConfig {
    std::string config_path;
    std::string source = "mvs";
    std::string backend = "auto";
    std::string model_onnx = "src/yolo11n.onnx";
    std::string model_rknn = "models/rknn/yolo11n.rknn";
    std::string output_dir = "outputs";
    std::string output_prefix = "central_control";
    std::string can_channel = "PCAN_USBBUS1";
    std::string mvs_serial;
    std::string window_name = "Central Control";

    int target_fps = 100;
    int duration_sec = 30;
    int input_width = 1280;
    int input_height = 720;
    int yolo_input_size = 640;
    int motion_points = 128;
    int blur_kernel = 21;
    int mvs_index = 0;

    float motion_threshold = 2.0F;
    float roi_conf_threshold = 0.25F;
    float bg_conf_threshold = 0.45F;
    float nms_iou = 0.45F;
    float mvs_exposure_us = -1.0F;
    float mvs_gain = -1.0F;
    float mvs_frame_rate = -1.0F;
    float yaw_zero = 0.0F;
    float pitch_zero = 0.0F;
    float yaw_rpm_per_deg = 25.0F;
    float pitch_rpm_per_deg = 25.0F;
    float h_fov = 45.0F;
    float v_fov = 34.0F;

    bool headless = false;
    bool record_rendered = false;
    bool mock_gimbal = true;
    bool use_yolo = true;
    bool list_sources = false;
};

struct FramePacket {
    std::uint64_t frame_id = 0;
    Clock::time_point capture_time = Clock::now();
    cv::Mat image;
};

struct MotionState {
    std::uint64_t frame_id = 0;
    std::vector<cv::Point2f> fixed_points;
    std::vector<float> magnitudes;
    int motion_count = 0;
    double motion_ms = 0.0;
    std::optional<cv::Rect> roi_box;
};

struct Detection {
    cv::Rect2f bbox;
    float confidence = 0.0F;
    std::string class_name = "person";
    std::string attention = "BG";
};

struct TrackView {
    int track_id = -1;
    cv::Rect2f bbox;
    std::deque<cv::Point2f> history;
    int age = 0;
};

struct DetectionState {
    std::uint64_t frame_id = 0;
    std::vector<Detection> detections;
    std::vector<TrackView> tracks;
    double detection_ms = 0.0;
};

struct RuntimeStats {
    std::uint64_t captured_frames = 0;
    std::uint64_t rendered_frames = 0;
    std::uint64_t written_frames = 0;
    std::uint64_t dropped_flow_jobs = 0;
    std::uint64_t dropped_detect_jobs = 0;
    double capture_fps = 0.0;
    double render_fps = 0.0;
    double detection_ms = 0.0;
    int inference_lag_frames = 0;
    double result_age_ms = 0.0;
};

struct SensorSnapshot {
    std::string name;
    double timestamp = 0.0;
    std::string payload;
};

struct TargetObservation {
    std::uint64_t frame_id = 0;
    cv::Rect2f bbox;
    float confidence = 0.0F;
    cv::Point2f image_point;
    float yaw_deg = 0.0F;
    float pitch_deg = 0.0F;
};

struct ControlCommand {
    std::string name;
    float yaw_rpm = 0.0F;
    float pitch_rpm = 0.0F;
};

class IFrameSource {
public:
    virtual ~IFrameSource() = default;
    virtual bool open(const AppConfig& config) = 0;
    virtual bool read(FramePacket& packet) = 0;
    virtual void close() = 0;
    virtual std::string describe() const = 0;
};

class IFlowEstimator {
public:
    virtual ~IFlowEstimator() = default;
    virtual MotionState process(const FramePacket& packet) = 0;
};

class IDetector {
public:
    virtual ~IDetector() = default;
    virtual bool initialize(const AppConfig& config) = 0;
    virtual std::vector<Detection> detect(const FramePacket& packet, const MotionState& motion, double& detection_ms) = 0;
    virtual std::string backend_name() const = 0;
};

class ITracker {
public:
    virtual ~ITracker() = default;
    virtual std::vector<TrackView> update(const std::vector<Detection>& detections, std::uint64_t frame_id) = 0;
};

class IRenderer {
public:
    virtual ~IRenderer() = default;
    virtual cv::Mat render(
        const FramePacket& packet,
        const std::optional<MotionState>& motion,
        const std::optional<DetectionState>& detection,
        const RuntimeStats& stats) = 0;
};

class IOutputSink {
public:
    virtual ~IOutputSink() = default;
    virtual bool open(const AppConfig& config) = 0;
    virtual void write(const cv::Mat& frame) = 0;
    virtual void close() = 0;
    virtual std::string output_path() const = 0;
};

class IActuator {
public:
    virtual ~IActuator() = default;
    virtual bool initialize(const AppConfig& config) = 0;
    virtual void apply(const ControlCommand& command) = 0;
    virtual void stop() = 0;
    virtual std::string describe() const = 0;
};

class ControlCenter {
public:
    void publish_sensor(const SensorSnapshot& snapshot);
    void publish_target(const std::optional<TargetObservation>& observation);
    void dispatch(const ControlCommand& command);
    void register_actuator(std::shared_ptr<IActuator> actuator);
    void stop_all();

private:
    std::mutex mutex_;
    std::vector<SensorSnapshot> sensors_;
    std::optional<TargetObservation> latest_target_;
    ControlCommand latest_command_;
    std::shared_ptr<IActuator> actuator_;
};

class Runtime {
public:
    explicit Runtime(AppConfig config);
    bool initialize();
    int run();

private:
    AppConfig config_;
    ControlCenter control_center_;
    std::shared_ptr<IFrameSource> frame_source_;
    std::shared_ptr<IFlowEstimator> flow_estimator_;
    std::shared_ptr<IDetector> detector_;
    std::shared_ptr<ITracker> tracker_;
    std::shared_ptr<IRenderer> renderer_;
    std::shared_ptr<IOutputSink> output_sink_;
    std::shared_ptr<IActuator> actuator_;
};

AppConfig parse_arguments(int argc, char** argv);
bool load_config_file(const std::filesystem::path& path, AppConfig& config, std::string& error);
std::string make_timestamped_output_path(const AppConfig& config);

std::shared_ptr<IFrameSource> create_frame_source(const AppConfig& config);
std::shared_ptr<IFlowEstimator> create_flow_estimator(const AppConfig& config);
std::shared_ptr<IDetector> create_detector(const AppConfig& config);
std::shared_ptr<ITracker> create_tracker(const AppConfig& config);
std::shared_ptr<IRenderer> create_renderer(const AppConfig& config);
std::shared_ptr<IOutputSink> create_output_sink(const AppConfig& config);
std::shared_ptr<IActuator> create_actuator(const AppConfig& config);
int list_available_sources(const AppConfig& config);

}  // namespace central_control
