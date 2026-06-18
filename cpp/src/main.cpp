#include <algorithm>
#include <atomic>
#include <condition_variable>
#include <deque>
#include <filesystem>
#include <chrono>
#include <cctype>
#include <cmath>
#include <cstdint>
#include <csignal>
#include <ctime>
#include <fstream>
#include <iostream>
#include <iomanip>
#include <memory>
#include <mutex>
#include <optional>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/videoio.hpp>

#include "cvproj/aravis_frame_source.hpp"
#include "cvproj/bytetrack_tracker.hpp"
#include "cvproj/frame_source.hpp"
#include "cvproj/hikrobot_mvs_source.hpp"
#include "cvproj/mjpeg_server.hpp"
#include "cvproj/motion_pipeline.hpp"
#include "cvproj/opencv_video_source.hpp"
#include "cvproj/realsense_t265_source.hpp"
#include "cvproj/socketcan_gimbal.hpp"
#include "cvproj/target_state_filter.hpp"
#include "cvproj/yolo_onnx_detector.hpp"

namespace fs = std::filesystem;


namespace {
volatile std::sig_atomic_t g_stop_requested = 0;

void handle_stop_signal(int) {
    g_stop_requested = 1;
}

bool stop_requested() {
    return g_stop_requested != 0;
}
}  // namespace

namespace cvproj {

struct AppConfig {
    std::string backend = "opencv";
    std::string source = "0";
    std::string serial_number;
    std::string telemetry_path;
    std::string record_path;
    bool output = false;
    double record_fps = 15.0;
    std::string pixel_format = "BayerRG8";
    bool swap_rb = false;
    std::string detector = "none";
    std::string model_path = "src/yolov8n.onnx";
    int width = 1440;
    int height = 1080;
    double fps = 240.0;
    double exposure_us = 3000.0;
    double gain_db = 12.0;
    int num_motion_points = 128;
    double motion_threshold = 2.0;
    float det_conf = 0.25F;
    float det_nms = 0.45F;
    int detect_interval = 3;
    int det_input_size = 640;
    std::string tracker = "bytetrack";
    float track_high_thresh = 0.45F;
    float track_low_thresh = 0.10F;
    float track_match_thresh = 0.30F;
    int track_buffer = 30;
    int track_min_hits = 2;
    std::string target_filter = "kalman";
    double target_filter_process_noise = 25.0;
    double target_filter_measurement_noise = 16.0;
    int target_filter_max_predict_frames = 10;
    double target_filter_timeout_ms = 350.0;
    double target_filter_motion_roi_gate = 1.5;
    double target_filter_min_motion_confidence = 0.15;
    std::string pipeline_mode = "serial";
    int async_slot_capacity = 1;
    int async_log_queue_capacity = 2;
    double async_control_fps = 30.0;
    double async_target_max_age_ms = 120.0;
    double async_detection_max_age_ms = 500.0;
    double async_render_fps = 15.0;
    bool async_drop_stale_detections = true;
    int max_frames = -1;
    bool auto_brightness = true;
    double target_luma = 82.0;
    double max_post_gain = 2.0;
    double gamma = 1.0;
    double clahe_clip = 1.0;
    int clahe_tile = 8;
    double downsample_scale = 1.0;
    int downsample_width = 0;
    int downsample_height = 0;
    std::string denoise_method = "bilateral";
    int denoise_kernel = 7;
    double denoise_sigma_color = 75.0;
    double denoise_sigma_space = 11.0;
    int denoise_passes = 2;
    bool livestream = false;
    bool headless = false;
    bool preview = false;
    std::string preview_host = "0.0.0.0";
    int preview_port = 8080;
    double preview_fps = 15.0;
    bool yaw_enable = false;
    bool yaw_dry_run = false;
    std::string yaw_can_iface = "can0";
    std::uint32_t yaw_can_id = 0x8001;
    bool yaw_invert = false;
    bool yaw_motor_invert = false;
    double yaw_max_angle_deg = 30.0;
    double yaw_limit_margin_deg = 2.0;
    double yaw_max_rpm = 120.0;
    double yaw_kp_rpm_per_deg = 12.0;
    double yaw_ki_rpm_per_deg_s = 0.0;
    double yaw_kd_rpm_per_deg_per_s = 0.0;
    double yaw_deadband_deg = 0.25;
    std::string yaw_control_mode = "visual-servo";
    std::string yaw_outer_rate_mode = "inference";
    double yaw_outer_max_stale_ms = 120.0;
    double yaw_outer_min_update_interval_ms = 0.0;
    double yaw_outer_kp = 3.0;
    double yaw_outer_ki = 0.0;
    double yaw_outer_kd = 0.0;
    double yaw_outer_ff_gain = 0.0;
    double yaw_outer_integral_limit_deg_s = 30.0;
    double yaw_outer_deadband_deg = 0.2;
    std::string yaw_inner_mode = "driver-speed";
    double yaw_inner_rate_hz = 1000.0;
    double yaw_inner_feedback_rate_hz = 1000.0;
    double yaw_inner_feedback_timeout_ms = 5.0;
    bool yaw_inner_realtime_priority = false;
    int yaw_inner_realtime_priority_value = 60;
    double yaw_inner_command_timeout_ms = 120.0;
    double yaw_inner_kp = 1.0;
    double yaw_inner_ki = 0.0;
    double yaw_inner_kd = 0.0;
    double yaw_inner_integral_limit_rpm = 10.0;
    double yaw_inner_output_limit_rpm = 30.0;
    double yaw_inner_deadband_rpm = 0.1;
    double yaw_max_accel_rpm_s = 30.0;
    double yaw_axis_deg_per_motor_rev = 360.0;
    bool yaw_feedback_required = true;
    double yaw_hfov_deg = 70.0;
    double yaw_fx_px = 0.0;
    double yaw_cx_px = -1.0;
    double yaw_feedback_zero_deg = 0.0;
    double yaw_test_target_x = -1.0;
    int t265_fisheye_index = 1;
    bool t265_pose = true;
};

void print_usage() {
    std::cout
        << "Usage: cvproj_capture [options]\n"
        << "  --config <file.conf>\n"
        << "  --backend <opencv|hikrobot|aravis|t265|realsense>\n"
        << "  --source <camera-index|video-path>\n"
        << "  --serial <hikrobot-serial>\n"
        << "  --record-fps <output-fps, default 15>\n"
        << "  --pixel-format <BayerRG8|Mono8|auto|RGB8Packed|BGR8Packed|...>\n"
        << "  --swap-rb\n"
        << "  --gain-db <camera-gain-db>\n"
        << "  --target-luma <0-255>\n"
        << "  --max-post-gain <scale>\n"
        << "  --gamma <value, default 1.0>\n"
        << "  --clahe-clip <0 disables CLAHE, default 1.0>\n"
        << "  --clahe-tile <tile-size, default 8>\n"
        << "  --no-auto-brightness\n"
        << "  --downsample-scale <0-1, default 1>\n"
        << "  --downsample-width <pixels>\n"
        << "  --downsample-height <pixels>\n"
        << "  --denoise <off|bilateral|gaussian|median>\n"
        << "  --denoise-kernel <odd-size, default 7>\n"
        << "  --denoise-sigma-color <default 75>\n"
        << "  --denoise-sigma-space <default 11>\n"
        << "  --denoise-passes <1|2, default 2>\n"
        << "  --detector <none|yolo>\n"
        << "  --model <onnx-path>\n"
        << "  --det-conf <threshold>\n"
        << "  --det-nms <threshold>\n"
        << "  --detect-interval <N>\n"
        << "  --det-input-size <pixels, default 640>\n"
        << "  --tracker <bytetrack|none>\n"
        << "  --track-high-thresh <default 0.45>\n"
        << "  --track-low-thresh <default 0.10>\n"
        << "  --track-match-thresh <default 0.30>\n"
        << "  --track-buffer <missed inference ticks, default 1>\n"
        << "  --track-min-hits <frames, default 2>\n"
        << "  --target-filter <kalman|off>\n"
        << "  --target-filter-process-noise <default 25>\n"
        << "  --target-filter-measurement-noise <default 16>\n"
        << "  --target-filter-max-predict-frames <default 10>\n"
        << "  --target-filter-timeout-ms <default 350>\n"
        << "  --target-filter-motion-roi-gate <default 1.5>\n"
        << "  --target-filter-min-motion-confidence <default 0.15>\n"
        << "  --pipeline-mode <serial|async>\n"
        << "  --async-slot-capacity <default 1>\n"
        << "  --async-log-queue-capacity <default 2>\n"
        << "  --async-control-fps <default 30>\n"
        << "  --async-target-max-age-ms <default 120>\n"
        << "  --async-detection-max-age-ms <default 500>\n"
        << "  --async-render-fps <default 15>\n"
        << "  --async-drop-stale-detections <0|1>\n"
        << "  --width <pixels>\n"
        << "  --height <pixels>\n"
        << "  --fps <target>\n"
        << "  --exposure-us <microseconds>\n"
        << "  --motion-thresh <pixels>\n"
        << "  --grid-points <count>\n"
        << "  --output\n"
        << "  --record <output.mp4>    (advanced/manual path)\n"
        << "  --telemetry <output.csv> (advanced/manual path)\n"
        << "  --livestream\n"
        << "  --http-preview <bind-ip:port>\n"
        << "  --preview-fps <fps, default 15>\n"
        << "  --yaw-enable\n"
        << "  --yaw-dry-run\n"
        << "  --yaw-can-iface <can0>\n"
        << "  --yaw-id <0x8001>\n"
        << "  --yaw-invert\n"
        << "  --yaw-motor-invert\n"
        << "  --yaw-max-angle <deg, software hard limit from config>\n"
        << "  --yaw-limit-margin-deg <deg, default 2>\n"
        << "  --yaw-max-rpm <rpm>\n"
        << "  --yaw-kp <rpm-per-deg>\n"
        << "  --yaw-ki <rpm-per-deg-s>\n"
        << "  --yaw-kd <rpm-per-deg-per-s>\n"
        << "  --yaw-deadband <deg>\n"
        << "  --yaw-control-mode <visual-servo|legacy>\n"
        << "  --yaw-outer-rate-mode <inference>\n"
        << "  --yaw-outer-kp <deg-s-per-deg>\n"
        << "  --yaw-outer-ki <deg-s-per-deg-s>\n"
        << "  --yaw-outer-kd <deg-s-per-deg-s>\n"
        << "  --yaw-inner-rate-hz <default 1000>\n"
        << "  --yaw-inner-mode <driver-speed|software-pi>\n"
        << "  --yaw-inner-kp <speed-loop kp>\n"
        << "  --yaw-inner-ki <speed-loop ki>\n"
        << "  --yaw-inner-kd <speed-loop kd>\n"
        << "  --yaw-hfov <deg>\n"
        << "  --yaw-fx-px <pixels, optional override>\n"
        << "  --yaw-cx-px <pixels, optional override>\n"
        << "  --yaw-feedback-zero-deg <motor feedback deg used as software center>\n"
        << "  --yaw-test-target-x <pixels, verification only>\n"
        << "  --t265-fisheye-index <1|2, default 1>\n"
        << "  --no-t265-pose\n"
        << "  --max-frames <N>\n"
        << "  --headless\n";
}


std::string trim_copy(const std::string& value) {
    const auto begin = std::find_if_not(value.begin(), value.end(), [](unsigned char ch) {
        return std::isspace(ch) != 0;
    });
    const auto end = std::find_if_not(value.rbegin(), value.rend(), [](unsigned char ch) {
        return std::isspace(ch) != 0;
    }).base();
    if (begin >= end) {
        return {};
    }
    return std::string(begin, end);
}

std::string strip_optional_quotes(const std::string& value) {
    if (value.size() >= 2 &&
        ((value.front() == '"' && value.back() == '"') || (value.front() == '\'' && value.back() == '\''))) {
        return value.substr(1, value.size() - 2);
    }
    return value;
}

bool parse_bool_literal(const std::string& value, bool& parsed) {
    std::string lower = value;
    std::transform(lower.begin(), lower.end(), lower.begin(), [](unsigned char ch) {
        return static_cast<char>(std::tolower(ch));
    });
    if (lower == "1" || lower == "true" || lower == "yes" || lower == "on") {
        parsed = true;
        return true;
    }
    if (lower == "0" || lower == "false" || lower == "no" || lower == "off") {
        parsed = false;
        return true;
    }
    return false;
}

void normalize_config_key(std::string& key) {
    key = trim_copy(key);
    while (key.rfind("--", 0) == 0) {
        key.erase(0, 2);
    }
    std::replace(key.begin(), key.end(), '_', '-');
}

bool is_flag_config_key(const std::string& key) {
    static const std::vector<std::string> flag_keys = {
        "swap-rb",
        "no-auto-brightness",
        "output",
        "livestream",
        "yaw-enable",
        "yaw-dry-run",
        "yaw-invert",
        "yaw-motor-invert",
        "no-t265-pose",
        "headless",
    };
    return std::find(flag_keys.begin(), flag_keys.end(), key) != flag_keys.end();
}

bool append_config_arg(const std::string& raw_line,
                       int line_number,
                       std::vector<std::string>& args,
                       std::string& error) {
    std::string line = raw_line;
    const auto comment = line.find('#');
    if (comment != std::string::npos) {
        line = line.substr(0, comment);
    }
    line = trim_copy(line);
    if (line.empty()) {
        return true;
    }

    const auto eq = line.find('=');
    if (eq == std::string::npos) {
        normalize_config_key(line);
        if (line.empty()) {
            error = "Invalid empty config key at line " + std::to_string(line_number);
            return false;
        }
        args.push_back("--" + line);
        return true;
    }

    std::string key = line.substr(0, eq);
    std::string value = strip_optional_quotes(trim_copy(line.substr(eq + 1)));
    normalize_config_key(key);
    if (key.empty()) {
        error = "Invalid empty config key at line " + std::to_string(line_number);
        return false;
    }

    bool bool_value = false;
    if (is_flag_config_key(key) && parse_bool_literal(value, bool_value)) {
        if (bool_value) {
            args.push_back("--" + key);
        }
        return true;
    }

    args.push_back("--" + key);
    args.push_back(value);
    return true;
}

bool load_config_args(const std::string& path, std::vector<std::string>& args, std::string& error) {
    std::ifstream file(path);
    if (!file.is_open()) {
        error = "Failed to open config file: " + path;
        return false;
    }

    std::string line;
    int line_number = 0;
    while (std::getline(file, line)) {
        ++line_number;
        if (!append_config_arg(line, line_number, args, error)) {
            error = path + ": " + error;
            return false;
        }
    }
    return true;
}

bool expand_config_args(int argc,
                        char** argv,
                        std::vector<std::string>& expanded_storage,
                        std::vector<char*>& expanded_argv,
                        std::string& error) {
    expanded_storage.clear();
    expanded_argv.clear();
    expanded_storage.push_back(argc > 0 ? argv[0] : "cvproj_capture");

    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        if (arg == "--config") {
            if (i + 1 >= argc) {
                error = "Missing value for --config";
                return false;
            }
            if (!load_config_args(argv[++i], expanded_storage, error)) {
                return false;
            }
            continue;
        }
        expanded_storage.push_back(arg);
    }

    expanded_argv.reserve(expanded_storage.size());
    for (auto& item : expanded_storage) {
        expanded_argv.push_back(item.data());
    }
    return true;
}

bool parse_bind_endpoint(const std::string& value, std::string& host, int& port) {
    const auto pos = value.rfind(':');
    if (pos == std::string::npos) {
        host = "0.0.0.0";
        port = std::stoi(value);
        return port > 0 && port <= 65535;
    }
    host = value.substr(0, pos);
    port = std::stoi(value.substr(pos + 1));
    if (host.empty()) {
        host = "0.0.0.0";
    }
    return port > 0 && port <= 65535;
}

bool parse_args(int argc, char** argv, AppConfig& config, std::string& error) {
    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        const auto need_value = [&](const char* name) -> const char* {
            if (i + 1 >= argc) {
                error = std::string("Missing value for ") + name;
                return nullptr;
            }
            return argv[++i];
        };

        if (arg == "--backend") {
            if (const char* value = need_value("--backend")) {
                config.backend = value;
            } else {
                return false;
            }
        } else if (arg == "--source") {
            if (const char* value = need_value("--source")) {
                config.source = value;
            } else {
                return false;
            }
        } else if (arg == "--serial") {
            if (const char* value = need_value("--serial")) {
                config.serial_number = value;
            } else {
                return false;
            }
        } else if (arg == "--pixel-format") {
            if (const char* value = need_value("--pixel-format")) {
                config.pixel_format = value;
            } else {
                return false;
            }
        } else if (arg == "--swap-rb") {
            config.swap_rb = true;
        } else if (arg == "--gain-db") {
            if (const char* value = need_value("--gain-db")) {
                config.gain_db = std::stod(value);
            } else {
                return false;
            }
        } else if (arg == "--target-luma") {
            if (const char* value = need_value("--target-luma")) {
                config.target_luma = std::stod(value);
            } else {
                return false;
            }
        } else if (arg == "--max-post-gain") {
            if (const char* value = need_value("--max-post-gain")) {
                config.max_post_gain = std::stod(value);
            } else {
                return false;
            }
        } else if (arg == "--gamma") {
            if (const char* value = need_value("--gamma")) {
                config.gamma = std::stod(value);
            } else {
                return false;
            }
        } else if (arg == "--clahe-clip") {
            if (const char* value = need_value("--clahe-clip")) {
                config.clahe_clip = std::stod(value);
            } else {
                return false;
            }
        } else if (arg == "--clahe-tile") {
            if (const char* value = need_value("--clahe-tile")) {
                config.clahe_tile = std::max(2, std::stoi(value));
            } else {
                return false;
            }
        } else if (arg == "--no-auto-brightness") {
            config.auto_brightness = false;
        } else if (arg == "--downsample-scale") {
            if (const char* value = need_value("--downsample-scale")) {
                config.downsample_scale = std::stod(value);
            } else {
                return false;
            }
        } else if (arg == "--downsample-width") {
            if (const char* value = need_value("--downsample-width")) {
                config.downsample_width = std::max(0, std::stoi(value));
            } else {
                return false;
            }
        } else if (arg == "--downsample-height") {
            if (const char* value = need_value("--downsample-height")) {
                config.downsample_height = std::max(0, std::stoi(value));
            } else {
                return false;
            }
        } else if (arg == "--denoise") {
            if (const char* value = need_value("--denoise")) {
                config.denoise_method = value;
            } else {
                return false;
            }
        } else if (arg == "--denoise-kernel") {
            if (const char* value = need_value("--denoise-kernel")) {
                config.denoise_kernel = std::max(1, std::stoi(value));
            } else {
                return false;
            }
        } else if (arg == "--denoise-sigma-color") {
            if (const char* value = need_value("--denoise-sigma-color")) {
                config.denoise_sigma_color = std::stod(value);
            } else {
                return false;
            }
        } else if (arg == "--denoise-sigma-space") {
            if (const char* value = need_value("--denoise-sigma-space")) {
                config.denoise_sigma_space = std::stod(value);
            } else {
                return false;
            }
        } else if (arg == "--denoise-passes") {
            if (const char* value = need_value("--denoise-passes")) {
                config.denoise_passes = std::clamp(std::stoi(value), 1, 2);
            } else {
                return false;
            }
        } else if (arg == "--detector") {
            if (const char* value = need_value("--detector")) {
                config.detector = value;
            } else {
                return false;
            }
        } else if (arg == "--model") {
            if (const char* value = need_value("--model")) {
                config.model_path = value;
            } else {
                return false;
            }
        } else if (arg == "--det-conf") {
            if (const char* value = need_value("--det-conf")) {
                config.det_conf = std::stof(value);
            } else {
                return false;
            }
        } else if (arg == "--det-nms") {
            if (const char* value = need_value("--det-nms")) {
                config.det_nms = std::stof(value);
            } else {
                return false;
            }
        } else if (arg == "--detect-interval") {
            if (const char* value = need_value("--detect-interval")) {
                config.detect_interval = std::max(1, std::stoi(value));
            } else {
                return false;
            }
        } else if (arg == "--det-input-size") {
            if (const char* value = need_value("--det-input-size")) {
                config.det_input_size = std::clamp(std::stoi(value), 160, 640);
            } else {
                return false;
            }
        } else if (arg == "--tracker") {
            if (const char* value = need_value("--tracker")) {
                config.tracker = value;
            } else {
                return false;
            }
        } else if (arg == "--track-high-thresh") {
            if (const char* value = need_value("--track-high-thresh")) {
                config.track_high_thresh = std::stof(value);
            } else {
                return false;
            }
        } else if (arg == "--track-low-thresh") {
            if (const char* value = need_value("--track-low-thresh")) {
                config.track_low_thresh = std::stof(value);
            } else {
                return false;
            }
        } else if (arg == "--track-match-thresh") {
            if (const char* value = need_value("--track-match-thresh")) {
                config.track_match_thresh = std::stof(value);
            } else {
                return false;
            }
        } else if (arg == "--track-buffer") {
            if (const char* value = need_value("--track-buffer")) {
                config.track_buffer = std::max(0, std::stoi(value));
            } else {
                return false;
            }
        } else if (arg == "--track-min-hits") {
            if (const char* value = need_value("--track-min-hits")) {
                config.track_min_hits = std::max(1, std::stoi(value));
            } else {
                return false;
            }
        } else if (arg == "--target-filter") {
            if (const char* value = need_value("--target-filter")) {
                config.target_filter = value;
            } else {
                return false;
            }
        } else if (arg == "--target-filter-process-noise") {
            if (const char* value = need_value("--target-filter-process-noise")) {
                config.target_filter_process_noise = std::stod(value);
            } else {
                return false;
            }
        } else if (arg == "--target-filter-measurement-noise") {
            if (const char* value = need_value("--target-filter-measurement-noise")) {
                config.target_filter_measurement_noise = std::stod(value);
            } else {
                return false;
            }
        } else if (arg == "--target-filter-max-predict-frames") {
            if (const char* value = need_value("--target-filter-max-predict-frames")) {
                config.target_filter_max_predict_frames = std::max(0, std::stoi(value));
            } else {
                return false;
            }
        } else if (arg == "--target-filter-timeout-ms") {
            if (const char* value = need_value("--target-filter-timeout-ms")) {
                config.target_filter_timeout_ms = std::stod(value);
            } else {
                return false;
            }
        } else if (arg == "--target-filter-motion-roi-gate") {
            if (const char* value = need_value("--target-filter-motion-roi-gate")) {
                config.target_filter_motion_roi_gate = std::stod(value);
            } else {
                return false;
            }
        } else if (arg == "--target-filter-min-motion-confidence") {
            if (const char* value = need_value("--target-filter-min-motion-confidence")) {
                config.target_filter_min_motion_confidence = std::stod(value);
            } else {
                return false;
            }
        } else if (arg == "--pipeline-mode") {
            if (const char* value = need_value("--pipeline-mode")) {
                config.pipeline_mode = value;
            } else {
                return false;
            }
        } else if (arg == "--async-slot-capacity") {
            if (const char* value = need_value("--async-slot-capacity")) {
                config.async_slot_capacity = std::max(1, std::stoi(value));
            } else {
                return false;
            }
        } else if (arg == "--async-log-queue-capacity") {
            if (const char* value = need_value("--async-log-queue-capacity")) {
                config.async_log_queue_capacity = std::max(1, std::stoi(value));
            } else {
                return false;
            }
        } else if (arg == "--async-control-fps") {
            if (const char* value = need_value("--async-control-fps")) {
                config.async_control_fps = std::stod(value);
            } else {
                return false;
            }
        } else if (arg == "--async-target-max-age-ms") {
            if (const char* value = need_value("--async-target-max-age-ms")) {
                config.async_target_max_age_ms = std::stod(value);
            } else {
                return false;
            }
        } else if (arg == "--async-detection-max-age-ms") {
            if (const char* value = need_value("--async-detection-max-age-ms")) {
                config.async_detection_max_age_ms = std::stod(value);
            } else {
                return false;
            }
        } else if (arg == "--async-render-fps") {
            if (const char* value = need_value("--async-render-fps")) {
                config.async_render_fps = std::stod(value);
            } else {
                return false;
            }
        } else if (arg == "--async-drop-stale-detections") {
            if (const char* value = need_value("--async-drop-stale-detections")) {
                bool parsed = true;
                if (!parse_bool_literal(value, parsed)) {
                    parsed = std::stoi(value) != 0;
                }
                config.async_drop_stale_detections = parsed;
            } else {
                return false;
            }
        } else if (arg == "--width") {
            if (const char* value = need_value("--width")) {
                config.width = std::stoi(value);
            } else {
                return false;
            }
        } else if (arg == "--height") {
            if (const char* value = need_value("--height")) {
                config.height = std::stoi(value);
            } else {
                return false;
            }
        } else if (arg == "--fps") {
            if (const char* value = need_value("--fps")) {
                config.fps = std::stod(value);
            } else {
                return false;
            }
        } else if (arg == "--exposure-us") {
            if (const char* value = need_value("--exposure-us")) {
                config.exposure_us = std::stod(value);
            } else {
                return false;
            }
        } else if (arg == "--motion-thresh") {
            if (const char* value = need_value("--motion-thresh")) {
                config.motion_threshold = std::stod(value);
            } else {
                return false;
            }
        } else if (arg == "--grid-points") {
            if (const char* value = need_value("--grid-points")) {
                config.num_motion_points = std::stoi(value);
            } else {
                return false;
            }
        } else if (arg == "--output") {
            config.output = true;
        } else if (arg == "--record") {
            if (const char* value = need_value("--record")) {
                config.record_path = value;
            } else {
                return false;
            }
        } else if (arg == "--telemetry") {
            if (const char* value = need_value("--telemetry")) {
                config.telemetry_path = value;
            } else {
                return false;
            }
        } else if (arg == "--livestream") {
            config.livestream = true;
        } else if (arg == "--http-preview") {
            if (const char* value = need_value("--http-preview")) {
                if (!parse_bind_endpoint(value, config.preview_host, config.preview_port)) {
                    error = "Invalid --http-preview endpoint: " + std::string(value);
                    return false;
                }
                config.preview = true;
            } else {
                return false;
            }
        } else if (arg == "--preview-fps") {
            if (const char* value = need_value("--preview-fps")) {
                config.preview_fps = std::stod(value);
            } else {
                return false;
            }
        } else if (arg == "--yaw-enable") {
            config.yaw_enable = true;
        } else if (arg == "--yaw-dry-run") {
            config.yaw_dry_run = true;
            config.yaw_enable = true;
        } else if (arg == "--yaw-can-iface") {
            if (const char* value = need_value("--yaw-can-iface")) {
                config.yaw_can_iface = value;
            } else {
                return false;
            }
        } else if (arg == "--yaw-id") {
            if (const char* value = need_value("--yaw-id")) {
                config.yaw_can_id = static_cast<std::uint32_t>(std::stoul(value, nullptr, 0));
            } else {
                return false;
            }
        } else if (arg == "--yaw-invert") {
            config.yaw_invert = true;
        } else if (arg == "--yaw-motor-invert") {
            config.yaw_motor_invert = true;
        } else if (arg == "--yaw-max-angle") {
            if (const char* value = need_value("--yaw-max-angle")) {
                config.yaw_max_angle_deg = std::stod(value);
            } else {
                return false;
            }
        } else if (arg == "--yaw-limit-margin-deg") {
            if (const char* value = need_value("--yaw-limit-margin-deg")) {
                config.yaw_limit_margin_deg = std::stod(value);
            } else {
                return false;
            }
        } else if (arg == "--yaw-max-rpm") {
            if (const char* value = need_value("--yaw-max-rpm")) {
                config.yaw_max_rpm = std::stod(value);
            } else {
                return false;
            }
        } else if (arg == "--yaw-kp") {
            if (const char* value = need_value("--yaw-kp")) {
                config.yaw_kp_rpm_per_deg = std::stod(value);
            } else {
                return false;
            }
        } else if (arg == "--yaw-ki") {
            if (const char* value = need_value("--yaw-ki")) {
                config.yaw_ki_rpm_per_deg_s = std::stod(value);
            } else {
                return false;
            }
        } else if (arg == "--yaw-kd") {
            if (const char* value = need_value("--yaw-kd")) {
                config.yaw_kd_rpm_per_deg_per_s = std::stod(value);
            } else {
                return false;
            }
        } else if (arg == "--yaw-deadband") {
            if (const char* value = need_value("--yaw-deadband")) {
                config.yaw_deadband_deg = std::stod(value);
            } else {
                return false;
            }
        } else if (arg == "--yaw-control-mode") {
            if (const char* value = need_value("--yaw-control-mode")) {
                config.yaw_control_mode = value;
            } else {
                return false;
            }
        } else if (arg == "--yaw-outer-rate-mode") {
            if (const char* value = need_value("--yaw-outer-rate-mode")) {
                config.yaw_outer_rate_mode = value;
            } else {
                return false;
            }
        } else if (arg == "--yaw-outer-max-stale-ms") {
            if (const char* value = need_value("--yaw-outer-max-stale-ms")) {
                config.yaw_outer_max_stale_ms = std::stod(value);
            } else {
                return false;
            }
        } else if (arg == "--yaw-outer-min-update-interval-ms") {
            if (const char* value = need_value("--yaw-outer-min-update-interval-ms")) {
                config.yaw_outer_min_update_interval_ms = std::stod(value);
            } else {
                return false;
            }
        } else if (arg == "--yaw-outer-kp") {
            if (const char* value = need_value("--yaw-outer-kp")) {
                config.yaw_outer_kp = std::stod(value);
            } else {
                return false;
            }
        } else if (arg == "--yaw-outer-ki") {
            if (const char* value = need_value("--yaw-outer-ki")) {
                config.yaw_outer_ki = std::stod(value);
            } else {
                return false;
            }
        } else if (arg == "--yaw-outer-kd") {
            if (const char* value = need_value("--yaw-outer-kd")) {
                config.yaw_outer_kd = std::stod(value);
            } else {
                return false;
            }
        } else if (arg == "--yaw-outer-ff-gain") {
            if (const char* value = need_value("--yaw-outer-ff-gain")) {
                config.yaw_outer_ff_gain = std::stod(value);
            } else {
                return false;
            }
        } else if (arg == "--yaw-outer-integral-limit-deg-s") {
            if (const char* value = need_value("--yaw-outer-integral-limit-deg-s")) {
                config.yaw_outer_integral_limit_deg_s = std::stod(value);
            } else {
                return false;
            }
        } else if (arg == "--yaw-outer-deadband-deg") {
            if (const char* value = need_value("--yaw-outer-deadband-deg")) {
                config.yaw_outer_deadband_deg = std::stod(value);
            } else {
                return false;
            }
        } else if (arg == "--yaw-inner-mode") {
            if (const char* value = need_value("--yaw-inner-mode")) {
                config.yaw_inner_mode = value;
            } else {
                return false;
            }
        } else if (arg == "--yaw-inner-rate-hz") {
            if (const char* value = need_value("--yaw-inner-rate-hz")) {
                config.yaw_inner_rate_hz = std::stod(value);
            } else {
                return false;
            }
        } else if (arg == "--yaw-inner-feedback-rate-hz") {
            if (const char* value = need_value("--yaw-inner-feedback-rate-hz")) {
                config.yaw_inner_feedback_rate_hz = std::stod(value);
            } else {
                return false;
            }
        } else if (arg == "--yaw-inner-feedback-timeout-ms") {
            if (const char* value = need_value("--yaw-inner-feedback-timeout-ms")) {
                config.yaw_inner_feedback_timeout_ms = std::stod(value);
            } else {
                return false;
            }
        } else if (arg == "--yaw-inner-realtime-priority") {
            if (const char* value = need_value("--yaw-inner-realtime-priority")) {
                bool parsed = false;
                config.yaw_inner_realtime_priority =
                    parse_bool_literal(value, parsed) ? parsed : std::stoi(value) != 0;
            } else {
                return false;
            }
        } else if (arg == "--yaw-inner-realtime-priority-value") {
            if (const char* value = need_value("--yaw-inner-realtime-priority-value")) {
                config.yaw_inner_realtime_priority_value = std::stoi(value);
            } else {
                return false;
            }
        } else if (arg == "--yaw-inner-command-timeout-ms") {
            if (const char* value = need_value("--yaw-inner-command-timeout-ms")) {
                config.yaw_inner_command_timeout_ms = std::stod(value);
            } else {
                return false;
            }
        } else if (arg == "--yaw-inner-kp") {
            if (const char* value = need_value("--yaw-inner-kp")) {
                config.yaw_inner_kp = std::stod(value);
            } else {
                return false;
            }
        } else if (arg == "--yaw-inner-ki") {
            if (const char* value = need_value("--yaw-inner-ki")) {
                config.yaw_inner_ki = std::stod(value);
            } else {
                return false;
            }
        } else if (arg == "--yaw-inner-kd") {
            if (const char* value = need_value("--yaw-inner-kd")) {
                config.yaw_inner_kd = std::stod(value);
            } else {
                return false;
            }
        } else if (arg == "--yaw-inner-integral-limit-rpm") {
            if (const char* value = need_value("--yaw-inner-integral-limit-rpm")) {
                config.yaw_inner_integral_limit_rpm = std::stod(value);
            } else {
                return false;
            }
        } else if (arg == "--yaw-inner-output-limit-rpm") {
            if (const char* value = need_value("--yaw-inner-output-limit-rpm")) {
                config.yaw_inner_output_limit_rpm = std::stod(value);
            } else {
                return false;
            }
        } else if (arg == "--yaw-inner-deadband-rpm") {
            if (const char* value = need_value("--yaw-inner-deadband-rpm")) {
                config.yaw_inner_deadband_rpm = std::stod(value);
            } else {
                return false;
            }
        } else if (arg == "--yaw-max-accel-rpm-s") {
            if (const char* value = need_value("--yaw-max-accel-rpm-s")) {
                config.yaw_max_accel_rpm_s = std::stod(value);
            } else {
                return false;
            }
        } else if (arg == "--yaw-axis-deg-per-motor-rev") {
            if (const char* value = need_value("--yaw-axis-deg-per-motor-rev")) {
                config.yaw_axis_deg_per_motor_rev = std::stod(value);
            } else {
                return false;
            }
        } else if (arg == "--yaw-feedback-required") {
            if (const char* value = need_value("--yaw-feedback-required")) {
                bool parsed = true;
                config.yaw_feedback_required =
                    parse_bool_literal(value, parsed) ? parsed : std::stoi(value) != 0;
            } else {
                return false;
            }
        } else if (arg == "--yaw-hfov") {
            if (const char* value = need_value("--yaw-hfov")) {
                config.yaw_hfov_deg = std::stod(value);
            } else {
                return false;
            }
        } else if (arg == "--yaw-fx-px") {
            if (const char* value = need_value("--yaw-fx-px")) {
                config.yaw_fx_px = std::stod(value);
            } else {
                return false;
            }
        } else if (arg == "--yaw-cx-px") {
            if (const char* value = need_value("--yaw-cx-px")) {
                config.yaw_cx_px = std::stod(value);
            } else {
                return false;
            }
        } else if (arg == "--yaw-feedback-zero-deg") {
            if (const char* value = need_value("--yaw-feedback-zero-deg")) {
                config.yaw_feedback_zero_deg = std::stod(value);
            } else {
                return false;
            }
        } else if (arg == "--yaw-test-target-x") {
            if (const char* value = need_value("--yaw-test-target-x")) {
                config.yaw_test_target_x = std::stod(value);
            } else {
                return false;
            }
        } else if (arg == "--t265-fisheye-index") {
            if (const char* value = need_value("--t265-fisheye-index")) {
                config.t265_fisheye_index = std::clamp(std::stoi(value), 1, 2);
            } else {
                return false;
            }
        } else if (arg == "--no-t265-pose") {
            config.t265_pose = false;
        } else if (arg == "--record-fps") {
            if (const char* value = need_value("--record-fps")) {
                config.record_fps = std::stod(value);
            } else {
                return false;
            }
        } else if (arg == "--max-frames") {
            if (const char* value = need_value("--max-frames")) {
                config.max_frames = std::stoi(value);
            } else {
                return false;
            }
        } else if (arg == "--headless") {
            config.headless = true;
        } else if (arg == "--help" || arg == "-h") {
            print_usage();
            return false;
        } else {
            error = "Unknown argument: " + arg;
            return false;
        }
    }

    return true;
}

std::unique_ptr<FrameSource> make_source(const AppConfig& config) {
    if (config.backend == "aravis") {
        AravisConfig aravis;
        if (!config.source.empty() && config.source != "0") {
            aravis.device_id = config.source;
        }
        aravis.serial_number = config.serial_number;
        aravis.width = config.width;
        aravis.height = config.height;
        aravis.target_fps = config.fps;
        aravis.exposure_us = config.exposure_us;
        aravis.gain_db = config.gain_db;
        aravis.pixel_format = config.pixel_format;
        return std::make_unique<AravisFrameSource>(aravis);
    }
    if (config.backend == "hikrobot") {
        HikrobotMvsConfig mvs;
        mvs.serial_number = config.serial_number;
        mvs.width = config.width;
        mvs.height = config.height;
        mvs.target_fps = config.fps;
        mvs.exposure_us = config.exposure_us;
        mvs.gain = config.gain_db;
        return std::make_unique<HikrobotMvsSource>(mvs);
    }
    if (config.backend == "t265" || config.backend == "realsense") {
        RealSenseT265Config t265;
        t265.serial_number = config.serial_number;
        t265.width = config.width;
        t265.height = config.height;
        t265.fps = config.fps;
        t265.fisheye_index = config.t265_fisheye_index;
        t265.enable_pose = config.t265_pose;
        return std::make_unique<RealSenseT265Source>(t265);
    }
    return std::make_unique<OpenCvVideoSource>(config.source, config.width, config.height, config.fps);
}


std::string make_timestamp_suffix() {
    const auto now = std::chrono::system_clock::now();
    const std::time_t now_time = std::chrono::system_clock::to_time_t(now);
    std::tm local_tm {};
#if defined(_WIN32)
    localtime_s(&local_tm, &now_time);
#else
    localtime_r(&now_time, &local_tm);
#endif
    std::ostringstream oss;
    oss << std::put_time(&local_tm, "%Y%m%d_%H%M%S");
    return oss.str();
}

void configure_auto_output_paths(AppConfig& config) {
    if (!config.output) {
        return;
    }

    const fs::path output_dir = "outputs";
    const fs::path base = output_dir / ("cvproj_" + make_timestamp_suffix());
    if (config.record_path.empty()) {
        config.record_path = (base.string() + ".mp4");
    }
    if (config.telemetry_path.empty()) {
        config.telemetry_path = (base.string() + ".targets.csv");
    }
}

fs::path make_metrics_path(const std::string& record_path) {
    fs::path path(record_path);
    if (path.extension().empty()) {
        path += ".csv";
        return path;
    }
    path.replace_extension(".csv");
    return path;
}

fs::path make_telemetry_path(const AppConfig& config) {
    if (!config.telemetry_path.empty()) {
        return fs::path(config.telemetry_path);
    }
    if (!config.record_path.empty()) {
        fs::path path(config.record_path);
        if (path.extension().empty()) {
            path += "_targets.csv";
        } else {
            path.replace_extension(".targets.csv");
        }
        return path;
    }
    return fs::path("outputs/livestream_targets.csv");
}

double rect_iou(const cv::Rect& a, const cv::Rect& b) {
    const cv::Rect overlap = a & b;
    if (overlap.area() <= 0) {
        return 0.0;
    }
    const double union_area = static_cast<double>(a.area() + b.area() - overlap.area());
    return union_area > 0.0 ? static_cast<double>(overlap.area()) / union_area : 0.0;
}

double center_distance(const cv::Rect& a, const cv::Rect& b) {
    const double ax = a.x + 0.5 * a.width;
    const double ay = a.y + 0.5 * a.height;
    const double bx = b.x + 0.5 * b.width;
    const double by = b.y + 0.5 * b.height;
    const double dx = ax - bx;
    const double dy = ay - by;
    return std::sqrt(dx * dx + dy * dy);
}

std::vector<TrackerDetection> build_target_observations(
    const MotionPipelineResult& result,
    const std::vector<Detection>& detections,
    bool allow_motion_fallback) {
    std::vector<TrackerDetection> observations;
    if (!detections.empty()) {
        observations.reserve(detections.size());
        for (const auto& det : detections) {
            observations.push_back(
                {det.box, det.confidence, det.class_id, det.class_name, "detector"});
        }
        return observations;
    }

    if (allow_motion_fallback && result.target_box.has_value()) {
        observations.push_back({*result.target_box, 1.0F, -1, "motion", "motion"});
    }
    return observations;
}

void draw_tracked_targets(cv::Mat& frame, const std::vector<TrackedTarget>& targets) {
    for (const auto& target : targets) {
        const cv::Scalar color = target.is_primary ? cv::Scalar(0, 255, 255) : cv::Scalar(255, 180, 0);
        cv::rectangle(frame, target.box, color, 2, cv::LINE_AA);
        const cv::Point target_center(target.box.x + target.box.width / 2, target.box.y + target.box.height / 2);
        if (target.is_primary) {
            const cv::Point frame_center(frame.cols / 2, frame.rows / 2);
            cv::circle(frame, target_center, 5, cv::Scalar(0, 255, 255), -1, cv::LINE_AA);
            cv::line(frame, frame_center, target_center, cv::Scalar(0, 255, 255), 1, cv::LINE_AA);
        }

    }
}

void draw_target_state(cv::Mat& frame, const TargetState& state, const MotionPipelineResult& motion) {
    if (motion.motion_roi.has_value()) {
        cv::rectangle(frame, *motion.motion_roi, cv::Scalar(0, 180, 0), 2, cv::LINE_AA);
    }
    if (!state.has_target) {
        return;
    }

    const auto draw_point = [&](const cv::Point2f& point, const cv::Scalar& color, const std::string& label) {
        if (point.x < 0.0F || point.y < 0.0F) {
            return;
        }
        const cv::Point p(cvRound(point.x), cvRound(point.y));
        cv::circle(frame, p, 5, color, -1, cv::LINE_AA);
        (void)label;
    };

    draw_point(state.raw_center, cv::Scalar(0, 255, 255), "raw");
    draw_point(state.predicted_center, cv::Scalar(255, 120, 0), "pred");
    draw_point(state.filtered_center, cv::Scalar(0, 0, 255), "filtered yaw");
}

void draw_system_status(cv::Mat& frame,
                        const std::string& mode_label,
                        bool recording,
                        bool preview,
                        const YawCommand& yaw) {
    const int x = 16;
    const int y = 114;
    cv::drawMarker(frame,
                   cv::Point(frame.cols / 2, frame.rows / 2),
                   cv::Scalar(255, 255, 255),
                   cv::MARKER_CROSS,
                   22,
                   1,
                   cv::LINE_AA);

    (void)x;
    (void)y;
    (void)mode_label;
    (void)recording;
    (void)preview;
    if (yaw.status != "disabled") {
        if (yaw.has_target && std::abs(yaw.target_yaw_deg) > 0.2) {
            const bool turn_right = yaw.target_yaw_deg < 0.0;
            const int arrow_y = std::max(50, frame.rows / 2);
            const int margin = 24;
            const int arrow_len = std::clamp(frame.cols / 5, 70, 150);
            const cv::Point tail = turn_right ? cv::Point(frame.cols - margin - arrow_len, arrow_y)
                                              : cv::Point(margin + arrow_len, arrow_y);
            const cv::Point head = turn_right ? cv::Point(frame.cols - margin, arrow_y)
                                              : cv::Point(margin, arrow_y);
            const cv::Scalar arrow_color = turn_right ? cv::Scalar(0, 180, 255) : cv::Scalar(255, 180, 0);
            cv::arrowedLine(frame, tail, head, arrow_color, 4, cv::LINE_AA, 0, 0.32);

        }
    }
}

struct RecordingOverlayStats {
    std::int64_t processed_index = 0;
    std::int64_t source_frame_id = -1;
    std::int64_t detection_frame_id = -1;
    std::int64_t tracking_frame_id = -1;
    double loop_fps = 0.0;
    double effective_fps = 0.0;
    double detection_fps = 0.0;
    double preprocess_ms = 0.0;
    double motion_ms = 0.0;
    double inference_ms = 0.0;
    double track_filter_ms = 0.0;
    double render_ms = 0.0;
    double frame_age_ms = 0.0;
    double queue_wait_ms = 0.0;
    double detection_age_ms = -1.0;
    int detection_count = 0;
    int tracker_detection_count = 0;
    int track_count = 0;
    int primary_track_id = -1;
    TargetState target_state;
    YawCommand yaw;
    std::uint64_t dropped_frames = 0;
    std::uint64_t dropped_detect_jobs = 0;
};

void draw_recording_dashboard(cv::Mat& frame, const RecordingOverlayStats& stats) {
    if (frame.empty()) {
        return;
    }

    const int panel_w = std::clamp(frame.cols * 42 / 100, 320, std::max(320, frame.cols - 24));
    const int panel_h = 282;
    const int x = std::max(12, frame.cols - panel_w - 12);
    const int y = 12;
    const cv::Rect panel(x, y, std::min(panel_w, frame.cols - x - 8), std::min(panel_h, frame.rows - y - 8));
    if (panel.width <= 0 || panel.height <= 0) {
        return;
    }

    cv::Mat overlay = frame.clone();
    cv::rectangle(overlay, panel, cv::Scalar(12, 16, 20), -1, cv::LINE_AA);
    cv::rectangle(overlay, panel, cv::Scalar(80, 120, 140), 1, cv::LINE_AA);
    cv::addWeighted(overlay, 0.88, frame, 0.12, 0.0, frame);

    int line_y = panel.y + 22;
    const int line_x = panel.x + 12;
    const int line_step = 18;
    const auto put = [&](const std::string& text, const cv::Scalar& color = cv::Scalar(230, 235, 230)) {
        if (line_y > panel.y + panel.height - 8) {
            return;
        }
        cv::putText(frame,
                    text,
                    cv::Point(line_x, line_y),
                    cv::FONT_HERSHEY_SIMPLEX,
                    0.46,
                    color,
                    1,
                    cv::LINE_AA);
        line_y += line_step;
    };
    const auto fmt1 = [](double value) {
        std::ostringstream oss;
        oss.setf(std::ios::fixed);
        oss.precision(1);
        oss << value;
        return oss.str();
    };
    const auto fmt2 = [](double value) {
        std::ostringstream oss;
        oss.setf(std::ios::fixed);
        oss.precision(2);
        oss << value;
        return oss.str();
    };
    const auto point_text = [&](const cv::Point2f& p) {
        if (p.x < 0.0F || p.y < 0.0F) {
            return std::string("none");
        }
        std::ostringstream oss;
        oss.setf(std::ios::fixed);
        oss.precision(1);
        oss << '(' << p.x << ',' << p.y << ')';
        return oss.str();
    };

    put("CV-proj telemetry", cv::Scalar(120, 230, 255));
    put("frame src=" + std::to_string(stats.source_frame_id) +
        " det=" + std::to_string(stats.detection_frame_id) +
        " trk=" + std::to_string(stats.tracking_frame_id));
    put("fps loop=" + fmt1(stats.loop_fps) +
        " cam=" + fmt1(stats.effective_fps) +
        " det=" + fmt1(stats.detection_fps), cv::Scalar(160, 255, 160));
    put("lat pre/mot/inf/trk/rnd ms " + fmt1(stats.preprocess_ms) + "/" +
        fmt1(stats.motion_ms) + "/" + fmt1(stats.inference_ms) + "/" +
        fmt1(stats.track_filter_ms) + "/" + fmt1(stats.render_ms));
    put("age frame=" + fmt1(stats.frame_age_ms) +
        " det=" + fmt1(stats.detection_age_ms) +
        " queue=" + fmt1(stats.queue_wait_ms) + " ms");
    put("drops frame=" + std::to_string(stats.dropped_frames) +
        " detect=" + std::to_string(stats.dropped_detect_jobs));

    put("det raw=" + std::to_string(stats.detection_count) +
        " used=" + std::to_string(stats.tracker_detection_count) +
        " tracks=" + std::to_string(stats.track_count) +
        " primary=" + std::to_string(stats.primary_track_id), cv::Scalar(255, 220, 130));
    put("target " + stats.target_state.source +
        " id=" + std::to_string(stats.target_state.track_id) +
        " age=" + fmt1(stats.target_state.age_ms) + "ms");
    put("raw " + point_text(stats.target_state.raw_center) +
        " filt " + point_text(stats.target_state.filtered_center));
    put("pred " + point_text(stats.target_state.predicted_center) +
        " vel=(" + fmt1(stats.target_state.velocity_px_s.x) + "," +
        fmt1(stats.target_state.velocity_px_s.y) + ")px/s");

    put("yaw " + stats.yaw.status +
        " alpha=" + fmt2(stats.yaw.alpha_deg) +
        " cmd=" + fmt2(stats.yaw.commanded_rpm_inner) +
        "rpm", cv::Scalar(255, 180, 150));
    put("yaw cur=" + fmt2(stats.yaw.current_yaw_deg) +
        " meas=" + fmt2(stats.yaw.measured_rpm) +
        "rpm lim=" + (stats.yaw.limited ? "1" : "0"));
    put("loop outer=" + fmt1(stats.yaw.outer_update_hz) +
        "Hz inner=" + fmt1(stats.yaw.inner_loop_hz) +
        "Hz fb=" + fmt1(stats.yaw.feedback_hz) + "Hz");
    put("seq outer=" + std::to_string(stats.yaw.outer_seq) +
        " jitter=" + fmt1(stats.yaw.inner_jitter_us) +
        "us fb_age=" + fmt1(stats.yaw.feedback_age_ms) + "ms");
}

cv::Mat apply_gamma_curve(const cv::Mat& input, double gamma) {
    if (std::abs(gamma - 1.0) < 1e-3) {
        return input.clone();
    }

    cv::Mat lut(1, 256, CV_8UC1);
    for (int i = 0; i < 256; ++i) {
        const double normalized = static_cast<double>(i) / 255.0;
        const double corrected = std::pow(normalized, gamma);
        lut.at<unsigned char>(i) = static_cast<unsigned char>(std::clamp(corrected * 255.0, 0.0, 255.0));
    }

    cv::Mat output;
    cv::LUT(input, lut, output);
    return output;
}




cv::Mat swap_red_blue(const cv::Mat& frame_bgr, const AppConfig& config) {
    if (frame_bgr.empty()) {
        return frame_bgr;
    }
    if (!config.swap_rb || frame_bgr.channels() != 3) {
        return frame_bgr.clone();
    }

    cv::Mat swapped;
    cv::cvtColor(frame_bgr, swapped, cv::COLOR_BGR2RGB);
    return swapped;
}

cv::Mat downsample_frame(const cv::Mat& frame_bgr, const AppConfig& config) {
    if (frame_bgr.empty()) {
        return frame_bgr;
    }

    int target_width = 0;
    int target_height = 0;
    if (config.downsample_width > 0 && config.downsample_height > 0) {
        target_width = config.downsample_width;
        target_height = config.downsample_height;
    } else if (config.downsample_scale > 0.0 && config.downsample_scale < 0.999) {
        const double scale = std::clamp(config.downsample_scale, 0.05, 1.0);
        target_width = std::max(1, static_cast<int>(std::lround(frame_bgr.cols * scale)));
        target_height = std::max(1, static_cast<int>(std::lround(frame_bgr.rows * scale)));
    } else {
        return frame_bgr.clone();
    }

    target_width = std::clamp(target_width, 1, frame_bgr.cols);
    target_height = std::clamp(target_height, 1, frame_bgr.rows);
    if (target_width == frame_bgr.cols && target_height == frame_bgr.rows) {
        return frame_bgr.clone();
    }

    cv::Mat downsampled;
    cv::resize(frame_bgr,
               downsampled,
               cv::Size(target_width, target_height),
               0.0,
               0.0,
               cv::INTER_AREA);
    return downsampled;
}

cv::Mat denoise_frame(const cv::Mat& frame_bgr, const AppConfig& config) {
    if (frame_bgr.empty()) {
        return frame_bgr;
    }

    std::string method = config.denoise_method;
    std::transform(method.begin(), method.end(), method.begin(), [](unsigned char ch) {
        return static_cast<char>(std::tolower(ch));
    });

    if (method.empty() || method == "off" || method == "none" || method == "disable") {
        return frame_bgr.clone();
    }

    int kernel = std::max(1, config.denoise_kernel);
    if (kernel % 2 == 0) {
        ++kernel;
    }
    kernel = std::clamp(kernel, 3, 15);

    cv::Mat denoised;
    if (method == "gaussian") {
        cv::GaussianBlur(frame_bgr, denoised, cv::Size(kernel, kernel), 0.0);
    } else if (method == "median") {
        cv::medianBlur(frame_bgr, denoised, std::min(kernel, 7));
    } else if (method == "bilateral") {
        cv::bilateralFilter(frame_bgr,
                            denoised,
                            std::min(kernel, 9),
                            std::max(1.0, config.denoise_sigma_color),
                            std::max(1.0, config.denoise_sigma_space));
    } else {
        return frame_bgr.clone();
    }

    return denoised;
}

cv::Mat enhance_for_visibility(const cv::Mat& frame_bgr, const AppConfig& config) {
    if (frame_bgr.empty()) {
        return frame_bgr;
    }
    if (!config.auto_brightness && std::abs(config.gamma - 1.0) < 1e-3) {
        return frame_bgr.clone();
    }

    cv::Mat lab;
    cv::cvtColor(frame_bgr, lab, cv::COLOR_BGR2Lab);

    std::vector<cv::Mat> channels;
    cv::split(lab, channels);

    cv::Mat enhanced_luma = channels[0].clone();
    if (config.clahe_clip > 0.0) {
        const int tile = std::max(2, config.clahe_tile);
        auto clahe = cv::createCLAHE(config.clahe_clip, cv::Size(tile, tile));
        clahe->apply(channels[0], enhanced_luma);
    }

    if (config.auto_brightness) {
        const double mean_luma = cv::mean(enhanced_luma)[0];
        if (mean_luma > 1.0) {
            const double scale = std::clamp(config.target_luma / mean_luma, 1.0, std::max(1.0, config.max_post_gain));
            enhanced_luma.convertTo(enhanced_luma, CV_8U, scale);
        }
    }

    if (config.gamma > 0.0 && std::abs(config.gamma - 1.0) > 1e-3) {
        enhanced_luma = apply_gamma_curve(enhanced_luma, config.gamma);
    }

    channels[0] = enhanced_luma;
    cv::merge(channels, lab);

    cv::Mat enhanced_bgr;
    cv::cvtColor(lab, enhanced_bgr, cv::COLOR_Lab2BGR);
    return enhanced_bgr;
}

template <typename T>
class LatestSlot {
public:
    explicit LatestSlot(std::size_t capacity) : capacity_(std::max<std::size_t>(1, capacity)) {}

    bool store(T value, std::atomic<std::uint64_t>* dropped_counter = nullptr) {
        std::lock_guard<std::mutex> lock(mutex_);
        if (stopped_) {
            return false;
        }
        if (queue_.size() >= capacity_) {
            queue_.pop_front();
            if (dropped_counter != nullptr) {
                dropped_counter->fetch_add(1, std::memory_order_relaxed);
            }
        }
        queue_.push_back(std::move(value));
        cv_.notify_one();
        return true;
    }

    bool wait_and_take_latest(T& out, int timeout_ms) {
        std::unique_lock<std::mutex> lock(mutex_);
        if (!cv_.wait_for(lock, std::chrono::milliseconds(timeout_ms), [&]() {
                return stopped_ || !queue_.empty();
            })) {
            return false;
        }
        if (queue_.empty()) {
            return false;
        }
        out = std::move(queue_.back());
        queue_.clear();
        return true;
    }

    void stop() {
        std::lock_guard<std::mutex> lock(mutex_);
        stopped_ = true;
        cv_.notify_all();
    }

private:
    std::size_t capacity_ = 1;
    std::deque<T> queue_;
    bool stopped_ = false;
    std::mutex mutex_;
    std::condition_variable cv_;
};

struct AsyncFramePacket {
    FramePacket packet;
    std::chrono::steady_clock::time_point capture_wall_time;
};

struct AsyncDetectionJob {
    cv::Mat frame_bgr;
    std::optional<cv::Rect> roi;
    std::optional<cv::Rect> motion_roi;
    cv::Size frame_size;
    std::chrono::steady_clock::time_point capture_wall_time;
    double capture_ts_ms = 0.0;
    std::int64_t frame_id = -1;
};

struct AsyncDetectionResult {
    std::vector<Detection> detections;
    double detection_ms = 0.0;
    double capture_ts_ms = 0.0;
    std::chrono::steady_clock::time_point capture_wall_time;
    cv::Size frame_size;
    std::optional<cv::Rect> roi;
    std::optional<cv::Rect> motion_roi;
    std::int64_t frame_id = -1;
    std::uint64_t detection_seq = 0;
};

double steady_ms_since_epoch(std::chrono::steady_clock::time_point t) {
    return std::chrono::duration<double, std::milli>(t.time_since_epoch()).count();
}

double elapsed_ms(std::chrono::steady_clock::time_point start,
                  std::chrono::steady_clock::time_point end) {
    return std::chrono::duration<double, std::milli>(end - start).count();
}

}  // namespace cvproj

int main(int argc, char** argv) {
    std::vector<std::string> expanded_args_storage;
    std::vector<char*> expanded_argv;
    std::string parse_error;
    if (!cvproj::expand_config_args(argc, argv, expanded_args_storage, expanded_argv, parse_error)) {
        std::cerr << parse_error << '\n';
        return 1;
    }

    cvproj::AppConfig config;
    if (!cvproj::parse_args(static_cast<int>(expanded_argv.size()), expanded_argv.data(), config, parse_error)) {
        if (!parse_error.empty()) {
            std::cerr << parse_error << '\n';
        }
        if (!parse_error.empty()) {
            cvproj::print_usage();
        }
        return parse_error.empty() ? 0 : 1;
    }

    std::signal(SIGINT, handle_stop_signal);
    std::signal(SIGTERM, handle_stop_signal);

    cvproj::configure_auto_output_paths(config);

    auto source = cvproj::make_source(config);
    std::string error;
    if (!source->open(&error)) {
        std::cerr << "Failed to open source " << source->name() << ": " << error << '\n';
        return 2;
    }
    cvproj::CameraIntrinsics source_intrinsics;
    const bool have_source_intrinsics = source->camera_intrinsics(source_intrinsics);
    if (have_source_intrinsics) {
        std::cout << "Source intrinsics: "
                  << source_intrinsics.width << "x" << source_intrinsics.height
                  << " fx=" << source_intrinsics.fx
                  << " fy=" << source_intrinsics.fy
                  << " cx=" << source_intrinsics.cx
                  << " cy=" << source_intrinsics.cy
                  << " model=" << source_intrinsics.model << '\n';
    }

    std::unique_ptr<cvproj::YoloOnnxDetector> detector;
    if (config.detector == "yolo") {
        const float detector_output_conf =
            config.tracker == "bytetrack" ? std::min(config.det_conf, config.track_low_thresh) : config.det_conf;
        detector = std::make_unique<cvproj::YoloOnnxDetector>(
            config.model_path, config.det_input_size, detector_output_conf, config.det_nms);
        if (!detector->open(&error)) {
            std::cerr << "Failed to open detector model " << config.model_path << ": " << error << '\n';
            return 4;
        }
        std::cout << "Detector: yolo model=" << config.model_path
                  << " input=" << config.det_input_size
                  << " conf=" << detector_output_conf
                  << " control_conf=" << config.det_conf << '\n';
    }

    cvproj::MotionPipelineConfig pipeline_config;
    pipeline_config.num_motion_points = config.num_motion_points;
    pipeline_config.motion_threshold = config.motion_threshold;
    cvproj::MotionPipeline pipeline(pipeline_config);
    cvproj::ByteTrackConfig tracker_config;
    tracker_config.high_threshold = config.track_high_thresh;
    tracker_config.low_threshold = config.track_low_thresh;
    tracker_config.match_threshold = config.track_match_thresh;
    tracker_config.track_buffer = config.track_buffer;
    tracker_config.min_hits = config.track_min_hits;
    cvproj::ByteTrackTracker target_tracker(tracker_config);
    if (config.tracker == "bytetrack") {
        const auto& active_tracker_config = target_tracker.config();
        std::cout << "Tracker: bytetrack"
                  << " high=" << active_tracker_config.high_threshold
                  << " low=" << active_tracker_config.low_threshold
                  << " match=" << active_tracker_config.match_threshold
                  << " buffer=" << active_tracker_config.track_buffer
                  << " min_hits=" << active_tracker_config.min_hits << '\n';
    } else {
        std::cout << "Tracker: none" << '\n';
    }
    cvproj::TargetStateFilterConfig target_filter_config;
    target_filter_config.mode = config.target_filter;
    target_filter_config.process_noise = config.target_filter_process_noise;
    target_filter_config.measurement_noise = config.target_filter_measurement_noise;
    target_filter_config.max_predict_frames = config.target_filter_max_predict_frames;
    target_filter_config.timeout_ms = config.target_filter_timeout_ms;
    target_filter_config.motion_roi_gate = config.target_filter_motion_roi_gate;
    target_filter_config.min_motion_confidence = config.target_filter_min_motion_confidence;
    cvproj::TargetStateFilter target_filter(target_filter_config);
    const auto& active_filter_config = target_filter.config();
    std::cout << "Target filter: " << active_filter_config.mode
              << " q=" << active_filter_config.process_noise
              << " r=" << active_filter_config.measurement_noise
              << " max_predict=" << active_filter_config.max_predict_frames
              << " timeout_ms=" << active_filter_config.timeout_ms
              << " roi_gate=" << active_filter_config.motion_roi_gate
              << " min_motion_conf=" << active_filter_config.min_motion_confidence << '\n';

    cvproj::MjpegServer preview_server;
    if (config.preview) {
        if (!preview_server.start(config.preview_host, config.preview_port, &error)) {
            std::cerr << "Failed to start HTTP preview: " << error << '\n';
            return 5;
        }
        std::cout << "HTTP preview: " << preview_server.url() << '\n';
    }

    std::unique_ptr<cvproj::SocketCanYawController> yaw_controller;
    cvproj::YawCommand last_yaw_command;
    if (config.yaw_enable) {
        cvproj::SocketCanYawConfig yaw_config;
        yaw_config.iface = config.yaw_can_iface;
        yaw_config.can_id = config.yaw_can_id;
        yaw_config.dry_run = config.yaw_dry_run;
        yaw_config.invert = config.yaw_invert;
        yaw_config.motor_invert = config.yaw_motor_invert;
        yaw_config.max_angle_deg = config.yaw_max_angle_deg;
        yaw_config.limit_margin_deg = config.yaw_limit_margin_deg;
        yaw_config.max_rpm = config.yaw_max_rpm;
        yaw_config.kp_rpm_per_deg = config.yaw_kp_rpm_per_deg;
        yaw_config.ki_rpm_per_deg_s = config.yaw_ki_rpm_per_deg_s;
        yaw_config.kd_rpm_per_deg_per_s = config.yaw_kd_rpm_per_deg_per_s;
        yaw_config.hold_deadband_deg = config.yaw_deadband_deg;
        yaw_config.control_mode = config.yaw_control_mode;
        yaw_config.outer_rate_mode = config.yaw_outer_rate_mode;
        yaw_config.outer_max_stale_ms = config.yaw_outer_max_stale_ms;
        yaw_config.outer_min_update_interval_ms = config.yaw_outer_min_update_interval_ms;
        yaw_config.outer_kp_deg_s_per_deg = config.yaw_outer_kp;
        yaw_config.outer_ki_deg_s_per_deg_s = config.yaw_outer_ki;
        yaw_config.outer_kd_deg_s_per_deg_s = config.yaw_outer_kd;
        yaw_config.outer_ff_gain = config.yaw_outer_ff_gain;
        yaw_config.outer_integral_limit_deg_s = config.yaw_outer_integral_limit_deg_s;
        yaw_config.outer_deadband_deg = config.yaw_outer_deadband_deg;
        yaw_config.inner_mode = config.yaw_inner_mode;
        yaw_config.inner_rate_hz = config.yaw_inner_rate_hz;
        yaw_config.inner_feedback_rate_hz = config.yaw_inner_feedback_rate_hz;
        yaw_config.inner_feedback_timeout_ms = config.yaw_inner_feedback_timeout_ms;
        yaw_config.inner_realtime_priority = config.yaw_inner_realtime_priority;
        yaw_config.inner_realtime_priority_value = config.yaw_inner_realtime_priority_value;
        yaw_config.inner_command_timeout_ms = config.yaw_inner_command_timeout_ms;
        yaw_config.inner_kp = config.yaw_inner_kp;
        yaw_config.inner_ki = config.yaw_inner_ki;
        yaw_config.inner_kd = config.yaw_inner_kd;
        yaw_config.inner_integral_limit_rpm = config.yaw_inner_integral_limit_rpm;
        yaw_config.inner_output_limit_rpm = config.yaw_inner_output_limit_rpm;
        yaw_config.inner_deadband_rpm = config.yaw_inner_deadband_rpm;
        yaw_config.max_accel_rpm_s = config.yaw_max_accel_rpm_s;
        yaw_config.axis_deg_per_motor_rev = config.yaw_axis_deg_per_motor_rev;
        yaw_config.feedback_required = config.yaw_feedback_required;
        yaw_config.horizontal_fov_deg = config.yaw_hfov_deg;
        yaw_config.focal_x_px = config.yaw_fx_px;
        yaw_config.center_x_px = config.yaw_cx_px;
        yaw_config.feedback_zero_deg = config.yaw_feedback_zero_deg;
        if (yaw_config.focal_x_px <= 0.0 && have_source_intrinsics && source_intrinsics.fx > 0.0) {
            yaw_config.focal_x_px = source_intrinsics.fx;
            yaw_config.center_x_px = source_intrinsics.cx;
            config.yaw_fx_px = yaw_config.focal_x_px;
            config.yaw_cx_px = yaw_config.center_x_px;
            std::cout << "Yaw pixel mapping: using source intrinsics fx="
                      << yaw_config.focal_x_px << " cx=" << yaw_config.center_x_px << '\n';
        }
        yaw_controller = std::make_unique<cvproj::SocketCanYawController>(yaw_config);
        if (!yaw_controller->open(&error)) {
            std::cerr << "Failed to open yaw controller: " << error << '\n';
            return 6;
        }
        last_yaw_command.status = config.yaw_dry_run ? "dry-run-ready" : "ready";
        std::cout << "Yaw controller: " << yaw_controller->name() << '\n';
    }

    cv::VideoWriter writer;
    std::ofstream metrics_file;
    std::ofstream telemetry_file;
    const bool livestream_enabled = config.livestream;
    const bool record_enabled = config.output || !config.record_path.empty();
    const bool telemetry_enabled = record_enabled || !config.telemetry_path.empty();
    const std::string window_name = "CV-proj-2026 C++ Motion Pipeline";
    const double output_record_fps = config.record_fps > 0.0 ? config.record_fps : 15.0;
    const fs::path telemetry_path = cvproj::make_telemetry_path(config);
    if (record_enabled) {
        std::cout << "Output video: " << config.record_path << '\n';
        std::cout << "Metrics CSV: " << cvproj::make_metrics_path(config.record_path).string() << '\n';
    }
    if (telemetry_enabled) {
        std::cout << "Targets CSV: " << telemetry_path.string() << '\n';
    }
    const bool async_pipeline = config.pipeline_mode == "async";
    std::cout << "Pipeline: " << (async_pipeline ? "async" : "serial");
    if (async_pipeline) {
        std::cout << " slot_capacity=" << config.async_slot_capacity
                  << " log_queue_capacity=" << config.async_log_queue_capacity
                  << " control_fps=" << config.async_control_fps
                  << " render_fps=" << config.async_render_fps
                  << " target_max_age_ms=" << config.async_target_max_age_ms
                  << " detection_max_age_ms=" << config.async_detection_max_age_ms
                  << " drop_stale_detections=" << (config.async_drop_stale_detections ? "true" : "false");
    }
    std::cout << '\n';

    std::int64_t processed_frames = 0;
    double last_report_fps = 0.0;
    double loop_fps_ema = 0.0;
    double effective_fps_ema = 0.0;
    double last_detection_ms = 0.0;
    double detection_fps_ema = 0.0;
    std::int64_t last_detection_frame_id = -1;
    std::int64_t active_detection_frame_id = -1;
    std::int64_t tracking_frame_id = -1;
    std::uint64_t last_detection_seq = 0;
    std::uint64_t last_tracker_detection_seq = 0;
    std::uint64_t last_yaw_outer_detection_seq = 0;
    double detection_age_ms = -1.0;
    std::chrono::steady_clock::time_point last_detection_capture_wall_time;
    std::vector<cvproj::Detection> last_detections;
    std::chrono::steady_clock::time_point last_preview_publish;
    std::chrono::steady_clock::time_point last_record_write;
    std::vector<cvproj::TrackedTarget> tracked_targets;
    cvproj::TargetState target_state;
    std::atomic<bool> async_running{async_pipeline};
    std::atomic<std::uint64_t> dropped_frames{0};
    std::atomic<std::uint64_t> dropped_detect_jobs{0};
    std::atomic<std::uint64_t> dropped_log_frames{0};
    cvproj::LatestSlot<cvproj::AsyncFramePacket> latest_frame_slot(
        static_cast<std::size_t>(std::max(1, config.async_slot_capacity)));
    cvproj::LatestSlot<cvproj::AsyncDetectionJob> latest_detect_slot(
        static_cast<std::size_t>(std::max(1, config.async_slot_capacity)));
    std::mutex async_detection_mutex;
    cvproj::AsyncDetectionResult async_detection_result;
    std::atomic<std::uint64_t> async_detection_seq_counter{0};
    std::thread capture_thread;
    std::thread detection_thread;
    if (async_pipeline) {
        capture_thread = std::thread([&]() {
            while (async_running.load(std::memory_order_relaxed) && !stop_requested()) {
                cvproj::FramePacket captured;
                std::string capture_error;
                if (!source->read_latest(captured, 2000, &capture_error)) {
                    if (!capture_error.empty()) {
                        std::cerr << "Capture thread: " << capture_error << '\n';
                    }
                    continue;
                }
                latest_frame_slot.store(
                    cvproj::AsyncFramePacket{std::move(captured), std::chrono::steady_clock::now()},
                    &dropped_frames);
            }
            latest_frame_slot.stop();
        });
        if (detector) {
            detection_thread = std::thread([&]() {
                while (async_running.load(std::memory_order_relaxed) && !stop_requested()) {
                    cvproj::AsyncDetectionJob job;
                    if (!latest_detect_slot.wait_and_take_latest(job, 2000)) {
                        continue;
                    }
                    const auto detect_start = std::chrono::steady_clock::now();
                    auto detections = detector->detect(job.frame_bgr, job.roi);
                    const double detection_ms = cvproj::elapsed_ms(detect_start, std::chrono::steady_clock::now());
                    {
                        std::lock_guard<std::mutex> lock(async_detection_mutex);
                        async_detection_result.detections = std::move(detections);
                        async_detection_result.detection_ms = detection_ms;
                        async_detection_result.frame_id = job.frame_id;
                        async_detection_result.capture_ts_ms = job.capture_ts_ms;
                        async_detection_result.capture_wall_time = job.capture_wall_time;
                        async_detection_result.frame_size = job.frame_size;
                        async_detection_result.roi = job.roi;
                        async_detection_result.motion_roi = job.motion_roi;
                        async_detection_result.detection_seq =
                            async_detection_seq_counter.fetch_add(1, std::memory_order_relaxed) + 1;
                    }
                }
            });
        }
    }

    while (!stop_requested() && (config.max_frames < 0 || processed_frames < config.max_frames)) {
        const auto loop_start = std::chrono::steady_clock::now();
        cvproj::FramePacket packet;
        auto capture_wall_time = loop_start;
        error.clear();
        if (async_pipeline) {
            cvproj::AsyncFramePacket async_frame;
            if (!latest_frame_slot.wait_and_take_latest(async_frame, 2000)) {
                std::cerr << "Stopping: async frame timeout" << '\n';
                break;
            }
            packet = std::move(async_frame.packet);
            capture_wall_time = async_frame.capture_wall_time;
        } else {
            if (!source->read_latest(packet, 2000, &error)) {
                std::cerr << "Stopping: " << error << '\n';
                break;
            }
            capture_wall_time = std::chrono::steady_clock::now();
        }
        const auto acquired_wall_time = std::chrono::steady_clock::now();
        const double queue_wait_ms = cvproj::elapsed_ms(capture_wall_time, acquired_wall_time);
        const double capture_ts_ms = packet.timestamp_seconds > 0.0
                                         ? packet.timestamp_seconds * 1000.0
                                         : cvproj::steady_ms_since_epoch(capture_wall_time);

        const auto preprocess_start = std::chrono::steady_clock::now();
        packet.frame_bgr = cvproj::swap_red_blue(packet.frame_bgr, config);
        packet.frame_bgr = cvproj::downsample_frame(packet.frame_bgr, config);
        packet.frame_bgr = cvproj::denoise_frame(packet.frame_bgr, config);
        packet.frame_bgr = cvproj::enhance_for_visibility(packet.frame_bgr, config);
        if (config.denoise_passes > 1) {
            packet.frame_bgr = cvproj::denoise_frame(packet.frame_bgr, config);
        }
        const auto preprocess_end = std::chrono::steady_clock::now();
        const double preprocess_ms = cvproj::elapsed_ms(preprocess_start, preprocess_end);

        const auto motion_start = std::chrono::steady_clock::now();
        auto result = pipeline.process(packet);
        const auto motion_end = std::chrono::steady_clock::now();
        const double motion_ms = cvproj::elapsed_ms(motion_start, motion_end);
        last_report_fps = result.fps;
        if (result.source_fps > 0.0) {
            effective_fps_ema = effective_fps_ema > 0.0 ? 0.9 * effective_fps_ema + 0.1 * result.source_fps
                                                         : result.source_fps;
        }

        if (detector && (processed_frames % config.detect_interval == 0)) {
            if (async_pipeline) {
                latest_detect_slot.store(
                    cvproj::AsyncDetectionJob{
                        packet.frame_bgr.clone(),
                        result.roi,
                        result.motion_roi,
                        packet.frame_bgr.size(),
                        capture_wall_time,
                        capture_ts_ms,
                        packet.frame_id},
                    &dropped_detect_jobs);
            } else {
                const auto detect_start = std::chrono::steady_clock::now();
                last_detections = detector->detect(packet.frame_bgr, result.roi);
                last_detection_ms = std::chrono::duration<double, std::milli>(
                                        std::chrono::steady_clock::now() - detect_start)
                                        .count();
                last_detection_frame_id = packet.frame_id;
                active_detection_frame_id = packet.frame_id;
                last_detection_seq += 1;
                last_detection_capture_wall_time = capture_wall_time;
                detection_age_ms = 0.0;
                const double detection_fps = last_detection_ms > 0.0 ? 1000.0 / last_detection_ms : 0.0;
                detection_fps_ema = detection_fps_ema > 0.0 ? 0.9 * detection_fps_ema + 0.1 * detection_fps
                                                            : detection_fps;
            }
        }
        if (async_pipeline && detector) {
            std::lock_guard<std::mutex> lock(async_detection_mutex);
            if (async_detection_result.detection_seq != 0 &&
                async_detection_result.detection_seq != last_detection_seq) {
                last_detections = async_detection_result.detections;
                last_detection_ms = async_detection_result.detection_ms;
                last_detection_frame_id = async_detection_result.frame_id;
                active_detection_frame_id = async_detection_result.frame_id;
                last_detection_seq = async_detection_result.detection_seq;
                last_detection_capture_wall_time = async_detection_result.capture_wall_time;
                detection_age_ms = cvproj::elapsed_ms(async_detection_result.capture_wall_time,
                                                      std::chrono::steady_clock::now());
            } else if (last_detection_frame_id >= 0) {
                detection_age_ms = cvproj::elapsed_ms(last_detection_capture_wall_time,
                                                      std::chrono::steady_clock::now());
            }
            const double detection_fps = last_detection_ms > 0.0 ? 1000.0 / last_detection_ms : 0.0;
            if (detection_fps > 0.0) {
                detection_fps_ema = detection_fps_ema > 0.0 ? 0.9 * detection_fps_ema + 0.1 * detection_fps
                                                            : detection_fps;
            }
        }
        const double inference_ms = last_detection_ms;

        const auto track_filter_start = std::chrono::steady_clock::now();
        tracking_frame_id = packet.frame_id;
        const bool has_new_detection_for_tracker =
            !detector || (last_detection_seq != 0 && last_detection_seq != last_tracker_detection_seq);
        const bool detection_is_stale =
            detector && detection_age_ms >= 0.0 && detection_age_ms > config.async_detection_max_age_ms;
        const std::vector<cvproj::Detection> tracker_detections =
            (has_new_detection_for_tracker && !detection_is_stale) ? last_detections
                                                                   : std::vector<cvproj::Detection>{};
        const auto observations = cvproj::build_target_observations(result, tracker_detections, !detector);
        const bool tracker_has_fresh_detection_tick = !detector || has_new_detection_for_tracker;
        if (has_new_detection_for_tracker) {
            last_tracker_detection_seq = last_detection_seq;
        }
        tracked_targets = target_tracker.update(
            observations, packet.frame_bgr.cols, packet.frame_bgr.rows, tracker_has_fresh_detection_tick);
        cvproj::draw_tracked_targets(result.annotated_frame, tracked_targets);

        std::string yaw_error;
        std::string yaw_target_source = "hold";
        auto primary = std::find_if(tracked_targets.begin(),
                                    tracked_targets.end(),
                                    [](const cvproj::TrackedTarget& target) { return target.is_primary; });
        std::optional<cvproj::TrackedTarget> primary_target;
        if (primary != tracked_targets.end()) {
            primary_target = *primary;
        }
        const double control_dt =
            result.source_delta_seconds > 0.0 ? result.source_delta_seconds : 1.0 / std::max(1.0, config.fps);
        target_state = target_filter.update(
            primary_target, result, packet.frame_bgr.size(), control_dt);
        if (async_pipeline && target_state.has_target && target_state.age_ms > config.async_target_max_age_ms) {
            target_state.has_target = false;
            target_state.source = "stale";
        }
        const auto track_filter_end = std::chrono::steady_clock::now();
        const double track_filter_ms = cvproj::elapsed_ms(track_filter_start, track_filter_end);
        cvproj::draw_target_state(result.annotated_frame, target_state, result);
        if (stop_requested()) {
        std::cout << "Stop requested; finalizing outputs..." << '\n';
    }

    if (yaw_controller) {
            double control_ts_ms = cvproj::steady_ms_since_epoch(std::chrono::steady_clock::now());
            if (config.yaw_test_target_x >= 0.0) {
                last_yaw_command = yaw_controller->update_from_target(
                    config.yaw_test_target_x, packet.frame_bgr.cols, control_dt, &yaw_error);
                last_yaw_command.status += "-test-target";
                yaw_target_source = "test-target";
            } else if (target_state.has_target) {
                const double target_cx = target_state.filtered_center.x;
                const cv::Point yaw_target_center(cvRound(target_state.filtered_center.x),
                                                  cvRound(target_state.filtered_center.y));
                cv::circle(result.annotated_frame, yaw_target_center, 7, cv::Scalar(0, 0, 255), 2, cv::LINE_AA);
                const bool has_new_inference_target =
                    last_detection_seq != 0 && last_detection_seq != last_yaw_outer_detection_seq &&
                    !detection_is_stale;
                if (has_new_inference_target || config.detector != "yolo") {
                    constexpr double kPi = 3.14159265358979323846;
                    const double yaw_cx = config.yaw_cx_px >= 0.0 ? config.yaw_cx_px
                                                                  : static_cast<double>(packet.frame_bgr.cols) * 0.5;
                    const double yaw_fx = config.yaw_fx_px > 0.0
                                              ? config.yaw_fx_px
                                              : yaw_cx / std::tan((config.yaw_hfov_deg * kPi / 180.0) * 0.5);
                    const double dx_left_positive = yaw_cx - target_cx;
                    const double alpha_deg = std::atan2(dx_left_positive, yaw_fx) * 180.0 / kPi;
                    const double alpha_rate_deg_s =
                        (-yaw_fx * static_cast<double>(target_state.velocity_px_s.x) /
                         (yaw_fx * yaw_fx + dx_left_positive * dx_left_positive)) *
                        180.0 / kPi;
                    last_yaw_command = yaw_controller->update_from_visual_error(
                        alpha_deg,
                        alpha_rate_deg_s,
                        target_state.age_ms,
                        active_detection_frame_id,
                        control_dt,
                        &yaw_error);
                    last_yaw_outer_detection_seq = last_detection_seq;
                } else {
                    last_yaw_command = yaw_controller->snapshot();
                }
                yaw_target_source = "filter:" + target_state.source + ":" + std::to_string(target_state.track_id);
            } else {
                last_yaw_command = yaw_controller->hold(control_dt, &yaw_error);
            }
            control_ts_ms = cvproj::steady_ms_since_epoch(std::chrono::steady_clock::now());
            (void)control_ts_ms;
        }
        (void)detector;

        const auto loop_end = std::chrono::steady_clock::now();
        const double loop_ms =
            std::chrono::duration<double, std::milli>(loop_end - loop_start).count();
        const double loop_fps = loop_ms > 0.0 ? 1000.0 / loop_ms : 0.0;
        loop_fps_ema = loop_fps_ema > 0.0 ? 0.9 * loop_fps_ema + 0.1 * loop_fps : loop_fps;

        cvproj::draw_system_status(result.annotated_frame,
                                   livestream_enabled ? "livestream" : "capture",
                                   record_enabled,
                                   config.preview,
                                   last_yaw_command);
        const auto render_done = std::chrono::steady_clock::now();
        const double render_ms = cvproj::elapsed_ms(track_filter_end, render_done);
        const double frame_age_ms = cvproj::elapsed_ms(capture_wall_time, render_done);
        const double control_ts_ms = cvproj::steady_ms_since_epoch(render_done);
        const auto overlay_primary = std::find_if(tracked_targets.begin(),
                                                  tracked_targets.end(),
                                                  [](const cvproj::TrackedTarget& track) {
                                                      return track.is_primary;
                                                  });
        cvproj::RecordingOverlayStats overlay_stats;
        overlay_stats.processed_index = processed_frames;
        overlay_stats.source_frame_id = packet.frame_id;
        overlay_stats.detection_frame_id = active_detection_frame_id;
        overlay_stats.tracking_frame_id = tracking_frame_id;
        overlay_stats.loop_fps = loop_fps_ema;
        overlay_stats.effective_fps = effective_fps_ema;
        overlay_stats.detection_fps = detection_fps_ema;
        overlay_stats.preprocess_ms = preprocess_ms;
        overlay_stats.motion_ms = motion_ms;
        overlay_stats.inference_ms = inference_ms;
        overlay_stats.track_filter_ms = track_filter_ms;
        overlay_stats.render_ms = render_ms;
        overlay_stats.frame_age_ms = frame_age_ms;
        overlay_stats.queue_wait_ms = queue_wait_ms;
        overlay_stats.detection_age_ms = detection_age_ms;
        overlay_stats.detection_count = static_cast<int>(last_detections.size());
        overlay_stats.tracker_detection_count = static_cast<int>(tracker_detections.size());
        overlay_stats.track_count = static_cast<int>(tracked_targets.size());
        overlay_stats.primary_track_id = overlay_primary != tracked_targets.end() ? overlay_primary->track_id : -1;
        overlay_stats.target_state = target_state;
        overlay_stats.yaw = last_yaw_command;
        overlay_stats.dropped_frames = dropped_frames.load(std::memory_order_relaxed);
        overlay_stats.dropped_detect_jobs = dropped_detect_jobs.load(std::memory_order_relaxed);
        cvproj::draw_recording_dashboard(result.annotated_frame, overlay_stats);

        if (record_enabled && !writer.isOpened()) {
            const fs::path record_path(config.record_path);
            if (record_path.has_parent_path()) {
                fs::create_directories(record_path.parent_path());
            }
            writer.open(config.record_path,
                        cv::VideoWriter::fourcc('m', 'p', '4', 'v'),
                        output_record_fps,
                        result.annotated_frame.size());
            if (!writer.isOpened()) {
                std::cerr << "Failed to open output video: " << config.record_path << '\n';
                break;
            }

            const fs::path metrics_path = cvproj::make_metrics_path(config.record_path);
            metrics_file.open(metrics_path);
            if (metrics_file.is_open()) {
                metrics_file
                    << "processed_index,source_frame_id,source_timestamp_s,source_delta_ms,effective_fps,"
                    << "pipeline_fps,loop_fps,processing_ms,detection_ms,detection_fps,roi_x,roi_y,roi_w,roi_h,target_x,target_y,"
                    << "target_w,target_h,target_cx,target_cy,detection_count,tracker_detection_count,"
                    << "primary_track_id,primary_track_state,primary_track_missed_frames,yaw_target_source,"
                    << "raw_cx,raw_cy,filtered_cx,filtered_cy,predicted_cx,predicted_cy,"
                    << "target_vx_px_s,target_vy_px_s,target_filter_source,target_age_ms,"
                    << "motion_confidence,global_dx,global_dy,"
                    << "capture_ts_ms,preprocess_ms,motion_ms,inference_ms,track_filter_ms,render_ms,"
                    << "control_ts_ms,frame_age_ms,target_age_ms_async,queue_wait_ms,"
                    << "detection_frame_id,tracking_frame_id,detection_age_ms,"
                    << "dropped_frames,dropped_detect_jobs,dropped_log_frames,"
                    << "yaw_has_target,yaw_target_deg,yaw_current_deg,yaw_error_deg,yaw_rpm,yaw_limited,yaw_status,"
                    << "outer_seq,outer_update_hz,outer_dt_ms,inner_loop_hz,inner_dt_us,inner_jitter_us,"
                    << "feedback_age_ms,feedback_hz,desired_rpm_outer,commanded_rpm_inner,measured_rpm,"
                    << "speed_error_rpm,yaw_alpha_deg,yaw_alpha_rate_deg_s,yaw_omega_axis_deg_s,"
                    << "missed_feedback_count,yaw_accel_limited\n";
                metrics_file << std::fixed << std::setprecision(6);
            }
        }

        if (telemetry_enabled && !telemetry_file.is_open()) {
            if (telemetry_path.has_parent_path()) {
                fs::create_directories(telemetry_path.parent_path());
            }
            telemetry_file.open(telemetry_path);
            if (!telemetry_file.is_open()) {
                std::cerr << "Failed to open telemetry output: " << telemetry_path << '\n';
                break;
            }
            telemetry_file << std::fixed << std::setprecision(6);
            telemetry_file
                << "processed_index,source_frame_id,source_timestamp_s,source_delta_ms,effective_fps,"
                << "target_index,track_id,is_primary,source,class_id,class_name,confidence,"
                << "track_state,track_age,track_hits,track_missed_frames,"
                << "x,y,w,h,cx,cy,norm_cx,norm_cy,offset_x,offset_y,roi_x,roi_y,roi_w,roi_h,yaw_target_source,"
                << "raw_cx,raw_cy,filtered_cx,filtered_cy,predicted_cx,predicted_cy,"
                << "target_vx_px_s,target_vy_px_s,target_filter_source,target_age_ms,"
                << "motion_confidence,global_dx,global_dy,"
                << "capture_ts_ms,preprocess_ms,motion_ms,inference_ms,track_filter_ms,render_ms,"
                << "control_ts_ms,frame_age_ms,target_age_ms_async,queue_wait_ms,"
                << "detection_frame_id,tracking_frame_id,detection_age_ms,"
                << "dropped_frames,dropped_detect_jobs,dropped_log_frames,"
                << "yaw_has_target,yaw_target_deg,yaw_current_deg,yaw_error_deg,yaw_rpm,yaw_limited,yaw_status,"
                << "outer_seq,outer_update_hz,outer_dt_ms,inner_loop_hz,inner_dt_us,inner_jitter_us,"
                << "feedback_age_ms,feedback_hz,desired_rpm_outer,commanded_rpm_inner,measured_rpm,"
                << "speed_error_rpm,yaw_alpha_deg,yaw_alpha_rate_deg_s,yaw_omega_axis_deg_s,"
                << "missed_feedback_count,yaw_accel_limited\n";
        }

        if (preview_server.running()) {
            const auto preview_now = std::chrono::steady_clock::now();
            const double preview_interval_ms =
                config.preview_fps > 0.0 ? 1000.0 / config.preview_fps : 0.0;
            const bool should_publish_preview =
                preview_interval_ms <= 0.0 || last_preview_publish.time_since_epoch().count() == 0 ||
                std::chrono::duration<double, std::milli>(preview_now - last_preview_publish).count() >=
                    preview_interval_ms;
            if (should_publish_preview) {
                preview_server.publish(result.annotated_frame);
                last_preview_publish = preview_now;
            }
        }

        if (writer.isOpened()) {
            const auto record_now = std::chrono::steady_clock::now();
            const double record_interval_ms = output_record_fps > 0.0 ? 1000.0 / output_record_fps : 0.0;
            const bool should_write_record =
                record_interval_ms <= 0.0 || last_record_write.time_since_epoch().count() == 0 ||
                std::chrono::duration<double, std::milli>(record_now - last_record_write).count() >=
                    record_interval_ms;
            if (should_write_record) {
                writer.write(result.annotated_frame);
                last_record_write = record_now;
            }
        }

        if (metrics_file.is_open()) {
            const auto roi = result.roi.value_or(cv::Rect(-1, -1, -1, -1));
            const auto target = result.target_box.value_or(cv::Rect(-1, -1, -1, -1));
            const int target_cx = target.x >= 0 ? target.x + target.width / 2 : -1;
            const int target_cy = target.y >= 0 ? target.y + target.height / 2 : -1;
            const auto primary = std::find_if(tracked_targets.begin(),
                                              tracked_targets.end(),
                                              [](const cvproj::TrackedTarget& track) { return track.is_primary; });
            const int primary_track_id = primary != tracked_targets.end() ? primary->track_id : -1;
            const std::string primary_track_state = primary != tracked_targets.end() ? primary->state : "none";
            const int primary_track_missed =
                primary != tracked_targets.end() ? primary->missed_frames : -1;
            metrics_file << processed_frames << ',' << packet.frame_id << ','
                         << result.source_timestamp_seconds << ','
                         << (result.source_delta_seconds * 1000.0) << ','
                         << effective_fps_ema << ',' << result.fps << ',' << loop_fps_ema << ','
                         << result.processing_ms << ',' << last_detection_ms << ',' << detection_fps_ema << ','
                         << roi.x << ',' << roi.y << ',' << roi.width
                         << ',' << roi.height << ',' << target.x << ',' << target.y << ','
                         << target.width << ',' << target.height << ',' << target_cx << ','
                         << target_cy << ',' << last_detections.size() << ',' << tracker_detections.size() << ','
                         << primary_track_id << ',' << primary_track_state << ','
                         << primary_track_missed << ',' << yaw_target_source << ','
                         << target_state.raw_center.x << ',' << target_state.raw_center.y << ','
                         << target_state.filtered_center.x << ',' << target_state.filtered_center.y << ','
                         << target_state.predicted_center.x << ',' << target_state.predicted_center.y << ','
                         << target_state.velocity_px_s.x << ',' << target_state.velocity_px_s.y << ','
                         << target_state.source << ',' << target_state.age_ms << ','
                         << result.motion_confidence << ',' << result.global_dx << ',' << result.global_dy << ','
                         << capture_ts_ms << ',' << preprocess_ms << ',' << motion_ms << ','
                         << inference_ms << ',' << track_filter_ms << ',' << render_ms << ','
                         << control_ts_ms << ',' << frame_age_ms << ',' << target_state.age_ms << ','
                         << queue_wait_ms << ','
                         << active_detection_frame_id << ','
                         << tracking_frame_id << ','
                         << detection_age_ms << ','
                         << dropped_frames.load(std::memory_order_relaxed) << ','
                         << dropped_detect_jobs.load(std::memory_order_relaxed) << ','
                         << dropped_log_frames.load(std::memory_order_relaxed) << ','
                         << (last_yaw_command.has_target ? 1 : 0) << ','
                         << last_yaw_command.target_yaw_deg << ','
                         << last_yaw_command.current_yaw_deg << ','
                         << last_yaw_command.error_deg << ','
                         << last_yaw_command.rpm << ','
                         << (last_yaw_command.limited ? 1 : 0) << ','
                         << last_yaw_command.status << ','
                         << last_yaw_command.outer_seq << ','
                         << last_yaw_command.outer_update_hz << ','
                         << last_yaw_command.outer_dt_ms << ','
                         << last_yaw_command.inner_loop_hz << ','
                         << last_yaw_command.inner_dt_us << ','
                         << last_yaw_command.inner_jitter_us << ','
                         << last_yaw_command.feedback_age_ms << ','
                         << last_yaw_command.feedback_hz << ','
                         << last_yaw_command.desired_rpm_outer << ','
                         << last_yaw_command.commanded_rpm_inner << ','
                         << last_yaw_command.measured_rpm << ','
                         << last_yaw_command.speed_error_rpm << ','
                         << last_yaw_command.alpha_deg << ','
                         << last_yaw_command.alpha_rate_deg_s << ','
                         << last_yaw_command.omega_axis_deg_s << ','
                         << last_yaw_command.missed_feedback_count << ','
                         << (last_yaw_command.accel_limited ? 1 : 0) << '\n';
        }

        if (telemetry_file.is_open()) {
            const auto roi = result.roi.value_or(cv::Rect(-1, -1, -1, -1));
            const double frame_cx = packet.frame_bgr.cols * 0.5;
            const double frame_cy = packet.frame_bgr.rows * 0.5;
            if (tracked_targets.empty()) {
                telemetry_file << processed_frames << ',' << packet.frame_id << ','
                               << result.source_timestamp_seconds << ','
                               << (result.source_delta_seconds * 1000.0) << ','
                               << effective_fps_ema
                               << ",-1,-1,0,none,-1,none,0.0,none,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,"
                               << roi.x << ',' << roi.y << ',' << roi.width << ',' << roi.height
                               << ',' << yaw_target_source
                               << ',' << target_state.raw_center.x << ',' << target_state.raw_center.y
                               << ',' << target_state.filtered_center.x << ',' << target_state.filtered_center.y
                               << ',' << target_state.predicted_center.x << ',' << target_state.predicted_center.y
                               << ',' << target_state.velocity_px_s.x << ',' << target_state.velocity_px_s.y
                               << ',' << target_state.source << ',' << target_state.age_ms
                               << ',' << result.motion_confidence << ',' << result.global_dx << ',' << result.global_dy
                               << ',' << capture_ts_ms << ',' << preprocess_ms << ',' << motion_ms
                               << ',' << inference_ms << ',' << track_filter_ms << ',' << render_ms
                               << ',' << control_ts_ms << ',' << frame_age_ms << ',' << target_state.age_ms
                               << ',' << queue_wait_ms
                               << ',' << active_detection_frame_id
                               << ',' << tracking_frame_id
                               << ',' << detection_age_ms
                               << ',' << dropped_frames.load(std::memory_order_relaxed)
                               << ',' << dropped_detect_jobs.load(std::memory_order_relaxed)
                               << ',' << dropped_log_frames.load(std::memory_order_relaxed)
                               << ',' << (last_yaw_command.has_target ? 1 : 0)
                               << ',' << last_yaw_command.target_yaw_deg
                               << ',' << last_yaw_command.current_yaw_deg
                               << ',' << last_yaw_command.error_deg
                               << ',' << last_yaw_command.rpm
                               << ',' << (last_yaw_command.limited ? 1 : 0)
                               << ',' << last_yaw_command.status
                               << ',' << last_yaw_command.outer_seq
                               << ',' << last_yaw_command.outer_update_hz
                               << ',' << last_yaw_command.outer_dt_ms
                               << ',' << last_yaw_command.inner_loop_hz
                               << ',' << last_yaw_command.inner_dt_us
                               << ',' << last_yaw_command.inner_jitter_us
                               << ',' << last_yaw_command.feedback_age_ms
                               << ',' << last_yaw_command.feedback_hz
                               << ',' << last_yaw_command.desired_rpm_outer
                               << ',' << last_yaw_command.commanded_rpm_inner
                               << ',' << last_yaw_command.measured_rpm
                               << ',' << last_yaw_command.speed_error_rpm
                               << ',' << last_yaw_command.alpha_deg
                               << ',' << last_yaw_command.alpha_rate_deg_s
                               << ',' << last_yaw_command.omega_axis_deg_s
                               << ',' << last_yaw_command.missed_feedback_count
                               << ',' << (last_yaw_command.accel_limited ? 1 : 0)
                               << '\n';
            } else {
                for (const auto& target : tracked_targets) {
                    const int cx = target.box.x + target.box.width / 2;
                    const int cy = target.box.y + target.box.height / 2;
                    const double norm_cx =
                        packet.frame_bgr.cols > 0 ? static_cast<double>(cx) / packet.frame_bgr.cols : -1.0;
                    const double norm_cy =
                        packet.frame_bgr.rows > 0 ? static_cast<double>(cy) / packet.frame_bgr.rows : -1.0;
                    const double offset_x = cx - frame_cx;
                    const double offset_y = cy - frame_cy;

                    telemetry_file << processed_frames << ',' << packet.frame_id << ','
                                   << result.source_timestamp_seconds << ','
                                   << (result.source_delta_seconds * 1000.0) << ','
                                   << effective_fps_ema << ',' << target.target_index << ','
                                   << target.track_id << ',' << (target.is_primary ? 1 : 0) << ','
                                   << target.source << ',' << target.class_id << ','
                                   << target.class_name << ',' << target.confidence << ','
                                   << target.state << ',' << target.age << ',' << target.hits << ','
                                   << target.missed_frames << ','
                                   << target.box.x << ',' << target.box.y << ',' << target.box.width
                                   << ',' << target.box.height << ',' << cx << ',' << cy << ','
                                   << norm_cx << ',' << norm_cy << ',' << offset_x << ',' << offset_y
                                   << ',' << roi.x << ',' << roi.y << ',' << roi.width << ','
                                   << roi.height
                                   << ',' << yaw_target_source
                                   << ',' << target_state.raw_center.x << ',' << target_state.raw_center.y
                                   << ',' << target_state.filtered_center.x << ',' << target_state.filtered_center.y
                                   << ',' << target_state.predicted_center.x << ',' << target_state.predicted_center.y
                                   << ',' << target_state.velocity_px_s.x << ',' << target_state.velocity_px_s.y
                                   << ',' << target_state.source << ',' << target_state.age_ms
                                   << ',' << result.motion_confidence << ',' << result.global_dx << ',' << result.global_dy
                                   << ',' << capture_ts_ms << ',' << preprocess_ms << ',' << motion_ms
                                   << ',' << inference_ms << ',' << track_filter_ms << ',' << render_ms
                                   << ',' << control_ts_ms << ',' << frame_age_ms << ',' << target_state.age_ms
                                   << ',' << queue_wait_ms
                                   << ',' << active_detection_frame_id
                                   << ',' << tracking_frame_id
                                   << ',' << detection_age_ms
                                   << ',' << dropped_frames.load(std::memory_order_relaxed)
                                   << ',' << dropped_detect_jobs.load(std::memory_order_relaxed)
                                   << ',' << dropped_log_frames.load(std::memory_order_relaxed)
                                   << ',' << (last_yaw_command.has_target ? 1 : 0)
                                   << ',' << last_yaw_command.target_yaw_deg
                                   << ',' << last_yaw_command.current_yaw_deg
                                   << ',' << last_yaw_command.error_deg
                                   << ',' << last_yaw_command.rpm
                                   << ',' << (last_yaw_command.limited ? 1 : 0)
                                   << ',' << last_yaw_command.status
                                   << ',' << last_yaw_command.outer_seq
                                   << ',' << last_yaw_command.outer_update_hz
                                   << ',' << last_yaw_command.outer_dt_ms
                                   << ',' << last_yaw_command.inner_loop_hz
                                   << ',' << last_yaw_command.inner_dt_us
                                   << ',' << last_yaw_command.inner_jitter_us
                                   << ',' << last_yaw_command.feedback_age_ms
                                   << ',' << last_yaw_command.feedback_hz
                                   << ',' << last_yaw_command.desired_rpm_outer
                                   << ',' << last_yaw_command.commanded_rpm_inner
                                   << ',' << last_yaw_command.measured_rpm
                                   << ',' << last_yaw_command.speed_error_rpm
                                   << ',' << last_yaw_command.alpha_deg
                                   << ',' << last_yaw_command.alpha_rate_deg_s
                                   << ',' << last_yaw_command.omega_axis_deg_s
                                   << ',' << last_yaw_command.missed_feedback_count
                                   << ',' << (last_yaw_command.accel_limited ? 1 : 0) << '\n';
                }
            }
        }

        if (!config.headless) {
            cv::imshow(window_name, result.annotated_frame);
            const int key = cv::waitKey(1);
            if (key == 'q' || key == 27) {
                break;
            }
        }

        ++processed_frames;
    }

    if (async_pipeline) {
        async_running.store(false, std::memory_order_relaxed);
        latest_frame_slot.stop();
        latest_detect_slot.stop();
        if (capture_thread.joinable()) {
            capture_thread.join();
        }
        if (detection_thread.joinable()) {
            detection_thread.join();
        }
    }

    if (yaw_controller) {
        yaw_controller->stop();
        yaw_controller->close();
    }
    preview_server.stop();
    source->close();
    if (writer.isOpened()) {
        writer.release();
    }
    if (metrics_file.is_open()) {
        metrics_file.close();
    }
    if (telemetry_file.is_open()) {
        telemetry_file.close();
    }
    if (!config.headless) {
        cv::destroyAllWindows();
    }

    std::cout << "Processed frames: " << processed_frames << '\n'
              << "Last reported pipeline FPS: " << last_report_fps << '\n'
              << "Last reported end-to-end loop FPS: " << loop_fps_ema << '\n'
              << "Last reported effective FPS: " << effective_fps_ema << '\n'
              << "Last reported detection FPS: " << detection_fps_ema << '\n'
              << "Telemetry: "
              << (telemetry_enabled ? telemetry_path.string() : std::string("disabled")) << '\n'
              << "HTTP preview: "
              << (config.preview ? preview_server.url() : std::string("disabled")) << '\n'
              << "Yaw controller: "
              << (yaw_controller ? yaw_controller->name() : std::string("disabled")) << '\n'
              << "Backend: " << source->name() << '\n';

    return processed_frames > 0 ? 0 : 3;
}
