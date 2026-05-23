#include "central_control/common.hpp"

#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <stdexcept>

#include <opencv2/core.hpp>

namespace central_control {

namespace {

bool parse_bool_flag(const std::string& value) {
    return value == "1" || value == "true" || value == "True" || value == "TRUE" || value == "on";
}

template <typename T>
void assign_if_has_node(const cv::FileNode& node, T& target) {
    if (!node.empty()) {
        node >> target;
    }
}

std::string read_cli_value(int& index, int argc, char** argv) {
    if (index + 1 >= argc) {
        return {};
    }
    ++index;
    return argv[index];
}

}  // namespace

bool load_config_file(const std::filesystem::path& path, AppConfig& config, std::string& error) {
    if (path.empty()) {
        return true;
    }
    if (!std::filesystem::exists(path)) {
        error = "config file not found: " + path.string();
        return false;
    }

    cv::FileStorage fs(path.string(), cv::FileStorage::READ);
    if (!fs.isOpened()) {
        error = "failed to open config file: " + path.string();
        return false;
    }

    assign_if_has_node(fs["source"], config.source);
    assign_if_has_node(fs["backend"], config.backend);
    assign_if_has_node(fs["model_onnx"], config.model_onnx);
    assign_if_has_node(fs["model_rknn"], config.model_rknn);
    assign_if_has_node(fs["output_dir"], config.output_dir);
    assign_if_has_node(fs["output_prefix"], config.output_prefix);
    assign_if_has_node(fs["can_channel"], config.can_channel);
    assign_if_has_node(fs["mvs_serial"], config.mvs_serial);
    assign_if_has_node(fs["window_name"], config.window_name);
    assign_if_has_node(fs["target_fps"], config.target_fps);
    assign_if_has_node(fs["duration_sec"], config.duration_sec);
    assign_if_has_node(fs["input_width"], config.input_width);
    assign_if_has_node(fs["input_height"], config.input_height);
    assign_if_has_node(fs["yolo_input_size"], config.yolo_input_size);
    assign_if_has_node(fs["motion_points"], config.motion_points);
    assign_if_has_node(fs["blur_kernel"], config.blur_kernel);
    assign_if_has_node(fs["mvs_index"], config.mvs_index);
    assign_if_has_node(fs["motion_threshold"], config.motion_threshold);
    assign_if_has_node(fs["roi_conf_threshold"], config.roi_conf_threshold);
    assign_if_has_node(fs["bg_conf_threshold"], config.bg_conf_threshold);
    assign_if_has_node(fs["nms_iou"], config.nms_iou);
    assign_if_has_node(fs["mvs_exposure_us"], config.mvs_exposure_us);
    assign_if_has_node(fs["mvs_gain"], config.mvs_gain);
    assign_if_has_node(fs["mvs_frame_rate"], config.mvs_frame_rate);
    assign_if_has_node(fs["yaw_zero"], config.yaw_zero);
    assign_if_has_node(fs["pitch_zero"], config.pitch_zero);
    assign_if_has_node(fs["yaw_rpm_per_deg"], config.yaw_rpm_per_deg);
    assign_if_has_node(fs["pitch_rpm_per_deg"], config.pitch_rpm_per_deg);
    assign_if_has_node(fs["h_fov"], config.h_fov);
    assign_if_has_node(fs["v_fov"], config.v_fov);
    assign_if_has_node(fs["headless"], config.headless);
    assign_if_has_node(fs["record_rendered"], config.record_rendered);
    assign_if_has_node(fs["mock_gimbal"], config.mock_gimbal);
    assign_if_has_node(fs["use_yolo"], config.use_yolo);
    return true;
}

AppConfig parse_arguments(int argc, char** argv) {
    AppConfig config;

    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        if (arg == "--config") {
            config.config_path = read_cli_value(i, argc, argv);
        }
    }

    if (!config.config_path.empty()) {
        std::string error;
        if (!load_config_file(config.config_path, config, error)) {
            throw std::runtime_error(error);
        }
    }

    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        if (arg == "--config") {
            ++i;
        } else if (arg == "--source") {
            config.source = read_cli_value(i, argc, argv);
        } else if (arg == "--backend") {
            config.backend = read_cli_value(i, argc, argv);
        } else if (arg == "--model-onnx") {
            config.model_onnx = read_cli_value(i, argc, argv);
        } else if (arg == "--model-rknn") {
            config.model_rknn = read_cli_value(i, argc, argv);
        } else if (arg == "--output-dir") {
            config.output_dir = read_cli_value(i, argc, argv);
        } else if (arg == "--output-prefix") {
            config.output_prefix = read_cli_value(i, argc, argv);
        } else if (arg == "--duration-sec") {
            config.duration_sec = std::stoi(read_cli_value(i, argc, argv));
        } else if (arg == "--target-fps") {
            config.target_fps = std::stoi(read_cli_value(i, argc, argv));
        } else if (arg == "--headless") {
            config.headless = true;
        } else if (arg == "--show-window") {
            config.headless = false;
        } else if (arg == "--record-rendered") {
            config.record_rendered = true;
        } else if (arg == "--no-record") {
            config.record_rendered = false;
        } else if (arg == "--mock-gimbal") {
            config.mock_gimbal = true;
        } else if (arg == "--real-gimbal") {
            config.mock_gimbal = false;
        } else if (arg == "--use-yolo") {
            config.use_yolo = true;
        } else if (arg == "--disable-yolo") {
            config.use_yolo = false;
        } else if (arg == "--list-sources") {
            config.list_sources = true;
        } else if (arg == "--can-channel") {
            config.can_channel = read_cli_value(i, argc, argv);
        } else if (arg == "--mvs-index") {
            config.mvs_index = std::stoi(read_cli_value(i, argc, argv));
        } else if (arg == "--mvs-serial") {
            config.mvs_serial = read_cli_value(i, argc, argv);
        } else if (arg == "--mvs-exposure-us") {
            config.mvs_exposure_us = std::stof(read_cli_value(i, argc, argv));
        } else if (arg == "--mvs-gain") {
            config.mvs_gain = std::stof(read_cli_value(i, argc, argv));
        } else if (arg == "--mvs-frame-rate") {
            config.mvs_frame_rate = std::stof(read_cli_value(i, argc, argv));
        } else if (arg == "--motion-threshold") {
            config.motion_threshold = std::stof(read_cli_value(i, argc, argv));
        } else if (arg == "--roi-conf-threshold") {
            config.roi_conf_threshold = std::stof(read_cli_value(i, argc, argv));
        } else if (arg == "--bg-conf-threshold") {
            config.bg_conf_threshold = std::stof(read_cli_value(i, argc, argv));
        } else if (arg == "--nms-iou") {
            config.nms_iou = std::stof(read_cli_value(i, argc, argv));
        } else if (arg == "--input-width") {
            config.input_width = std::stoi(read_cli_value(i, argc, argv));
        } else if (arg == "--input-height") {
            config.input_height = std::stoi(read_cli_value(i, argc, argv));
        } else if (arg == "--yolo-input-size") {
            config.yolo_input_size = std::stoi(read_cli_value(i, argc, argv));
        } else if (arg == "--motion-points") {
            config.motion_points = std::stoi(read_cli_value(i, argc, argv));
        } else if (arg == "--blur-kernel") {
            config.blur_kernel = std::stoi(read_cli_value(i, argc, argv));
        } else if (arg == "--h-fov") {
            config.h_fov = std::stof(read_cli_value(i, argc, argv));
        } else if (arg == "--v-fov") {
            config.v_fov = std::stof(read_cli_value(i, argc, argv));
        } else if (arg == "--yaw-zero") {
            config.yaw_zero = std::stof(read_cli_value(i, argc, argv));
        } else if (arg == "--pitch-zero") {
            config.pitch_zero = std::stof(read_cli_value(i, argc, argv));
        } else if (arg == "--yaw-rpm-per-deg") {
            config.yaw_rpm_per_deg = std::stof(read_cli_value(i, argc, argv));
        } else if (arg == "--pitch-rpm-per-deg") {
            config.pitch_rpm_per_deg = std::stof(read_cli_value(i, argc, argv));
        } else if (arg == "--record-rendered=true") {
            config.record_rendered = true;
        } else if (arg == "--record-rendered=false") {
            config.record_rendered = false;
        } else if (arg == "--headless=true") {
            config.headless = true;
        } else if (arg == "--headless=false") {
            config.headless = false;
        } else if (arg.rfind("--mock-gimbal=", 0) == 0) {
            config.mock_gimbal = parse_bool_flag(arg.substr(14));
        }
    }

    config.blur_kernel = (config.blur_kernel % 2 == 0) ? (config.blur_kernel + 1) : config.blur_kernel;
    config.target_fps = std::max(1, config.target_fps);
    config.duration_sec = std::max(1, config.duration_sec);
    config.yolo_input_size = std::max(32, config.yolo_input_size);
    config.motion_points = std::max(16, config.motion_points);
    return config;
}

std::string make_timestamped_output_path(const AppConfig& config) {
    const auto now = std::chrono::system_clock::now();
    const std::time_t tt = std::chrono::system_clock::to_time_t(now);
    std::tm tm_value{};
#if defined(_WIN32)
    localtime_s(&tm_value, &tt);
#else
    localtime_r(&tt, &tm_value);
#endif
    std::ostringstream oss;
    oss << config.output_prefix << "_" << std::put_time(&tm_value, "%Y%m%d_%H%M%S") << ".mp4";
    std::filesystem::path out_dir(config.output_dir);
    return (out_dir / oss.str()).string();
}

void ControlCenter::publish_sensor(const SensorSnapshot& snapshot) {
    std::lock_guard<std::mutex> lock(mutex_);
    sensors_.push_back(snapshot);
    if (sensors_.size() > 128) {
        sensors_.erase(sensors_.begin(), sensors_.begin() + 64);
    }
}

void ControlCenter::publish_target(const std::optional<TargetObservation>& observation) {
    std::lock_guard<std::mutex> lock(mutex_);
    latest_target_ = observation;
}

void ControlCenter::dispatch(const ControlCommand& command) {
    std::shared_ptr<IActuator> actuator;
    {
        std::lock_guard<std::mutex> lock(mutex_);
        latest_command_ = command;
        actuator = actuator_;
    }
    if (actuator) {
        actuator->apply(command);
    }
}

void ControlCenter::register_actuator(std::shared_ptr<IActuator> actuator) {
    std::lock_guard<std::mutex> lock(mutex_);
    actuator_ = std::move(actuator);
}

void ControlCenter::stop_all() {
    std::shared_ptr<IActuator> actuator;
    {
        std::lock_guard<std::mutex> lock(mutex_);
        actuator = actuator_;
    }
    if (actuator) {
        actuator->stop();
    }
}

}  // namespace central_control
