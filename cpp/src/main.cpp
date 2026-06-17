#include <algorithm>
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
#include <optional>
#include <sstream>
#include <string>
#include <vector>

#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/videoio.hpp>

#include "cvproj/aravis_frame_source.hpp"
#include "cvproj/frame_source.hpp"
#include "cvproj/hikrobot_mvs_source.hpp"
#include "cvproj/mjpeg_server.hpp"
#include "cvproj/motion_pipeline.hpp"
#include "cvproj/opencv_video_source.hpp"
#include "cvproj/socketcan_gimbal.hpp"
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
    std::string model_path = "src/yolo11n.onnx";
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
    double yaw_max_angle_deg = 10.0;
    double yaw_max_rpm = 120.0;
    double yaw_kp_rpm_per_deg = 12.0;
    double yaw_deadband_deg = 0.25;
    double yaw_hfov_deg = 70.0;
};

void print_usage() {
    std::cout
        << "Usage: cvproj_capture [options]\n"
        << "  --config <file.conf>\n"
        << "  --backend <opencv|hikrobot|aravis>\n"
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
        << "  --yaw-max-angle <deg, default 10>\n"
        << "  --yaw-max-rpm <rpm>\n"
        << "  --yaw-kp <rpm-per-deg>\n"
        << "  --yaw-deadband <deg>\n"
        << "  --yaw-hfov <deg>\n"
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
        } else if (arg == "--yaw-max-angle") {
            if (const char* value = need_value("--yaw-max-angle")) {
                config.yaw_max_angle_deg = std::stod(value);
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
        } else if (arg == "--yaw-deadband") {
            if (const char* value = need_value("--yaw-deadband")) {
                config.yaw_deadband_deg = std::stod(value);
            } else {
                return false;
            }
        } else if (arg == "--yaw-hfov") {
            if (const char* value = need_value("--yaw-hfov")) {
                config.yaw_hfov_deg = std::stod(value);
            } else {
                return false;
            }
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

struct TargetObservation {
    cv::Rect box;
    float confidence = 0.0F;
    int class_id = -1;
    std::string class_name = "motion";
    std::string source = "motion";
};

struct TrackedTarget {
    int target_index = -1;
    int track_id = -1;
    bool is_primary = false;
    cv::Rect box;
    float confidence = 0.0F;
    int class_id = -1;
    std::string class_name;
    std::string source;
};

class TargetTracker {
public:
    std::vector<TrackedTarget> update(const std::vector<TargetObservation>& observations) {
        for (auto& track : tracks_) {
            ++track.missed_frames;
        }

        std::vector<bool> track_taken(tracks_.size(), false);
        std::vector<TrackedTarget> result;
        result.reserve(observations.size());

        for (std::size_t i = 0; i < observations.size(); ++i) {
            const auto& obs = observations[i];
            int best_index = -1;
            double best_score = -1.0;

            for (std::size_t t = 0; t < tracks_.size(); ++t) {
                if (track_taken[t]) {
                    continue;
                }

                const double iou = rect_iou(obs.box, tracks_[t].box);
                const double dist = center_distance(obs.box, tracks_[t].box);
                const double max_dist =
                    std::max(80.0, 0.75 * std::sqrt(static_cast<double>(std::max(1, obs.box.area()))));
                if (iou < 0.10 && dist > max_dist) {
                    continue;
                }

                const double score = iou * 1000.0 - dist;
                if (score > best_score) {
                    best_score = score;
                    best_index = static_cast<int>(t);
                }
            }

            TrackState* track = nullptr;
            if (best_index >= 0) {
                track_taken[best_index] = true;
                track = &tracks_[best_index];
            } else {
                tracks_.push_back({next_track_id_++, obs.box, obs.class_name, obs.source, 0});
                track_taken.push_back(true);
                track = &tracks_.back();
            }

            track->box = obs.box;
            track->class_name = obs.class_name;
            track->source = obs.source;
            track->missed_frames = 0;

            TrackedTarget tracked;
            tracked.target_index = static_cast<int>(i);
            tracked.track_id = track->track_id;
            tracked.is_primary = i == 0;
            tracked.box = obs.box;
            tracked.confidence = obs.confidence;
            tracked.class_id = obs.class_id;
            tracked.class_name = obs.class_name;
            tracked.source = obs.source;
            result.push_back(std::move(tracked));
        }

        tracks_.erase(std::remove_if(tracks_.begin(),
                                     tracks_.end(),
                                     [](const TrackState& track) { return track.missed_frames > 8; }),
                      tracks_.end());
        return result;
    }

private:
    struct TrackState {
        int track_id = -1;
        cv::Rect box;
        std::string class_name;
        std::string source;
        int missed_frames = 0;
    };

    int next_track_id_ = 1;
    std::vector<TrackState> tracks_;
};

std::vector<TargetObservation> build_target_observations(
    const MotionPipelineResult& result,
    const std::vector<Detection>& detections) {
    std::vector<TargetObservation> observations;
    if (!detections.empty()) {
        observations.reserve(detections.size());
        for (const auto& det : detections) {
            observations.push_back(
                {det.box, det.confidence, det.class_id, det.class_name, "detector"});
        }
        return observations;
    }

    if (result.target_box.has_value()) {
        observations.push_back({*result.target_box, 1.0F, -1, "motion", "motion"});
    }
    return observations;
}

void draw_tracked_targets(cv::Mat& frame, const std::vector<TrackedTarget>& targets) {
    for (const auto& target : targets) {
        const cv::Scalar color = target.is_primary ? cv::Scalar(0, 255, 255) : cv::Scalar(255, 180, 0);
        cv::rectangle(frame, target.box, color, 2, cv::LINE_AA);

        std::ostringstream label;
        label << "ID " << target.track_id << " " << target.source;
        if (target.confidence > 0.0F && target.source == "detector") {
            label.setf(std::ios::fixed);
            label.precision(2);
            label << ' ' << target.confidence;
        }

        cv::putText(frame,
                    label.str(),
                    cv::Point(target.box.x, std::max(20, target.box.y - 6)),
                    cv::FONT_HERSHEY_SIMPLEX,
                    0.6,
                    color,
                    2,
                    cv::LINE_AA);
    }
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

    std::ostringstream status;
    status.setf(std::ios::fixed);
    status.precision(1);
    status << "Mode: " << mode_label
           << "  Rec: " << (recording ? "on" : "off")
           << "  Preview: " << (preview ? "on" : "off")
           << "  Yaw: " << yaw.status;
    if (yaw.status != "disabled") {
        status << " target=" << yaw.target_yaw_deg
               << "deg cur=" << yaw.current_yaw_deg
               << "deg rpm=" << yaw.rpm;
        if (yaw.limited) {
            status << " limited";
        }
    }

    cv::putText(frame,
                status.str(),
                cv::Point(x, y),
                cv::FONT_HERSHEY_SIMPLEX,
                0.58,
                cv::Scalar(120, 255, 120),
                2,
                cv::LINE_AA);
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

    std::unique_ptr<cvproj::YoloOnnxDetector> detector;
    if (config.detector == "yolo") {
        detector = std::make_unique<cvproj::YoloOnnxDetector>(config.model_path, config.det_input_size, config.det_conf, config.det_nms);
        if (!detector->open(&error)) {
            std::cerr << "Failed to open detector model " << config.model_path << ": " << error << '\n';
            return 4;
        }
    }

    cvproj::MotionPipelineConfig pipeline_config;
    pipeline_config.num_motion_points = config.num_motion_points;
    pipeline_config.motion_threshold = config.motion_threshold;
    cvproj::MotionPipeline pipeline(pipeline_config);
    cvproj::TargetTracker target_tracker;

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
        yaw_config.max_angle_deg = config.yaw_max_angle_deg;
        yaw_config.max_rpm = config.yaw_max_rpm;
        yaw_config.kp_rpm_per_deg = config.yaw_kp_rpm_per_deg;
        yaw_config.hold_deadband_deg = config.yaw_deadband_deg;
        yaw_config.horizontal_fov_deg = config.yaw_hfov_deg;
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

    std::int64_t processed_frames = 0;
    double last_report_fps = 0.0;
    double loop_fps_ema = 0.0;
    double effective_fps_ema = 0.0;
    double last_detection_ms = 0.0;
    double detection_fps_ema = 0.0;
    std::vector<cvproj::Detection> last_detections;
    std::chrono::steady_clock::time_point last_preview_publish;
    std::chrono::steady_clock::time_point last_record_write;
    std::vector<cvproj::TrackedTarget> tracked_targets;

    while (!stop_requested() && (config.max_frames < 0 || processed_frames < config.max_frames)) {
        const auto loop_start = std::chrono::steady_clock::now();
        cvproj::FramePacket packet;
        error.clear();
        if (!source->read_latest(packet, 2000, &error)) {
            std::cerr << "Stopping: " << error << '\n';
            break;
        }

        packet.frame_bgr = cvproj::swap_red_blue(packet.frame_bgr, config);
        packet.frame_bgr = cvproj::downsample_frame(packet.frame_bgr, config);
        packet.frame_bgr = cvproj::denoise_frame(packet.frame_bgr, config);
        packet.frame_bgr = cvproj::enhance_for_visibility(packet.frame_bgr, config);
        if (config.denoise_passes > 1) {
            packet.frame_bgr = cvproj::denoise_frame(packet.frame_bgr, config);
        }

        auto result = pipeline.process(packet);
        last_report_fps = result.fps;
        if (result.source_fps > 0.0) {
            effective_fps_ema = effective_fps_ema > 0.0 ? 0.9 * effective_fps_ema + 0.1 * result.source_fps
                                                         : result.source_fps;
        }

        if (detector && (processed_frames % config.detect_interval == 0)) {
            const auto detect_start = std::chrono::steady_clock::now();
            last_detections = detector->detect(packet.frame_bgr, result.roi);
            last_detection_ms = std::chrono::duration<double, std::milli>(
                                    std::chrono::steady_clock::now() - detect_start)
                                    .count();
            const double detection_fps = last_detection_ms > 0.0 ? 1000.0 / last_detection_ms : 0.0;
            detection_fps_ema = detection_fps_ema > 0.0 ? 0.9 * detection_fps_ema + 0.1 * detection_fps
                                                        : detection_fps;
        }

        const auto observations = cvproj::build_target_observations(result, last_detections);
        tracked_targets = target_tracker.update(observations);
        cvproj::draw_tracked_targets(result.annotated_frame, tracked_targets);

        std::string yaw_error;
        if (stop_requested()) {
        std::cout << "Stop requested; finalizing outputs..." << '\n';
    }

    if (yaw_controller) {
            const double control_dt =
                result.source_delta_seconds > 0.0 ? result.source_delta_seconds : 1.0 / std::max(1.0, config.fps);
            auto primary = std::find_if(tracked_targets.begin(),
                                        tracked_targets.end(),
                                        [](const cvproj::TrackedTarget& target) { return target.is_primary; });
            if (primary != tracked_targets.end()) {
                const double target_cx = primary->box.x + 0.5 * primary->box.width;
                last_yaw_command = yaw_controller->update_from_target(
                    target_cx, packet.frame_bgr.cols, control_dt, &yaw_error);
            } else {
                last_yaw_command = yaw_controller->hold(control_dt, &yaw_error);
            }
        }
        if (detector) {
            cv::putText(result.annotated_frame,
                        "Detector: YOLO ONNX  count=" + std::to_string(last_detections.size()),
                        cv::Point(16, 58),
                        cv::FONT_HERSHEY_SIMPLEX,
                        0.65,
                        cv::Scalar(255, 200, 0),
                        2,
                        cv::LINE_AA);
        }

        const auto loop_end = std::chrono::steady_clock::now();
        const double loop_ms =
            std::chrono::duration<double, std::milli>(loop_end - loop_start).count();
        const double loop_fps = loop_ms > 0.0 ? 1000.0 / loop_ms : 0.0;
        loop_fps_ema = loop_fps_ema > 0.0 ? 0.9 * loop_fps_ema + 0.1 * loop_fps : loop_fps;

        std::ostringstream fps_label;
        fps_label.setf(std::ios::fixed);
        fps_label.precision(1);
        fps_label << "Loop FPS: " << loop_fps_ema;
        if (effective_fps_ema > 0.0) {
            fps_label << "  Effective FPS: " << effective_fps_ema;
        }
        if (livestream_enabled) {
            fps_label << "  Mode: livestream";
        }
        cv::putText(result.annotated_frame,
                    fps_label.str(),
                    cv::Point(16, 86),
                    cv::FONT_HERSHEY_SIMPLEX,
                    0.65,
                    cv::Scalar(0, 220, 255),
                    2,
                    cv::LINE_AA);

        cvproj::draw_system_status(result.annotated_frame,
                                   livestream_enabled ? "livestream" : "capture",
                                   record_enabled,
                                   config.preview,
                                   last_yaw_command);

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
                    << "target_w,target_h,target_cx,target_cy,detection_count,"
                    << "yaw_has_target,yaw_target_deg,yaw_current_deg,yaw_error_deg,yaw_rpm,yaw_limited,yaw_status\n";
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
                << "x,y,w,h,cx,cy,norm_cx,norm_cy,offset_x,offset_y,roi_x,roi_y,roi_w,roi_h,"
                << "yaw_has_target,yaw_target_deg,yaw_current_deg,yaw_error_deg,yaw_rpm,yaw_limited,yaw_status\n";
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
            metrics_file << processed_frames << ',' << packet.frame_id << ','
                         << result.source_timestamp_seconds << ','
                         << (result.source_delta_seconds * 1000.0) << ','
                         << effective_fps_ema << ',' << result.fps << ',' << loop_fps_ema << ','
                         << result.processing_ms << ',' << last_detection_ms << ',' << detection_fps_ema << ','
                         << roi.x << ',' << roi.y << ',' << roi.width
                         << ',' << roi.height << ',' << target.x << ',' << target.y << ','
                         << target.width << ',' << target.height << ',' << target_cx << ','
                         << target_cy << ',' << last_detections.size() << ','
                         << (last_yaw_command.has_target ? 1 : 0) << ','
                         << last_yaw_command.target_yaw_deg << ','
                         << last_yaw_command.current_yaw_deg << ','
                         << last_yaw_command.error_deg << ','
                         << last_yaw_command.rpm << ','
                         << (last_yaw_command.limited ? 1 : 0) << ','
                         << last_yaw_command.status << '\n';
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
                               << ",-1,-1,0,none,-1,none,0.0,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,"
                               << roi.x << ',' << roi.y << ',' << roi.width << ',' << roi.height
                               << ',' << (last_yaw_command.has_target ? 1 : 0)
                               << ',' << last_yaw_command.target_yaw_deg
                               << ',' << last_yaw_command.current_yaw_deg
                               << ',' << last_yaw_command.error_deg
                               << ',' << last_yaw_command.rpm
                               << ',' << (last_yaw_command.limited ? 1 : 0)
                               << ',' << last_yaw_command.status
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
                                   << target.box.x << ',' << target.box.y << ',' << target.box.width
                                   << ',' << target.box.height << ',' << cx << ',' << cy << ','
                                   << norm_cx << ',' << norm_cy << ',' << offset_x << ',' << offset_y
                                   << ',' << roi.x << ',' << roi.y << ',' << roi.width << ','
                                   << roi.height
                                   << ',' << (last_yaw_command.has_target ? 1 : 0)
                                   << ',' << last_yaw_command.target_yaw_deg
                                   << ',' << last_yaw_command.current_yaw_deg
                                   << ',' << last_yaw_command.error_deg
                                   << ',' << last_yaw_command.rpm
                                   << ',' << (last_yaw_command.limited ? 1 : 0)
                                   << ',' << last_yaw_command.status << '\n';
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
