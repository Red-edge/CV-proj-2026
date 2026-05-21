#include <algorithm>
#include <filesystem>
#include <chrono>
#include <cmath>
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
#include "cvproj/motion_pipeline.hpp"
#include "cvproj/opencv_video_source.hpp"
#include "cvproj/yolo_onnx_detector.hpp"

namespace fs = std::filesystem;

namespace cvproj {

struct AppConfig {
    std::string backend = "opencv";
    std::string source = "0";
    std::string serial_number;
    std::string telemetry_path;
    std::string record_path;
    double record_fps = -1.0;
    std::string pixel_format = "Mono8";
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
    int max_frames = -1;
    bool auto_brightness = true;
    double target_luma = 96.0;
    double max_post_gain = 6.0;
    double gamma = 0.8;
    bool livestream = false;
    bool headless = false;
};

void print_usage() {
    std::cout
        << "Usage: cvproj_capture [options]\n"
        << "  --backend <opencv|hikrobot|aravis>\n"
        << "  --source <camera-index|video-path>\n"
        << "  --serial <hikrobot-serial>\n"
        << "  --record-fps <output-fps>\n"
        << "  --pixel-format <Mono8|BayerRG8|...>\n"
        << "  --gain-db <camera-gain-db>\n"
        << "  --target-luma <0-255>\n"
        << "  --max-post-gain <scale>\n"
        << "  --gamma <value>\n"
        << "  --no-auto-brightness\n"
        << "  --detector <none|yolo>\n"
        << "  --model <onnx-path>\n"
        << "  --det-conf <threshold>\n"
        << "  --det-nms <threshold>\n"
        << "  --detect-interval <N>\n"
        << "  --width <pixels>\n"
        << "  --height <pixels>\n"
        << "  --fps <target>\n"
        << "  --exposure-us <microseconds>\n"
        << "  --motion-thresh <pixels>\n"
        << "  --grid-points <count>\n"
        << "  --record <output.mp4>\n"
        << "  --telemetry <output.csv>\n"
        << "  --livestream\n"
        << "  --max-frames <N>\n"
        << "  --headless\n";
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
        } else if (arg == "--no-auto-brightness") {
            config.auto_brightness = false;
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

    auto clahe = cv::createCLAHE(2.0, cv::Size(8, 8));
    cv::Mat enhanced_luma;
    clahe->apply(channels[0], enhanced_luma);

    if (config.auto_brightness) {
        const double mean_luma = cv::mean(enhanced_luma)[0];
        if (mean_luma > 1.0) {
            const double scale = std::clamp(config.target_luma / mean_luma, 1.0, config.max_post_gain);
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
    cvproj::AppConfig config;
    std::string parse_error;
    if (!cvproj::parse_args(argc, argv, config, parse_error)) {
        if (!parse_error.empty()) {
            std::cerr << parse_error << '\n';
        }
        if (!parse_error.empty()) {
            cvproj::print_usage();
        }
        return parse_error.empty() ? 0 : 1;
    }

    auto source = cvproj::make_source(config);
    std::string error;
    if (!source->open(&error)) {
        std::cerr << "Failed to open source " << source->name() << ": " << error << '\n';
        return 2;
    }

    std::unique_ptr<cvproj::YoloOnnxDetector> detector;
    if (config.detector == "yolo") {
        detector = std::make_unique<cvproj::YoloOnnxDetector>(config.model_path, 640, config.det_conf, config.det_nms);
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

    cv::VideoWriter writer;
    std::ofstream metrics_file;
    std::ofstream telemetry_file;
    const bool livestream_enabled = config.livestream;
    const bool record_enabled = !livestream_enabled && !config.record_path.empty();
    const bool telemetry_enabled = livestream_enabled || !config.telemetry_path.empty();
    const std::string window_name = "CV-proj-2026 C++ Motion Pipeline";
    const double output_record_fps =
        config.record_fps > 0.0 ? config.record_fps : std::max(1.0, config.fps);
    const fs::path telemetry_path = cvproj::make_telemetry_path(config);

    std::int64_t processed_frames = 0;
    double last_report_fps = 0.0;
    double loop_fps_ema = 0.0;
    double effective_fps_ema = 0.0;
    std::vector<cvproj::Detection> last_detections;
    std::vector<cvproj::TrackedTarget> tracked_targets;

    while (config.max_frames < 0 || processed_frames < config.max_frames) {
        const auto loop_start = std::chrono::steady_clock::now();
        cvproj::FramePacket packet;
        error.clear();
        if (!source->read_latest(packet, 2000, &error)) {
            std::cerr << "Stopping: " << error << '\n';
            break;
        }

        packet.frame_bgr = cvproj::enhance_for_visibility(packet.frame_bgr, config);

        auto result = pipeline.process(packet);
        last_report_fps = result.fps;
        if (result.source_fps > 0.0) {
            effective_fps_ema = effective_fps_ema > 0.0 ? 0.9 * effective_fps_ema + 0.1 * result.source_fps
                                                         : result.source_fps;
        }

        if (detector && (processed_frames % config.detect_interval == 0)) {
            last_detections = detector->detect(packet.frame_bgr, result.roi);
        }

        const auto observations = cvproj::build_target_observations(result, last_detections);
        tracked_targets = target_tracker.update(observations);
        cvproj::draw_tracked_targets(result.annotated_frame, tracked_targets);
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
                    << "pipeline_fps,loop_fps,processing_ms,roi_x,roi_y,roi_w,roi_h,target_x,target_y,"
                    << "target_w,target_h,target_cx,target_cy,detection_count\n";
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
                << "x,y,w,h,cx,cy,norm_cx,norm_cy,offset_x,offset_y,roi_x,roi_y,roi_w,roi_h\n";
        }

        if (writer.isOpened()) {
            writer.write(result.annotated_frame);
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
                         << result.processing_ms << ',' << roi.x << ',' << roi.y << ',' << roi.width
                         << ',' << roi.height << ',' << target.x << ',' << target.y << ','
                         << target.width << ',' << target.height << ',' << target_cx << ','
                         << target_cy << ',' << last_detections.size() << '\n';
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
                                   << roi.height << '\n';
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
              << "Telemetry: "
              << (telemetry_enabled ? telemetry_path.string() : std::string("disabled")) << '\n'
              << "Backend: " << source->name() << '\n';

    return processed_frames > 0 ? 0 : 3;
}
