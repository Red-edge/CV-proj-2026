#include "cvproj/motion_pipeline.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <numeric>
#include <sstream>

#include <opencv2/imgproc.hpp>
#include <opencv2/video/tracking.hpp>

namespace cvproj {

namespace {
double median_of(std::vector<float> values) {
    if (values.empty()) {
        return 0.0;
    }
    const auto mid = values.begin() + static_cast<std::ptrdiff_t>(values.size() / 2);
    std::nth_element(values.begin(), mid, values.end());
    return static_cast<double>(*mid);
}
}  // namespace

MotionPipeline::MotionPipeline(MotionPipelineConfig config) : config_(std::move(config)) {
    if (config_.blur_kernel % 2 == 0) {
        ++config_.blur_kernel;
    }
}

MotionPipelineResult MotionPipeline::process(const FramePacket& packet) {
    const auto t0 = std::chrono::steady_clock::now();

    MotionPipelineResult result;
    result.source_timestamp_seconds = packet.timestamp_seconds;
    if (packet.timestamp_seconds > 0.0 && last_frame_timestamp_seconds_ > 0.0 &&
        packet.timestamp_seconds > last_frame_timestamp_seconds_) {
        result.source_delta_seconds = packet.timestamp_seconds - last_frame_timestamp_seconds_;
        result.source_fps =
            result.source_delta_seconds > 0.0 ? 1.0 / result.source_delta_seconds : 0.0;
    }
    if (packet.timestamp_seconds > 0.0) {
        last_frame_timestamp_seconds_ = packet.timestamp_seconds;
    }

    const cv::Mat& frame_bgr = packet.frame_bgr;
    result.annotated_frame = frame_bgr.clone();

    if (frame_bgr.empty()) {
        return result;
    }

    if (fixed_points_.empty()) {
        fixed_points_ = generate_fixed_motion_points(frame_bgr.size());
    }

    cv::Mat gray;
    cv::cvtColor(frame_bgr, gray, cv::COLOR_BGR2GRAY);

    std::vector<float> magnitudes(fixed_points_.size(), 0.0F);
    std::vector<cv::Point2f> compensated_points = fixed_points_;

    if (!prev_gray_.empty()) {
        std::vector<cv::Point2f> next_points;
        std::vector<unsigned char> status;
        std::vector<float> errors;
        cv::calcOpticalFlowPyrLK(
            prev_gray_,
            gray,
            fixed_points_,
            next_points,
            status,
            errors,
            cv::Size(15, 15),
            2,
            cv::TermCriteria(cv::TermCriteria::COUNT | cv::TermCriteria::EPS, 10, 0.03));

        std::vector<float> dxs;
        std::vector<float> dys;
        dxs.reserve(next_points.size());
        dys.reserve(next_points.size());
        for (std::size_t i = 0; i < next_points.size(); ++i) {
            if (i >= status.size() || status[i] == 0) {
                continue;
            }
            dxs.push_back(next_points[i].x - fixed_points_[i].x);
            dys.push_back(next_points[i].y - fixed_points_[i].y);
        }

        const double global_dx = median_of(dxs);
        const double global_dy = median_of(dys);

        for (std::size_t i = 0; i < fixed_points_.size() && i < next_points.size(); ++i) {
            if (i >= status.size() || status[i] == 0) {
                continue;
            }
            const float dx = static_cast<float>((next_points[i].x - fixed_points_[i].x) - global_dx);
            const float dy = static_cast<float>((next_points[i].y - fixed_points_[i].y) - global_dy);
            magnitudes[i] = std::sqrt(dx * dx + dy * dy);
            compensated_points[i] = cv::Point2f(fixed_points_[i].x + dx, fixed_points_[i].y + dy);
        }
    }

    prev_gray_ = gray;

    int motion_count = 0;
    for (const float mag : magnitudes) {
        if (mag >= config_.motion_threshold) {
            ++motion_count;
        }
    }
    result.motion_count = motion_count;

    result.roi = get_roi_from_motion_points(fixed_points_, magnitudes, frame_bgr.size());
    auto blob_box = detect_target_blob(compensated_points, magnitudes, frame_bgr.size());
    if (blob_box.has_value()) {
        result.target_box = smooth_track(*blob_box, result.source_delta_seconds);
    }

    const auto t1 = std::chrono::steady_clock::now();
    result.processing_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    const double instant_fps = result.processing_ms > 0.0 ? 1000.0 / result.processing_ms : 0.0;
    fps_ema_ = fps_ema_ > 0.0 ? 0.9 * fps_ema_ + 0.1 * instant_fps : instant_fps;
    result.fps = fps_ema_;

    draw_overlay(result.annotated_frame, fixed_points_, magnitudes, result);
    return result;
}

void MotionPipeline::reset() {
    fixed_points_.clear();
    prev_gray_.release();
    fps_ema_ = 0.0;
    last_frame_timestamp_seconds_ = 0.0;
    smoothed_track_.reset();
}

std::vector<cv::Point2f> MotionPipeline::generate_fixed_motion_points(const cv::Size& size) const {
    const double width = static_cast<double>(size.width);
    const double height = static_cast<double>(size.height);
    int cols = static_cast<int>(std::round(std::sqrt(config_.num_motion_points * width / height)));
    cols = std::max(4, std::min(cols, config_.num_motion_points));
    const int rows = static_cast<int>(std::ceil(static_cast<double>(config_.num_motion_points) / cols));

    const double cell_w = width / cols;
    const double cell_h = height / rows;

    std::vector<cv::Point2f> points;
    points.reserve(config_.num_motion_points);
    for (int r = 0; r < rows; ++r) {
        for (int c = 0; c < cols; ++c) {
            if (static_cast<int>(points.size()) >= config_.num_motion_points) {
                break;
            }
            points.emplace_back(static_cast<float>((c + 0.5) * cell_w),
                                static_cast<float>((r + 0.5) * cell_h));
        }
    }
    return points;
}

std::optional<cv::Rect> MotionPipeline::get_roi_from_motion_points(const std::vector<cv::Point2f>& points,
                                                                   const std::vector<float>& magnitudes,
                                                                   const cv::Size& size) const {
    if (points.empty() || magnitudes.empty()) {
        return std::nullopt;
    }

    std::vector<float> motion_x;
    motion_x.reserve(points.size());
    for (std::size_t i = 0; i < points.size() && i < magnitudes.size(); ++i) {
        if (magnitudes[i] >= config_.motion_threshold) {
            motion_x.push_back(points[i].x);
        }
    }
    if (motion_x.empty()) {
        return std::nullopt;
    }

    const int roi_w = std::max(32, static_cast<int>(std::round(size.width * config_.roi_width_ratio)));
    int best_x = 0;
    int best_count = -1;
    for (int x = 0; x <= std::max(0, size.width - roi_w); x += 2) {
        int count = 0;
        for (const float px : motion_x) {
            if (px >= x && px < x + roi_w) {
                ++count;
            }
        }
        if (count > best_count) {
            best_count = count;
            best_x = x;
        }
    }

    return cv::Rect(best_x, 0, roi_w, size.height);
}

std::optional<cv::Rect> MotionPipeline::detect_target_blob(const std::vector<cv::Point2f>& points,
                                                           const std::vector<float>& magnitudes,
                                                           const cv::Size& size) const {
    cv::Mat motion_mask(size, CV_8UC1, cv::Scalar(0));
    for (std::size_t i = 0; i < points.size() && i < magnitudes.size(); ++i) {
        if (magnitudes[i] < config_.motion_threshold) {
            continue;
        }
        cv::circle(motion_mask, points[i], 10, cv::Scalar(255), -1, cv::LINE_AA);
    }

    cv::morphologyEx(motion_mask,
                     motion_mask,
                     cv::MORPH_CLOSE,
                     cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(9, 9)));

    std::vector<std::vector<cv::Point>> contours;
    cv::findContours(motion_mask, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

    double best_area = 0.0;
    std::optional<cv::Rect> best_rect;
    for (const auto& contour : contours) {
        const double area = cv::contourArea(contour);
        if (area < config_.min_blob_area || area < best_area) {
            continue;
        }
        best_area = area;
        best_rect = cv::boundingRect(contour);
    }
    return best_rect;
}

cv::Rect MotionPipeline::smooth_track(const cv::Rect& current, double delta_seconds) {
    if (!smoothed_track_.has_value()) {
        smoothed_track_ = current;
        return current;
    }

    const double clamped_delta = delta_seconds > 0.0 ? std::min(delta_seconds, 0.25) : 1.0 / 60.0;
    const double reference_frames = clamped_delta * 60.0;
    const double base = std::clamp(1.0 - config_.track_smooth_alpha, 0.001, 0.999);
    const float alpha = static_cast<float>(1.0 - std::pow(base, reference_frames));
    cv::Rect2f value = *smoothed_track_;
    value.x = (1.0F - alpha) * value.x + alpha * static_cast<float>(current.x);
    value.y = (1.0F - alpha) * value.y + alpha * static_cast<float>(current.y);
    value.width = (1.0F - alpha) * value.width + alpha * static_cast<float>(current.width);
    value.height = (1.0F - alpha) * value.height + alpha * static_cast<float>(current.height);
    smoothed_track_ = value;
    return cv::Rect(cvRound(value.x), cvRound(value.y), cvRound(value.width), cvRound(value.height));
}

void MotionPipeline::draw_overlay(cv::Mat& vis,
                                  const std::vector<cv::Point2f>& points,
                                  const std::vector<float>& magnitudes,
                                  const MotionPipelineResult& result) const {
    for (std::size_t i = 0; i < points.size() && i < magnitudes.size(); ++i) {
        const float mag = magnitudes[i];
        cv::Scalar color(0, 255, 0);
        if (mag >= config_.motion_threshold) {
            color = cv::Scalar(0, 0, 255);
        } else if (mag >= config_.motion_threshold * 0.5) {
            color = cv::Scalar(0, 255, 255);
        }
        cv::circle(vis, points[i], config_.overlay_radius, color, -1, cv::LINE_AA);
    }

    if (result.roi.has_value()) {
        cv::rectangle(vis, *result.roi, cv::Scalar(255, 0, 255), 2, cv::LINE_AA);
    }

    if (result.target_box.has_value()) {
        cv::rectangle(vis, *result.target_box, cv::Scalar(0, 255, 255), 2, cv::LINE_AA);
        const cv::Point center(result.target_box->x + result.target_box->width / 2,
                               result.target_box->y + result.target_box->height / 2);
        cv::drawMarker(vis, center, cv::Scalar(0, 255, 0), cv::MARKER_CROSS, 18, 2, cv::LINE_AA);
    }

    std::ostringstream oss;
    oss.setf(std::ios::fixed);
    oss.precision(1);
    oss << "FPS: " << result.fps << "  MotionPts: " << result.motion_count;
    oss.precision(2);
    oss << "  Proc: " << result.processing_ms << " ms";
    if (result.source_fps > 0.0) {
        oss.precision(1);
        oss << "  Src: " << result.source_fps;
    }

    cv::putText(vis,
                oss.str(),
                cv::Point(16, 30),
                cv::FONT_HERSHEY_SIMPLEX,
                0.7,
                cv::Scalar(0, 255, 255),
                2,
                cv::LINE_AA);
}

}  // namespace cvproj
