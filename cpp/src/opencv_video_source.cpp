#include "cvproj/opencv_video_source.hpp"

#include <chrono>
#include <cctype>

namespace cvproj {

namespace {
double now_seconds() {
    using clock = std::chrono::steady_clock;
    const auto now = clock::now().time_since_epoch();
    return std::chrono::duration<double>(now).count();
}
}  // namespace

OpenCvVideoSource::OpenCvVideoSource(std::string source, int width, int height, double fps_hint)
    : source_(std::move(source)), width_(width), height_(height), fps_hint_(fps_hint) {}

bool OpenCvVideoSource::open(std::string* error) {
    close();

    const bool numeric = is_integer_source(source_);
    if (numeric) {
        capture_.open(std::stoi(source_));
    } else {
        capture_.open(source_);
    }

    if (!capture_.isOpened()) {
        if (error) {
            *error = "Failed to open OpenCV source: " + source_;
        }
        return false;
    }

    if (width_ > 0) {
        capture_.set(cv::CAP_PROP_FRAME_WIDTH, width_);
    }
    if (height_ > 0) {
        capture_.set(cv::CAP_PROP_FRAME_HEIGHT, height_);
    }
    if (numeric) {
        capture_.set(cv::CAP_PROP_BUFFERSIZE, 1);
    }
    if (fps_hint_ > 0.0) {
        capture_.set(cv::CAP_PROP_FPS, fps_hint_);
    }

    frame_counter_ = 0;
    return true;
}

bool OpenCvVideoSource::read(FramePacket& packet, int, std::string* error) {
    cv::Mat frame;
    if (!capture_.read(frame) || frame.empty()) {
        if (error) {
            *error = "OpenCV source ended or returned empty frame";
        }
        return false;
    }

    packet.frame_bgr = frame;
    packet.frame_id = frame_counter_++;
    packet.timestamp_seconds = now_seconds();
    return true;
}

void OpenCvVideoSource::close() {
    if (capture_.isOpened()) {
        capture_.release();
    }
}

std::string OpenCvVideoSource::name() const {
    return "opencv:" + source_;
}

bool OpenCvVideoSource::is_integer_source(const std::string& value) {
    if (value.empty()) {
        return false;
    }
    for (const unsigned char c : value) {
        if (!std::isdigit(c)) {
            return false;
        }
    }
    return true;
}

}  // namespace cvproj
