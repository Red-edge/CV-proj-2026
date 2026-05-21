#pragma once

#include <opencv2/videoio.hpp>

#include "cvproj/frame_source.hpp"

namespace cvproj {

class OpenCvVideoSource final : public FrameSource {
public:
    OpenCvVideoSource(std::string source, int width, int height, double fps_hint);

    bool open(std::string* error) override;
    bool read(FramePacket& packet, int timeout_ms, std::string* error) override;
    void close() override;
    std::string name() const override;

private:
    static bool is_integer_source(const std::string& value);

    std::string source_;
    int width_ = 0;
    int height_ = 0;
    double fps_hint_ = 0.0;
    std::int64_t frame_counter_ = 0;
    cv::VideoCapture capture_;
};

}  // namespace cvproj
