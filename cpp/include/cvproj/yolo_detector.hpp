#pragma once

#include <optional>
#include <string>
#include <vector>

#include <opencv2/core.hpp>

namespace cvproj {

struct Detection {
    cv::Rect box;
    float confidence = 0.0F;
    int class_id = 0;
    std::string class_name;
};

class DetectorBackend {
public:
    virtual ~DetectorBackend() = default;

    virtual bool open(std::string* error) = 0;
    virtual bool is_ready() const = 0;
    virtual std::vector<Detection> detect(const cv::Mat& frame_bgr, const std::optional<cv::Rect>& roi) = 0;
    virtual std::string backend_name() const = 0;
};

}  // namespace cvproj
