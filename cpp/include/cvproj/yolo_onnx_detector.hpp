#pragma once

#include <optional>
#include <string>
#include <vector>

#include <opencv2/core.hpp>
#include <opencv2/dnn.hpp>

namespace cvproj {

struct Detection {
    cv::Rect box;
    float confidence = 0.0F;
    int class_id = 0;
    std::string class_name;
};

class YoloOnnxDetector {
public:
    YoloOnnxDetector(std::string model_path, int input_size, float conf_threshold, float nms_threshold);

    bool open(std::string* error);
    bool is_ready() const;
    std::vector<Detection> detect(const cv::Mat& frame_bgr, const std::optional<cv::Rect>& roi) const;

private:
    std::string model_path_;
    int input_size_ = 640;
    float conf_threshold_ = 0.25F;
    float nms_threshold_ = 0.45F;
    cv::dnn::Net net_;
};

}  // namespace cvproj
