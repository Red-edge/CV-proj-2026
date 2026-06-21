#pragma once

#include <string>

#include <opencv2/dnn.hpp>

#include "cvproj/yolo_detector.hpp"

namespace cvproj {

class YoloOnnxDetector final : public DetectorBackend {
public:
    YoloOnnxDetector(std::string model_path, int input_size, float conf_threshold, float nms_threshold);

    bool open(std::string* error) override;
    bool is_ready() const override;
    std::vector<Detection> detect(const cv::Mat& frame_bgr, const std::optional<cv::Rect>& roi) override;
    std::string backend_name() const override { return "onnx"; }

private:
    std::string model_path_;
    int input_size_ = 640;
    float conf_threshold_ = 0.25F;
    float nms_threshold_ = 0.45F;
    cv::dnn::Net net_;
};

}  // namespace cvproj
