#pragma once

#include <string>

#include "cvproj/yolo_detector.hpp"

#if defined(CVPROJ_HAS_RKNN)
#include "rknn_api.h"
#endif

namespace cvproj {

class YoloRknnDetector final : public DetectorBackend {
public:
    YoloRknnDetector(std::string model_path, int input_size, float conf_threshold, float nms_threshold);
    ~YoloRknnDetector() override;

    bool open(std::string* error) override;
    bool is_ready() const override;
    std::vector<Detection> detect(const cv::Mat& frame_bgr, const std::optional<cv::Rect>& roi) override;
    std::string backend_name() const override { return "rknn"; }

private:
    void close();

    std::string model_path_;
    int input_size_ = 640;
    float conf_threshold_ = 0.25F;
    float nms_threshold_ = 0.45F;
    bool ready_ = false;

#if defined(CVPROJ_HAS_RKNN)
    rknn_context ctx_ = 0;
    rknn_input_output_num io_num_{};
    rknn_tensor_attr input_attr_{};
    rknn_tensor_attr output_attr_{};
#endif
};

}  // namespace cvproj
