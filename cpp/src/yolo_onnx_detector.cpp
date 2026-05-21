#include "cvproj/yolo_onnx_detector.hpp"

#include <algorithm>
#include <array>
#include <utility>

#include <opencv2/imgproc.hpp>

namespace cvproj {

namespace {
constexpr int kPersonClassId = 0;
constexpr std::array<const char*, 80> kCocoNames = {
    "person",        "bicycle",      "car",           "motorcycle",  "airplane",     "bus",          "train",
    "truck",         "boat",         "traffic light", "fire hydrant","stop sign",    "parking meter","bench",
    "bird",          "cat",          "dog",           "horse",       "sheep",        "cow",          "elephant",
    "bear",          "zebra",        "giraffe",       "backpack",    "umbrella",     "handbag",      "tie",
    "suitcase",      "frisbee",      "skis",          "snowboard",   "sports ball",  "kite",         "baseball bat",
    "baseball glove","skateboard",   "surfboard",     "tennis racket","bottle",      "wine glass",   "cup",
    "fork",          "knife",        "spoon",         "bowl",        "banana",       "apple",        "sandwich",
    "orange",        "broccoli",     "carrot",        "hot dog",     "pizza",        "donut",        "cake",
    "chair",         "couch",        "potted plant",  "bed",         "dining table", "toilet",       "tv",
    "laptop",        "mouse",        "remote",        "keyboard",    "cell phone",   "microwave",    "oven",
    "toaster",       "sink",         "refrigerator",  "book",        "clock",        "vase",         "scissors",
    "teddy bear",    "hair drier",   "toothbrush"};
}  // namespace

YoloOnnxDetector::YoloOnnxDetector(std::string model_path,
                                   int input_size,
                                   float conf_threshold,
                                   float nms_threshold)
    : model_path_(std::move(model_path)),
      input_size_(input_size),
      conf_threshold_(conf_threshold),
      nms_threshold_(nms_threshold) {}

bool YoloOnnxDetector::open(std::string* error) {
    try {
        net_ = cv::dnn::readNetFromONNX(model_path_);
        net_.setPreferableBackend(cv::dnn::DNN_BACKEND_OPENCV);
        net_.setPreferableTarget(cv::dnn::DNN_TARGET_CPU);
        return true;
    } catch (const std::exception& ex) {
        if (error) {
            *error = ex.what();
        }
        return false;
    }
}

bool YoloOnnxDetector::is_ready() const {
    return !net_.empty();
}

std::vector<Detection> YoloOnnxDetector::detect(const cv::Mat& frame_bgr, const std::optional<cv::Rect>& roi) const {
    if (frame_bgr.empty() || net_.empty()) {
        return {};
    }

    const cv::Rect full_frame(0, 0, frame_bgr.cols, frame_bgr.rows);
    const cv::Rect valid_roi = roi.has_value() ? (*roi & full_frame) : full_frame;
    if (valid_roi.width <= 0 || valid_roi.height <= 0) {
        return {};
    }

    const cv::Mat input = frame_bgr(valid_roi);
    cv::Mat blob =
        cv::dnn::blobFromImage(input, 1.0 / 255.0, cv::Size(input_size_, input_size_), cv::Scalar(), true, false);

    cv::dnn::Net net = net_;
    net.setInput(blob);
    cv::Mat output = net.forward();
    if (output.dims != 3 || output.size[1] < 5) {
        return {};
    }

    const int channels = output.size[1];
    const int candidates = output.size[2];
    cv::Mat reshaped(channels, candidates, CV_32F, output.ptr<float>());

    std::vector<cv::Rect> boxes;
    std::vector<float> scores;
    boxes.reserve(candidates);
    scores.reserve(candidates);

    const float scale_x = static_cast<float>(input.cols) / static_cast<float>(input_size_);
    const float scale_y = static_cast<float>(input.rows) / static_cast<float>(input_size_);

    for (int i = 0; i < candidates; ++i) {
        const float confidence = reshaped.at<float>(4 + kPersonClassId, i);
        if (confidence < conf_threshold_) {
            continue;
        }

        const float cx = reshaped.at<float>(0, i) * scale_x;
        const float cy = reshaped.at<float>(1, i) * scale_y;
        const float w = reshaped.at<float>(2, i) * scale_x;
        const float h = reshaped.at<float>(3, i) * scale_y;

        const int x = std::max(0, cvRound(cx - 0.5F * w));
        const int y = std::max(0, cvRound(cy - 0.5F * h));
        const int width = std::min(input.cols - x, std::max(0, cvRound(w)));
        const int height = std::min(input.rows - y, std::max(0, cvRound(h)));
        if (width <= 0 || height <= 0) {
            continue;
        }

        boxes.emplace_back(x, y, width, height);
        scores.push_back(confidence);
    }

    std::vector<int> keep;
    cv::dnn::NMSBoxes(boxes, scores, conf_threshold_, nms_threshold_, keep);

    std::vector<Detection> detections;
    detections.reserve(keep.size());
    for (const int idx : keep) {
        Detection det;
        det.box = boxes[idx];
        det.box.x += valid_roi.x;
        det.box.y += valid_roi.y;
        det.confidence = scores[idx];
        det.class_id = kPersonClassId;
        det.class_name = kCocoNames[kPersonClassId];
        detections.push_back(std::move(det));
    }

    std::sort(detections.begin(), detections.end(), [](const Detection& a, const Detection& b) {
        return a.confidence > b.confidence;
    });
    return detections;
}

}  // namespace cvproj
