#include "cvproj/yolo_rknn_detector.hpp"

#include <algorithm>
#include <array>
#include <cstring>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <limits>
#include <sstream>
#include <utility>

#include <opencv2/dnn.hpp>
#include <opencv2/imgproc.hpp>

namespace cvproj {
namespace {
constexpr int kPersonClassId = 0;

struct LetterboxInfo {
    float scale = 1.0F;
    int pad_x = 0;
    int pad_y = 0;
};

cv::Mat letterbox_rgb(const cv::Mat& input_bgr, int input_size, LetterboxInfo& info) {
    const float scale = std::min(static_cast<float>(input_size) / static_cast<float>(input_bgr.cols),
                                 static_cast<float>(input_size) / static_cast<float>(input_bgr.rows));
    const int resized_w = std::max(1, cvRound(static_cast<float>(input_bgr.cols) * scale));
    const int resized_h = std::max(1, cvRound(static_cast<float>(input_bgr.rows) * scale));
    info.scale = scale;
    info.pad_x = (input_size - resized_w) / 2;
    info.pad_y = (input_size - resized_h) / 2;

    cv::Mat resized;
    cv::resize(input_bgr, resized, cv::Size(resized_w, resized_h), 0.0, 0.0, cv::INTER_LINEAR);
    cv::Mat canvas(input_size, input_size, input_bgr.type(), cv::Scalar(114, 114, 114));
    resized.copyTo(canvas(cv::Rect(info.pad_x, info.pad_y, resized_w, resized_h)));
    cv::Mat rgb;
    cv::cvtColor(canvas, rgb, cv::COLOR_BGR2RGB);
    return rgb;
}

#if defined(CVPROJ_HAS_RKNN)
std::string tensor_attr_string(const rknn_tensor_attr& attr) {
    std::ostringstream oss;
    oss << "name=" << attr.name << " dims=[";
    for (uint32_t i = 0; i < attr.n_dims; ++i) {
        if (i) {
            oss << 'x';
        }
        oss << attr.dims[i];
    }
    oss << "] elems=" << attr.n_elems << " size=" << attr.size
        << " fmt=" << get_format_string(attr.fmt)
        << " type=" << get_type_string(attr.type)
        << " qnt=" << get_qnt_type_string(attr.qnt_type)
        << " zp=" << attr.zp
        << " scale=" << attr.scale;
    return oss.str();
}
#endif
}  // namespace

YoloRknnDetector::YoloRknnDetector(std::string model_path,
                                   int input_size,
                                   float conf_threshold,
                                   float nms_threshold)
    : model_path_(std::move(model_path)),
      input_size_(input_size),
      conf_threshold_(conf_threshold),
      nms_threshold_(nms_threshold) {}

YoloRknnDetector::~YoloRknnDetector() {
    close();
}

bool YoloRknnDetector::open(std::string* error) {
#if !defined(CVPROJ_HAS_RKNN)
    if (error) {
        *error = "RKNN runtime was not found at build time";
    }
    return false;
#else
    close();

    std::ifstream file(model_path_, std::ios::binary | std::ios::ate);
    if (!file.is_open()) {
        if (error) {
            *error = "failed to open RKNN model: " + model_path_;
        }
        return false;
    }
    const std::streamsize size = file.tellg();
    if (size <= 0) {
        if (error) {
            *error = "empty RKNN model: " + model_path_;
        }
        return false;
    }
    file.seekg(0, std::ios::beg);
    std::vector<unsigned char> model(static_cast<std::size_t>(size));
    if (!file.read(reinterpret_cast<char*>(model.data()), size)) {
        if (error) {
            *error = "failed to read RKNN model: " + model_path_;
        }
        return false;
    }

    int ret = rknn_init(&ctx_, model.data(), static_cast<uint32_t>(model.size()), 0, nullptr);
    if (ret != RKNN_SUCC) {
        if (error) {
            *error = "rknn_init failed: " + std::to_string(ret);
        }
        ctx_ = 0;
        return false;
    }
    (void)rknn_set_core_mask(ctx_, RKNN_NPU_CORE_AUTO);

    ret = rknn_query(ctx_, RKNN_QUERY_IN_OUT_NUM, &io_num_, sizeof(io_num_));
    if (ret != RKNN_SUCC || io_num_.n_input < 1 || io_num_.n_output < 1) {
        if (error) {
            *error = "rknn_query IN_OUT_NUM failed: " + std::to_string(ret);
        }
        close();
        return false;
    }

    std::memset(&input_attr_, 0, sizeof(input_attr_));
    input_attr_.index = 0;
    ret = rknn_query(ctx_, RKNN_QUERY_INPUT_ATTR, &input_attr_, sizeof(input_attr_));
    if (ret != RKNN_SUCC) {
        if (error) {
            *error = "rknn_query INPUT_ATTR failed: " + std::to_string(ret);
        }
        close();
        return false;
    }

    std::memset(&output_attr_, 0, sizeof(output_attr_));
    output_attr_.index = 0;
    ret = rknn_query(ctx_, RKNN_QUERY_OUTPUT_ATTR, &output_attr_, sizeof(output_attr_));
    if (ret != RKNN_SUCC) {
        if (error) {
            *error = "rknn_query OUTPUT_ATTR failed: " + std::to_string(ret);
        }
        close();
        return false;
    }

    std::cout << "RKNN io: inputs=" << io_num_.n_input << " outputs=" << io_num_.n_output << '\n';
    std::cout << "RKNN input[0]: " << tensor_attr_string(input_attr_) << '\n';
    std::cout << "RKNN output[0]: " << tensor_attr_string(output_attr_) << '\n';
    ready_ = true;
    return true;
#endif
}

bool YoloRknnDetector::is_ready() const {
    return ready_;
}

std::vector<Detection> YoloRknnDetector::detect(const cv::Mat& frame_bgr, const std::optional<cv::Rect>& roi) {
#if !defined(CVPROJ_HAS_RKNN)
    (void)frame_bgr;
    (void)roi;
    return {};
#else
    if (frame_bgr.empty() || !ready_) {
        return {};
    }
    const cv::Rect full_frame(0, 0, frame_bgr.cols, frame_bgr.rows);
    const cv::Rect valid_roi = roi.has_value() ? (*roi & full_frame) : full_frame;
    if (valid_roi.width <= 0 || valid_roi.height <= 0) {
        return {};
    }

    const cv::Mat input = frame_bgr(valid_roi);
    LetterboxInfo letterbox_info;
    cv::Mat network_input_u8 = letterbox_rgb(input, input_size_, letterbox_info);
    if (!network_input_u8.isContinuous()) {
        network_input_u8 = network_input_u8.clone();
    }
    cv::Mat network_input_i8;
    bool use_int8_input = input_attr_.type == RKNN_TENSOR_INT8;
    if (use_int8_input) {
        network_input_i8.create(network_input_u8.size(), CV_8SC3);
        const auto* src = network_input_u8.ptr<unsigned char>();
        auto* dst = network_input_i8.ptr<signed char>();
        const std::size_t count = network_input_u8.total() * network_input_u8.channels();
        for (std::size_t i = 0; i < count; ++i) {
            dst[i] = static_cast<signed char>(static_cast<int>(src[i]) - 128);
        }
    }

    rknn_input rknn_in{};
    rknn_in.index = 0;
    rknn_in.buf = use_int8_input ? static_cast<void*>(network_input_i8.data)
                                 : static_cast<void*>(network_input_u8.data);
    rknn_in.size = static_cast<uint32_t>(network_input_u8.total() * network_input_u8.elemSize());
    rknn_in.pass_through = use_int8_input ? 1 : 0;
    rknn_in.type = use_int8_input ? RKNN_TENSOR_INT8 : RKNN_TENSOR_UINT8;
    rknn_in.fmt = RKNN_TENSOR_NHWC;
    int ret = rknn_inputs_set(ctx_, 1, &rknn_in);
    if (ret != RKNN_SUCC) {
        return {};
    }

    ret = rknn_run(ctx_, nullptr);
    if (ret != RKNN_SUCC) {
        return {};
    }

    std::vector<rknn_output> outputs(io_num_.n_output);
    for (uint32_t i = 0; i < io_num_.n_output; ++i) {
        outputs[i].index = i;
        outputs[i].want_float = 1;
        outputs[i].is_prealloc = 0;
    }
    ret = rknn_outputs_get(ctx_, io_num_.n_output, outputs.data(), nullptr);
    if (ret != RKNN_SUCC || outputs.empty() || outputs[0].buf == nullptr) {
        return {};
    }

    const auto release_outputs = [&]() {
        rknn_outputs_release(ctx_, io_num_.n_output, outputs.data());
    };
    const float* data = static_cast<const float*>(outputs[0].buf);
    const std::size_t attr_float_count = output_attr_.n_elems > 0 ? output_attr_.n_elems : 0;
    const std::size_t size_float_count = outputs[0].size / sizeof(float);
    const std::size_t float_count = attr_float_count > 0 ? attr_float_count : size_float_count;
    if (float_count < 5 || float_count % 5 != 0) {
        release_outputs();
        return {};
    }

    int channels = 5;
    int candidates = static_cast<int>(float_count / 5);
    bool channel_first = true;
    if (output_attr_.n_dims >= 2) {
        for (uint32_t i = 0; i < output_attr_.n_dims; ++i) {
            if (output_attr_.dims[i] == 5 && i + 1 < output_attr_.n_dims) {
                channels = 5;
                candidates = static_cast<int>(output_attr_.dims[i + 1]);
                channel_first = true;
                break;
            }
            if (output_attr_.dims[i] == 5 && i > 0) {
                channels = 5;
                candidates = static_cast<int>(output_attr_.dims[i - 1]);
                channel_first = false;
                break;
            }
        }
    }

    auto layout_value_at = [&](bool use_channel_first, int channel, int candidate) -> float {
        if (use_channel_first) {
            return data[channel * candidates + candidate];
        }
        return data[candidate * channels + channel];
    };

    auto value_at = [&](int channel, int candidate) -> float {
        return layout_value_at(channel_first, channel, candidate);
    };

    float score_max = -std::numeric_limits<float>::infinity();
    for (int i = 0; i < candidates; ++i) {
        score_max = std::max(score_max, value_at(4 + kPersonClassId, i));
    }
    static bool warned_invalid_scores = false;
    if (score_max <= 0.0F) {
        if (!warned_invalid_scores) {
            std::cerr << "WARN: RKNN output score channel max is " << score_max
                      << "; this model cannot produce valid detections with YOLOv8 postprocess. "
                         "Use a RKNN export whose output keeps confidence precision, "
                         "for example FP16/FP32 output or split/dequantized score output."
                      << '\n';
            warned_invalid_scores = true;
        }
        release_outputs();
        return {};
    }

    static bool debug_printed = false;
    const bool debug = std::getenv("CVPROJ_RKNN_DEBUG") != nullptr;
    if (debug && !debug_printed) {
        std::array<float, 5> mins{};
        std::array<float, 5> maxs{};
        mins.fill(std::numeric_limits<float>::infinity());
        maxs.fill(-std::numeric_limits<float>::infinity());
        for (int i = 0; i < candidates; ++i) {
            for (int c = 0; c < 5; ++c) {
                const float v = value_at(c, i);
                mins[c] = std::min(mins[c], v);
                maxs[c] = std::max(maxs[c], v);
            }
        }
        std::cout << "RKNN output buffer_size=" << outputs[0].size
                  << " attr_elems=" << output_attr_.n_elems
                  << " parsed_float_count=" << float_count
                  << " candidates=" << candidates
                  << " channel_first=" << (channel_first ? 1 : 0) << '\n';
        std::cout << "RKNN score max=" << score_max << '\n';
        for (int c = 0; c < 5; ++c) {
            std::cout << "RKNN output ch" << c << " min=" << mins[c] << " max=" << maxs[c] << '\n';
        }
        debug_printed = true;
    }

    std::vector<cv::Rect> boxes;
    std::vector<float> scores;
    boxes.reserve(candidates);
    scores.reserve(candidates);
    for (int i = 0; i < candidates; ++i) {
        const float confidence = value_at(4 + kPersonClassId, i);
        if (confidence < conf_threshold_) {
            continue;
        }
        const float cx = (value_at(0, i) - static_cast<float>(letterbox_info.pad_x)) / letterbox_info.scale;
        const float cy = (value_at(1, i) - static_cast<float>(letterbox_info.pad_y)) / letterbox_info.scale;
        const float w = value_at(2, i) / letterbox_info.scale;
        const float h = value_at(3, i) / letterbox_info.scale;
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
        det.class_name = "person";
        detections.push_back(std::move(det));
    }
    release_outputs();

    std::sort(detections.begin(), detections.end(), [](const Detection& a, const Detection& b) {
        return a.confidence > b.confidence;
    });
    return detections;
#endif
}

void YoloRknnDetector::close() {
#if defined(CVPROJ_HAS_RKNN)
    if (ctx_ != 0) {
        rknn_destroy(ctx_);
        ctx_ = 0;
    }
#endif
    ready_ = false;
}

}  // namespace cvproj
