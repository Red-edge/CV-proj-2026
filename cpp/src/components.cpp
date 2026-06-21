#include "central_control/common.hpp"

#include <algorithm>
#include <atomic>
#include <cctype>
#include <cmath>
#include <cstring>
#include <filesystem>
#include <iostream>
#include <limits>
#include <numeric>
#include <sstream>

#include <opencv2/dnn.hpp>
#include <opencv2/core/utils/logger.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/video.hpp>
#include <opencv2/videoio.hpp>

#if defined(CENTRAL_CONTROL_WITH_ORT)
#include <onnxruntime_cxx_api.h>
#endif

#if defined(CENTRAL_CONTROL_WITH_MVS)
#include "CameraParams.h"
#include "MvCameraControl.h"
#include "PixelType.h"
#endif

namespace central_control {

namespace {

double to_ms(const Clock::duration& duration) {
    return std::chrono::duration<double, std::milli>(duration).count();
}

cv::Mat resize_keep_aspect(const cv::Mat& image, int target_w, int target_h, float& scale, int& pad_x, int& pad_y) {
    const int image_w = image.cols;
    const int image_h = image.rows;
    scale = std::min(static_cast<float>(target_w) / static_cast<float>(image_w),
                     static_cast<float>(target_h) / static_cast<float>(image_h));
    const int resized_w = std::max(1, static_cast<int>(std::round(image_w * scale)));
    const int resized_h = std::max(1, static_cast<int>(std::round(image_h * scale)));
    pad_x = (target_w - resized_w) / 2;
    pad_y = (target_h - resized_h) / 2;

    cv::Mat resized;
    cv::resize(image, resized, cv::Size(resized_w, resized_h), 0.0, 0.0, cv::INTER_LINEAR);
    cv::Mat canvas(target_h, target_w, CV_8UC3, cv::Scalar(114, 114, 114));
    resized.copyTo(canvas(cv::Rect(pad_x, pad_y, resized_w, resized_h)));
    return canvas;
}

float compute_iou(const cv::Rect2f& a, const cv::Rect2f& b) {
    const float inter = (a & b).area();
    const float uni = a.area() + b.area() - inter;
    return uni > 0.0F ? inter / uni : 0.0F;
}

cv::Scalar motion_color(float magnitude, float threshold) {
    if (magnitude < threshold * 0.5F) {
        return {0, 255, 0};
    }
    if (magnitude < threshold) {
        return {0, 255, 255};
    }
    return {0, 0, 255};
}

class VideoCaptureSource final : public IFrameSource {
public:
    bool open(const AppConfig& config) override {
        description_ = "VideoCapture";
        if (config.source.size() == 1 && std::isdigit(config.source[0])) {
            capture_.open(std::stoi(config.source));
            description_ += " index=" + config.source;
        } else {
            capture_.open(config.source);
            description_ += " path=" + config.source;
        }
        return capture_.isOpened();
    }

    bool read(FramePacket& packet) override {
        cv::Mat frame;
        if (!capture_.read(frame) || frame.empty()) {
            return false;
        }
        packet.image = frame;
        return true;
    }

    void close() override {
        capture_.release();
    }

    std::string describe() const override {
        return description_;
    }

private:
    cv::VideoCapture capture_;
    std::string description_;
};

#if defined(CENTRAL_CONTROL_WITH_MVS)
class MvsFrameSource final : public IFrameSource {
public:
    bool open(const AppConfig& config) override {
        config_ = config;
        if (MV_CC_Initialize() != MV_OK) {
            std::cerr << "MVS SDK initialize failed" << std::endl;
            return false;
        }

        MV_CC_DEVICE_INFO_LIST device_list{};
        const unsigned int layer_mask = MV_GIGE_DEVICE | MV_USB_DEVICE;
        if (MV_CC_EnumDevices(layer_mask, &device_list) != MV_OK || device_list.nDeviceNum == 0) {
            std::cerr << "No MVS devices enumerated" << std::endl;
            MV_CC_Finalize();
            return false;
        }

        int selected_index = -1;
        for (unsigned int i = 0; i < device_list.nDeviceNum; ++i) {
            auto* info = device_list.pDeviceInfo[i];
            if (info == nullptr) {
                continue;
            }
            if (!config.mvs_serial.empty()) {
                std::string serial;
                if ((info->nTLayerType & MV_USB_DEVICE) != 0U) {
                    serial = reinterpret_cast<const char*>(info->SpecialInfo.stUsb3VInfo.chSerialNumber);
                } else if ((info->nTLayerType & MV_GIGE_DEVICE) != 0U) {
                    serial = reinterpret_cast<const char*>(info->SpecialInfo.stGigEInfo.chSerialNumber);
                }
                if (serial == config.mvs_serial) {
                    selected_index = static_cast<int>(i);
                    break;
                }
            } else if (static_cast<int>(i) == config.mvs_index) {
                selected_index = static_cast<int>(i);
                break;
            }
        }

        if (selected_index < 0) {
            std::cerr << "Requested MVS device not found" << std::endl;
            MV_CC_Finalize();
            return false;
        }

        MV_CC_DEVICE_INFO* selected = device_list.pDeviceInfo[selected_index];
        if (MV_CC_CreateHandle(&handle_, selected) != MV_OK) {
            std::cerr << "MV_CC_CreateHandle failed" << std::endl;
            cleanup();
            return false;
        }
        if (MV_CC_OpenDevice(handle_, MV_ACCESS_Exclusive, 0) != MV_OK) {
            std::cerr << "MV_CC_OpenDevice failed" << std::endl;
            cleanup();
            return false;
        }

        MV_CC_SetEnumValueByString(handle_, "TriggerMode", "Off");
        MV_CC_SetEnumValueByString(handle_, "ExposureAuto", "Off");
        MV_CC_SetEnumValueByString(handle_, "GainAuto", "Off");

        if (config.mvs_exposure_us > 0.0F) {
            MV_CC_SetFloatValue(handle_, "ExposureTime", config.mvs_exposure_us);
        }
        if (config.mvs_gain >= 0.0F) {
            MV_CC_SetFloatValue(handle_, "Gain", config.mvs_gain);
        }
        if (config.mvs_frame_rate > 0.0F) {
            MV_CC_SetBoolValue(handle_, "AcquisitionFrameRateEnable", true);
            MV_CC_SetFloatValue(handle_, "AcquisitionFrameRate", config.mvs_frame_rate);
        }

        MV_CC_SetImageNodeNum(handle_, 8);
        if (MV_CC_StartGrabbing(handle_) != MV_OK) {
            std::cerr << "MV_CC_StartGrabbing failed" << std::endl;
            cleanup();
            return false;
        }

        if ((selected->nTLayerType & MV_USB_DEVICE) != 0U) {
            description_ = "MVS USB model=" + std::string(reinterpret_cast<const char*>(selected->SpecialInfo.stUsb3VInfo.chModelName));
        } else {
            description_ = "MVS GigE model=" + std::string(reinterpret_cast<const char*>(selected->SpecialInfo.stGigEInfo.chModelName));
        }
        return true;
    }

    bool read(FramePacket& packet) override {
        if (handle_ == nullptr) {
            return false;
        }

        MV_FRAME_OUT frame_out{};
        const int ret = MV_CC_GetImageBuffer(handle_, &frame_out, 1000);
        if (ret != MV_OK) {
            return false;
        }

        const auto release_guard = [&]() { MV_CC_FreeImageBuffer(handle_, &frame_out); };
        const unsigned int width = frame_out.stFrameInfo.nExtendWidth > 0 ? frame_out.stFrameInfo.nExtendWidth : frame_out.stFrameInfo.nWidth;
        const unsigned int height = frame_out.stFrameInfo.nExtendHeight > 0 ? frame_out.stFrameInfo.nExtendHeight : frame_out.stFrameInfo.nHeight;

        cv::Mat converted;
        if (frame_out.stFrameInfo.enPixelType == PixelType_Gvsp_BGR8_Packed) {
            converted = cv::Mat(static_cast<int>(height), static_cast<int>(width), CV_8UC3, frame_out.pBufAddr).clone();
        } else if (frame_out.stFrameInfo.enPixelType == PixelType_Gvsp_Mono8) {
            cv::Mat mono(static_cast<int>(height), static_cast<int>(width), CV_8UC1, frame_out.pBufAddr);
            cv::cvtColor(mono, converted, cv::COLOR_GRAY2BGR);
        } else {
            std::vector<unsigned char> buffer(width * height * 3U);
            MV_CC_PIXEL_CONVERT_PARAM_EX convert_param{};
            convert_param.nWidth = width;
            convert_param.nHeight = height;
            convert_param.enSrcPixelType = frame_out.stFrameInfo.enPixelType;
            convert_param.pSrcData = frame_out.pBufAddr;
            convert_param.nSrcDataLen = frame_out.stFrameInfo.nFrameLen;
            convert_param.enDstPixelType = PixelType_Gvsp_BGR8_Packed;
            convert_param.pDstBuffer = buffer.data();
            convert_param.nDstBufferSize = static_cast<unsigned int>(buffer.size());
            if (MV_CC_ConvertPixelTypeEx(handle_, &convert_param) != MV_OK) {
                release_guard();
                return false;
            }
            converted = cv::Mat(static_cast<int>(height), static_cast<int>(width), CV_8UC3, buffer.data()).clone();
        }
        release_guard();

        if (converted.empty()) {
            return false;
        }
        packet.image = converted;
        return true;
    }

    void close() override {
        cleanup();
    }

    std::string describe() const override {
        return description_;
    }

private:
    void cleanup() {
        if (handle_ != nullptr) {
            MV_CC_StopGrabbing(handle_);
            MV_CC_CloseDevice(handle_);
            MV_CC_DestroyHandle(handle_);
            handle_ = nullptr;
        }
        MV_CC_Finalize();
    }

    void* handle_ = nullptr;
    AppConfig config_{};
    std::string description_ = "MVS";
};
#else
class MvsFrameSource final : public IFrameSource {
public:
    bool open(const AppConfig&) override {
        std::cerr << "MVS support was not compiled in." << std::endl;
        return false;
    }
    bool read(FramePacket&) override { return false; }
    void close() override {}
    std::string describe() const override { return "MVS unavailable"; }
};
#endif

class SparseFlowEstimator final : public IFlowEstimator {
public:
    explicit SparseFlowEstimator(const AppConfig& config)
        : config_(config) {}

    MotionState process(const FramePacket& packet) override {
        const auto start = Clock::now();
        MotionState state;
        state.frame_id = packet.frame_id;

        cv::Mat frame = packet.image;
        if (frame.empty()) {
            return state;
        }

        if (fixed_points_.empty()) {
            fixed_points_ = generate_fixed_points(frame.cols, frame.rows, config_.motion_points);
        }
        state.fixed_points = fixed_points_;

        cv::Mat gray;
        if (frame.channels() == 3) {
            cv::cvtColor(frame, gray, cv::COLOR_BGR2GRAY);
        } else {
            gray = frame;
        }

        state.magnitudes.assign(fixed_points_.size(), 0.0F);
        if (!prev_gray_.empty()) {
            std::vector<cv::Point2f> next_points;
            std::vector<unsigned char> status;
            std::vector<float> errors;
            cv::calcOpticalFlowPyrLK(
                prev_gray_, gray, fixed_points_, next_points, status, errors,
                cv::Size(15, 15), 2,
                cv::TermCriteria(cv::TermCriteria::COUNT | cv::TermCriteria::EPS, 10, 0.03));

            std::vector<float> active_x;
            active_x.reserve(fixed_points_.size());
            for (std::size_t i = 0; i < fixed_points_.size(); ++i) {
                if (i >= next_points.size() || i >= status.size() || status[i] == 0U) {
                    continue;
                }
                const cv::Point2f delta = next_points[i] - fixed_points_[i];
                const float magnitude = std::sqrt(delta.x * delta.x + delta.y * delta.y);
                state.magnitudes[i] = magnitude;
                if (magnitude >= config_.motion_threshold) {
                    active_x.push_back(fixed_points_[i].x);
                    state.motion_count += 1;
                }
            }
            if (!active_x.empty()) {
                const int roi_width = std::max(32, frame.cols / 3);
                int best_x = 0;
                int best_count = -1;
                for (int x = 0; x <= frame.cols - roi_width; x += 2) {
                    const int count = static_cast<int>(std::count_if(active_x.begin(), active_x.end(),
                        [&](float point_x) { return point_x >= static_cast<float>(x) && point_x < static_cast<float>(x + roi_width); }));
                    if (count > best_count) {
                        best_count = count;
                        best_x = x;
                    }
                }
                state.roi_box = cv::Rect(best_x, 0, roi_width, frame.rows);
            }
        }

        prev_gray_ = gray.clone();
        state.motion_ms = to_ms(Clock::now() - start);
        return state;
    }

private:
    static std::vector<cv::Point2f> generate_fixed_points(int width, int height, int count) {
        const int cols = std::max(4, static_cast<int>(std::round(std::sqrt(static_cast<double>(count) * width / std::max(1, height)))));
        const int rows = std::max(4, static_cast<int>(std::ceil(static_cast<double>(count) / cols)));
        const float cell_w = static_cast<float>(width) / static_cast<float>(cols);
        const float cell_h = static_cast<float>(height) / static_cast<float>(rows);
        std::vector<cv::Point2f> points;
        points.reserve(count);
        for (int r = 0; r < rows && static_cast<int>(points.size()) < count; ++r) {
            for (int c = 0; c < cols && static_cast<int>(points.size()) < count; ++c) {
                points.emplace_back((static_cast<float>(c) + 0.5F) * cell_w, (static_cast<float>(r) + 0.5F) * cell_h);
            }
        }
        return points;
    }

    AppConfig config_;
    std::vector<cv::Point2f> fixed_points_;
    cv::Mat prev_gray_;
};

class OrtCudaDetector final : public IDetector {
public:
    bool initialize(const AppConfig& config) override {
        config_ = config;
#if defined(CENTRAL_CONTROL_WITH_ORT)
        try {
            env_ = std::make_unique<Ort::Env>(ORT_LOGGING_LEVEL_WARNING, "central_control");
            Ort::SessionOptions options;
            options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_EXTENDED);
            if (config.backend == "cuda" || config.backend == "auto") {
                OrtCUDAProviderOptions cuda_options{};
                cuda_options.device_id = 0;
                options.AppendExecutionProvider_CUDA(cuda_options);
                actual_backend_ = "onnxruntime-cuda";
            } else {
                actual_backend_ = "onnxruntime-cpu";
            }

#if defined(_WIN32)
            const std::wstring model_path = std::filesystem::path(config.model_onnx).wstring();
            session_ = std::make_unique<Ort::Session>(*env_, model_path.c_str(), options);
#else
            session_ = std::make_unique<Ort::Session>(*env_, config.model_onnx.c_str(), options);
#endif
            Ort::AllocatorWithDefaultOptions allocator;
            auto input_name = session_->GetInputNameAllocated(0, allocator);
            auto output_name = session_->GetOutputNameAllocated(0, allocator);
            input_name_ = input_name.get();
            output_name_ = output_name.get();
            const auto input_info = session_->GetInputTypeInfo(0).GetTensorTypeAndShapeInfo().GetShape();
            if (input_info.size() >= 4) {
                input_height_ = input_info[2] > 0 ? static_cast<int>(input_info[2]) : config.yolo_input_size;
                input_width_ = input_info[3] > 0 ? static_cast<int>(input_info[3]) : config.yolo_input_size;
            }
            return true;
        } catch (const std::exception& exc) {
            std::cerr << "Failed to initialize ONNX Runtime detector: " << exc.what()
                      << ". Falling back to OpenCV DNN." << std::endl;
            actual_backend_.clear();
            env_.reset();
            session_.reset();
        }
#endif

        try {
            net_ = cv::dnn::readNetFromONNX(config.model_onnx);
        } catch (const std::exception& exc) {
            std::cerr << "Failed to load ONNX model: " << exc.what() << std::endl;
            return false;
        }

        actual_backend_ = "opencv-dnn-cpu";
        if (config.backend == "cuda" || config.backend == "auto") {
            const std::string build_info = cv::getBuildInformation();
            const bool has_cuda = build_info.find("NVIDIA CUDA:                   YES") != std::string::npos;
            if (has_cuda) {
                net_.setPreferableBackend(cv::dnn::DNN_BACKEND_CUDA);
                net_.setPreferableTarget(cv::dnn::DNN_TARGET_CUDA);
                actual_backend_ = "opencv-dnn-cuda";
            } else {
                net_.setPreferableBackend(cv::dnn::DNN_BACKEND_OPENCV);
                net_.setPreferableTarget(cv::dnn::DNN_TARGET_CPU);
                actual_backend_ = "opencv-dnn-cpu";
            }
        } else {
            net_.setPreferableBackend(cv::dnn::DNN_BACKEND_OPENCV);
            net_.setPreferableTarget(cv::dnn::DNN_TARGET_CPU);
        }

        return true;
    }

    std::vector<Detection> detect(const FramePacket& packet, const MotionState& motion, double& detection_ms) override {
        std::vector<Detection> detections;
        if (packet.image.empty()) {
            detection_ms = 0.0;
            return detections;
        }

        const auto start = Clock::now();
        cv::Mat attended = packet.image.clone();
        if (motion.roi_box.has_value()) {
            cv::Mat blurred;
            cv::GaussianBlur(packet.image, blurred, cv::Size(config_.blur_kernel, config_.blur_kernel), 0.0);
            attended = blurred;
            packet.image(motion.roi_box.value()).copyTo(attended(motion.roi_box.value()));
        }

#if defined(CENTRAL_CONTROL_WITH_ORT)
        if (session_) {
            float scale = 1.0F;
            int pad_x = 0;
            int pad_y = 0;
            cv::Mat input = resize_keep_aspect(attended, input_width_, input_height_, scale, pad_x, pad_y);
            cv::Mat blob = cv::dnn::blobFromImage(input, 1.0 / 255.0, cv::Size(input_width_, input_height_), cv::Scalar(), true, false);
            const std::size_t tensor_size = static_cast<std::size_t>(blob.total());
            std::vector<int64_t> input_shape{1, 3, input_height_, input_width_};
            Ort::MemoryInfo memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
            Ort::Value input_tensor = Ort::Value::CreateTensor<float>(
                memory_info,
                reinterpret_cast<float*>(blob.data),
                tensor_size,
                input_shape.data(),
                input_shape.size());

            const char* input_names[] = {input_name_.c_str()};
            const char* output_names[] = {output_name_.c_str()};
            auto output_tensors = session_->Run(
                Ort::RunOptions{nullptr},
                input_names,
                &input_tensor,
                1,
                output_names,
                1);

            if (!output_tensors.empty() && output_tensors[0].IsTensor()) {
                auto shape = output_tensors[0].GetTensorTypeAndShapeInfo().GetShape();
                const float* data = output_tensors[0].GetTensorData<float>();
                int rows = 0;
                int cols = 0;
                std::vector<float> decoded_buffer;

                if (shape.size() == 3) {
                    const int dim1 = static_cast<int>(shape[1]);
                    const int dim2 = static_cast<int>(shape[2]);
                    if (dim1 < dim2) {
                        rows = dim2;
                        cols = dim1;
                        decoded_buffer.resize(static_cast<std::size_t>(rows) * static_cast<std::size_t>(cols));
                        for (int c = 0; c < cols; ++c) {
                            for (int r = 0; r < rows; ++r) {
                                decoded_buffer[static_cast<std::size_t>(r) * static_cast<std::size_t>(cols) + static_cast<std::size_t>(c)] =
                                    data[static_cast<std::size_t>(c) * static_cast<std::size_t>(rows) + static_cast<std::size_t>(r)];
                            }
                        }
                        data = decoded_buffer.data();
                    } else {
                        rows = dim1;
                        cols = dim2;
                    }
                } else if (shape.size() == 2) {
                    rows = static_cast<int>(shape[0]);
                    cols = static_cast<int>(shape[1]);
                }

                decode_yolo_rows(data, rows, cols, packet, motion, scale, pad_x, pad_y, detections);
            }

            detection_ms = to_ms(Clock::now() - start);
            return detections;
        }
#endif

        float scale = 1.0F;
        int pad_x = 0;
        int pad_y = 0;
        cv::Mat input = resize_keep_aspect(attended, config_.yolo_input_size, config_.yolo_input_size, scale, pad_x, pad_y);
        cv::Mat blob = cv::dnn::blobFromImage(input, 1.0 / 255.0, cv::Size(config_.yolo_input_size, config_.yolo_input_size), cv::Scalar(), true, false);
        net_.setInput(blob);
        cv::Mat output = net_.forward();

        std::vector<cv::Rect> boxes;
        std::vector<float> confidences;
        std::vector<std::string> attentions;

        const int dimensions = output.dims;
        int rows = 0;
        int cols = 0;
        cv::Mat decoded;
        if (dimensions == 3 && output.size[1] < output.size[2]) {
            decoded = cv::Mat(output.size[1], output.size[2], CV_32F, output.ptr<float>()).t();
            rows = decoded.rows;
            cols = decoded.cols;
        } else if (dimensions == 3) {
            decoded = cv::Mat(output.size[1], output.size[2], CV_32F, output.ptr<float>());
            rows = decoded.rows;
            cols = decoded.cols;
        } else if (dimensions == 2) {
            decoded = output;
            rows = decoded.rows;
            cols = decoded.cols;
        } else {
            detection_ms = to_ms(Clock::now() - start);
            return detections;
        }

        decode_yolo_rows(decoded.ptr<float>(0), rows, cols, packet, motion, scale, pad_x, pad_y, detections);
        detection_ms = to_ms(Clock::now() - start);
        return detections;
    }

    std::string backend_name() const override {
        return actual_backend_;
    }

private:
    void decode_yolo_rows(const float* data,
                          int rows,
                          int cols,
                          const FramePacket& packet,
                          const MotionState& motion,
                          float scale,
                          int pad_x,
                          int pad_y,
                          std::vector<Detection>& detections) const {
        if (rows <= 0 || cols <= 0 || data == nullptr) {
            return;
        }

        std::vector<cv::Rect> boxes;
        std::vector<float> confidences;
        std::vector<std::string> attentions;

        for (int i = 0; i < rows; ++i) {
            const float* row = data + static_cast<std::size_t>(i) * static_cast<std::size_t>(cols);
            if (cols < 6) {
                continue;
            }
            const int class_offset = 4;
            int best_class = -1;
            float best_conf = 0.0F;
            for (int c = class_offset; c < cols; ++c) {
                if (row[c] > best_conf) {
                    best_conf = row[c];
                    best_class = c - class_offset;
                }
            }
            if (best_class != 0) {
                continue;
            }
            if (best_conf < std::min(config_.bg_conf_threshold, config_.roi_conf_threshold)) {
                continue;
            }

            const float cx = row[0];
            const float cy = row[1];
            const float w = row[2];
            const float h = row[3];

            float x1 = (cx - w * 0.5F - static_cast<float>(pad_x)) / scale;
            float y1 = (cy - h * 0.5F - static_cast<float>(pad_y)) / scale;
            float x2 = (cx + w * 0.5F - static_cast<float>(pad_x)) / scale;
            float y2 = (cy + h * 0.5F - static_cast<float>(pad_y)) / scale;

            x1 = std::clamp(x1, 0.0F, static_cast<float>(packet.image.cols - 1));
            y1 = std::clamp(y1, 0.0F, static_cast<float>(packet.image.rows - 1));
            x2 = std::clamp(x2, 0.0F, static_cast<float>(packet.image.cols - 1));
            y2 = std::clamp(y2, 0.0F, static_cast<float>(packet.image.rows - 1));
            if (x2 <= x1 || y2 <= y1) {
                continue;
            }

            const cv::Rect2f box(x1, y1, x2 - x1, y2 - y1);
            const bool in_roi = motion.roi_box.has_value() && ((box & cv::Rect2f(motion.roi_box.value())).area() > 0.0F);
            const float threshold = in_roi ? config_.roi_conf_threshold : config_.bg_conf_threshold;
            if (best_conf < threshold) {
                continue;
            }

            boxes.emplace_back(static_cast<int>(x1), static_cast<int>(y1), static_cast<int>(x2 - x1), static_cast<int>(y2 - y1));
            confidences.push_back(best_conf);
            attentions.emplace_back(in_roi ? "ROI" : "BG");
        }

        std::vector<int> indices;
        cv::dnn::NMSBoxes(boxes, confidences, 0.1F, config_.nms_iou, indices);
        detections.reserve(indices.size());
        for (const int index : indices) {
            Detection det;
            det.bbox = cv::Rect2f(static_cast<float>(boxes[index].x),
                                  static_cast<float>(boxes[index].y),
                                  static_cast<float>(boxes[index].width),
                                  static_cast<float>(boxes[index].height));
            det.confidence = confidences[index];
            det.class_name = "person";
            det.attention = attentions[index];
            detections.push_back(det);
        }
    }

    AppConfig config_{};
    cv::dnn::Net net_;
    std::string actual_backend_ = "uninitialized";
#if defined(CENTRAL_CONTROL_WITH_ORT)
    std::unique_ptr<Ort::Env> env_;
    std::unique_ptr<Ort::Session> session_;
    std::string input_name_;
    std::string output_name_;
    int input_width_ = 640;
    int input_height_ = 640;
#endif
};

class RknnDetector final : public IDetector {
public:
    bool initialize(const AppConfig&) override {
        std::cerr << "RKNN detector is a placeholder in this Windows-first migration. Use backend=cuda/cpu for now." << std::endl;
        return true;
    }

    std::vector<Detection> detect(const FramePacket&, const MotionState&, double& detection_ms) override {
        detection_ms = 0.0;
        return {};
    }

    std::string backend_name() const override {
        return "rknn-placeholder";
    }
};

struct TrackInternal {
    int track_id = -1;
    cv::Rect2f bbox;
    cv::KalmanFilter kf;
    int age = 1;
    int time_since_update = 0;
    std::deque<cv::Point2f> history;
};

class SimpleTracker final : public ITracker {
public:
    std::vector<TrackView> update(const std::vector<Detection>& detections, std::uint64_t) override {
        for (auto& track : tracks_) {
            predict(track);
        }

        std::vector<int> assigned_track(tracks_.size(), -1);
        std::vector<int> assigned_det(detections.size(), -1);

        while (true) {
            float best_iou = 0.3F;
            int best_track = -1;
            int best_det = -1;
            for (std::size_t ti = 0; ti < tracks_.size(); ++ti) {
                if (assigned_track[ti] >= 0) {
                    continue;
                }
                for (std::size_t di = 0; di < detections.size(); ++di) {
                    if (assigned_det[di] >= 0) {
                        continue;
                    }
                    const float iou = compute_iou(tracks_[ti].bbox, detections[di].bbox);
                    if (iou > best_iou) {
                        best_iou = iou;
                        best_track = static_cast<int>(ti);
                        best_det = static_cast<int>(di);
                    }
                }
            }
            if (best_track < 0 || best_det < 0) {
                break;
            }
            assigned_track[best_track] = best_det;
            assigned_det[best_det] = best_track;
            correct(tracks_[best_track], detections[best_det].bbox);
        }

        for (std::size_t di = 0; di < detections.size(); ++di) {
            if (assigned_det[di] < 0) {
                tracks_.push_back(make_track(detections[di].bbox));
            }
        }

        for (auto it = tracks_.begin(); it != tracks_.end();) {
            if (it->time_since_update > 30) {
                it = tracks_.erase(it);
            } else {
                ++it;
            }
        }

        std::vector<TrackView> result;
        for (const auto& track : tracks_) {
            TrackView view;
            view.track_id = track.track_id;
            view.bbox = track.bbox;
            view.history = track.history;
            view.age = track.age;
            result.push_back(view);
        }
        return result;
    }

private:
    static TrackInternal make_track(const cv::Rect2f& bbox) {
        TrackInternal track;
        track.track_id = next_track_id_++;
        track.bbox = bbox;
        track.kf.init(8, 4);
        track.kf.transitionMatrix = (cv::Mat_<float>(8, 8) <<
            1, 0, 0, 0, 1, 0, 0, 0,
            0, 1, 0, 0, 0, 1, 0, 0,
            0, 0, 1, 0, 0, 0, 1, 0,
            0, 0, 0, 1, 0, 0, 0, 1,
            0, 0, 0, 0, 1, 0, 0, 0,
            0, 0, 0, 0, 0, 1, 0, 0,
            0, 0, 0, 0, 0, 0, 1, 0,
            0, 0, 0, 0, 0, 0, 0, 1);
        cv::setIdentity(track.kf.measurementMatrix);
        cv::setIdentity(track.kf.processNoiseCov, cv::Scalar(1e-2));
        cv::setIdentity(track.kf.measurementNoiseCov, cv::Scalar(1e-1));
        cv::setIdentity(track.kf.errorCovPost, cv::Scalar(1.0));
        track.kf.statePost = (cv::Mat_<float>(8, 1) <<
            bbox.x, bbox.y, bbox.width, bbox.height, 0, 0, 0, 0);
        const cv::Point2f center(bbox.x + bbox.width * 0.5F, bbox.y + bbox.height * 0.5F);
        track.history.push_back(center);
        return track;
    }

    static void predict(TrackInternal& track) {
        const cv::Mat prediction = track.kf.predict();
        track.bbox = cv::Rect2f(prediction.at<float>(0), prediction.at<float>(1), prediction.at<float>(2), prediction.at<float>(3));
        track.time_since_update += 1;
        track.age += 1;
        const cv::Point2f center(track.bbox.x + track.bbox.width * 0.5F, track.bbox.y + track.bbox.height * 0.5F);
        track.history.push_back(center);
        if (track.history.size() > 30) {
            track.history.pop_front();
        }
    }

    static void correct(TrackInternal& track, const cv::Rect2f& bbox) {
        cv::Mat measurement = (cv::Mat_<float>(4, 1) << bbox.x, bbox.y, bbox.width, bbox.height);
        track.kf.correct(measurement);
        track.bbox = bbox;
        track.time_since_update = 0;
        const cv::Point2f center(bbox.x + bbox.width * 0.5F, bbox.y + bbox.height * 0.5F);
        track.history.push_back(center);
        if (track.history.size() > 30) {
            track.history.pop_front();
        }
    }

    inline static int next_track_id_ = 0;
    std::vector<TrackInternal> tracks_;
};

class OverlayRenderer final : public IRenderer {
public:
    explicit OverlayRenderer(const AppConfig& config)
        : config_(config) {}

    cv::Mat render(
        const FramePacket& packet,
        const std::optional<MotionState>& motion,
        const std::optional<DetectionState>& detection,
        const RuntimeStats& stats) override {
        cv::Mat vis = packet.image.clone();
        if (vis.empty()) {
            return vis;
        }

        if (motion.has_value()) {
            for (std::size_t i = 0; i < motion->fixed_points.size(); ++i) {
                const auto point = motion->fixed_points[i];
                const float magnitude = i < motion->magnitudes.size() ? motion->magnitudes[i] : 0.0F;
                cv::circle(vis, point, 3, motion_color(magnitude, config_.motion_threshold), -1);
            }
            if (motion->roi_box.has_value()) {
                cv::rectangle(vis, motion->roi_box.value(), cv::Scalar(255, 0, 255), 3);
                cv::putText(vis, "ROI", cv::Point(motion->roi_box->x, std::max(20, motion->roi_box->y - 8)),
                            cv::FONT_HERSHEY_SIMPLEX, 0.55, cv::Scalar(255, 0, 255), 2, cv::LINE_AA);
            }
        }

        if (detection.has_value()) {
            for (const auto& det : detection->detections) {
                const cv::Scalar color = det.attention == "ROI" ? cv::Scalar(0, 255, 255) : cv::Scalar(0, 165, 255);
                cv::rectangle(vis, det.bbox, color, 2);
                std::ostringstream label;
                label << det.class_name << " " << std::fixed << std::setprecision(2) << det.confidence << " [" << det.attention << "]";
                cv::putText(vis, label.str(), cv::Point(static_cast<int>(det.bbox.x), std::max(20, static_cast<int>(det.bbox.y) - 8)),
                            cv::FONT_HERSHEY_SIMPLEX, 0.5, color, 2, cv::LINE_AA);
            }
            for (const auto& track : detection->tracks) {
                cv::rectangle(vis, track.bbox, cv::Scalar(255, 0, 0), 2);
                cv::putText(vis, "Track " + std::to_string(track.track_id),
                            cv::Point(static_cast<int>(track.bbox.x), std::max(20, static_cast<int>(track.bbox.y) - 26)),
                            cv::FONT_HERSHEY_SIMPLEX, 0.55, cv::Scalar(255, 0, 0), 2, cv::LINE_AA);
                for (std::size_t i = 1; i < track.history.size(); ++i) {
                    cv::line(vis, track.history[i - 1], track.history[i], cv::Scalar(255, 0, 0), 2);
                }
            }
        }

        draw_stats(vis, motion, detection, stats);
        return vis;
    }

private:
    void draw_stats(const cv::Mat& target,
                    const std::optional<MotionState>& motion,
                    const std::optional<DetectionState>& detection,
                    const RuntimeStats& stats) const {
        cv::Mat& vis = const_cast<cv::Mat&>(target);
        const std::vector<std::string> lines = {
            "capture_fps: " + format_value(stats.capture_fps),
            "render_fps: " + format_value(stats.render_fps),
            "motion_pts: " + std::to_string(motion.has_value() ? motion->motion_count : 0),
            "detections: " + std::to_string(detection.has_value() ? detection->detections.size() : 0),
            "tracks: " + std::to_string(detection.has_value() ? detection->tracks.size() : 0),
            "detect_ms: " + format_value(stats.detection_ms),
            "lag_frames: " + std::to_string(stats.inference_lag_frames),
            "result_age_ms: " + format_value(stats.result_age_ms),
            "dropped_detect_jobs: " + std::to_string(stats.dropped_detect_jobs),
            "recording: " + std::string(config_.record_rendered ? "ON" : "OFF"),
            "headless: " + std::string(config_.headless ? "ON" : "OFF")
        };

        const int panel_w = 360;
        const int panel_h = 18 + static_cast<int>(lines.size()) * 24;
        cv::Mat overlay = vis.clone();
        cv::rectangle(overlay, cv::Rect(8, 8, panel_w, panel_h), cv::Scalar(0, 0, 0), -1);
        cv::addWeighted(overlay, 0.58, vis, 0.42, 0.0, vis);
        int y = 32;
        for (const auto& line : lines) {
            cv::putText(vis, line, cv::Point(18, y), cv::FONT_HERSHEY_SIMPLEX, 0.58, cv::Scalar(0, 255, 255), 2, cv::LINE_AA);
            y += 24;
        }
    }

    static std::string format_value(double value) {
        std::ostringstream oss;
        oss << std::fixed << std::setprecision(1) << value;
        return oss.str();
    }

    AppConfig config_;
};

class VideoFileSink final : public IOutputSink {
public:
    bool open(const AppConfig& config) override {
        config_ = config;
        path_ = make_timestamped_output_path(config);
        std::filesystem::create_directories(std::filesystem::path(path_).parent_path());
        return true;
    }

    void write(const cv::Mat& frame) override {
        if (frame.empty()) {
            return;
        }
        if (!writer_.isOpened()) {
            writer_.open(path_, cv::VideoWriter::fourcc('m', 'p', '4', 'v'),
                         static_cast<double>(config_.target_fps), frame.size());
            if (!writer_.isOpened()) {
                std::cerr << "Failed to open video writer: " << path_ << std::endl;
                return;
            }
            writer_frame_size_ = frame.size();
        }

        if (frame.size() == writer_frame_size_) {
            writer_.write(frame);
        } else {
            cv::Mat resized;
            cv::resize(frame, resized, writer_frame_size_, 0.0, 0.0, cv::INTER_AREA);
            writer_.write(resized);
        }
    }

    void close() override {
        writer_.release();
    }

    std::string output_path() const override {
        return path_;
    }

private:
    AppConfig config_{};
    cv::VideoWriter writer_;
    std::string path_;
    cv::Size writer_frame_size_{};
};

class NullSink final : public IOutputSink {
public:
    bool open(const AppConfig&) override { return true; }
    void write(const cv::Mat&) override {}
    void close() override {}
    std::string output_path() const override { return {}; }
};

class MockActuator final : public IActuator {
public:
    bool initialize(const AppConfig&) override {
        return true;
    }

    void apply(const ControlCommand& command) override {
        last_command_ = command;
    }

    void stop() override {
        last_command_ = ControlCommand{"stop", 0.0F, 0.0F};
    }

    std::string describe() const override {
        return "MockActuator";
    }

private:
    ControlCommand last_command_{};
};

class PcanActuator final : public IActuator {
public:
    bool initialize(const AppConfig&) override {
        std::cerr << "PCAN integration placeholder: build with PCAN SDK to enable real gimbal control." << std::endl;
        return true;
    }

    void apply(const ControlCommand&) override {}
    void stop() override {}
    std::string describe() const override { return "PCAN placeholder"; }
};

}  // namespace

std::shared_ptr<IFrameSource> create_frame_source(const AppConfig& config) {
    if (config.source == "mvs") {
        return std::make_shared<MvsFrameSource>();
    }
    return std::make_shared<VideoCaptureSource>();
}

std::shared_ptr<IFlowEstimator> create_flow_estimator(const AppConfig& config) {
    return std::make_shared<SparseFlowEstimator>(config);
}

std::shared_ptr<IDetector> create_detector(const AppConfig& config) {
    if (config.backend == "rknn") {
        return std::make_shared<RknnDetector>();
    }
    return std::make_shared<OrtCudaDetector>();
}

std::shared_ptr<ITracker> create_tracker(const AppConfig&) {
    return std::make_shared<SimpleTracker>();
}

std::shared_ptr<IRenderer> create_renderer(const AppConfig& config) {
    return std::make_shared<OverlayRenderer>(config);
}

std::shared_ptr<IOutputSink> create_output_sink(const AppConfig& config) {
    if (config.record_rendered) {
        return std::make_shared<VideoFileSink>();
    }
    return std::make_shared<NullSink>();
}

std::shared_ptr<IActuator> create_actuator(const AppConfig& config) {
    if (config.mock_gimbal) {
        return std::make_shared<MockActuator>();
    }
    return std::make_shared<PcanActuator>();
}

int list_available_sources(const AppConfig& config) {
#if defined(CENTRAL_CONTROL_WITH_MVS)
    if (config.source == "mvs" || config.source.empty()) {
        if (MV_CC_Initialize() != MV_OK) {
            std::cerr << "MVS SDK initialize failed" << std::endl;
            return 1;
        }
        MV_CC_DEVICE_INFO_LIST device_list{};
        const unsigned int layer_mask = MV_GIGE_DEVICE | MV_USB_DEVICE;
        const int ret = MV_CC_EnumDevices(layer_mask, &device_list);
        if (ret != MV_OK) {
            std::cerr << "Failed to enumerate MVS devices: " << ret << std::endl;
            MV_CC_Finalize();
            return 1;
        }
        std::cout << "Detected MVS cameras:" << std::endl;
        for (unsigned int i = 0; i < device_list.nDeviceNum; ++i) {
            auto* info = device_list.pDeviceInfo[i];
            if (info == nullptr) {
                continue;
            }
            if ((info->nTLayerType & MV_USB_DEVICE) != 0U) {
                std::cout << "  [" << i << "] model=" << reinterpret_cast<const char*>(info->SpecialInfo.stUsb3VInfo.chModelName)
                          << " serial=" << reinterpret_cast<const char*>(info->SpecialInfo.stUsb3VInfo.chSerialNumber) << std::endl;
            } else if ((info->nTLayerType & MV_GIGE_DEVICE) != 0U) {
                std::cout << "  [" << i << "] model=" << reinterpret_cast<const char*>(info->SpecialInfo.stGigEInfo.chModelName)
                          << " serial=" << reinterpret_cast<const char*>(info->SpecialInfo.stGigEInfo.chSerialNumber) << std::endl;
            }
        }
        MV_CC_Finalize();
        return 0;
    }
#endif
    std::cout << "Non-MVS source listing is not implemented; pass --source mvs for device enumeration." << std::endl;
    return 0;
}

}  // namespace central_control
