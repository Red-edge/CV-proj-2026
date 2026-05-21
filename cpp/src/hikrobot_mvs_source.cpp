#include "cvproj/hikrobot_mvs_source.hpp"

#include <chrono>
#include <sstream>
#include <utility>

#include <opencv2/imgproc.hpp>

#ifdef CVPROJ_HAS_HIKROBOT_MVS
#include "MvCameraControl.h"
#endif

namespace cvproj {

namespace {
double now_seconds() {
    using clock = std::chrono::steady_clock;
    const auto now = clock::now().time_since_epoch();
    return std::chrono::duration<double>(now).count();
}
}  // namespace

struct HikrobotMvsSource::Impl {
#ifdef CVPROJ_HAS_HIKROBOT_MVS
    void* handle = nullptr;
    std::vector<unsigned char> converted_buffer;
    std::int64_t frame_counter = 0;
#endif
};

HikrobotMvsSource::HikrobotMvsSource(HikrobotMvsConfig config)
    : config_(std::move(config)), impl_(std::make_unique<Impl>()) {}

HikrobotMvsSource::~HikrobotMvsSource() {
    close();
}

bool HikrobotMvsSource::open(std::string* error) {
#ifdef CVPROJ_HAS_HIKROBOT_MVS
    close();

    MV_CC_DEVICE_INFO_LIST devices{};
    const int ret = MV_CC_EnumDevices(MV_GIGE_DEVICE | MV_USB_DEVICE, &devices);
    if (ret != MV_OK) {
        if (error) {
            *error = "MV_CC_EnumDevices failed with code " + std::to_string(ret);
        }
        return false;
    }
    if (devices.nDeviceNum == 0) {
        if (error) {
            *error = "No Hikrobot devices found through MVS SDK";
        }
        return false;
    }

    MV_CC_DEVICE_INFO* selected = nullptr;
    for (unsigned int i = 0; i < devices.nDeviceNum; ++i) {
        auto* info = devices.pDeviceInfo[i];
        if (info == nullptr) {
            continue;
        }
        if (config_.serial_number.empty()) {
            selected = info;
            break;
        }
        if (info->nTLayerType == MV_USB_DEVICE) {
            const std::string serial(reinterpret_cast<const char*>(info->SpecialInfo.stUsb3VInfo.chSerialNumber));
            if (serial.find(config_.serial_number) != std::string::npos) {
                selected = info;
                break;
            }
        }
    }

    if (selected == nullptr) {
        if (error) {
            *error = "Requested Hikrobot serial not found: " + config_.serial_number;
        }
        return false;
    }

    int status = MV_CC_CreateHandle(&impl_->handle, selected);
    if (status != MV_OK) {
        if (error) {
            *error = "MV_CC_CreateHandle failed with code " + std::to_string(status);
        }
        return false;
    }

    status = MV_CC_OpenDevice(impl_->handle, MV_ACCESS_Exclusive, 0);
    if (status != MV_OK) {
        if (error) {
            *error = "MV_CC_OpenDevice failed with code " + std::to_string(status);
        }
        close();
        return false;
    }

    MV_CC_SetEnumValue(impl_->handle, "TriggerMode", 0);
    MV_CC_SetIntValueEx(impl_->handle, "Width", config_.width);
    MV_CC_SetIntValueEx(impl_->handle, "Height", config_.height);
    MV_CC_SetBoolValue(impl_->handle, "AcquisitionFrameRateEnable", true);
    MV_CC_SetFloatValue(impl_->handle, "AcquisitionFrameRate", static_cast<float>(config_.target_fps));

    if (config_.exposure_us > 0.0) {
        MV_CC_SetEnumValue(impl_->handle, "ExposureAuto", 0);
        MV_CC_SetFloatValue(impl_->handle, "ExposureTime", static_cast<float>(config_.exposure_us));
    }
    if (config_.gain > 0.0) {
        MV_CC_SetFloatValue(impl_->handle, "Gain", static_cast<float>(config_.gain));
    }

    status = MV_CC_StartGrabbing(impl_->handle);
    if (status != MV_OK) {
        if (error) {
            *error = "MV_CC_StartGrabbing failed with code " + std::to_string(status);
        }
        close();
        return false;
    }

    impl_->frame_counter = 0;
    return true;
#else
    if (error) {
        *error = "This build does not include Hikrobot MVS support. Set HIKROBOT_MVS_ROOT and rebuild.";
    }
    return false;
#endif
}

bool HikrobotMvsSource::read(FramePacket& packet, int timeout_ms, std::string* error) {
    return read_impl(packet, timeout_ms, false, error);
}

bool HikrobotMvsSource::read_latest(FramePacket& packet, int timeout_ms, std::string* error) {
    return read_impl(packet, timeout_ms, true, error);
}

bool HikrobotMvsSource::read_impl(FramePacket& packet,
                                  int timeout_ms,
                                  bool latest_only,
                                  std::string* error) {
#ifdef CVPROJ_HAS_HIKROBOT_MVS
    if (impl_->handle == nullptr) {
        if (error) {
            *error = "Hikrobot camera is not open";
        }
        return false;
    }

    MV_FRAME_OUT frame_out{};
    const int ret = MV_CC_GetImageBuffer(impl_->handle, &frame_out, timeout_ms);
    if (ret != MV_OK) {
        if (error) {
            *error = "MV_CC_GetImageBuffer failed with code " + std::to_string(ret);
        }
        return false;
    }

    if (latest_only) {
        while (true) {
            MV_FRAME_OUT newer_frame{};
            const int newer_ret = MV_CC_GetImageBuffer(impl_->handle, &newer_frame, 0);
            if (newer_ret != MV_OK) {
                break;
            }

            MV_CC_FreeImageBuffer(impl_->handle, &frame_out);
            frame_out = newer_frame;
        }
    }

    cv::Mat bgr;
    const auto width = static_cast<int>(frame_out.stFrameInfo.nWidth);
    const auto height = static_cast<int>(frame_out.stFrameInfo.nHeight);
    const auto pixel_type = frame_out.stFrameInfo.enPixelType;

    if (pixel_type == PixelType_Gvsp_BGR8_Packed) {
        bgr = cv::Mat(height, width, CV_8UC3, frame_out.pBufAddr).clone();
    } else if (pixel_type == PixelType_Gvsp_Mono8) {
        cv::Mat mono(height, width, CV_8UC1, frame_out.pBufAddr);
        cv::cvtColor(mono, bgr, cv::COLOR_GRAY2BGR);
    } else {
        MV_CC_PIXEL_CONVERT_PARAM convert{};
        convert.nWidth = frame_out.stFrameInfo.nWidth;
        convert.nHeight = frame_out.stFrameInfo.nHeight;
        convert.pSrcData = frame_out.pBufAddr;
        convert.nSrcDataLen = frame_out.stFrameInfo.nFrameLen;
        convert.enSrcPixelType = frame_out.stFrameInfo.enPixelType;
        convert.enDstPixelType = PixelType_Gvsp_BGR8_Packed;

        const std::size_t dst_size = static_cast<std::size_t>(width) * static_cast<std::size_t>(height) * 3U;
        impl_->converted_buffer.resize(dst_size);
        convert.pDstBuffer = impl_->converted_buffer.data();
        convert.nDstBufferSize = static_cast<unsigned int>(impl_->converted_buffer.size());

        const int cvt_ret = MV_CC_ConvertPixelType(impl_->handle, &convert);
        if (cvt_ret != MV_OK) {
            MV_CC_FreeImageBuffer(impl_->handle, &frame_out);
            if (error) {
                *error = "MV_CC_ConvertPixelType failed with code " + std::to_string(cvt_ret);
            }
            return false;
        }
        bgr = cv::Mat(height, width, CV_8UC3, impl_->converted_buffer.data()).clone();
    }

    packet.frame_bgr = std::move(bgr);
    packet.frame_id = impl_->frame_counter++;
    packet.timestamp_seconds = now_seconds();

    MV_CC_FreeImageBuffer(impl_->handle, &frame_out);
    return true;
#else
    if (error) {
        *error = "This build does not include Hikrobot MVS support.";
    }
    return false;
#endif
}

void HikrobotMvsSource::close() {
#ifdef CVPROJ_HAS_HIKROBOT_MVS
    if (impl_ == nullptr || impl_->handle == nullptr) {
        return;
    }
    MV_CC_StopGrabbing(impl_->handle);
    MV_CC_CloseDevice(impl_->handle);
    MV_CC_DestroyHandle(impl_->handle);
    impl_->handle = nullptr;
    impl_->converted_buffer.clear();
#endif
}

std::string HikrobotMvsSource::name() const {
    std::ostringstream oss;
    oss << "hikrobot-mvs";
    if (!config_.serial_number.empty()) {
        oss << ":" << config_.serial_number;
    }
    return oss.str();
}

}  // namespace cvproj
