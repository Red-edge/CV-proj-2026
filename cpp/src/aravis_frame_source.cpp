#include "cvproj/aravis_frame_source.hpp"

#include <chrono>
#include <sstream>
#include <string>
#include <utility>

#include <opencv2/imgproc.hpp>

#ifdef CVPROJ_HAS_ARAVIS
#include <arv.h>
#endif

namespace cvproj {

namespace {
double now_seconds() {
    using clock = std::chrono::steady_clock;
    const auto now = clock::now().time_since_epoch();
    return std::chrono::duration<double>(now).count();
}

#ifdef CVPROJ_HAS_ARAVIS
std::string to_error_message(GError* error) {
    if (error == nullptr) {
        return "unknown Aravis error";
    }
    const std::string message = error->message != nullptr ? error->message : "unknown Aravis error";
    g_clear_error(&error);
    return message;
}
#endif
}  // namespace

struct AravisFrameSource::Impl {
#ifdef CVPROJ_HAS_ARAVIS
    ArvCamera* camera = nullptr;
    ArvStream* stream = nullptr;
    std::int64_t frame_counter = 0;
#endif
};

AravisFrameSource::AravisFrameSource(AravisConfig config)
    : config_(std::move(config)), impl_(std::make_unique<Impl>()) {}

AravisFrameSource::~AravisFrameSource() {
    close();
}

std::string AravisFrameSource::resolve_device_id(std::string* error) {
#ifdef CVPROJ_HAS_ARAVIS
    arv_update_device_list();
    const unsigned int device_count = arv_get_n_devices();
    if (device_count == 0) {
        if (error) {
            *error = "Aravis found no GenICam devices";
        }
        return {};
    }

    for (unsigned int i = 0; i < device_count; ++i) {
        const char* device_id = arv_get_device_id(i);
        const char* serial = arv_get_device_serial_nbr(i);

        if (!config_.device_id.empty() && device_id != nullptr && config_.device_id == device_id) {
            return device_id;
        }
        if (!config_.serial_number.empty() && serial != nullptr && config_.serial_number == serial) {
            return device_id != nullptr ? std::string(device_id) : std::string();
        }
    }

    if (!config_.device_id.empty()) {
        return config_.device_id;
    }

    const char* first_device = arv_get_device_id(0);
    return first_device != nullptr ? std::string(first_device) : std::string();
#else
    if (error) {
        *error = "This build does not include Aravis support.";
    }
    return {};
#endif
}

bool AravisFrameSource::open(std::string* error) {
#ifdef CVPROJ_HAS_ARAVIS
    close();

    const std::string device_id = resolve_device_id(error);
    if (device_id.empty()) {
        if (error && error->empty()) {
            *error = "Unable to resolve Aravis device ID";
        }
        return false;
    }

    GError* gerror = nullptr;
    impl_->camera = arv_camera_new(device_id.c_str(), &gerror);
    if (impl_->camera == nullptr) {
        if (error) {
            *error = "arv_camera_new failed: " + to_error_message(gerror);
        }
        return false;
    }

    arv_camera_set_acquisition_mode(impl_->camera, ARV_ACQUISITION_MODE_CONTINUOUS, &gerror);
    if (gerror != nullptr && error) {
        *error = "Failed to set acquisition mode: " + to_error_message(gerror);
        close();
        return false;
    }

    arv_camera_set_exposure_mode(impl_->camera, ARV_EXPOSURE_MODE_TIMED, &gerror);
    if (gerror != nullptr) {
        g_clear_error(&gerror);
    }

    arv_camera_set_exposure_time_auto(impl_->camera, ARV_AUTO_OFF, &gerror);
    if (gerror != nullptr) {
        g_clear_error(&gerror);
    }

    if (arv_camera_is_gain_auto_available(impl_->camera, &gerror)) {
        arv_camera_set_gain_auto(impl_->camera, ARV_AUTO_OFF, &gerror);
        if (gerror != nullptr) {
            g_clear_error(&gerror);
        }
    } else if (gerror != nullptr) {
        g_clear_error(&gerror);
    }

    if (!config_.pixel_format.empty()) {
        arv_camera_set_pixel_format_from_string(impl_->camera, config_.pixel_format.c_str(), &gerror);
        if (gerror != nullptr) {
            g_clear_error(&gerror);
        }
    }

    int offset_x = 0;
    int offset_y = 0;
    int current_width = 0;
    int current_height = 0;
    arv_camera_get_region(impl_->camera, &offset_x, &offset_y, &current_width, &current_height, &gerror);
    if (gerror != nullptr) {
        g_clear_error(&gerror);
    }

    if (config_.width > 0 && config_.height > 0) {
        arv_camera_set_region(impl_->camera, offset_x, offset_y, config_.width, config_.height, &gerror);
        if (gerror != nullptr) {
            g_clear_error(&gerror);
        }
    }

    arv_camera_set_frame_rate_enable(impl_->camera, TRUE, &gerror);
    if (gerror != nullptr) {
        g_clear_error(&gerror);
    }

    if (config_.target_fps > 0.0) {
        arv_camera_set_frame_rate(impl_->camera, config_.target_fps, &gerror);
        if (gerror != nullptr) {
            g_clear_error(&gerror);
        }
    }

    if (config_.exposure_us > 0.0) {
        arv_camera_set_exposure_time(impl_->camera, config_.exposure_us, &gerror);
        if (gerror != nullptr) {
            g_clear_error(&gerror);
        }
    }

    if (config_.gain_db > 0.0 && arv_camera_is_gain_available(impl_->camera, &gerror)) {
        arv_camera_set_gain(impl_->camera, config_.gain_db, &gerror);
        if (gerror != nullptr) {
            g_clear_error(&gerror);
        }
    } else if (gerror != nullptr) {
        g_clear_error(&gerror);
    }

    const guint payload = arv_camera_get_payload(impl_->camera, &gerror);
    if (gerror != nullptr) {
        if (error) {
            *error = "Failed to query payload: " + to_error_message(gerror);
        }
        close();
        return false;
    }

    impl_->stream = arv_camera_create_stream(impl_->camera, nullptr, nullptr, &gerror);
    if (impl_->stream == nullptr || gerror != nullptr) {
        if (error) {
            *error = "Failed to create Aravis stream: " + to_error_message(gerror);
        }
        close();
        return false;
    }

    arv_stream_set_emit_signals(impl_->stream, FALSE);
    for (int i = 0; i < config_.stream_buffers; ++i) {
        arv_stream_push_buffer(impl_->stream, arv_buffer_new_allocate(payload));
    }

    arv_camera_start_acquisition(impl_->camera, &gerror);
    if (gerror != nullptr) {
        if (error) {
            *error = "Failed to start acquisition: " + to_error_message(gerror);
        }
        close();
        return false;
    }

    impl_->frame_counter = 0;
    return true;
#else
    if (error) {
        *error = "This build does not include Aravis support.";
    }
    return false;
#endif
}

bool AravisFrameSource::read(FramePacket& packet, int timeout_ms, std::string* error) {
    return read_impl(packet, timeout_ms, false, error);
}

bool AravisFrameSource::read_latest(FramePacket& packet, int timeout_ms, std::string* error) {
    return read_impl(packet, timeout_ms, true, error);
}

bool AravisFrameSource::read_impl(FramePacket& packet,
                                  int timeout_ms,
                                  bool latest_only,
                                  std::string* error) {
#ifdef CVPROJ_HAS_ARAVIS
    if (impl_->camera == nullptr || impl_->stream == nullptr) {
        if (error) {
            *error = "Aravis camera is not open";
        }
        return false;
    }

    const auto deadline =
        std::chrono::steady_clock::now() + std::chrono::milliseconds(std::max(1, timeout_ms));
    ArvBuffer* buffer = nullptr;
    ArvBufferStatus status = ARV_BUFFER_STATUS_UNKNOWN;
    while (std::chrono::steady_clock::now() < deadline) {
        buffer = arv_stream_timeout_pop_buffer(impl_->stream, 2000);
        if (buffer == nullptr) {
            continue;
        }

        status = arv_buffer_get_status(buffer);
        if (status == ARV_BUFFER_STATUS_SUCCESS) {
            break;
        }

        arv_stream_push_buffer(impl_->stream, buffer);
        buffer = nullptr;
    }

    if (buffer != nullptr && latest_only) {
        while (ArvBuffer* candidate = arv_stream_try_pop_buffer(impl_->stream)) {
            const auto candidate_status = arv_buffer_get_status(candidate);
            if (candidate_status == ARV_BUFFER_STATUS_SUCCESS) {
                arv_stream_push_buffer(impl_->stream, buffer);
                buffer = candidate;
                status = candidate_status;
                continue;
            }

            arv_stream_push_buffer(impl_->stream, candidate);
        }
    }

    if (buffer == nullptr) {
        if (error) {
            if (status != ARV_BUFFER_STATUS_UNKNOWN) {
                std::ostringstream oss;
                oss << "Aravis did not return a valid frame before timeout, last status="
                    << static_cast<int>(status);
                *error = oss.str();
            } else {
                *error = "Timed out waiting for Aravis buffer";
            }
        }
        return false;
    }

    size_t size = 0;
    const void* image_data = arv_buffer_get_image_data(buffer, &size);
    const int width = arv_buffer_get_image_width(buffer);
    const int height = arv_buffer_get_image_height(buffer);
    const ArvPixelFormat pixel_format = arv_buffer_get_image_pixel_format(buffer);

    cv::Mat frame_bgr;
    if (image_data == nullptr || width <= 0 || height <= 0) {
        arv_stream_push_buffer(impl_->stream, buffer);
        if (error) {
            *error = "Aravis returned empty image payload";
        }
        return false;
    }

    if (pixel_format == ARV_PIXEL_FORMAT_MONO_8) {
        cv::Mat mono(height, width, CV_8UC1, const_cast<void*>(image_data));
        cv::cvtColor(mono, frame_bgr, cv::COLOR_GRAY2BGR);
    } else if (pixel_format == ARV_PIXEL_FORMAT_BAYER_RG_8) {
        cv::Mat raw(height, width, CV_8UC1, const_cast<void*>(image_data));
        cv::cvtColor(raw, frame_bgr, cv::COLOR_BayerRG2BGR);
    } else if (pixel_format == ARV_PIXEL_FORMAT_BAYER_BG_8) {
        cv::Mat raw(height, width, CV_8UC1, const_cast<void*>(image_data));
        cv::cvtColor(raw, frame_bgr, cv::COLOR_BayerBG2BGR);
    } else if (pixel_format == ARV_PIXEL_FORMAT_BAYER_GR_8) {
        cv::Mat raw(height, width, CV_8UC1, const_cast<void*>(image_data));
        cv::cvtColor(raw, frame_bgr, cv::COLOR_BayerGR2BGR);
    } else if (pixel_format == ARV_PIXEL_FORMAT_BAYER_GB_8) {
        cv::Mat raw(height, width, CV_8UC1, const_cast<void*>(image_data));
        cv::cvtColor(raw, frame_bgr, cv::COLOR_BayerGB2BGR);
    } else if (pixel_format == ARV_PIXEL_FORMAT_BGR_8_PACKED) {
        frame_bgr = cv::Mat(height, width, CV_8UC3, const_cast<void*>(image_data)).clone();
    } else if (pixel_format == ARV_PIXEL_FORMAT_RGB_8_PACKED) {
        cv::Mat rgb(height, width, CV_8UC3, const_cast<void*>(image_data));
        cv::cvtColor(rgb, frame_bgr, cv::COLOR_RGB2BGR);
    } else if (size == static_cast<size_t>(width) * static_cast<size_t>(height)) {
        cv::Mat mono(height, width, CV_8UC1, const_cast<void*>(image_data));
        cv::cvtColor(mono, frame_bgr, cv::COLOR_GRAY2BGR);
    } else {
        arv_stream_push_buffer(impl_->stream, buffer);
        if (error) {
            std::ostringstream oss;
            oss << "Unsupported Aravis pixel format 0x" << std::hex << pixel_format;
            *error = oss.str();
        }
        return false;
    }

    packet.frame_bgr = std::move(frame_bgr);
    packet.frame_id = static_cast<std::int64_t>(arv_buffer_get_frame_id(buffer));
    if (packet.frame_id <= 0) {
        packet.frame_id = impl_->frame_counter++;
    } else {
        ++impl_->frame_counter;
    }
    packet.timestamp_seconds = now_seconds();

    arv_stream_push_buffer(impl_->stream, buffer);
    return true;
#else
    if (error) {
        *error = "This build does not include Aravis support.";
    }
    return false;
#endif
}

void AravisFrameSource::close() {
#ifdef CVPROJ_HAS_ARAVIS
    if (impl_ == nullptr) {
        return;
    }
    if (impl_->camera != nullptr) {
        GError* gerror = nullptr;
        arv_camera_stop_acquisition(impl_->camera, &gerror);
        if (gerror != nullptr) {
            g_clear_error(&gerror);
        }
    }
    if (impl_->stream != nullptr) {
        g_object_unref(impl_->stream);
        impl_->stream = nullptr;
    }
    if (impl_->camera != nullptr) {
        g_object_unref(impl_->camera);
        impl_->camera = nullptr;
    }
    arv_shutdown();
#endif
}

std::string AravisFrameSource::name() const {
    if (!config_.device_id.empty()) {
        return "aravis:" + config_.device_id;
    }
    if (!config_.serial_number.empty()) {
        return "aravis:" + config_.serial_number;
    }
    return "aravis:auto";
}

}  // namespace cvproj
