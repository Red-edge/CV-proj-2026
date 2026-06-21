#include "cvproj/realsense_t265_source.hpp"

#include <chrono>
#include <cmath>
#include <iostream>
#include <sstream>
#include <utility>

#include <opencv2/imgproc.hpp>

#if defined(CVPROJ_HAS_REALSENSE2)
#include <librealsense2/rs.hpp>
#endif

namespace cvproj {

namespace {
double now_seconds() {
    using clock = std::chrono::steady_clock;
    return std::chrono::duration<double>(clock::now().time_since_epoch()).count();
}

std::string distortion_model_name(int model) {
#if defined(CVPROJ_HAS_REALSENSE2)
    switch (model) {
    case RS2_DISTORTION_NONE:
        return "none";
    case RS2_DISTORTION_MODIFIED_BROWN_CONRADY:
        return "modified_brown_conrady";
    case RS2_DISTORTION_INVERSE_BROWN_CONRADY:
        return "inverse_brown_conrady";
    case RS2_DISTORTION_FTHETA:
        return "ftheta";
    case RS2_DISTORTION_BROWN_CONRADY:
        return "brown_conrady";
    case RS2_DISTORTION_KANNALA_BRANDT4:
        return "kannala_brandt4";
    default:
        return "unknown";
    }
#else
    (void)model;
    return "unavailable";
#endif
}
}  // namespace

struct RealSenseT265Source::Impl {
#if defined(CVPROJ_HAS_REALSENSE2)
    rs2::pipeline pipe;
    rs2::pipeline_profile profile;
#endif
    std::int64_t frame_counter = 0;
};

RealSenseT265Source::RealSenseT265Source(RealSenseT265Config config)
    : config_(std::move(config)), impl_(std::make_unique<Impl>()) {}

RealSenseT265Source::~RealSenseT265Source() {
    close();
}

bool RealSenseT265Source::open(std::string* error) {
    close();
    impl_ = std::make_unique<Impl>();
    have_intrinsics_ = false;
    last_pose_timestamp_seconds_ = 0.0;

#if defined(CVPROJ_HAS_REALSENSE2)
    try {
        rs2::config cfg;
        if (!config_.serial_number.empty()) {
            cfg.enable_device(config_.serial_number);
        }
        const int fps = static_cast<int>(std::lround(config_.fps));
        // T265 firmware requires the stereo fisheye pair to be enabled together.
        // The frame source still exports only config_.fisheye_index to the pipeline.
        cfg.enable_stream(RS2_STREAM_FISHEYE, 1, config_.width, config_.height, RS2_FORMAT_Y8, fps);
        cfg.enable_stream(RS2_STREAM_FISHEYE, 2, config_.width, config_.height, RS2_FORMAT_Y8, fps);
        if (config_.enable_pose) {
            cfg.enable_stream(RS2_STREAM_POSE);
        }

        impl_->profile = impl_->pipe.start(cfg);
        const auto stream = impl_->profile.get_stream(RS2_STREAM_FISHEYE, config_.fisheye_index)
                                .as<rs2::video_stream_profile>();
        const rs2_intrinsics rs_intr = stream.get_intrinsics();
        intrinsics_.width = rs_intr.width;
        intrinsics_.height = rs_intr.height;
        intrinsics_.fx = rs_intr.fx;
        intrinsics_.fy = rs_intr.fy;
        intrinsics_.cx = rs_intr.ppx;
        intrinsics_.cy = rs_intr.ppy;
        intrinsics_.model = distortion_model_name(rs_intr.model);
        have_intrinsics_ = true;
        impl_->frame_counter = 0;

        std::ostringstream oss;
        oss << "T265 fisheye " << config_.fisheye_index << " intrinsics: "
            << intrinsics_.width << "x" << intrinsics_.height
            << " fx=" << intrinsics_.fx
            << " fy=" << intrinsics_.fy
            << " cx=" << intrinsics_.cx
            << " cy=" << intrinsics_.cy
            << " model=" << intrinsics_.model;
        std::cout << oss.str() << '\n';
        return true;
    } catch (const rs2::error& e) {
        if (error) {
            *error = std::string("librealsense error: ") + e.what();
        }
        return false;
    } catch (const std::exception& e) {
        if (error) {
            *error = std::string("failed to open T265: ") + e.what();
        }
        return false;
    }
#else
    if (error) {
        *error = "RealSense T265 backend was not built. Install librealsense2 and rebuild with CVPROJ_ENABLE_REALSENSE=ON.";
    }
    return false;
#endif
}

bool RealSenseT265Source::read(FramePacket& packet, int timeout_ms, std::string* error) {
    return read_impl(packet, timeout_ms, false, error);
}

bool RealSenseT265Source::read_latest(FramePacket& packet, int timeout_ms, std::string* error) {
    return read_impl(packet, timeout_ms, true, error);
}

bool RealSenseT265Source::read_impl(FramePacket& packet, int timeout_ms, bool latest_only, std::string* error) {
#if defined(CVPROJ_HAS_REALSENSE2)
    try {
        rs2::frameset frames;
        if (!impl_->pipe.poll_for_frames(&frames)) {
            if (!impl_->pipe.try_wait_for_frames(&frames, timeout_ms)) {
                if (error) {
                    *error = "Timed out waiting for T265 fisheye frame";
                }
                return false;
            }
        }

        if (latest_only) {
            rs2::frameset newer;
            while (impl_->pipe.poll_for_frames(&newer)) {
                frames = newer;
            }
        }

        const rs2::video_frame fisheye = frames.get_fisheye_frame(config_.fisheye_index);
        if (!fisheye) {
            if (error) {
                *error = "T265 frameset did not contain requested fisheye frame";
            }
            return false;
        }

        cv::Mat gray(cv::Size(fisheye.get_width(), fisheye.get_height()),
                     CV_8UC1,
                     const_cast<void*>(fisheye.get_data()),
                     static_cast<std::size_t>(fisheye.get_stride_in_bytes()));
        cv::cvtColor(gray, packet.frame_bgr, cv::COLOR_GRAY2BGR);
        packet.frame_id = impl_->frame_counter++;
        packet.timestamp_seconds =
            fisheye.get_timestamp() > 0.0 ? fisheye.get_timestamp() / 1000.0 : now_seconds();
        if (have_intrinsics_) {
            packet.intrinsics = intrinsics_;
        }
        if (config_.enable_pose) {
            const rs2::pose_frame pose = frames.first_or_default(RS2_STREAM_POSE);
            if (pose) {
                const rs2_pose data = pose.get_pose_data();
                CameraMotion motion;
                motion.valid = true;
                motion.timestamp_seconds =
                    pose.get_timestamp() > 0.0 ? pose.get_timestamp() / 1000.0 : packet.timestamp_seconds;
                motion.dt_seconds = packet.timestamp_seconds > 0.0 && last_pose_timestamp_seconds_ > 0.0
                                        ? std::max(0.0, packet.timestamp_seconds - last_pose_timestamp_seconds_)
                                        : 0.0;
                motion.angular_velocity_x_rad_s = data.angular_velocity.x;
                motion.angular_velocity_y_rad_s = data.angular_velocity.y;
                motion.angular_velocity_z_rad_s = data.angular_velocity.z;
                packet.camera_motion = motion;
                last_pose_timestamp_seconds_ = packet.timestamp_seconds;
            }
        }
        return true;
    } catch (const rs2::error& e) {
        if (error) {
            *error = std::string("T265 read failed: ") + e.what();
        }
        return false;
    } catch (const std::exception& e) {
        if (error) {
            *error = std::string("T265 read failed: ") + e.what();
        }
        return false;
    }
#else
    (void)packet;
    (void)timeout_ms;
    (void)latest_only;
    if (error) {
        *error = "RealSense T265 backend was not built";
    }
    return false;
#endif
}

bool RealSenseT265Source::camera_intrinsics(CameraIntrinsics& intrinsics) const {
    if (!have_intrinsics_) {
        return false;
    }
    intrinsics = intrinsics_;
    return true;
}

void RealSenseT265Source::close() {
#if defined(CVPROJ_HAS_REALSENSE2)
    if (impl_) {
        try {
            impl_->pipe.stop();
        } catch (...) {
        }
    }
#endif
}

std::string RealSenseT265Source::name() const {
    std::ostringstream oss;
    oss << "t265:fisheye" << config_.fisheye_index;
    if (!config_.serial_number.empty()) {
        oss << ":" << config_.serial_number;
    }
    return oss.str();
}

}  // namespace cvproj
