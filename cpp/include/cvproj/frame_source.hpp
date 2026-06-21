#pragma once

#include <cstdint>
#include <optional>
#include <string>

#include <opencv2/core.hpp>

namespace cvproj {

struct CameraIntrinsics {
    int width = 0;
    int height = 0;
    double fx = 0.0;
    double fy = 0.0;
    double cx = 0.0;
    double cy = 0.0;
    std::string model;
};

struct CameraMotion {
    bool valid = false;
    double dt_seconds = 0.0;
    double angular_velocity_x_rad_s = 0.0;
    double angular_velocity_y_rad_s = 0.0;
    double angular_velocity_z_rad_s = 0.0;
    double timestamp_seconds = 0.0;
};

struct FramePacket {
    cv::Mat frame_bgr;
    std::int64_t frame_id = -1;
    double timestamp_seconds = 0.0;
    std::optional<CameraIntrinsics> intrinsics;
    std::optional<CameraMotion> camera_motion;
};

class FrameSource {
public:
    virtual ~FrameSource() = default;

    virtual bool open(std::string* error) = 0;
    virtual bool read(FramePacket& packet, int timeout_ms, std::string* error) = 0;
    virtual bool read_latest(FramePacket& packet, int timeout_ms, std::string* error) {
        return read(packet, timeout_ms, error);
    }
    virtual bool camera_intrinsics(CameraIntrinsics& intrinsics) const {
        (void)intrinsics;
        return false;
    }
    virtual void close() = 0;
    virtual std::string name() const = 0;
};

}  // namespace cvproj
