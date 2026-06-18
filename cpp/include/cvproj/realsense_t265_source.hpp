#pragma once

#include <memory>
#include <string>

#include "cvproj/frame_source.hpp"

namespace cvproj {

struct RealSenseT265Config {
    std::string serial_number;
    int width = 848;
    int height = 800;
    double fps = 30.0;
    int fisheye_index = 1;
    bool enable_pose = true;
};

class RealSenseT265Source final : public FrameSource {
public:
    explicit RealSenseT265Source(RealSenseT265Config config);
    ~RealSenseT265Source() override;

    bool open(std::string* error) override;
    bool read(FramePacket& packet, int timeout_ms, std::string* error) override;
    bool read_latest(FramePacket& packet, int timeout_ms, std::string* error) override;
    bool camera_intrinsics(CameraIntrinsics& intrinsics) const override;
    void close() override;
    std::string name() const override;

private:
    struct Impl;

    bool read_impl(FramePacket& packet, int timeout_ms, bool latest_only, std::string* error);

    RealSenseT265Config config_;
    CameraIntrinsics intrinsics_;
    bool have_intrinsics_ = false;
    std::unique_ptr<Impl> impl_;
};

}  // namespace cvproj
