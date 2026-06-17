#pragma once

#include <memory>
#include <string>
#include <vector>

#include "cvproj/frame_source.hpp"

namespace cvproj {

struct HikrobotMvsConfig {
    std::string serial_number;
    int width = 1440;
    int height = 1080;
    double target_fps = 240.0;
    double exposure_us = 0.0;
    double gain = 0.0;
};

class HikrobotMvsSource final : public FrameSource {
public:
    explicit HikrobotMvsSource(HikrobotMvsConfig config);
    ~HikrobotMvsSource() override;

    bool open(std::string* error) override;
    bool read(FramePacket& packet, int timeout_ms, std::string* error) override;
    bool read_latest(FramePacket& packet, int timeout_ms, std::string* error) override;
    void close() override;
    std::string name() const override;

private:
    struct Impl;

    bool read_impl(FramePacket& packet, int timeout_ms, bool latest_only, std::string* error);

    HikrobotMvsConfig config_;
    std::unique_ptr<Impl> impl_;
};

}  // namespace cvproj
