#pragma once

#include <memory>
#include <string>

#include "cvproj/frame_source.hpp"

namespace cvproj {

struct AravisConfig {
    std::string device_id;
    std::string serial_number;
    int width = 960;
    int height = 540;
    double target_fps = 240.0;
    double exposure_us = 3000.0;
    double gain_db = 12.0;
    std::string pixel_format = "Mono8";
    int stream_buffers = 32;
};

class AravisFrameSource final : public FrameSource {
public:
    explicit AravisFrameSource(AravisConfig config);
    ~AravisFrameSource() override;

    bool open(std::string* error) override;
    bool read(FramePacket& packet, int timeout_ms, std::string* error) override;
    bool read_latest(FramePacket& packet, int timeout_ms, std::string* error) override;
    void close() override;
    std::string name() const override;

private:
    struct Impl;

    bool read_impl(FramePacket& packet, int timeout_ms, bool latest_only, std::string* error);
    std::string resolve_device_id(std::string* error);

    AravisConfig config_;
    std::unique_ptr<Impl> impl_;
};

}  // namespace cvproj
