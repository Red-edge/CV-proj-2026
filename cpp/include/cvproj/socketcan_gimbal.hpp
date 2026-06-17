#pragma once

#include <cstdint>
#include <string>

namespace cvproj {

struct YawCommand {
    bool has_target = false;
    double target_yaw_deg = 0.0;
    double current_yaw_deg = 0.0;
    double error_deg = 0.0;
    double rpm = 0.0;
    bool limited = false;
    bool sent = false;
    std::string status = "disabled";
};

struct SocketCanYawConfig {
    std::string iface = "can0";
    std::uint32_t can_id = 0x8001;
    bool extended_id = true;
    bool dry_run = false;
    bool invert = false;
    double max_angle_deg = 10.0;
    double max_rpm = 120.0;
    double kp_rpm_per_deg = 12.0;
    double hold_deadband_deg = 0.25;
    int safe_torque_raw = 2000;
    double horizontal_fov_deg = 70.0;
};

class SocketCanYawController {
public:
    explicit SocketCanYawController(SocketCanYawConfig config);
    ~SocketCanYawController();

    SocketCanYawController(const SocketCanYawController&) = delete;
    SocketCanYawController& operator=(const SocketCanYawController&) = delete;

    bool open(std::string* error);
    void close();
    bool is_open() const;
    YawCommand update_from_target(double target_cx,
                                  int frame_width,
                                  double dt_seconds,
                                  std::string* error);
    YawCommand hold(double dt_seconds, std::string* error);
    void stop();
    std::string name() const;

private:
    bool send_speed(double rpm, std::string* error);
    bool send_stop(std::string* error);
    double pixel_to_yaw(double target_cx, int frame_width) const;

    SocketCanYawConfig config_;
    int socket_fd_ = -1;
    double current_yaw_deg_ = 0.0;
};

}  // namespace cvproj
