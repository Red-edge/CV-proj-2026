#include "cvproj/socketcan_gimbal.hpp"

#include <algorithm>
#include <cerrno>
#include <cmath>
#include <cstring>
#include <iomanip>
#include <sstream>
#include <utility>

#if defined(__linux__)
#include <linux/can.h>
#include <linux/can/raw.h>
#include <net/if.h>
#include <sys/ioctl.h>
#include <sys/socket.h>
#include <unistd.h>
#endif

namespace cvproj {

namespace {
template <typename T>
void write_le(std::uint8_t* dst, T value) {
    for (std::size_t i = 0; i < sizeof(T); ++i) {
        dst[i] = static_cast<std::uint8_t>((static_cast<std::uint64_t>(value) >> (8 * i)) & 0xFF);
    }
}
}  // namespace

SocketCanYawController::SocketCanYawController(SocketCanYawConfig config) : config_(std::move(config)) {
    config_.max_angle_deg = std::abs(config_.max_angle_deg);
    config_.max_rpm = std::abs(config_.max_rpm);
    config_.kp_rpm_per_deg = std::abs(config_.kp_rpm_per_deg);
}

SocketCanYawController::~SocketCanYawController() {
    close();
}

bool SocketCanYawController::open(std::string* error) {
    close();
    current_yaw_deg_ = 0.0;
    if (config_.dry_run) {
        return true;
    }

#if defined(__linux__)
    socket_fd_ = ::socket(PF_CAN, SOCK_RAW, CAN_RAW);
    if (socket_fd_ < 0) {
        if (error) {
            *error = std::string("socket(PF_CAN) failed: ") + std::strerror(errno);
        }
        return false;
    }

    ifreq ifr {};
    std::snprintf(ifr.ifr_name, sizeof(ifr.ifr_name), "%s", config_.iface.c_str());
    if (::ioctl(socket_fd_, SIOCGIFINDEX, &ifr) < 0) {
        if (error) {
            *error = "CAN interface not found: " + config_.iface;
        }
        close();
        return false;
    }

    sockaddr_can addr {};
    addr.can_family = AF_CAN;
    addr.can_ifindex = ifr.ifr_ifindex;
    if (::bind(socket_fd_, reinterpret_cast<sockaddr*>(&addr), sizeof(addr)) < 0) {
        if (error) {
            *error = std::string("bind CAN interface failed: ") + std::strerror(errno);
        }
        close();
        return false;
    }
    return true;
#else
    if (error) {
        *error = "SocketCAN is only available on Linux. Use --yaw-dry-run on this platform.";
    }
    return false;
#endif
}

void SocketCanYawController::close() {
    if (socket_fd_ >= 0) {
#if defined(__linux__)
        ::close(socket_fd_);
#endif
        socket_fd_ = -1;
    }
}

bool SocketCanYawController::is_open() const {
    return config_.dry_run || socket_fd_ >= 0;
}

YawCommand SocketCanYawController::update_from_target(double target_cx,
                                                      int frame_width,
                                                      double dt_seconds,
                                                      std::string* error) {
    YawCommand cmd;
    cmd.has_target = true;
    const double raw_target = pixel_to_yaw(target_cx, frame_width);
    cmd.limited = std::abs(raw_target) > config_.max_angle_deg;
    cmd.target_yaw_deg = std::clamp(raw_target, -config_.max_angle_deg, config_.max_angle_deg);
    cmd.current_yaw_deg = current_yaw_deg_;
    cmd.error_deg = cmd.target_yaw_deg - current_yaw_deg_;

    if (std::abs(cmd.error_deg) <= config_.hold_deadband_deg) {
        cmd.rpm = 0.0;
    } else {
        cmd.rpm = std::clamp(cmd.error_deg * config_.kp_rpm_per_deg, -config_.max_rpm, config_.max_rpm);
    }

    if (!send_speed(cmd.rpm, error)) {
        cmd.status = error && !error->empty() ? *error : "send_failed";
        return cmd;
    }

    const double dt = dt_seconds > 0.0 ? std::min(dt_seconds, 0.1) : 1.0 / 60.0;
    current_yaw_deg_ += cmd.rpm * 6.0 * dt;
    current_yaw_deg_ = std::clamp(current_yaw_deg_, -config_.max_angle_deg, config_.max_angle_deg);
    cmd.current_yaw_deg = current_yaw_deg_;
    cmd.sent = true;
    cmd.status = config_.dry_run ? "dry-run" : "sent";
    return cmd;
}

YawCommand SocketCanYawController::hold(double dt_seconds, std::string* error) {
    (void)dt_seconds;
    YawCommand cmd;
    cmd.current_yaw_deg = current_yaw_deg_;
    cmd.target_yaw_deg = current_yaw_deg_;
    cmd.error_deg = 0.0;
    cmd.rpm = 0.0;
    if (!send_speed(0.0, error)) {
        cmd.status = error && !error->empty() ? *error : "send_failed";
        return cmd;
    }
    cmd.sent = true;
    cmd.status = config_.dry_run ? "dry-run-hold" : "hold";
    return cmd;
}

void SocketCanYawController::stop() {
    std::string ignored;
    send_speed(0.0, &ignored);
    send_stop(&ignored);
}

std::string SocketCanYawController::name() const {
    std::ostringstream oss;
    oss << (config_.dry_run ? "yaw-dry-run:" : "socketcan:")
        << config_.iface << " id=0x" << std::hex << config_.can_id << std::dec
        << " max_angle=" << config_.max_angle_deg;
    if (config_.invert) {
        oss << " inverted";
    }
    return oss.str();
}

bool SocketCanYawController::send_speed(double rpm, std::string* error) {
    const double safe_rpm = std::clamp(rpm, -config_.max_rpm, config_.max_rpm);
    if (config_.dry_run) {
        return true;
    }
    if (socket_fd_ < 0) {
        if (error) {
            *error = "CAN socket is not open";
        }
        return false;
    }

#if defined(__linux__)
    can_frame frame {};
    frame.can_id = config_.can_id | (config_.extended_id ? CAN_EFF_FLAG : 0U);
    frame.can_dlc = 8;
    frame.data[0] = 0x07;
    frame.data[1] = 0x35;
    const auto speed_raw =
        static_cast<std::int16_t>(std::clamp(safe_rpm / 0.015, -32768.0, 32767.0));
    const auto torque_raw =
        static_cast<std::int16_t>(std::clamp(static_cast<double>(config_.safe_torque_raw), -32768.0, 32767.0));
    write_le<std::uint16_t>(&frame.data[2], static_cast<std::uint16_t>(speed_raw));
    write_le<std::uint16_t>(&frame.data[4], static_cast<std::uint16_t>(torque_raw));
    write_le<std::uint16_t>(&frame.data[6], 0x8000);

    const ssize_t n = ::write(socket_fd_, &frame, sizeof(frame));
    if (n != static_cast<ssize_t>(sizeof(frame))) {
        if (error) {
            *error = std::string("CAN write failed: ") + std::strerror(errno);
        }
        return false;
    }
    return true;
#else
    (void)error;
    return false;
#endif
}

bool SocketCanYawController::send_stop(std::string* error) {
    if (config_.dry_run) {
        return true;
    }
    if (socket_fd_ < 0) {
        if (error) {
            *error = "CAN socket is not open";
        }
        return false;
    }

#if defined(__linux__)
    can_frame frame {};
    frame.can_id = config_.can_id | (config_.extended_id ? CAN_EFF_FLAG : 0U);
    frame.can_dlc = 3;
    frame.data[0] = 0x01;
    frame.data[1] = 0x00;
    frame.data[2] = 0x00;
    const ssize_t n = ::write(socket_fd_, &frame, sizeof(frame));
    if (n != static_cast<ssize_t>(sizeof(frame))) {
        if (error) {
            *error = std::string("CAN stop write failed: ") + std::strerror(errno);
        }
        return false;
    }
    return true;
#else
    (void)error;
    return false;
#endif
}

double SocketCanYawController::pixel_to_yaw(double target_cx, int frame_width) const {
    if (frame_width <= 0) {
        return 0.0;
    }
    constexpr double kPi = 3.14159265358979323846;
    const double cx = static_cast<double>(frame_width) * 0.5;
    const double fx = cx / std::tan((config_.horizontal_fov_deg * kPi / 180.0) * 0.5);
    const double sign = config_.invert ? -1.0 : 1.0;
    return sign * std::atan2(target_cx - cx, fx) * 180.0 / kPi;
}

}  // namespace cvproj
