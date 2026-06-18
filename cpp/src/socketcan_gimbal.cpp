#include "cvproj/socketcan_gimbal.hpp"

#include <algorithm>
#include <cerrno>
#include <cmath>
#include <cstring>
#include <iomanip>
#include <limits>
#include <sstream>
#include <thread>
#include <utility>

#if defined(__linux__)
#include <fcntl.h>
#include <linux/can.h>
#include <linux/can/raw.h>
#include <net/if.h>
#include <sys/ioctl.h>
#include <sys/socket.h>
#include <unistd.h>
#include <pthread.h>
#include <sched.h>
#endif

namespace cvproj {

namespace {
template <typename T>
void write_le(std::uint8_t* dst, T value) {
    for (std::size_t i = 0; i < sizeof(T); ++i) {
        dst[i] = static_cast<std::uint8_t>((static_cast<std::uint64_t>(value) >> (8 * i)) & 0xFF);
    }
}

double feedback_raw_from_logical(double logical_yaw_deg, const SocketCanYawConfig& config) {
    return config.feedback_zero_deg + logical_yaw_deg;
}
}  // namespace

SocketCanYawController::SocketCanYawController(SocketCanYawConfig config) : config_(std::move(config)) {
    config_.max_angle_deg = std::abs(config_.max_angle_deg);
    config_.limit_margin_deg = std::clamp(std::abs(config_.limit_margin_deg), 0.0, config_.max_angle_deg);
    config_.max_rpm = std::abs(config_.max_rpm);
    config_.kp_rpm_per_deg = std::abs(config_.kp_rpm_per_deg);
    config_.outer_integral_limit_deg_s = std::abs(config_.outer_integral_limit_deg_s);
    config_.inner_integral_limit_rpm = std::abs(config_.inner_integral_limit_rpm);
    config_.inner_output_limit_rpm = std::abs(config_.inner_output_limit_rpm);
    config_.max_accel_rpm_s = std::abs(config_.max_accel_rpm_s);
    config_.axis_deg_per_motor_rev = std::max(1.0, std::abs(config_.axis_deg_per_motor_rev));
    config_.inner_rate_hz = std::clamp(config_.inner_rate_hz, 1.0, 2000.0);
    config_.inner_feedback_rate_hz = std::clamp(config_.inner_feedback_rate_hz, 1.0, config_.inner_rate_hz);
}

SocketCanYawController::~SocketCanYawController() {
    close();
}

bool SocketCanYawController::open(std::string* error) {
    close();
    current_yaw_deg_ = 0.0;
    feedback_raw_deg_ = config_.feedback_zero_deg;
    feedback_velocity_rpm_ = 0.0;
    feedback_torque_raw_ = 0;
    have_feedback_ = false;
    direction_error_count_ = 0;
    last_commanded_rpm_ = 0.0;
    last_desired_rpm_outer_ = 0.0;
    outer_desired_rpm_ = 0.0;
    outer_target_valid_ = false;
    outer_seq_ = 0;
    target_frame_id_ = -1;
    missed_feedback_count_ = 0;
    const auto now = std::chrono::steady_clock::now();
    last_outer_update_time_ = now;
    last_inner_tick_time_ = now;
    last_feedback_time_ = now;
    last_feedback_query_time_ = now;
    reset_pid();
    reset_inner_pid();
    if (config_.dry_run) {
        have_feedback_ = true;
        start_inner_loop();
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
    const int flags = ::fcntl(socket_fd_, F_GETFL, 0);
    if (flags >= 0) {
        ::fcntl(socket_fd_, F_SETFL, flags | O_NONBLOCK);
    }
    if (config_.query_feedback) {
        std::string ignored;
        for (int attempt = 0; attempt < 5 && !have_feedback_; ++attempt) {
            send_status_query(&ignored);
#if defined(__linux__)
            ::usleep(50000);
#endif
            read_feedback(0.0);
        }
        if (!have_feedback_) {
            if (error) {
                *error = "No yaw feedback received from motor; refusing to enable real yaw control";
            }
            close();
            return false;
        }
    }
    start_inner_loop();
    return true;
#else
    if (error) {
        *error = "SocketCAN is only available on Linux. Use --yaw-dry-run on this platform.";
    }
    return false;
#endif
}

void SocketCanYawController::close() {
    stop_inner_loop();
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
    return update_from_visual_error(pixel_to_yaw(target_cx, frame_width), 0.0, 0.0, -1, dt_seconds, error);
}

YawCommand SocketCanYawController::update_from_visual_error(double alpha_deg,
                                                            double alpha_rate_deg_s,
                                                            double target_age_ms,
                                                            std::int64_t target_frame_id,
                                                            double dt_seconds,
                                                            std::string* error) {
    (void)error;
    const auto now = std::chrono::steady_clock::now();
    std::lock_guard<std::mutex> lock(mutex_);
    const double dt = dt_seconds > 0.0 ? std::clamp(dt_seconds, 0.001, 0.5) : 1.0 / 30.0;
    const double elapsed_since_outer_ms =
        std::chrono::duration<double, std::milli>(now - last_outer_update_time_).count();
    if (config_.outer_min_update_interval_ms > 0.0 &&
        elapsed_since_outer_ms < config_.outer_min_update_interval_ms) {
        return make_snapshot_locked();
    }

    bool limited = false;
    const double omega_axis_deg_s = compute_outer_axis_deg_s(alpha_deg, alpha_rate_deg_s, dt, &limited);
    const double desired_rpm = std::clamp(
        omega_axis_deg_s * 60.0 / config_.axis_deg_per_motor_rev,
        -config_.max_rpm,
        config_.max_rpm);

    YawCommand cmd = make_snapshot_locked();
    cmd.has_target = true;
    cmd.target_yaw_deg = alpha_deg;
    cmd.error_deg = alpha_deg;
    cmd.alpha_deg = alpha_deg;
    cmd.alpha_rate_deg_s = alpha_rate_deg_s;
    cmd.omega_axis_deg_s = omega_axis_deg_s;
    cmd.desired_rpm_outer = desired_rpm;
    cmd.target_frame_id = target_frame_id;
    cmd.limited = limited;
    cmd.status = config_.dry_run ? "dry-run-outer" : "outer";
    if (target_age_ms > config_.outer_max_stale_ms) {
        cmd.has_target = false;
        cmd.status = "outer-stale-target";
        set_outer_command_locked(cmd, 0.0, false);
        reset_pid();
        return make_snapshot_locked();
    }
    set_outer_command_locked(cmd, desired_rpm, true);
    return make_snapshot_locked();
}

YawCommand SocketCanYawController::hold(double dt_seconds, std::string* error) {
    (void)dt_seconds;
    (void)error;
    std::lock_guard<std::mutex> lock(mutex_);
    reset_pid();
    reset_inner_pid();
    direction_error_count_ = 0;
    YawCommand cmd = make_snapshot_locked();
    cmd.has_target = false;
    cmd.target_yaw_deg = current_yaw_deg_;
    cmd.error_deg = 0.0;
    cmd.rpm = 0.0;
    cmd.status = config_.dry_run ? "dry-run-hold" : "hold";
    set_outer_command_locked(cmd, 0.0, false);
    return make_snapshot_locked();
}

void SocketCanYawController::reset_pid() {
    integral_error_deg_s_ = 0.0;
    previous_error_deg_ = 0.0;
    have_previous_error_ = false;
    last_outer_p_ = 0.0;
    last_outer_i_ = 0.0;
    last_outer_d_ = 0.0;
}

void SocketCanYawController::reset_inner_pid() {
    inner_integral_error_rpm_s_ = 0.0;
    previous_speed_error_rpm_ = 0.0;
    have_previous_speed_error_ = false;
    last_inner_p_ = 0.0;
    last_inner_i_ = 0.0;
    last_inner_d_ = 0.0;
}

double SocketCanYawController::compute_outer_axis_deg_s(double alpha_deg,
                                                        double alpha_rate_deg_s,
                                                        double dt_seconds,
                                                        bool* limited) {
    const double dt = dt_seconds > 0.0 ? std::clamp(dt_seconds, 0.001, 0.5) : 1.0 / 30.0;
    if (std::abs(alpha_deg) <= config_.outer_deadband_deg) {
        reset_pid();
        return 0.0;
    }
    const double derivative = have_previous_error_ ? (alpha_deg - previous_error_deg_) / dt : alpha_rate_deg_s;
    integral_error_deg_s_ = std::clamp(integral_error_deg_s_ + alpha_deg * dt,
                                       -config_.outer_integral_limit_deg_s,
                                       config_.outer_integral_limit_deg_s);
    last_outer_p_ = config_.outer_kp_deg_s_per_deg * alpha_deg;
    last_outer_i_ = config_.outer_ki_deg_s_per_deg_s * integral_error_deg_s_;
    last_outer_d_ = config_.outer_kd_deg_s_per_deg_s * derivative;
    const double ff = config_.outer_ff_gain * alpha_rate_deg_s;
    const double max_axis_deg_s = config_.max_rpm * config_.axis_deg_per_motor_rev / 60.0;
    const double raw = last_outer_p_ + last_outer_i_ + last_outer_d_ + ff;
    const double clamped = std::clamp(raw, -max_axis_deg_s, max_axis_deg_s);
    if (limited && raw != clamped) {
        *limited = true;
    }
    previous_error_deg_ = alpha_deg;
    have_previous_error_ = true;
    last_alpha_deg_ = alpha_deg;
    last_alpha_rate_deg_s_ = alpha_rate_deg_s;
    last_omega_axis_deg_s_ = clamped;
    return clamped;
}

double SocketCanYawController::compute_inner_command_rpm(double desired_rpm,
                                                         double measured_rpm,
                                                         double dt_seconds) {
    const double dt = dt_seconds > 0.0 ? std::clamp(dt_seconds, 0.0002, 0.1) : 0.001;
    const double error = desired_rpm - measured_rpm;
    last_speed_error_rpm_ = error;
    if (std::abs(error) <= config_.inner_deadband_rpm) {
        last_inner_p_ = 0.0;
        last_inner_d_ = 0.0;
        return desired_rpm;
    }
    const double derivative = have_previous_speed_error_ ? (error - previous_speed_error_rpm_) / dt : 0.0;
    inner_integral_error_rpm_s_ = std::clamp(inner_integral_error_rpm_s_ + error * dt,
                                            -config_.inner_integral_limit_rpm,
                                            config_.inner_integral_limit_rpm);
    last_inner_p_ = config_.inner_kp * error;
    last_inner_i_ = config_.inner_ki * inner_integral_error_rpm_s_;
    last_inner_d_ = config_.inner_kd * derivative;
    previous_speed_error_rpm_ = error;
    have_previous_speed_error_ = true;
    const double correction = last_inner_p_ + last_inner_i_ + last_inner_d_;
    const double output_limit = config_.inner_output_limit_rpm > 0.0 ? config_.inner_output_limit_rpm : config_.max_rpm;
    return std::clamp(desired_rpm + correction, -output_limit, output_limit);
}

double SocketCanYawController::limit_rpm_for_angle(double rpm,
                                                   double dt_seconds,
                                                   bool* limited,
                                                   std::string* status) const {
    std::lock_guard<std::mutex> lock(mutex_);
    return limit_rpm_for_angle_locked(rpm, dt_seconds, limited, status);
}

double SocketCanYawController::limit_rpm_for_angle_locked(double rpm,
                                                          double dt_seconds,
                                                          bool* limited,
                                                          std::string* status) const {
    const double hard_limit = config_.max_angle_deg;
    const double dt = dt_seconds > 0.0 ? std::clamp(dt_seconds, 0.005, 0.1) : 1.0 / 30.0;
    double safe_rpm = std::clamp(rpm, -config_.max_rpm, config_.max_rpm);
    if (status) {
        *status = "sent";
    }

    if (hard_limit <= 0.0 || safe_rpm == 0.0) {
        return 0.0;
    }

    if (current_yaw_deg_ >= hard_limit && safe_rpm > 0.0) {
        if (limited) {
            *limited = true;
        }
        if (status) {
            *status = "left-limit";
        }
        return 0.0;
    }
    if (current_yaw_deg_ <= -hard_limit && safe_rpm < 0.0) {
        if (limited) {
            *limited = true;
        }
        if (status) {
            *status = "right-limit";
        }
        return 0.0;
    }

    const double deg_per_rpm_second = 6.0;
    if (safe_rpm > 0.0) {
        const double remaining_deg = hard_limit - current_yaw_deg_;
        const double right_recovery_deg = -hard_limit - current_yaw_deg_;
        const double recovery_limited_remaining =
            current_yaw_deg_ < -hard_limit ? right_recovery_deg : remaining_deg;
        const double usable_remaining = std::max(0.0, recovery_limited_remaining);
        double max_safe_rpm = usable_remaining / (deg_per_rpm_second * dt);
        if (config_.limit_margin_deg > 0.0 && usable_remaining < config_.limit_margin_deg) {
            max_safe_rpm = std::min(max_safe_rpm, config_.max_rpm * usable_remaining / config_.limit_margin_deg);
        }
        if (safe_rpm > max_safe_rpm) {
            safe_rpm = max_safe_rpm;
            if (limited) {
                *limited = true;
            }
            if (status) {
                *status = current_yaw_deg_ < -hard_limit ? "recover-right-limit" : "left-limit-approach";
            }
        }
    } else if (safe_rpm < 0.0) {
        const double remaining_deg = current_yaw_deg_ + hard_limit;
        const double left_recovery_deg = current_yaw_deg_ - hard_limit;
        const double recovery_limited_remaining =
            current_yaw_deg_ > hard_limit ? left_recovery_deg : remaining_deg;
        const double usable_remaining = std::max(0.0, recovery_limited_remaining);
        double max_safe_rpm = usable_remaining / (deg_per_rpm_second * dt);
        if (config_.limit_margin_deg > 0.0 && usable_remaining < config_.limit_margin_deg) {
            max_safe_rpm = std::min(max_safe_rpm, config_.max_rpm * usable_remaining / config_.limit_margin_deg);
        }
        if (-safe_rpm > max_safe_rpm) {
            safe_rpm = -max_safe_rpm;
            if (limited) {
                *limited = true;
            }
            if (status) {
                *status = current_yaw_deg_ > hard_limit ? "recover-left-limit" : "right-limit-approach";
            }
        }
    }

    return std::clamp(safe_rpm, -config_.max_rpm, config_.max_rpm);
}

double SocketCanYawController::apply_accel_limit(double target_rpm, double dt_seconds, bool* limited) {
    if (config_.max_accel_rpm_s <= 0.0 || dt_seconds <= 0.0) {
        return target_rpm;
    }
    const double max_delta = config_.max_accel_rpm_s * dt_seconds;
    const double delta = target_rpm - last_commanded_rpm_;
    if (std::abs(delta) <= max_delta) {
        return target_rpm;
    }
    if (limited) {
        *limited = true;
    }
    return last_commanded_rpm_ + std::copysign(max_delta, delta);
}

void SocketCanYawController::set_outer_command_locked(const YawCommand& command,
                                                      double desired_rpm,
                                                      bool target_valid) {
    const auto now = std::chrono::steady_clock::now();
    const double dt_ms = std::chrono::duration<double, std::milli>(now - last_outer_update_time_).count();
    outer_dt_ms_ = dt_ms > 0.0 ? dt_ms : 0.0;
    outer_update_hz_ = dt_ms > 0.0 ? 1000.0 / dt_ms : 0.0;
    last_outer_update_time_ = now;
    outer_target_valid_ = target_valid;
    outer_desired_rpm_ = std::clamp(desired_rpm, -config_.max_rpm, config_.max_rpm);
    last_desired_rpm_outer_ = outer_desired_rpm_;
    last_alpha_deg_ = command.alpha_deg;
    last_alpha_rate_deg_s_ = command.alpha_rate_deg_s;
    last_omega_axis_deg_s_ = command.omega_axis_deg_s;
    last_limited_ = command.limited;
    last_status_ = command.status;
    target_frame_id_ = command.target_frame_id;
    if (target_valid) {
        ++outer_seq_;
    }
}

YawCommand SocketCanYawController::make_snapshot_locked() const {
    YawCommand cmd;
    cmd.has_target = outer_target_valid_;
    cmd.target_yaw_deg = last_alpha_deg_;
    cmd.current_yaw_deg = current_yaw_deg_;
    cmd.feedback_raw_deg = feedback_raw_deg_;
    cmd.feedback_zero_deg = config_.feedback_zero_deg;
    cmd.max_angle_deg = config_.max_angle_deg;
    cmd.error_deg = last_alpha_deg_;
    cmd.rpm = last_commanded_rpm_;
    cmd.desired_rpm_outer = last_desired_rpm_outer_;
    cmd.commanded_rpm_inner = last_commanded_rpm_;
    cmd.measured_rpm = feedback_velocity_rpm_;
    cmd.speed_error_rpm = last_speed_error_rpm_;
    cmd.alpha_deg = last_alpha_deg_;
    cmd.alpha_rate_deg_s = last_alpha_rate_deg_s_;
    cmd.omega_axis_deg_s = last_omega_axis_deg_s_;
    cmd.outer_update_hz = outer_update_hz_;
    cmd.outer_dt_ms = outer_dt_ms_;
    cmd.inner_loop_hz = inner_loop_hz_;
    cmd.inner_dt_us = inner_dt_us_;
    cmd.inner_jitter_us = inner_jitter_us_;
    const auto now = std::chrono::steady_clock::now();
    cmd.feedback_age_ms = have_feedback_
                              ? std::chrono::duration<double, std::milli>(now - last_feedback_time_).count()
                              : -1.0;
    cmd.feedback_hz = feedback_hz_;
    cmd.outer_seq = outer_seq_;
    cmd.target_frame_id = target_frame_id_;
    cmd.missed_feedback_count = missed_feedback_count_;
    cmd.limited = last_limited_;
    cmd.accel_limited = last_accel_limited_;
    cmd.sent = is_open();
    cmd.status = last_status_;
    return cmd;
}

YawCommand SocketCanYawController::snapshot() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return make_snapshot_locked();
}

void SocketCanYawController::start_inner_loop() {
    stop_inner_loop();
    {
        std::lock_guard<std::mutex> lock(mutex_);
        inner_running_ = true;
        last_inner_tick_time_ = std::chrono::steady_clock::now();
    }
    inner_thread_ = std::thread(&SocketCanYawController::inner_loop, this);
}

void SocketCanYawController::stop_inner_loop() {
    {
        std::lock_guard<std::mutex> lock(mutex_);
        inner_running_ = false;
        outer_target_valid_ = false;
        outer_desired_rpm_ = 0.0;
    }
    if (inner_thread_.joinable()) {
        inner_thread_.join();
    }
}

void SocketCanYawController::inner_loop() {
#if defined(__linux__)
    if (config_.inner_realtime_priority) {
        sched_param param {};
        param.sched_priority = std::clamp(config_.inner_realtime_priority_value, 1, 99);
        (void)::pthread_setschedparam(::pthread_self(), SCHED_FIFO, &param);
    }
#endif
    const auto period = std::chrono::microseconds(
        std::max<int>(1, static_cast<int>(std::llround(1000000.0 / config_.inner_rate_hz))));
    auto next_tick = std::chrono::steady_clock::now() + period;
    while (true) {
        const auto tick_start = std::chrono::steady_clock::now();
        std::string ignored;
        bool should_query_feedback = false;
        double command_rpm_to_send = 0.0;
        double dry_run_dt = 0.001;
        {
            std::lock_guard<std::mutex> lock(mutex_);
            if (!inner_running_) {
                break;
            }
            const double dt = std::max(
                0.0002,
                std::chrono::duration<double>(tick_start - last_inner_tick_time_).count());
            inner_dt_us_ = dt * 1000000.0;
            inner_loop_hz_ = dt > 0.0 ? 1.0 / dt : 0.0;
            inner_jitter_us_ = inner_dt_us_ - static_cast<double>(period.count());
            last_inner_tick_time_ = tick_start;

            const double outer_age_ms =
                std::chrono::duration<double, std::milli>(tick_start - last_outer_update_time_).count();
            double desired_rpm = outer_target_valid_ ? outer_desired_rpm_ : 0.0;
            std::string status = outer_target_valid_ ? "inner" : "hold";
            bool limited = last_limited_;
            if (outer_target_valid_ && outer_age_ms > config_.inner_command_timeout_ms) {
                desired_rpm = 0.0;
                outer_target_valid_ = false;
                reset_pid();
                reset_inner_pid();
                status = "inner-command-timeout";
                limited = true;
            }

            if (!config_.dry_run && config_.query_feedback) {
                const double feedback_period_ms = 1000.0 / std::max(1.0, config_.inner_feedback_rate_hz);
                const double since_query_ms =
                    std::chrono::duration<double, std::milli>(tick_start - last_feedback_query_time_).count();
                if (since_query_ms >= feedback_period_ms) {
                    should_query_feedback = true;
                    last_feedback_query_time_ = tick_start;
                }
            }

            const double feedback_age_ms = have_feedback_
                                               ? std::chrono::duration<double, std::milli>(
                                                     tick_start - last_feedback_time_)
                                                     .count()
                                               : std::numeric_limits<double>::infinity();
            if (!config_.dry_run && config_.feedback_required &&
                (!have_feedback_ || feedback_age_ms > config_.inner_feedback_timeout_ms)) {
                desired_rpm = 0.0;
                outer_target_valid_ = false;
                reset_pid();
                reset_inner_pid();
                status = "no-fresh-feedback";
                limited = true;
            }

            double command_rpm = desired_rpm;
            last_speed_error_rpm_ = desired_rpm - feedback_velocity_rpm_;
            if (config_.inner_mode == "software-pi" || config_.inner_mode == "software-pid") {
                command_rpm = compute_inner_command_rpm(desired_rpm, feedback_velocity_rpm_, dt);
            } else {
                last_inner_p_ = 0.0;
                last_inner_i_ = 0.0;
                last_inner_d_ = 0.0;
            }
            command_rpm = std::clamp(command_rpm, -config_.max_rpm, config_.max_rpm);
            command_rpm = limit_rpm_for_angle_locked(command_rpm, dt, &limited, &status);
            bool accel_limited = false;
            command_rpm = apply_accel_limit(command_rpm, dt, &accel_limited);
            last_accel_limited_ = accel_limited;
            last_limited_ = limited || accel_limited;
            last_status_ = config_.dry_run ? (outer_target_valid_ ? "dry-run-inner" : "dry-run-hold") : status;
            last_desired_rpm_outer_ = desired_rpm;
            last_commanded_rpm_ = command_rpm;
            command_rpm_to_send = command_rpm;
            dry_run_dt = dt;
        }

        if (should_query_feedback) {
            send_status_query(&ignored);
        }
        FeedbackSample sample;
        const bool feedback_updated =
            !config_.dry_run && config_.query_feedback ? read_feedback_sample(sample, 0) : false;
        send_speed(command_rpm_to_send, &ignored);
        {
            std::lock_guard<std::mutex> lock(mutex_);
            if (feedback_updated) {
                apply_feedback_sample_locked(sample);
            } else if (!config_.dry_run && config_.query_feedback) {
                ++missed_feedback_count_;
            }
            if (config_.dry_run) {
                current_yaw_deg_ += command_rpm_to_send * config_.axis_deg_per_motor_rev / 60.0 * dry_run_dt;
                current_yaw_deg_ = std::clamp(current_yaw_deg_, -config_.max_angle_deg, config_.max_angle_deg);
                feedback_raw_deg_ = feedback_raw_from_logical(current_yaw_deg_, config_);
                feedback_velocity_rpm_ = command_rpm_to_send;
                feedback_torque_raw_ = 0;
                have_feedback_ = true;
                last_feedback_time_ = tick_start;
                feedback_hz_ = inner_loop_hz_;
            }
        }
        std::this_thread::sleep_until(next_tick);
        next_tick += period;
        const auto now = std::chrono::steady_clock::now();
        if (now > next_tick + period) {
            next_tick = now + period;
        }
    }
    std::string ignored;
    send_speed(0.0, &ignored);
}

void SocketCanYawController::stop() {
    std::string ignored;
    stop_inner_loop();
    send_speed(0.0, &ignored);
    send_stop(&ignored);
}

std::string SocketCanYawController::name() const {
    std::ostringstream oss;
    oss << (config_.dry_run ? "yaw-dry-run:" : "socketcan:")
        << config_.iface << " id=0x" << std::hex << config_.can_id << std::dec
        << " max_angle=" << config_.max_angle_deg
        << " limit_margin=" << config_.limit_margin_deg
        << " max_rpm=" << config_.max_rpm
        << " control=" << config_.control_mode
        << " outer_pid=(" << config_.outer_kp_deg_s_per_deg
        << "," << config_.outer_ki_deg_s_per_deg_s
        << "," << config_.outer_kd_deg_s_per_deg_s << ")"
        << " inner=" << config_.inner_mode
        << " inner_hz=" << config_.inner_rate_hz
        << " feedback_zero=" << config_.feedback_zero_deg;
    if (config_.invert) {
        oss << " yaw-inverted";
    }
    if (config_.motor_invert) {
        oss << " motor-inverted";
    }
    return oss.str();
}

bool SocketCanYawController::send_speed(double rpm, std::string* error) {
    const double safe_rpm = std::clamp(rpm, -config_.max_rpm, config_.max_rpm);
    const double motor_rpm = config_.motor_invert ? -safe_rpm : safe_rpm;
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
        static_cast<std::int16_t>(std::clamp(motor_rpm / 0.015, -32768.0, 32767.0));
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

bool SocketCanYawController::send_status_query(std::string* error) {
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
    frame.can_dlc = 2;
    frame.data[0] = 0x17;
    frame.data[1] = 0x01;
    const ssize_t n = ::write(socket_fd_, &frame, sizeof(frame));
    if (n != static_cast<ssize_t>(sizeof(frame))) {
        if (error) {
            *error = std::string("CAN status-query write failed: ") + std::strerror(errno);
        }
        return false;
    }
    return true;
#else
    (void)error;
    return false;
#endif
}

bool SocketCanYawController::read_feedback(double dt_seconds, int wait_timeout_us) {
    (void)dt_seconds;
    FeedbackSample sample;
    const bool updated = read_feedback_sample(sample, wait_timeout_us);
    if (updated) {
        apply_feedback_sample_locked(sample);
    }
    return updated;
}

void SocketCanYawController::apply_feedback_sample_locked(const FeedbackSample& sample) {
    const double feedback_dt_ms =
        have_feedback_ ? std::chrono::duration<double, std::milli>(sample.received_time - last_feedback_time_).count()
                       : 0.0;
    feedback_raw_deg_ = sample.raw_yaw_deg;
    current_yaw_deg_ = sample.raw_yaw_deg - config_.feedback_zero_deg;
    feedback_velocity_rpm_ = sample.velocity_rpm;
    feedback_torque_raw_ = sample.torque_raw;
    if (feedback_dt_ms > 0.0) {
        feedback_hz_ = 1000.0 / feedback_dt_ms;
    }
    last_feedback_time_ = sample.received_time;
    have_feedback_ = true;
}

bool SocketCanYawController::read_feedback_sample(FeedbackSample& sample, int wait_timeout_us) {
    if (config_.dry_run || socket_fd_ < 0) {
        return false;
    }

#if defined(__linux__)
    bool updated = false;
    int waited_us = 0;
    for (;;) {
        can_frame frame {};
        const ssize_t n = ::read(socket_fd_, &frame, sizeof(frame));
        if (n < 0) {
            if (errno == EAGAIN || errno == EWOULDBLOCK) {
                if (wait_timeout_us > 0 && !updated && waited_us < wait_timeout_us) {
                    ::usleep(1000);
                    waited_us += 1000;
                    continue;
                }
                break;
            }
            break;
        }
        if (n != static_cast<ssize_t>(sizeof(frame))) {
            break;
        }
        const std::uint32_t frame_id = frame.can_id & CAN_EFF_MASK;
        const bool status_id_matches = frame_id == config_.can_id || frame_id == 0x100;
        if (!status_id_matches || frame.can_dlc < 6 || frame.data[0] != 0x27) {
            continue;
        }
        const auto pos_raw = static_cast<std::int16_t>(
            static_cast<std::uint16_t>(frame.data[2]) |
            (static_cast<std::uint16_t>(frame.data[3]) << 8));
        if (frame.data[1] == 0x01) {
            const auto vel_raw = frame.can_dlc >= 6
                                     ? static_cast<std::int16_t>(
                                           static_cast<std::uint16_t>(frame.data[4]) |
                                           (static_cast<std::uint16_t>(frame.data[5]) << 8))
                                     : static_cast<std::int16_t>(0);
            const auto torque_raw = frame.can_dlc >= 8
                                        ? static_cast<std::int16_t>(
                                              static_cast<std::uint16_t>(frame.data[6]) |
                                              (static_cast<std::uint16_t>(frame.data[7]) << 8))
                                        : static_cast<std::int16_t>(0);
            const double raw_yaw_deg = static_cast<double>(pos_raw) * 0.0001 * 360.0;
            sample.raw_yaw_deg = raw_yaw_deg;
            sample.velocity_rpm = static_cast<double>(vel_raw) * 0.00025 * 60.0;
            sample.torque_raw = torque_raw;
            sample.received_time = std::chrono::steady_clock::now();
            updated = true;
        }
    }
    return updated;
#else
    (void)wait_timeout_us;
    (void)sample;
    return false;
#endif
}

double SocketCanYawController::pixel_to_yaw(double target_cx, int frame_width) const {
    if (frame_width <= 0) {
        return 0.0;
    }
    constexpr double kPi = 3.14159265358979323846;
    const double cx = config_.center_x_px >= 0.0 ? config_.center_x_px : static_cast<double>(frame_width) * 0.5;
    const double fx = config_.focal_x_px > 0.0
                          ? config_.focal_x_px
                          : cx / std::tan((config_.horizontal_fov_deg * kPi / 180.0) * 0.5);
    const double sign = config_.invert ? -1.0 : 1.0;
    // Project convention: +yaw means a left turn. The motor feedback also
    // increases when the yaw axis turns left, so feedback relative angle is
    // always raw_feedback - software_zero.
    return sign * std::atan2(cx - target_cx, fx) * 180.0 / kPi;
}

}  // namespace cvproj
