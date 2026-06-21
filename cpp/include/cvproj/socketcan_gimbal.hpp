#pragma once

#include <cstdint>
#include <chrono>
#include <mutex>
#include <optional>
#include <string>
#include <thread>

namespace cvproj {

struct YawCommand {
    bool has_target = false;
    double target_yaw_deg = 0.0;
    double current_yaw_deg = 0.0;
    double feedback_raw_deg = 0.0;
    double feedback_zero_deg = 0.0;
    double max_angle_deg = 0.0;
    double error_deg = 0.0;
    double rpm = 0.0;
    double desired_rpm_outer = 0.0;
    double commanded_rpm_inner = 0.0;
    double measured_rpm = 0.0;
    double speed_error_rpm = 0.0;
    double alpha_deg = 0.0;
    double alpha_rate_deg_s = 0.0;
    double omega_axis_deg_s = 0.0;
    double outer_update_hz = 0.0;
    double outer_dt_ms = 0.0;
    double inner_loop_hz = 0.0;
    double inner_dt_us = 0.0;
    double inner_jitter_us = 0.0;
    double feedback_age_ms = 0.0;
    double feedback_hz = 0.0;
    std::uint64_t outer_seq = 0;
    std::int64_t target_frame_id = -1;
    std::uint64_t missed_feedback_count = 0;
    bool limited = false;
    bool accel_limited = false;
    bool sent = false;
    std::string status = "disabled";
};

struct SocketCanYawConfig {
    std::string iface = "can0";
    std::uint32_t can_id = 0x8001;
    bool extended_id = true;
    bool dry_run = false;
    bool invert = false;
    bool motor_invert = false;
    double max_angle_deg = 30.0;
    double limit_margin_deg = 2.0;
    double max_rpm = 120.0;
    double kp_rpm_per_deg = 12.0;
    double ki_rpm_per_deg_s = 0.0;
    double kd_rpm_per_deg_per_s = 0.0;
    double hold_deadband_deg = 0.25;
    int safe_torque_raw = 2000;
    double horizontal_fov_deg = 70.0;
    double focal_x_px = 0.0;
    double center_x_px = -1.0;
    double feedback_zero_deg = 0.0;
    bool query_feedback = true;
    std::string control_mode = "visual-servo";
    std::string outer_rate_mode = "inference";
    double outer_max_stale_ms = 120.0;
    double outer_min_update_interval_ms = 0.0;
    double outer_kp_deg_s_per_deg = 3.0;
    double outer_ki_deg_s_per_deg_s = 0.0;
    double outer_kd_deg_s_per_deg_s = 0.0;
    double outer_ff_gain = 0.0;
    double outer_integral_limit_deg_s = 30.0;
    double outer_deadband_deg = 0.2;
    std::string inner_mode = "driver-speed";
    double inner_rate_hz = 1000.0;
    double inner_feedback_rate_hz = 1000.0;
    double inner_feedback_timeout_ms = 5.0;
    bool inner_realtime_priority = false;
    int inner_realtime_priority_value = 60;
    double inner_command_timeout_ms = 120.0;
    double inner_kp = 1.0;
    double inner_ki = 0.0;
    double inner_kd = 0.0;
    double inner_integral_limit_rpm = 10.0;
    double inner_output_limit_rpm = 30.0;
    double inner_deadband_rpm = 0.1;
    double max_accel_rpm_s = 30.0;
    double axis_deg_per_motor_rev = 360.0;
    bool feedback_required = true;
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
    YawCommand update_from_visual_error(double alpha_deg,
                                        double alpha_rate_deg_s,
                                        double target_age_ms,
                                        std::int64_t target_frame_id,
                                        double dt_seconds,
                                        std::string* error);
    YawCommand hold(double dt_seconds, std::string* error);
    void stop();
    std::string name() const;
    YawCommand snapshot() const;

private:
    struct FeedbackSample {
        double raw_yaw_deg = 0.0;
        double velocity_rpm = 0.0;
        int torque_raw = 0;
        std::chrono::steady_clock::time_point received_time;
    };

    bool send_speed(double rpm, std::string* error);
    bool send_stop(std::string* error);
    bool send_status_query(std::string* error);
    bool read_feedback(double dt_seconds, int wait_timeout_us = 0);
    bool read_feedback_sample(FeedbackSample& sample, int wait_timeout_us = 0);
    void apply_feedback_sample_locked(const FeedbackSample& sample);
    void start_inner_loop();
    void stop_inner_loop();
    void inner_loop();
    double compute_outer_axis_deg_s(double alpha_deg, double alpha_rate_deg_s, double dt_seconds, bool* limited);
    double compute_inner_command_rpm(double desired_rpm, double measured_rpm, double dt_seconds);
    void reset_pid();
    void reset_inner_pid();
    YawCommand make_snapshot_locked() const;
    void set_outer_command_locked(const YawCommand& command, double desired_rpm, bool target_valid);
    double limit_rpm_for_angle(double rpm, double dt_seconds, bool* limited, std::string* status) const;
    double limit_rpm_for_angle_locked(double rpm, double dt_seconds, bool* limited, std::string* status) const;
    double apply_accel_limit(double target_rpm, double dt_seconds, bool* limited);
    double pixel_to_yaw(double target_cx, int frame_width) const;

    SocketCanYawConfig config_;
    int socket_fd_ = -1;
    double current_yaw_deg_ = 0.0;
    double feedback_raw_deg_ = 0.0;
    double feedback_velocity_rpm_ = 0.0;
    int feedback_torque_raw_ = 0;
    double integral_error_deg_s_ = 0.0;
    double previous_error_deg_ = 0.0;
    bool have_previous_error_ = false;
    double inner_integral_error_rpm_s_ = 0.0;
    double previous_speed_error_rpm_ = 0.0;
    bool have_previous_speed_error_ = false;
    bool have_feedback_ = false;
    int direction_error_count_ = 0;
    mutable std::mutex mutex_;
    std::thread inner_thread_;
    bool inner_running_ = false;
    bool outer_target_valid_ = false;
    double outer_desired_rpm_ = 0.0;
    double last_commanded_rpm_ = 0.0;
    double last_desired_rpm_outer_ = 0.0;
    double last_speed_error_rpm_ = 0.0;
    double last_alpha_deg_ = 0.0;
    double last_alpha_rate_deg_s_ = 0.0;
    double last_omega_axis_deg_s_ = 0.0;
    double last_outer_p_ = 0.0;
    double last_outer_i_ = 0.0;
    double last_outer_d_ = 0.0;
    double last_inner_p_ = 0.0;
    double last_inner_i_ = 0.0;
    double last_inner_d_ = 0.0;
    double outer_update_hz_ = 0.0;
    double outer_dt_ms_ = 0.0;
    double inner_loop_hz_ = 0.0;
    double inner_dt_us_ = 0.0;
    double inner_jitter_us_ = 0.0;
    double feedback_hz_ = 0.0;
    bool last_limited_ = false;
    bool last_accel_limited_ = false;
    std::string last_status_ = "disabled";
    std::uint64_t outer_seq_ = 0;
    std::int64_t target_frame_id_ = -1;
    std::uint64_t missed_feedback_count_ = 0;
    std::chrono::steady_clock::time_point last_outer_update_time_;
    std::chrono::steady_clock::time_point last_inner_tick_time_;
    std::chrono::steady_clock::time_point last_feedback_time_;
    std::chrono::steady_clock::time_point last_feedback_query_time_;
};

}  // namespace cvproj
