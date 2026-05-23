#include "central_control/common.hpp"

#include <atomic>
#include <condition_variable>
#include <iostream>
#include <map>
#include <thread>

#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

namespace central_control {

namespace {

template <typename T>
class BoundedQueue {
public:
    explicit BoundedQueue(std::size_t capacity)
        : capacity_(capacity) {}

    bool push(T value, std::atomic<std::uint64_t>* dropped_counter = nullptr) {
        std::lock_guard<std::mutex> lock(mutex_);
        if (stopped_) {
            return false;
        }
        if (queue_.size() >= capacity_) {
            queue_.pop_front();
            if (dropped_counter != nullptr) {
                dropped_counter->fetch_add(1);
            }
        }
        queue_.push_back(std::move(value));
        cv_.notify_one();
        return true;
    }

    bool pop(T& value) {
        std::unique_lock<std::mutex> lock(mutex_);
        cv_.wait(lock, [&]() { return stopped_ || !queue_.empty(); });
        if (queue_.empty()) {
            return false;
        }
        value = std::move(queue_.front());
        queue_.pop_front();
        return true;
    }

    void stop() {
        std::lock_guard<std::mutex> lock(mutex_);
        stopped_ = true;
        cv_.notify_all();
    }

private:
    std::size_t capacity_;
    std::deque<T> queue_;
    bool stopped_ = false;
    std::mutex mutex_;
    std::condition_variable cv_;
};

template <typename T>
class LatestSlot {
public:
    bool store(T value, std::atomic<std::uint64_t>* dropped_counter = nullptr) {
        std::lock_guard<std::mutex> lock(mutex_);
        if (stopped_) {
            return false;
        }
        if (value_.has_value() && dropped_counter != nullptr) {
            dropped_counter->fetch_add(1);
        }
        value_ = std::move(value);
        cv_.notify_one();
        return true;
    }

    bool wait_and_take(T& out) {
        std::unique_lock<std::mutex> lock(mutex_);
        cv_.wait(lock, [&]() { return stopped_ || value_.has_value(); });
        if (!value_.has_value()) {
            return false;
        }
        out = std::move(value_.value());
        value_.reset();
        return true;
    }

    void stop() {
        std::lock_guard<std::mutex> lock(mutex_);
        stopped_ = true;
        cv_.notify_all();
    }

private:
    std::optional<T> value_;
    bool stopped_ = false;
    std::mutex mutex_;
    std::condition_variable cv_;
};

template <typename T>
class RecentStore {
public:
    explicit RecentStore(std::size_t max_entries)
        : max_entries_(max_entries) {}

    void put(std::uint64_t key, T value) {
        std::lock_guard<std::mutex> lock(mutex_);
        values_[key] = std::move(value);
        while (values_.size() > max_entries_) {
            values_.erase(values_.begin());
        }
    }

    std::optional<T> latest_not_after(std::uint64_t key) const {
        std::lock_guard<std::mutex> lock(mutex_);
        auto it = values_.upper_bound(key);
        if (it == values_.begin()) {
            return std::nullopt;
        }
        --it;
        return it->second;
    }

private:
    std::size_t max_entries_;
    mutable std::mutex mutex_;
    std::map<std::uint64_t, T> values_;
};

struct DetectJob {
    FramePacket frame;
    MotionState motion;
};

double elapsed_ms(const Clock::time_point& start, const Clock::time_point& end) {
    return std::chrono::duration<double, std::milli>(end - start).count();
}

}  // namespace

Runtime::Runtime(AppConfig config)
    : config_(std::move(config)) {}

bool Runtime::initialize() {
    frame_source_ = create_frame_source(config_);
    flow_estimator_ = create_flow_estimator(config_);
    detector_ = create_detector(config_);
    tracker_ = create_tracker(config_);
    renderer_ = create_renderer(config_);
    output_sink_ = create_output_sink(config_);
    actuator_ = create_actuator(config_);

    if (!frame_source_ || !flow_estimator_ || !detector_ || !tracker_ || !renderer_ || !output_sink_ || !actuator_) {
        std::cerr << "Failed to create runtime components" << std::endl;
        return false;
    }
    if (!frame_source_->open(config_)) {
        std::cerr << "Failed to open frame source: " << config_.source << std::endl;
        return false;
    }
    if (!detector_->initialize(config_)) {
        std::cerr << "Detector initialization failed" << std::endl;
        return false;
    }
    if (!output_sink_->open(config_)) {
        std::cerr << "Output sink initialization failed" << std::endl;
        return false;
    }
    if (!actuator_->initialize(config_)) {
        std::cerr << "Actuator initialization failed" << std::endl;
        return false;
    }
    control_center_.register_actuator(actuator_);
    control_center_.publish_sensor(SensorSnapshot{"frame_source", 0.0, frame_source_->describe()});
    control_center_.publish_sensor(SensorSnapshot{"detector_backend", 0.0, detector_->backend_name()});
    std::cout << "Frame source: " << frame_source_->describe() << std::endl;
    std::cout << "Detector backend: " << detector_->backend_name() << std::endl;
    if (config_.record_rendered) {
        std::cout << "Recording output to: " << output_sink_->output_path() << std::endl;
    }
    return true;
}

int Runtime::run() {
    BoundedQueue<FramePacket> render_queue(512);
    BoundedQueue<FramePacket> flow_queue(512);
    LatestSlot<DetectJob> detect_slot;
    RecentStore<MotionState> recent_motion(256);
    RecentStore<DetectionState> recent_detection(256);

    std::atomic<bool> running{true};
    RuntimeStats stats;
    std::mutex stats_mutex;
    std::atomic<std::uint64_t> dropped_flow_jobs{0};
    std::atomic<std::uint64_t> dropped_detect_jobs{0};

    const auto run_start = Clock::now();

    auto capture_thread = std::thread([&]() {
        std::uint64_t frame_id = 0;
        auto last_fps_tick = Clock::now();
        std::uint64_t last_fps_count = 0;
        while (running.load()) {
            if (elapsed_ms(run_start, Clock::now()) >= static_cast<double>(config_.duration_sec) * 1000.0) {
                running.store(false);
                break;
            }

            FramePacket packet;
            if (!frame_source_->read(packet) || packet.image.empty()) {
                std::this_thread::sleep_for(std::chrono::milliseconds(5));
                continue;
            }
            packet.frame_id = frame_id++;
            packet.capture_time = Clock::now();

            render_queue.push(packet, &dropped_flow_jobs);
            flow_queue.push(packet, &dropped_flow_jobs);

            {
                std::lock_guard<std::mutex> lock(stats_mutex);
                stats.captured_frames += 1;
                const auto now = Clock::now();
                const double window_ms = elapsed_ms(last_fps_tick, now);
                if (window_ms >= 500.0) {
                    const std::uint64_t delta = stats.captured_frames - last_fps_count;
                    stats.capture_fps = static_cast<double>(delta) * 1000.0 / window_ms;
                    last_fps_count = stats.captured_frames;
                    last_fps_tick = now;
                }
            }
        }
        render_queue.stop();
        flow_queue.stop();
        detect_slot.stop();
    });

    auto flow_thread = std::thread([&]() {
        FramePacket packet;
        while (flow_queue.pop(packet)) {
            MotionState motion = flow_estimator_->process(packet);
            recent_motion.put(packet.frame_id, motion);
            detect_slot.store(DetectJob{packet, motion}, &dropped_detect_jobs);
        }
    });

    auto detect_thread = std::thread([&]() {
        DetectJob job;
        while (detect_slot.wait_and_take(job)) {
            double detection_ms = 0.0;
            auto detections = detector_->detect(job.frame, job.motion, detection_ms);
            auto tracks = tracker_->update(detections, job.frame.frame_id);

            DetectionState state;
            state.frame_id = job.frame.frame_id;
            state.detections = std::move(detections);
            state.tracks = std::move(tracks);
            state.detection_ms = detection_ms;
            recent_detection.put(job.frame.frame_id, state);

            std::lock_guard<std::mutex> lock(stats_mutex);
            stats.detection_ms = detection_ms;
        }
    });

    auto render_thread = std::thread([&]() {
        FramePacket packet;
        auto last_render_tick = Clock::now();
        std::uint64_t last_render_count = 0;

        if (!config_.headless) {
            cv::namedWindow(config_.window_name, cv::WINDOW_NORMAL);
        }

        while (render_queue.pop(packet)) {
            const auto motion = recent_motion.latest_not_after(packet.frame_id);
            const auto detection = recent_detection.latest_not_after(packet.frame_id);

            RuntimeStats snapshot;
            {
                std::lock_guard<std::mutex> lock(stats_mutex);
                stats.rendered_frames += 1;
                snapshot = stats;
                snapshot.dropped_flow_jobs = dropped_flow_jobs.load();
                snapshot.dropped_detect_jobs = dropped_detect_jobs.load();
                const auto now = Clock::now();
                const double window_ms = elapsed_ms(last_render_tick, now);
                if (window_ms >= 500.0) {
                    const std::uint64_t delta = stats.rendered_frames - last_render_count;
                    stats.render_fps = static_cast<double>(delta) * 1000.0 / window_ms;
                    last_render_count = stats.rendered_frames;
                    last_render_tick = now;
                }
                snapshot.render_fps = stats.render_fps;
                snapshot.capture_fps = stats.capture_fps;
                if (detection.has_value()) {
                    snapshot.inference_lag_frames = static_cast<int>(packet.frame_id - detection->frame_id);
                    const auto detection_age = elapsed_ms(packet.capture_time, Clock::now());
                    snapshot.result_age_ms = detection_age;
                    snapshot.detection_ms = detection->detection_ms;
                }
            }

            cv::Mat rendered = renderer_->render(packet, motion, detection, snapshot);
            if (config_.record_rendered) {
                output_sink_->write(rendered);
                std::lock_guard<std::mutex> lock(stats_mutex);
                stats.written_frames += 1;
            }
            if (!config_.headless && !rendered.empty()) {
                cv::imshow(config_.window_name, rendered);
                const int key = cv::waitKey(1) & 0xFF;
                if (key == 'q' || key == 'Q') {
                    running.store(false);
                    render_queue.stop();
                    flow_queue.stop();
                    detect_slot.stop();
                    break;
                }
            }

            if (detection.has_value() && !detection->detections.empty()) {
                const auto& det = detection->detections.front();
                const float cx = det.bbox.x + det.bbox.width * 0.5F;
                const float cy = det.bbox.y + det.bbox.height * 0.5F;
                const float yaw_deg = ((cx / static_cast<float>(packet.image.cols)) - 0.5F) * config_.h_fov;
                const float pitch_deg = (0.5F - (cy / static_cast<float>(packet.image.rows))) * config_.v_fov;
                control_center_.publish_target(TargetObservation{
                    packet.frame_id, det.bbox, det.confidence, cv::Point2f(cx, cy), yaw_deg, pitch_deg});
                control_center_.dispatch(ControlCommand{
                    "gimbal_track",
                    -config_.yaw_rpm_per_deg * yaw_deg + config_.yaw_zero,
                    -config_.pitch_rpm_per_deg * pitch_deg + config_.pitch_zero});
            } else {
                control_center_.dispatch(ControlCommand{"gimbal_idle", 0.0F, 0.0F});
            }
        }
        running.store(false);
        if (!config_.headless) {
            cv::destroyAllWindows();
        }
    });

    capture_thread.join();
    flow_thread.join();
    detect_thread.join();
    render_thread.join();

    control_center_.stop_all();
    output_sink_->close();
    frame_source_->close();

    RuntimeStats final_stats;
    {
        std::lock_guard<std::mutex> lock(stats_mutex);
        final_stats = stats;
    }
    final_stats.dropped_flow_jobs = dropped_flow_jobs.load();
    final_stats.dropped_detect_jobs = dropped_detect_jobs.load();
    std::cout << "Run summary:" << std::endl;
    std::cout << "  captured_frames=" << final_stats.captured_frames << std::endl;
    std::cout << "  rendered_frames=" << final_stats.rendered_frames << std::endl;
    std::cout << "  written_frames=" << final_stats.written_frames << std::endl;
    std::cout << "  capture_fps=" << final_stats.capture_fps << std::endl;
    std::cout << "  render_fps=" << final_stats.render_fps << std::endl;
    std::cout << "  detect_ms=" << final_stats.detection_ms << std::endl;
    std::cout << "  dropped_flow_jobs=" << final_stats.dropped_flow_jobs << std::endl;
    std::cout << "  dropped_detect_jobs=" << final_stats.dropped_detect_jobs << std::endl;
    if (config_.record_rendered) {
        std::cout << "  output=" << output_sink_->output_path() << std::endl;
    }
    return 0;
}

}  // namespace central_control
