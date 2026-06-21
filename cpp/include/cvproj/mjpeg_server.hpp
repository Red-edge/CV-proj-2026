#pragma once

#include <atomic>
#include <condition_variable>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

#include <opencv2/core.hpp>

namespace cvproj {

class MjpegServer {
public:
    MjpegServer() = default;
    ~MjpegServer();

    MjpegServer(const MjpegServer&) = delete;
    MjpegServer& operator=(const MjpegServer&) = delete;

    bool start(const std::string& bind_host, int port, std::string* error);
    void stop();
    bool running() const;
    void publish(const cv::Mat& frame_bgr);
    std::string url() const;

private:
    void accept_loop();
    void client_loop(int client_fd);

    std::string bind_host_ = "0.0.0.0";
    int port_ = 8080;
    int server_fd_ = -1;
    std::atomic<bool> running_{false};
    std::thread accept_thread_;

    mutable std::mutex frame_mutex_;
    std::condition_variable frame_cv_;
    std::vector<unsigned char> latest_jpeg_;
    std::uint64_t frame_version_ = 0;
};

}  // namespace cvproj
