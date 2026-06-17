#include "cvproj/mjpeg_server.hpp"

#include <cerrno>
#include <cstring>
#include <iostream>
#include <csignal>
#include <sstream>

#include <opencv2/imgcodecs.hpp>

#if defined(_WIN32)
#error "MjpegServer requires a POSIX socket platform"
#else
#include <arpa/inet.h>
#include <netinet/in.h>
#include <sys/socket.h>
#include <unistd.h>
#endif

namespace cvproj {

namespace {
bool send_all(int fd, const char* data, std::size_t size) {
    std::size_t sent = 0;
    while (sent < size) {
        const ssize_t n = ::send(fd, data + sent, size - sent, 0);
        if (n <= 0) {
            return false;
        }
        sent += static_cast<std::size_t>(n);
    }
    return true;
}

bool send_string(int fd, const std::string& text) {
    return send_all(fd, text.data(), text.size());
}
}  // namespace

MjpegServer::~MjpegServer() {
    stop();
}

bool MjpegServer::start(const std::string& bind_host, int port, std::string* error) {
    if (running_) {
        return true;
    }

    bind_host_ = bind_host.empty() ? "0.0.0.0" : bind_host;
    port_ = port;
    std::signal(SIGPIPE, SIG_IGN);

    server_fd_ = ::socket(AF_INET, SOCK_STREAM, 0);
    if (server_fd_ < 0) {
        if (error) {
            *error = std::string("socket failed: ") + std::strerror(errno);
        }
        return false;
    }

    int reuse = 1;
    ::setsockopt(server_fd_, SOL_SOCKET, SO_REUSEADDR, &reuse, sizeof(reuse));

    sockaddr_in addr {};
    addr.sin_family = AF_INET;
    addr.sin_port = htons(static_cast<uint16_t>(port_));
    if (bind_host_ == "0.0.0.0" || bind_host_ == "*") {
        addr.sin_addr.s_addr = htonl(INADDR_ANY);
    } else if (::inet_pton(AF_INET, bind_host_.c_str(), &addr.sin_addr) != 1) {
        if (error) {
            *error = "Invalid IPv4 bind host: " + bind_host_;
        }
        ::close(server_fd_);
        server_fd_ = -1;
        return false;
    }

    if (::bind(server_fd_, reinterpret_cast<sockaddr*>(&addr), sizeof(addr)) < 0) {
        if (error) {
            *error = std::string("bind failed: ") + std::strerror(errno);
        }
        ::close(server_fd_);
        server_fd_ = -1;
        return false;
    }

    if (::listen(server_fd_, 8) < 0) {
        if (error) {
            *error = std::string("listen failed: ") + std::strerror(errno);
        }
        ::close(server_fd_);
        server_fd_ = -1;
        return false;
    }

    running_ = true;
    accept_thread_ = std::thread(&MjpegServer::accept_loop, this);
    return true;
}

void MjpegServer::stop() {
    if (!running_.exchange(false)) {
        return;
    }
    frame_cv_.notify_all();
    if (server_fd_ >= 0) {
        ::shutdown(server_fd_, SHUT_RDWR);
        ::close(server_fd_);
        server_fd_ = -1;
    }
    if (accept_thread_.joinable()) {
        accept_thread_.join();
    }
}

bool MjpegServer::running() const {
    return running_;
}

void MjpegServer::publish(const cv::Mat& frame_bgr) {
    if (!running_ || frame_bgr.empty()) {
        return;
    }

    std::vector<unsigned char> jpeg;
    std::vector<int> params {cv::IMWRITE_JPEG_QUALITY, 82};
    if (!cv::imencode(".jpg", frame_bgr, jpeg, params)) {
        return;
    }

    {
        std::lock_guard<std::mutex> lock(frame_mutex_);
        latest_jpeg_ = std::move(jpeg);
        ++frame_version_;
    }
    frame_cv_.notify_all();
}

std::string MjpegServer::url() const {
    std::ostringstream oss;
    oss << "http://" << (bind_host_ == "0.0.0.0" ? "<rock5b-ip>" : bind_host_) << ':' << port_ << '/';
    return oss.str();
}

void MjpegServer::accept_loop() {
    while (running_) {
        sockaddr_in client_addr {};
        socklen_t client_len = sizeof(client_addr);
        const int client_fd = ::accept(server_fd_, reinterpret_cast<sockaddr*>(&client_addr), &client_len);
        if (client_fd < 0) {
            if (running_) {
                std::cerr << "MJPEG accept failed: " << std::strerror(errno) << '\n';
            }
            continue;
        }
        std::thread(&MjpegServer::client_loop, this, client_fd).detach();
    }
}

void MjpegServer::client_loop(int client_fd) {
    char request[1024];
    const ssize_t n = ::recv(client_fd, request, sizeof(request) - 1, 0);
    if (n <= 0) {
        ::close(client_fd);
        return;
    }
    request[n] = '\0';
    const std::string request_text(request);

    if (request_text.find("GET /stream") == std::string::npos) {
        const std::string page =
            "HTTP/1.1 200 OK\r\n"
            "Content-Type: text/html; charset=utf-8\r\n"
            "Cache-Control: no-cache\r\n\r\n"
            "<!doctype html><html><head><meta name=\"viewport\" content=\"width=device-width,initial-scale=1\">"
            "<title>CV-proj-2026</title>"
            "<style>body{margin:0;background:#101214;color:#e8ecef;font-family:system-ui,sans-serif}"
            "header{padding:10px 14px;background:#1c2329}img{display:block;width:100%;height:auto}</style>"
            "</head><body><header>CV-proj-2026 realtime preview</header>"
            "<img src=\"/stream\" alt=\"annotated stream\"></body></html>";
        send_string(client_fd, page);
        ::close(client_fd);
        return;
    }

    const std::string header =
        "HTTP/1.1 200 OK\r\n"
        "Connection: close\r\n"
        "Cache-Control: no-cache, no-store, must-revalidate\r\n"
        "Pragma: no-cache\r\n"
        "Content-Type: multipart/x-mixed-replace; boundary=frame\r\n\r\n";
    if (!send_string(client_fd, header)) {
        ::close(client_fd);
        return;
    }

    std::uint64_t sent_version = 0;
    while (running_) {
        std::vector<unsigned char> jpeg;
        {
            std::unique_lock<std::mutex> lock(frame_mutex_);
            frame_cv_.wait(lock, [&] { return !running_ || frame_version_ != sent_version; });
            if (!running_) {
                break;
            }
            sent_version = frame_version_;
            jpeg = latest_jpeg_;
        }
        if (jpeg.empty()) {
            continue;
        }

        std::ostringstream part;
        part << "--frame\r\n"
             << "Content-Type: image/jpeg\r\n"
             << "Content-Length: " << jpeg.size() << "\r\n\r\n";
        if (!send_string(client_fd, part.str()) ||
            !send_all(client_fd, reinterpret_cast<const char*>(jpeg.data()), jpeg.size()) ||
            !send_string(client_fd, "\r\n")) {
            break;
        }
    }
    ::close(client_fd);
}

}  // namespace cvproj
