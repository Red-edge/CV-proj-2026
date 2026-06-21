#pragma once

#include <string>
#include <vector>

#include <opencv2/core.hpp>

namespace cvproj {

struct ByteTrackConfig {
    float high_threshold = 0.45F;
    float low_threshold = 0.10F;
    float match_threshold = 0.30F;
    int track_buffer = 30;
    int min_hits = 2;
};

struct TrackerDetection {
    cv::Rect box;
    float confidence = 0.0F;
    int class_id = -1;
    std::string class_name = "unknown";
    std::string source = "detector";
};

struct TrackedTarget {
    int target_index = -1;
    int track_id = -1;
    bool is_primary = false;
    cv::Rect box;
    float confidence = 0.0F;
    int class_id = -1;
    std::string class_name;
    std::string source;
    std::string state = "tentative";
    int age = 0;
    int hits = 0;
    int missed_frames = 0;
};

class ByteTrackTracker {
public:
    explicit ByteTrackTracker(ByteTrackConfig config = {});

    std::vector<TrackedTarget> update(const std::vector<TrackerDetection>& detections,
                                      int frame_width,
                                      int frame_height,
                                      bool detections_are_fresh = true,
                                      cv::Point2f prediction_flow = cv::Point2f(0.0F, 0.0F));
    const ByteTrackConfig& config() const;

private:
    struct TrackState {
        int track_id = -1;
        cv::Rect2f box;
        cv::Point2f velocity{0.0F, 0.0F};
        float confidence = 0.0F;
        int class_id = -1;
        std::string class_name;
        std::string source;
        int age = 0;
        int hits = 0;
        int missed_frames = 0;
        bool updated = false;
        bool detector_tick = false;
    };

    std::vector<std::pair<int, int>> match_tracks(const std::vector<int>& track_indices,
                                                  const std::vector<int>& detection_indices,
                                                  const std::vector<TrackerDetection>& detections) const;
    void update_primary(int frame_width, int frame_height);

    ByteTrackConfig config_;
    int next_track_id_ = 1;
    int primary_track_id_ = -1;
    std::vector<TrackState> tracks_;
};

}  // namespace cvproj
