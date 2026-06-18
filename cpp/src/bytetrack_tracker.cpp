#include "cvproj/bytetrack_tracker.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <set>

namespace cvproj {
namespace {

double rect_iou(const cv::Rect2f& a, const cv::Rect2f& b) {
    const float x1 = std::max(a.x, b.x);
    const float y1 = std::max(a.y, b.y);
    const float x2 = std::min(a.x + a.width, b.x + b.width);
    const float y2 = std::min(a.y + a.height, b.y + b.height);
    const float w = std::max(0.0F, x2 - x1);
    const float h = std::max(0.0F, y2 - y1);
    const double intersection = static_cast<double>(w) * static_cast<double>(h);
    const double union_area = static_cast<double>(a.area() + b.area()) - intersection;
    return union_area > 0.0 ? intersection / union_area : 0.0;
}

cv::Rect clamp_rect(const cv::Rect2f& box, int frame_width, int frame_height) {
    const int x = std::clamp(cvRound(box.x), 0, std::max(0, frame_width - 1));
    const int y = std::clamp(cvRound(box.y), 0, std::max(0, frame_height - 1));
    const int right = std::clamp(cvRound(box.x + box.width), x + 1, std::max(x + 1, frame_width));
    const int bottom = std::clamp(cvRound(box.y + box.height), y + 1, std::max(y + 1, frame_height));
    return cv::Rect(x, y, right - x, bottom - y);
}

cv::Point2f center_of(const cv::Rect2f& box) {
    return cv::Point2f(box.x + 0.5F * box.width, box.y + 0.5F * box.height);
}

}  // namespace

ByteTrackTracker::ByteTrackTracker(ByteTrackConfig config) : config_(config) {
    config_.high_threshold = std::clamp(config_.high_threshold, 0.0F, 1.0F);
    config_.low_threshold = std::clamp(config_.low_threshold, 0.0F, config_.high_threshold);
    config_.match_threshold = std::clamp(config_.match_threshold, 0.0F, 1.0F);
    config_.track_buffer = std::max(0, config_.track_buffer);
    config_.min_hits = std::max(1, config_.min_hits);
}

const ByteTrackConfig& ByteTrackTracker::config() const {
    return config_;
}

std::vector<std::pair<int, int>> ByteTrackTracker::match_tracks(
    const std::vector<int>& track_indices,
    const std::vector<int>& detection_indices,
    const std::vector<TrackerDetection>& detections) const {
    struct Candidate {
        int track_index = -1;
        int detection_index = -1;
        double iou = 0.0;
    };

    std::vector<Candidate> candidates;
    for (const int track_index : track_indices) {
        for (const int detection_index : detection_indices) {
            const double iou = rect_iou(tracks_[track_index].box, detections[detection_index].box);
            if (iou >= config_.match_threshold) {
                candidates.push_back({track_index, detection_index, iou});
            }
        }
    }

    std::sort(candidates.begin(), candidates.end(), [](const Candidate& a, const Candidate& b) {
        return a.iou > b.iou;
    });

    std::set<int> used_tracks;
    std::set<int> used_detections;
    std::vector<std::pair<int, int>> matches;
    for (const auto& candidate : candidates) {
        if (used_tracks.count(candidate.track_index) || used_detections.count(candidate.detection_index)) {
            continue;
        }
        used_tracks.insert(candidate.track_index);
        used_detections.insert(candidate.detection_index);
        matches.emplace_back(candidate.track_index, candidate.detection_index);
    }
    return matches;
}

std::vector<TrackedTarget> ByteTrackTracker::update(const std::vector<TrackerDetection>& detections,
                                                    int frame_width,
                                                    int frame_height,
                                                    bool detections_are_fresh) {
    for (auto& track : tracks_) {
        track.updated = false;
        track.detector_tick = detections_are_fresh;
        track.box.x += track.velocity.x;
        track.box.y += track.velocity.y;
        ++track.age;
        if (detections_are_fresh) {
            ++track.missed_frames;
        }
    }

    std::vector<int> high_detections;
    std::vector<int> low_detections;
    for (std::size_t i = 0; i < detections.size(); ++i) {
        if (detections[i].confidence >= config_.high_threshold) {
            high_detections.push_back(static_cast<int>(i));
        } else if (detections[i].confidence >= config_.low_threshold) {
            low_detections.push_back(static_cast<int>(i));
        }
    }

    std::vector<int> active_tracks;
    active_tracks.reserve(tracks_.size());
    for (std::size_t i = 0; i < tracks_.size(); ++i) {
        active_tracks.push_back(static_cast<int>(i));
    }

    std::set<int> matched_tracks;
    std::set<int> matched_high_detections;
    auto apply_match = [&](int track_index, int detection_index) {
        auto& track = tracks_[track_index];
        const auto& detection = detections[detection_index];
        const cv::Point2f previous_center = center_of(track.box);
        const cv::Rect2f new_box(detection.box);
        const cv::Point2f new_center = center_of(new_box);
        track.velocity = new_center - previous_center;
        track.box = new_box;
        track.confidence = detection.confidence;
        track.class_id = detection.class_id;
        track.class_name = detection.class_name;
        track.source = detection.source;
        track.missed_frames = 0;
        track.updated = true;
        ++track.hits;
    };

    for (const auto& [track_index, detection_index] : match_tracks(active_tracks, high_detections, detections)) {
        apply_match(track_index, detection_index);
        matched_tracks.insert(track_index);
        matched_high_detections.insert(detection_index);
    }

    std::vector<int> unmatched_tracks;
    for (std::size_t i = 0; i < tracks_.size(); ++i) {
        if (!matched_tracks.count(static_cast<int>(i))) {
            unmatched_tracks.push_back(static_cast<int>(i));
        }
    }

    for (const auto& [track_index, detection_index] : match_tracks(unmatched_tracks, low_detections, detections)) {
        apply_match(track_index, detection_index);
        matched_tracks.insert(track_index);
    }

    for (const int detection_index : high_detections) {
        if (matched_high_detections.count(detection_index)) {
            continue;
        }
        const auto& detection = detections[detection_index];
        TrackState track;
        track.track_id = next_track_id_++;
        track.box = cv::Rect2f(detection.box);
        track.confidence = detection.confidence;
        track.class_id = detection.class_id;
        track.class_name = detection.class_name;
        track.source = detection.source;
        track.age = 1;
        track.hits = 1;
        track.missed_frames = 0;
        track.updated = true;
        tracks_.push_back(std::move(track));
    }

    tracks_.erase(std::remove_if(tracks_.begin(),
                                 tracks_.end(),
                                 [&](const TrackState& track) {
                                     if (track.missed_frames <= config_.track_buffer) {
                                         return false;
                                     }
                                     if (track.track_id == primary_track_id_) {
                                         primary_track_id_ = -1;
                                     }
                                     return true;
                                 }),
                  tracks_.end());

    update_primary(frame_width, frame_height);

    std::vector<TrackedTarget> result;
    result.reserve(tracks_.size());
    for (std::size_t i = 0; i < tracks_.size(); ++i) {
        const auto& track = tracks_[i];
        if (track.hits < config_.min_hits && track.track_id != primary_track_id_) {
            continue;
        }

        TrackedTarget target;
        target.target_index = static_cast<int>(i);
        target.track_id = track.track_id;
        target.is_primary = track.track_id == primary_track_id_;
        target.box = clamp_rect(track.box, frame_width, frame_height);
        target.confidence = track.confidence;
        target.class_id = track.class_id;
        target.class_name = track.class_name;
        target.source = track.source;
        if (track.updated) {
            target.state = track.hits >= config_.min_hits ? "tracked" : "tentative";
        } else {
            target.state = track.detector_tick ? "lost" : "predicted";
        }
        target.age = track.age;
        target.hits = track.hits;
        target.missed_frames = track.missed_frames;
        result.push_back(std::move(target));
    }

    std::sort(result.begin(), result.end(), [](const TrackedTarget& a, const TrackedTarget& b) {
        if (a.is_primary != b.is_primary) {
            return a.is_primary;
        }
        if (a.missed_frames != b.missed_frames) {
            return a.missed_frames < b.missed_frames;
        }
        return a.confidence > b.confidence;
    });
    return result;
}

void ByteTrackTracker::update_primary(int frame_width, int frame_height) {
    auto is_confirmed = [&](const TrackState& track) {
        return track.hits >= config_.min_hits &&
               track.confidence >= config_.high_threshold &&
               track.missed_frames <= config_.track_buffer;
    };

    auto current = std::find_if(tracks_.begin(), tracks_.end(), [&](const TrackState& track) {
        return track.track_id == primary_track_id_;
    });
    if (current != tracks_.end() && current->missed_frames <= config_.track_buffer) {
        return;
    }

    primary_track_id_ = -1;
    const cv::Point2f frame_center(static_cast<float>(frame_width) * 0.5F,
                                   static_cast<float>(frame_height) * 0.5F);
    double best_score = std::numeric_limits<double>::max();
    for (const auto& track : tracks_) {
        if (!is_confirmed(track)) {
            continue;
        }
        const cv::Point2f center = center_of(track.box);
        const double dx = static_cast<double>(center.x - frame_center.x);
        const double dy = static_cast<double>(center.y - frame_center.y);
        const double score = std::sqrt(dx * dx + dy * dy);
        if (score < best_score) {
            best_score = score;
            primary_track_id_ = track.track_id;
        }
    }
}

}  // namespace cvproj
