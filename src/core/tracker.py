import numpy as np
from typing import Dict, List, Tuple, Optional
from collections import defaultdict
import supervision as sv


class ObjectTracker:
    def __init__(self, max_disappeared: int = 30, max_distance: float = 50):
        self.byte_tracker = sv.ByteTrack(
            track_activation_threshold=0.25,
            lost_track_buffer=max_disappeared,
            minimum_matching_threshold=0.8,
            frame_rate=30
        )
        self.tracks = defaultdict(list)
        self.max_distance = max_distance
        
    def update(self, detections: List[Dict], frame_shape: Tuple[int, int]) -> List[Dict]:
        if not detections:
            return []
        
        # Convert detections to supervision format
        xyxy = np.array([d['bbox'] for d in detections])
        confidence = np.array([d['confidence'] for d in detections])
        class_id = np.array([d['class_id'] for d in detections])
        
        detections_sv = sv.Detections(
            xyxy=xyxy,
            confidence=confidence,
            class_id=class_id
        )
        
        # Update tracks
        tracks = self.byte_tracker.update_with_detections(detections_sv)
        
        # Build tracked objects with IDs
        tracked_objects = []
        for i, track_id in enumerate(tracks.tracker_id):
            if track_id is not None:
                center_x = int((tracks.xyxy[i][0] + tracks.xyxy[i][2]) / 2)
                center_y = int((tracks.xyxy[i][1] + tracks.xyxy[i][3]) / 2)
                
                # Store trajectory
                self.tracks[track_id].append((center_x, center_y))
                
                tracked_obj = {
                    'id': track_id,
                    'bbox': tracks.xyxy[i],
                    'center': (center_x, center_y),
                    'class_id': tracks.class_id[i] if tracks.class_id is not None else 0,
                    'confidence': tracks.confidence[i] if tracks.confidence is not None else 1.0,
                    'trajectory': list(self.tracks[track_id])
                }
                tracked_objects.append(tracked_obj)
        
        return tracked_objects
    
    def get_all_trajectories(self) -> Dict[int, List[Tuple[int, int]]]:
        return dict(self.tracks)