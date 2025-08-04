import pytest
import numpy as np
from unittest.mock import Mock, patch, MagicMock

from src.core import ObjectTracker


class TestObjectTracker:
    """Unit tests for ObjectTracker."""
    
    @patch('supervision.ByteTrack')
    def test_init(self, mock_bytetrack):
        """Test tracker initialization."""
        tracker = ObjectTracker(max_disappeared=30, max_distance=50)
        
        mock_bytetrack.assert_called_once()
        assert hasattr(tracker, 'tracks')
        assert hasattr(tracker, 'max_distance')
    
    @patch('supervision.ByteTrack')
    @patch('supervision.Detections')
    def test_update_with_detections(self, mock_detections_class, mock_bytetrack):
        """Test updating tracker with detections."""
        # Setup mock ByteTracker
        mock_tracker_instance = MagicMock()
        mock_bytetrack.return_value = mock_tracker_instance
        
        # Mock tracked objects
        mock_tracks = MagicMock()
        mock_tracks.tracker_id = np.array([1, 2])
        mock_tracks.xyxy = np.array([[100, 100, 200, 200], [300, 300, 400, 400]])
        mock_tracks.class_id = np.array([0, 0])
        mock_tracks.confidence = np.array([0.9, 0.8])
        
        mock_tracker_instance.update_with_detections.return_value = mock_tracks
        
        # Create tracker and test
        tracker = ObjectTracker()
        detections = [
            {'bbox': np.array([100, 100, 200, 200]), 'confidence': 0.9, 'class_id': 0},
            {'bbox': np.array([300, 300, 400, 400]), 'confidence': 0.8, 'class_id': 0}
        ]
        
        tracked_objects = tracker.update(detections, (480, 640))
        
        assert len(tracked_objects) == 2
        assert tracked_objects[0]['id'] == 1
        assert tracked_objects[0]['center'] == (150, 150)
        assert tracked_objects[1]['id'] == 2
        assert tracked_objects[1]['center'] == (350, 350)
    
    @patch('supervision.ByteTrack')
    def test_update_no_detections(self, mock_bytetrack):
        """Test updating tracker with no detections."""
        tracker = ObjectTracker()
        tracked_objects = tracker.update([], (480, 640))
        
        assert len(tracked_objects) == 0
    
    @patch('supervision.ByteTrack')
    def test_get_all_trajectories(self, mock_bytetrack):
        """Test getting all trajectories."""
        tracker = ObjectTracker()
        
        # Manually add some tracks
        tracker.tracks[1] = [(100, 100), (110, 110), (120, 120)]
        tracker.tracks[2] = [(200, 200), (210, 210)]
        
        trajectories = tracker.get_all_trajectories()
        
        assert len(trajectories) == 2
        assert len(trajectories[1]) == 3
        assert len(trajectories[2]) == 2
        assert trajectories[1][0] == (100, 100)