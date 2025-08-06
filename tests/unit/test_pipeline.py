import pytest
import numpy as np
from unittest.mock import Mock, MagicMock, patch
import cv2

from src.processors import TrackingPipeline
from src.sources import VideoFileSource


class TestTrackingPipeline:
    """Unit tests for TrackingPipeline."""
    
    def setup_method(self):
        """Setup test fixtures."""
        self.mock_detector = Mock()
        self.mock_tracker = Mock()
        self.mock_visualizer = Mock()
        
        self.pipeline = TrackingPipeline(
            detector=self.mock_detector,
            tracker=self.mock_tracker,
            visualizer=self.mock_visualizer
        )
    
    def test_init(self):
        """Test pipeline initialization."""
        assert self.pipeline.detector == self.mock_detector
        assert self.pipeline.tracker == self.mock_tracker
        assert self.pipeline.visualizer == self.mock_visualizer
        assert len(self.pipeline.frame_processors) == 0
    
    def test_add_frame_processor(self):
        """Test adding custom frame processor."""
        def custom_processor(frame, metadata):
            return frame
        
        self.pipeline.add_frame_processor(custom_processor)
        assert len(self.pipeline.frame_processors) == 1
        assert self.pipeline.frame_processors[0] == custom_processor
    
    def test_process_frame(self):
        """Test processing a single frame."""
        # Setup test data
        test_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        test_detections = [{'bbox': np.array([100, 100, 200, 200])}]
        test_tracked = [{'id': 1, 'bbox': np.array([100, 100, 200, 200])}]
        test_trajectories = {1: [(150, 150), (160, 160)]}
        vis_frame = np.ones((480, 640, 3), dtype=np.uint8)
        
        # Setup mocks
        self.mock_detector.detect.return_value = test_detections
        self.mock_tracker.update.return_value = test_tracked
        self.mock_tracker.get_all_trajectories.return_value = test_trajectories
        self.mock_visualizer.draw_frame.return_value = vis_frame
        
        # Process frame
        result = self.pipeline.process_frame(test_frame)
        
        # Verify calls
        self.mock_detector.detect.assert_called_once_with(test_frame)
        self.mock_tracker.update.assert_called_once_with(test_detections, (480, 640))
        self.mock_tracker.get_all_trajectories.assert_called_once()
        # Updated to match new draw_frame signature with trajectories
        self.mock_visualizer.draw_frame.assert_called_once_with(test_frame, test_tracked, test_trajectories)
        
        # Check result
        assert 'frame' in result
        assert 'detections' in result
        assert 'tracked_objects' in result
        assert result['detections'] == test_detections
        assert result['tracked_objects'] == test_tracked
    
    @patch('cv2.VideoWriter')
    @patch('cv2.imshow')
    @patch('cv2.waitKey')
    @patch('cv2.destroyAllWindows')
    def test_process_video(self, mock_destroy, mock_waitkey, mock_imshow, mock_writer):
        """Test processing a video source."""
        # Setup mock video source
        mock_source = Mock()
        mock_source.get_properties.return_value = {
            'width': 640, 'height': 480, 'fps': 30, 'frame_count': 3
        }
        
        # Mock frames
        frames = [
            (True, np.zeros((480, 640, 3), dtype=np.uint8)),
            (True, np.zeros((480, 640, 3), dtype=np.uint8)),
            (True, np.zeros((480, 640, 3), dtype=np.uint8)),
            (False, None)
        ]
        mock_source.read.side_effect = frames
        
        # Setup pipeline mocks
        self.mock_detector.detect.return_value = []
        self.mock_tracker.update.return_value = []
        self.mock_tracker.get_all_trajectories.return_value = {}
        self.mock_visualizer.draw_frame.return_value = np.zeros((480, 640, 3), dtype=np.uint8)
        
        # Mock cv2 functions
        mock_waitkey.return_value = -1  # No key pressed
        
        # Process video
        results = self.pipeline.process_video(mock_source, show_preview=False)
        
        # Verify results
        assert results['frames_processed'] == 3
        assert results['total_objects_tracked'] == 0
        assert self.mock_detector.detect.call_count == 3
    
    def test_generate_heatmap(self):
        """Test heatmap generation."""
        test_trajectories = {1: [(100, 100), (110, 110)]}
        test_heatmap = np.zeros((480, 640, 3), dtype=np.uint8)
        
        self.mock_tracker.get_all_trajectories.return_value = test_trajectories
        self.mock_visualizer.create_heatmap.return_value = test_heatmap
        
        heatmap = self.pipeline.generate_heatmap((480, 640))
        
        self.mock_visualizer.create_heatmap.assert_called_once_with((480, 640), test_trajectories)
        assert heatmap.shape == (480, 640, 3)