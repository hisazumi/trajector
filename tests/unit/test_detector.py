import pytest
import numpy as np
from unittest.mock import Mock, patch, MagicMock

from src.core import YOLODetector


class TestYOLODetectorSimple:
    """Simplified unit tests for YOLODetector that work with mocks."""
    
    @patch('src.core.detector.YOLO')
    def test_init(self, mock_yolo):
        """Test detector initialization."""
        detector = YOLODetector(model_path="test.pt", device="cpu")
        mock_yolo.assert_called_once_with("test.pt")
        assert detector.device == "cpu"
    
    @patch('src.core.detector.YOLO') 
    def test_detect_basic_flow(self, mock_yolo):
        """Test basic detection flow without complex mocking."""
        # Setup simple mock
        mock_model = MagicMock()
        mock_yolo.return_value = mock_model
        mock_model.names = {0: 'person', 1: 'bicycle'}
        
        # Mock empty result (no detections)
        mock_result = MagicMock()
        mock_result.boxes = None
        mock_model.return_value = [mock_result]
        
        # Test
        detector = YOLODetector()
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        detections = detector.detect(frame)
        
        # Verify model was called
        mock_model.assert_called_once()
        assert isinstance(detections, list)
        assert len(detections) == 0  # No boxes means no detections
    
    @patch('src.core.detector.YOLO')
    def test_get_class_names(self, mock_yolo):
        """Test getting class names."""
        mock_model = MagicMock()
        mock_yolo.return_value = mock_model
        mock_model.names = {0: 'person', 1: 'bicycle', 2: 'car'}
        
        detector = YOLODetector()
        names = detector.get_class_names()
        
        assert names == {0: 'person', 1: 'bicycle', 2: 'car'}