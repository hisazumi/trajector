import pytest
import numpy as np
from unittest.mock import Mock, patch, MagicMock
import cv2

from src.sources import VideoSource, VideoFileSource, WebcamSource


class TestVideoFileSource:
    """Unit tests for VideoFileSource."""
    
    @patch('cv2.VideoCapture')
    @patch('pathlib.Path.exists')
    def test_init_success(self, mock_exists, mock_capture):
        """Test successful initialization."""
        mock_exists.return_value = True
        mock_cap = MagicMock()
        mock_cap.isOpened.return_value = True
        mock_capture.return_value = mock_cap
        
        source = VideoFileSource("test.mp4")
        
        assert source.file_path.name == "test.mp4"
        mock_capture.assert_called_once_with("test.mp4")
        
    @patch('pathlib.Path.exists')
    def test_init_file_not_found(self, mock_exists):
        """Test initialization with non-existent file."""
        mock_exists.return_value = False
        
        with pytest.raises(FileNotFoundError):
            VideoFileSource("nonexistent.mp4")
    
    @patch('cv2.VideoCapture')
    @patch('pathlib.Path.exists')
    def test_read_frame(self, mock_exists, mock_capture):
        """Test reading a frame."""
        mock_exists.return_value = True
        mock_cap = MagicMock()
        mock_cap.isOpened.return_value = True
        
        # Mock frame data
        test_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        mock_cap.read.return_value = (True, test_frame)
        mock_capture.return_value = mock_cap
        
        source = VideoFileSource("test.mp4")
        ret, frame = source.read()
        
        assert ret is True
        assert frame is not None
        assert frame.shape == (480, 640, 3)
    
    @patch('cv2.VideoCapture')
    @patch('pathlib.Path.exists')
    def test_get_properties(self, mock_exists, mock_capture):
        """Test getting video properties."""
        mock_exists.return_value = True
        mock_cap = MagicMock()
        mock_cap.isOpened.return_value = True
        mock_cap.get.side_effect = lambda prop: {
            cv2.CAP_PROP_FRAME_WIDTH: 1920,
            cv2.CAP_PROP_FRAME_HEIGHT: 1080,
            cv2.CAP_PROP_FPS: 30,
            cv2.CAP_PROP_FRAME_COUNT: 300,
            cv2.CAP_PROP_POS_FRAMES: 0
        }.get(prop, 0)
        mock_capture.return_value = mock_cap
        
        source = VideoFileSource("test.mp4")
        props = source.get_properties()
        
        assert props['width'] == 1920
        assert props['height'] == 1080
        assert props['fps'] == 30
        assert props['frame_count'] == 300
        assert props['source_type'] == 'file'


class TestWebcamSource:
    """Unit tests for WebcamSource."""
    
    @patch('cv2.VideoCapture')
    def test_init_success(self, mock_capture):
        """Test successful webcam initialization."""
        mock_cap = MagicMock()
        mock_cap.isOpened.return_value = True
        mock_capture.return_value = mock_cap
        
        source = WebcamSource(0)
        
        assert source.camera_index == 0
        mock_capture.assert_called_once_with(0)
    
    @patch('cv2.VideoCapture')
    def test_init_failure(self, mock_capture):
        """Test webcam initialization failure."""
        mock_cap = MagicMock()
        mock_cap.isOpened.return_value = False
        mock_capture.return_value = mock_cap
        
        with pytest.raises(RuntimeError):
            WebcamSource(0)
    
    @patch('cv2.VideoCapture')
    def test_context_manager(self, mock_capture):
        """Test context manager functionality."""
        mock_cap = MagicMock()
        mock_cap.isOpened.return_value = True
        mock_capture.return_value = mock_cap
        
        with WebcamSource(0) as source:
            assert source.is_open
        
        mock_cap.release.assert_called_once()


class TestVideoSourceAbstract:
    """Test abstract base class behavior."""
    
    def test_cannot_instantiate(self):
        """Test that VideoSource cannot be instantiated directly."""
        with pytest.raises(TypeError):
            VideoSource()