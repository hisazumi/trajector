import cv2
from pathlib import Path
from typing import Optional, Tuple, Dict, Any
import numpy as np

from .base import VideoSource


class VideoFileSource(VideoSource):
    """Video file source implementation."""
    
    def __init__(self, file_path: str, **kwargs):
        """Initialize video file source.
        
        Args:
            file_path: Path to the video file
            **kwargs: Additional OpenCV VideoCapture parameters
        """
        self.file_path = Path(file_path)
        if not self.file_path.exists():
            raise FileNotFoundError(f"Video file not found: {file_path}")
        
        self.cap = cv2.VideoCapture(str(self.file_path))
        if not self.cap.isOpened():
            raise RuntimeError(f"Failed to open video file: {file_path}")
        
        # Apply any additional parameters
        for key, value in kwargs.items():
            if hasattr(cv2, f'CAP_PROP_{key.upper()}'):
                prop = getattr(cv2, f'CAP_PROP_{key.upper()}')
                self.cap.set(prop, value)
    
    def read(self) -> Tuple[bool, Optional[np.ndarray]]:
        """Read a frame from the video file."""
        ret, frame = self.cap.read()
        return ret, frame if ret else None
    
    def release(self) -> None:
        """Release the video capture object."""
        if self.cap is not None:
            self.cap.release()
    
    def get_properties(self) -> Dict[str, Any]:
        """Get video properties."""
        return {
            'width': int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            'height': int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            'fps': int(self.cap.get(cv2.CAP_PROP_FPS)),
            'frame_count': int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT)),
            'current_frame': int(self.cap.get(cv2.CAP_PROP_POS_FRAMES)),
            'source_type': 'file',
            'source_path': str(self.file_path)
        }
    
    @property
    def is_open(self) -> bool:
        """Check if video capture is open."""
        return self.cap is not None and self.cap.isOpened()
    
    def seek(self, frame_number: int) -> bool:
        """Seek to a specific frame number.
        
        Args:
            frame_number: Target frame number
            
        Returns:
            True if successful, False otherwise
        """
        return self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)