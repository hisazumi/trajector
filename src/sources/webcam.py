import cv2
from typing import Optional, Tuple, Dict, Any
import numpy as np

from .base import VideoSource


class WebcamSource(VideoSource):
    """Webcam source implementation."""
    
    def __init__(self, camera_index: int = 0, **kwargs):
        """Initialize webcam source.
        
        Args:
            camera_index: Camera device index (default: 0)
            **kwargs: Additional OpenCV VideoCapture parameters
        """
        self.camera_index = camera_index
        self.cap = cv2.VideoCapture(camera_index)
        
        if not self.cap.isOpened():
            raise RuntimeError(f"Failed to open camera {camera_index}")
        
        # Apply any additional parameters
        for key, value in kwargs.items():
            if hasattr(cv2, f'CAP_PROP_{key.upper()}'):
                prop = getattr(cv2, f'CAP_PROP_{key.upper()}')
                self.cap.set(prop, value)
    
    def read(self) -> Tuple[bool, Optional[np.ndarray]]:
        """Read a frame from the webcam."""
        ret, frame = self.cap.read()
        return ret, frame if ret else None
    
    def release(self) -> None:
        """Release the video capture object."""
        if self.cap is not None:
            self.cap.release()
    
    def get_properties(self) -> Dict[str, Any]:
        """Get webcam properties."""
        return {
            'width': int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            'height': int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            'fps': int(self.cap.get(cv2.CAP_PROP_FPS)) or 30,
            'frame_count': -1,  # Infinite for live sources
            'current_frame': int(self.cap.get(cv2.CAP_PROP_POS_FRAMES)),
            'source_type': 'webcam',
            'camera_index': self.camera_index
        }
    
    @property
    def is_open(self) -> bool:
        """Check if video capture is open."""
        return self.cap is not None and self.cap.isOpened()