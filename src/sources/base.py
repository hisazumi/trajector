from abc import ABC, abstractmethod
from typing import Optional, Tuple, Dict, Any
import numpy as np


class VideoSource(ABC):
    """Abstract base class for video sources."""
    
    @abstractmethod
    def __init__(self, **kwargs):
        """Initialize the video source."""
        pass
    
    @abstractmethod
    def read(self) -> Tuple[bool, Optional[np.ndarray]]:
        """Read a frame from the video source.
        
        Returns:
            Tuple of (success, frame) where success is a boolean
            and frame is a numpy array or None if unsuccessful.
        """
        pass
    
    @abstractmethod
    def release(self) -> None:
        """Release the video source resources."""
        pass
    
    @abstractmethod
    def get_properties(self) -> Dict[str, Any]:
        """Get video source properties.
        
        Returns:
            Dictionary containing properties like width, height, fps, etc.
        """
        pass
    
    @property
    @abstractmethod
    def is_open(self) -> bool:
        """Check if the video source is open and ready."""
        pass
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.release()