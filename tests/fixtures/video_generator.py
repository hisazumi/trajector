import cv2
import numpy as np
from pathlib import Path


def create_test_video(output_path: str, width: int = 640, height: int = 480, 
                     fps: int = 30, duration: float = 1.0, num_objects: int = 2):
    """Create a test video with moving objects for testing.
    
    Args:
        output_path: Path to save the video
        width: Video width
        height: Video height
        fps: Frames per second
        duration: Duration in seconds
        num_objects: Number of moving objects
    """
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    total_frames = int(fps * duration)
    
    for frame_num in range(total_frames):
        # Create white background
        frame = np.ones((height, width, 3), dtype=np.uint8) * 255
        
        # Add moving objects
        t = frame_num / total_frames
        
        for i in range(num_objects):
            # Different motion patterns for each object
            if i == 0:
                # Horizontal motion
                x = int(50 + (width - 100) * t)
                y = height // 3
            else:
                # Sine wave motion
                x = int(50 + (width - 100) * t)
                y = int(height // 2 + 100 * np.sin(2 * np.pi * t * 2))
            
            # Draw circle (simulating person/object)
            color = (0, 0, 255) if i == 0 else (255, 0, 0)
            cv2.circle(frame, (x, y), 30, color, -1)
            
            # Add label
            cv2.putText(frame, f"Object {i+1}", (x-30, y-35), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
        
        out.write(frame)
    
    out.release()
    return output_path


if __name__ == "__main__":
    # Create test videos when run directly
    create_test_video("tests/fixtures/test_video_short.mp4", duration=1.0)
    create_test_video("tests/fixtures/test_video_long.mp4", duration=5.0)
    print("Test videos created!")