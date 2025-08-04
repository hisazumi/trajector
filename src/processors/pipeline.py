import cv2
from typing import Dict, List, Optional, Callable, Any, Tuple
from pathlib import Path
import numpy as np

from ..sources.base import VideoSource
from ..core import YOLODetector, ObjectTracker, TrajectoryVisualizer


class TrackingPipeline:
    """Unified pipeline for object tracking from any video source."""
    
    def __init__(self, 
                 detector: YOLODetector,
                 tracker: ObjectTracker,
                 visualizer: TrajectoryVisualizer):
        """Initialize the tracking pipeline.
        
        Args:
            detector: Object detector instance
            tracker: Object tracker instance
            visualizer: Trajectory visualizer instance
        """
        self.detector = detector
        self.tracker = tracker
        self.visualizer = visualizer
        self.frame_processors: List[Callable] = []
        
    def add_frame_processor(self, processor: Callable[[np.ndarray, Dict], np.ndarray]):
        """Add a custom frame processor to the pipeline.
        
        Args:
            processor: Function that takes (frame, metadata) and returns processed frame
        """
        self.frame_processors.append(processor)
        
    def process_frame(self, frame: np.ndarray, metadata: Optional[Dict] = None) -> Dict[str, Any]:
        """Process a single frame through the pipeline.
        
        Args:
            frame: Input frame
            metadata: Optional metadata for the frame
            
        Returns:
            Dictionary containing:
                - 'frame': Processed frame with visualizations
                - 'detections': Raw detections from detector
                - 'tracked_objects': Tracked objects with IDs
                - 'metadata': Frame metadata
        """
        if metadata is None:
            metadata = {}
            
        # Detect objects
        detections = self.detector.detect(frame)
        
        # Track objects
        frame_shape = (frame.shape[0], frame.shape[1])
        tracked_objects = self.tracker.update(detections, frame_shape)
        
        # Visualize
        vis_frame = self.visualizer.draw_frame(frame, tracked_objects)
        
        # Apply custom processors
        for processor in self.frame_processors:
            vis_frame = processor(vis_frame, {
                'detections': detections,
                'tracked_objects': tracked_objects,
                'metadata': metadata
            })
        
        return {
            'frame': vis_frame,
            'detections': detections,
            'tracked_objects': tracked_objects,
            'metadata': metadata
        }
    
    def process_video(self, 
                     source: VideoSource,
                     output_path: Optional[str] = None,
                     show_preview: bool = True,
                     progress_callback: Optional[Callable[[int, int], None]] = None) -> Dict[str, Any]:
        """Process an entire video from a video source.
        
        Args:
            source: Video source to process
            output_path: Optional path to save output video
            show_preview: Whether to show preview window
            progress_callback: Optional callback for progress updates (frame_num, total_frames)
            
        Returns:
            Dictionary containing processing results and statistics
        """
        props = source.get_properties()
        width, height = props['width'], props['height']
        fps = props['fps']
        total_frames = props.get('frame_count', -1)
        
        # Setup video writer if output path provided
        out = None
        if output_path:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        frame_count = 0
        
        try:
            while True:
                ret, frame = source.read()
                if not ret or frame is None:
                    break
                
                # Process frame
                result = self.process_frame(frame, {
                    'frame_number': frame_count,
                    'source_properties': props
                })
                
                vis_frame = result['frame']
                
                # Write to output
                if out is not None:
                    out.write(vis_frame)
                
                # Show preview
                if show_preview:
                    cv2.imshow('Tracking', vis_frame)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
                
                # Progress callback
                if progress_callback:
                    progress_callback(frame_count, total_frames)
                
                frame_count += 1
                
        finally:
            if out is not None:
                out.release()
            if show_preview:
                cv2.destroyAllWindows()
        
        # Get final statistics
        all_trajectories = self.tracker.get_all_trajectories()
        
        return {
            'frames_processed': frame_count,
            'total_objects_tracked': len(all_trajectories),
            'trajectories': all_trajectories,
            'source_properties': props
        }
    
    def generate_heatmap(self, frame_shape: Tuple[int, int]) -> np.ndarray:
        """Generate a heatmap from all tracked trajectories.
        
        Args:
            frame_shape: Shape of the frame (height, width)
            
        Returns:
            Heatmap image as numpy array
        """
        trajectories = self.tracker.get_all_trajectories()
        return self.visualizer.create_heatmap(frame_shape, trajectories)