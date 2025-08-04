import pytest
import numpy as np
from pathlib import Path
import cv2
import tempfile

from src.sources import VideoFileSource, WebcamSource
from src.core import YOLODetector, ObjectTracker, TrajectoryVisualizer
from src.processors import TrackingPipeline
from tests.fixtures.video_generator import create_test_video


class TestFullPipeline:
    """Integration tests for the complete tracking pipeline."""
    
    @pytest.fixture
    def test_video(self):
        """Create a temporary test video."""
        with tempfile.NamedTemporaryFile(suffix='.mp4', delete=False) as tmp:
            video_path = tmp.name
        
        create_test_video(video_path, duration=2.0, num_objects=2)
        yield video_path
        
        # Cleanup
        Path(video_path).unlink(missing_ok=True)
    
    @pytest.fixture
    def pipeline_components(self):
        """Create pipeline components with real implementations."""
        # Note: In actual tests, you might want to use lighter models or mocks
        detector = YOLODetector(model_path="yolov8n.pt", device="cpu")
        tracker = ObjectTracker(max_disappeared=30, max_distance=50)
        visualizer = TrajectoryVisualizer(
            trajectory_length=50,
            trajectory_color=(0, 255, 0),
            trajectory_thickness=2
        )
        return detector, tracker, visualizer
    
    def test_video_file_pipeline(self, test_video, pipeline_components):
        """Test pipeline with video file source."""
        detector, tracker, visualizer = pipeline_components
        pipeline = TrackingPipeline(detector, tracker, visualizer)
        
        # Process video
        with VideoFileSource(test_video) as source:
            results = pipeline.process_video(source, show_preview=False)
        
        # Verify results
        assert results['frames_processed'] > 0
        assert 'total_objects_tracked' in results
        assert 'trajectories' in results
        assert results['source_properties']['source_type'] == 'file'
    
    def test_pipeline_with_output(self, test_video, pipeline_components):
        """Test pipeline with video output."""
        detector, tracker, visualizer = pipeline_components
        pipeline = TrackingPipeline(detector, tracker, visualizer)
        
        with tempfile.NamedTemporaryFile(suffix='_output.mp4', delete=False) as tmp:
            output_path = tmp.name
        
        try:
            # Process video with output
            with VideoFileSource(test_video) as source:
                results = pipeline.process_video(
                    source, 
                    output_path=output_path,
                    show_preview=False
                )
            
            # Verify output file exists and has content
            assert Path(output_path).exists()
            assert Path(output_path).stat().st_size > 0
            
            # Verify output video properties
            cap = cv2.VideoCapture(output_path)
            assert cap.isOpened()
            assert cap.get(cv2.CAP_PROP_FRAME_COUNT) > 0
            cap.release()
            
        finally:
            Path(output_path).unlink(missing_ok=True)
    
    def test_custom_frame_processor(self, test_video, pipeline_components):
        """Test pipeline with custom frame processor."""
        detector, tracker, visualizer = pipeline_components
        pipeline = TrackingPipeline(detector, tracker, visualizer)
        
        # Add custom processor that adds text
        def add_timestamp(frame, metadata):
            frame_num = metadata['metadata'].get('frame_number', 0)
            cv2.putText(frame, f"Frame: {frame_num}", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            return frame
        
        pipeline.add_frame_processor(add_timestamp)
        
        # Process single frame
        test_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        result = pipeline.process_frame(test_frame, {'frame_number': 42})
        
        # Result should have the custom processing applied
        assert result['metadata']['frame_number'] == 42
    
    def test_heatmap_generation(self, test_video, pipeline_components):
        """Test heatmap generation after processing."""
        detector, tracker, visualizer = pipeline_components
        pipeline = TrackingPipeline(detector, tracker, visualizer)
        
        # Process video
        with VideoFileSource(test_video) as source:
            results = pipeline.process_video(source, show_preview=False)
        
        # Generate heatmap
        heatmap = pipeline.generate_heatmap((480, 640))
        
        # Verify heatmap
        assert heatmap is not None
        assert heatmap.shape == (480, 640, 3)
        assert heatmap.dtype == np.uint8
    
    def test_progress_callback(self, test_video, pipeline_components):
        """Test progress callback functionality."""
        detector, tracker, visualizer = pipeline_components
        pipeline = TrackingPipeline(detector, tracker, visualizer)
        
        progress_calls = []
        
        def progress_callback(current, total):
            progress_calls.append((current, total))
        
        # Process video with callback
        with VideoFileSource(test_video) as source:
            pipeline.process_video(
                source,
                show_preview=False,
                progress_callback=progress_callback
            )
        
        # Verify callbacks were made
        assert len(progress_calls) > 0
        assert progress_calls[0][0] == 0  # First frame
        assert progress_calls[-1][0] == len(progress_calls) - 1  # Last frame