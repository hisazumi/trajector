#!/usr/bin/env python3
import streamlit as st
import cv2
import numpy as np
import tempfile
from pathlib import Path
import yaml
from datetime import datetime
import io
from PIL import Image

from .sources import VideoFileSource, WebcamSource
from .core import YOLODetector, ObjectTracker, TrajectoryVisualizer
from .processors import TrackingPipeline


def load_config():
    """Load default configuration."""
    with open('config/config.yaml', 'r') as f:
        return yaml.safe_load(f)


def create_pipeline(config):
    """Create tracking pipeline from configuration."""
    detector = YOLODetector(
        model_path=config['detector']['model_path'],
        device=config['detector']['device']
    )
    
    tracker = ObjectTracker(
        max_disappeared=config['tracker']['max_disappeared'],
        max_distance=config['tracker']['max_distance']
    )
    
    visualizer = TrajectoryVisualizer(
        trajectory_length=config['visualizer']['trajectory_length'],
        trajectory_color=config['visualizer']['trajectory_color'],
        trajectory_thickness=config['visualizer']['trajectory_thickness'],
        show_bbox=config['visualizer']['show_bbox'],
        show_id=config['visualizer']['show_id'],
        show_trajectory=config['visualizer']['show_trajectory'],
        show_heatmap=config['visualizer'].get('show_heatmap', False),
        heatmap_alpha=config['visualizer'].get('heatmap_alpha', 0.6),
        heatmap_update_interval=config['visualizer'].get('heatmap_update_interval', 10)
    )
    
    return TrackingPipeline(detector, tracker, visualizer)


def process_video_file(uploaded_file, config, progress_bar, status_text):
    """Process uploaded video file."""
    # Save uploaded file to temporary location
    with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp_file:
        tmp_file.write(uploaded_file.read())
        tmp_path = tmp_file.name
    
    # Create output path
    output_dir = Path('output')
    output_dir.mkdir(exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = str(output_dir / f"tracked_{timestamp}.mp4")
    
    # Create pipeline
    pipeline = create_pipeline(config)
    
    # Process video
    try:
        with VideoFileSource(tmp_path) as source:
            props = source.get_properties()
            total_frames = props['frame_count']
            
            # Setup video writer
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(
                output_path, fourcc,
                props['fps'], (props['width'], props['height'])
            )
            
            # Process frames
            frame_container = st.empty()
            tracked_objects = set()
            frame_count = 0
            
            while True:
                ret, frame = source.read()
                if not ret:
                    break
                
                # Process frame
                result = pipeline.process_frame(frame, {
                    'frame_number': frame_count,
                    'source_properties': props
                })
                
                vis_frame = result['frame']
                out.write(vis_frame)
                
                # Update tracked objects
                for obj in result.get('tracked_objects', []):
                    tracked_objects.add(obj['id'])
                
                # Update progress
                frame_count += 1
                progress = frame_count / total_frames if total_frames > 0 else 0
                progress_bar.progress(progress)
                status_text.text(f"Processing frame {frame_count}/{total_frames} | Objects tracked: {len(tracked_objects)}")
                
                # Show preview every 10th frame
                if frame_count % 10 == 0:
                    frame_rgb = cv2.cvtColor(vis_frame, cv2.COLOR_BGR2RGB)
                    frame_container.image(frame_rgb, channels="RGB", use_container_width=True)
            
            out.release()
            
            # Generate heatmap
            heatmap = pipeline.generate_heatmap((props['height'], props['width']))
            heatmap_path = Path(output_path).with_suffix('.heatmap.png')
            cv2.imwrite(str(heatmap_path), heatmap)
            
            # Clean up temp file
            Path(tmp_path).unlink()
            
            return {
                'output_path': output_path,
                'heatmap_path': str(heatmap_path),
                'frames_processed': frame_count,
                'objects_tracked': len(tracked_objects),
                'heatmap': heatmap
            }
            
    except Exception as e:
        st.error(f"Error processing video: {e}")
        return None


def process_webcam(config, frame_placeholder, info_placeholder):
    """Process webcam stream."""
    pipeline = create_pipeline(config)
    
    cap = cv2.VideoCapture(0)
    frame_count = 0
    tracked_objects = set()
    
    if 'recording' not in st.session_state:
        st.session_state.recording = False
        st.session_state.recorded_frames = []
    
    try:
        while st.session_state.webcam_active:
            ret, frame = cap.read()
            if not ret:
                st.error("Failed to capture frame from webcam")
                break
            
            # Get frame properties
            props = {
                'width': frame.shape[1],
                'height': frame.shape[0],
                'fps': 30
            }
            
            # Process frame
            result = pipeline.process_frame(frame, {
                'frame_number': frame_count,
                'source_properties': props
            })
            
            vis_frame = result['frame']
            
            # Update tracked objects
            for obj in result.get('tracked_objects', []):
                tracked_objects.add(obj['id'])
            
            # Record frame if recording
            if st.session_state.recording:
                st.session_state.recorded_frames.append(vis_frame.copy())
            
            # Display frame
            frame_rgb = cv2.cvtColor(vis_frame, cv2.COLOR_BGR2RGB)
            frame_placeholder.image(frame_rgb, channels="RGB", use_container_width=True)
            
            # Update info
            info_placeholder.text(f"Frame: {frame_count} | Objects: {len(result.get('tracked_objects', []))} | Total tracked: {len(tracked_objects)}")
            
            frame_count += 1
            
    finally:
        cap.release()
        
        # Save recording if frames were captured
        if st.session_state.recording and st.session_state.recorded_frames:
            output_dir = Path('output')
            output_dir.mkdir(exist_ok=True)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = str(output_dir / f"webcam_{timestamp}.mp4")
            
            # Write video
            if st.session_state.recorded_frames:
                h, w = st.session_state.recorded_frames[0].shape[:2]
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                out = cv2.VideoWriter(output_path, fourcc, 30, (w, h))
                
                for frame in st.session_state.recorded_frames:
                    out.write(frame)
                
                out.release()
                st.success(f"Recording saved to {output_path}")
            
            st.session_state.recorded_frames = []
            st.session_state.recording = False


def main():
    st.set_page_config(
        page_title="Trajector - Object Tracking",
        page_icon="🎯",
        layout="wide"
    )
    
    st.title("🎯 Trajector - Object Tracking and Trajectory Visualization")
    st.markdown("Track objects in videos and visualize their trajectories")
    
    # Load config
    config = load_config()
    
    # Sidebar for settings
    with st.sidebar:
        st.header("⚙️ Settings")
        
        st.subheader("Tracker Settings")
        config['tracker']['max_disappeared'] = st.slider(
            "Max Disappeared Frames",
            min_value=5, max_value=100,
            value=config['tracker']['max_disappeared'],
            help="Maximum frames an object can be missing before removal"
        )
        config['tracker']['max_distance'] = st.slider(
            "Max Distance",
            min_value=10, max_value=200,
            value=config['tracker']['max_distance'],
            help="Maximum distance for object association"
        )
        
        st.subheader("Visualizer Settings")
        config['visualizer']['trajectory_length'] = st.slider(
            "Trajectory Length",
            min_value=10, max_value=100,
            value=config['visualizer']['trajectory_length'],
            help="Number of points to show in trajectory"
        )
        config['visualizer']['show_bbox'] = st.checkbox(
            "Show Bounding Box",
            value=config['visualizer']['show_bbox']
        )
        config['visualizer']['show_id'] = st.checkbox(
            "Show Object ID",
            value=config['visualizer']['show_id']
        )
        config['visualizer']['show_trajectory'] = st.checkbox(
            "Show Trajectory",
            value=config['visualizer']['show_trajectory']
        )
        
        st.subheader("Heatmap Settings")
        show_heatmap = st.checkbox(
            "Show Dynamic Heatmap",
            value=config['visualizer'].get('show_heatmap', False)
        )
        config['visualizer']['show_heatmap'] = show_heatmap
        
        if show_heatmap:
            config['visualizer']['heatmap_alpha'] = st.slider(
                "Heatmap Opacity",
                min_value=0.0, max_value=1.0,
                value=config['visualizer'].get('heatmap_alpha', 0.6),
                step=0.1
            )
    
    # Main content
    tab1, tab2, tab3 = st.tabs(["📹 Video File", "📷 Webcam", "📊 Examples"])
    
    with tab1:
        st.header("Upload and Process Video")
        
        uploaded_file = st.file_uploader(
            "Choose a video file",
            type=['mp4', 'avi', 'mov', 'mkv'],
            help="Upload a video file to process"
        )
        
        if uploaded_file is not None:
            st.video(uploaded_file)
            
            col1, col2 = st.columns(2)
            with col1:
                if st.button("🚀 Process Video", type="primary"):
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    
                    with st.spinner("Processing video..."):
                        result = process_video_file(
                            uploaded_file, config,
                            progress_bar, status_text
                        )
                    
                    if result:
                        st.success(f"✅ Processing complete!")
                        st.info(f"📊 Frames: {result['frames_processed']} | Objects tracked: {result['objects_tracked']}")
                        
                        # Display results
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            st.subheader("Processed Video")
                            with open(result['output_path'], 'rb') as f:
                                st.download_button(
                                    label="📥 Download Processed Video",
                                    data=f,
                                    file_name=Path(result['output_path']).name,
                                    mime="video/mp4"
                                )
                        
                        with col2:
                            st.subheader("Trajectory Heatmap")
                            heatmap_rgb = cv2.cvtColor(result['heatmap'], cv2.COLOR_BGR2RGB)
                            st.image(heatmap_rgb, use_container_width=True)
                            
                            with open(result['heatmap_path'], 'rb') as f:
                                st.download_button(
                                    label="📥 Download Heatmap",
                                    data=f,
                                    file_name=Path(result['heatmap_path']).name,
                                    mime="image/png"
                                )
    
    with tab2:
        st.header("Real-time Webcam Tracking")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            if st.button("▶️ Start Webcam", type="primary"):
                st.session_state.webcam_active = True
        
        with col2:
            if st.button("⏹️ Stop Webcam"):
                st.session_state.webcam_active = False
        
        with col3:
            if st.button("🔴 Start Recording"):
                st.session_state.recording = True
                st.session_state.recorded_frames = []
        
        with col4:
            if st.button("⏸️ Stop Recording"):
                st.session_state.recording = False
        
        if st.session_state.get('recording', False):
            st.warning("🔴 Recording in progress...")
        
        # Webcam display area
        frame_placeholder = st.empty()
        info_placeholder = st.empty()
        
        if st.session_state.get('webcam_active', False):
            process_webcam(config, frame_placeholder, info_placeholder)
    
    with tab3:
        st.header("Example Results")
        
        example_dir = Path('examples')
        if example_dir.exists():
            example_files = list(example_dir.glob('*.mp4'))
            
            if example_files:
                selected_example = st.selectbox(
                    "Select an example video",
                    example_files,
                    format_func=lambda x: x.name
                )
                
                if selected_example:
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.subheader("Original Video")
                        st.video(str(selected_example))
                    
                    with col2:
                        # Check for processed version
                        processed_path = selected_example.parent / f"{selected_example.stem}_tracked{selected_example.suffix}"
                        if processed_path.exists():
                            st.subheader("Processed Video")
                            st.video(str(processed_path))
                        
                        # Check for heatmap
                        heatmap_path = selected_example.parent / f"{selected_example.stem}.heatmap.png"
                        if heatmap_path.exists():
                            st.subheader("Trajectory Heatmap")
                            st.image(str(heatmap_path))
            else:
                st.info("No example videos found in the examples directory")
        else:
            st.info("Examples directory not found")
    
    # Footer
    st.markdown("---")
    st.markdown("Built with Streamlit, OpenCV, and YOLOv8")


if __name__ == '__main__':
    main()