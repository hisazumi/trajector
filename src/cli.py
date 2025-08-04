#!/usr/bin/env python3
import argparse
import sys
from pathlib import Path
from datetime import datetime
import cv2
import yaml

from .sources import VideoFileSource, WebcamSource
from .core import YOLODetector, ObjectTracker, TrajectoryVisualizer
from .processors import TrackingPipeline


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def create_pipeline(config: dict) -> TrackingPipeline:
    """Create tracking pipeline from configuration."""
    # Initialize components
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
        show_trajectory=config['visualizer']['show_trajectory']
    )
    
    return TrackingPipeline(detector, tracker, visualizer)


def add_status_overlay(frame, metadata):
    """Add status overlay to frame."""
    info = metadata.get('metadata', {})
    tracked_objects = metadata.get('tracked_objects', [])
    
    status_text = f"Frame: {info.get('frame_number', 0)} | Objects: {len(tracked_objects)}"
    cv2.putText(frame, status_text, (10, 30), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    return frame


def process_file(args):
    """Process a video file."""
    config = load_config(args.config)
    pipeline = create_pipeline(config)
    
    # Add status overlay if requested
    if args.show_status:
        pipeline.add_frame_processor(add_status_overlay)
    
    # Set output path
    if args.output is None:
        input_path = Path(args.input)
        args.output = str(input_path.parent / f"{input_path.stem}_tracked{input_path.suffix}")
    
    print(f"Processing: {args.input}")
    print(f"Output: {args.output}")
    
    # Progress callback
    def show_progress(current, total):
        if total > 0 and current % 30 == 0:
            progress = current / total * 100
            print(f"Progress: {current}/{total} frames ({progress:.1f}%)")
    
    # Process video
    try:
        with VideoFileSource(args.input) as source:
            results = pipeline.process_video(
                source,
                output_path=args.output,
                show_preview=not args.no_preview,
                progress_callback=show_progress if not args.quiet else None
            )
    except Exception as e:
        print(f"Error processing video: {e}", file=sys.stderr)
        return 1
    
    print(f"\nProcessing complete!")
    print(f"Frames processed: {results['frames_processed']}")
    print(f"Objects tracked: {results['total_objects_tracked']}")
    
    # Generate heatmap if requested
    if args.heatmap:
        props = results['source_properties']
        heatmap = pipeline.generate_heatmap((props['height'], props['width']))
        heatmap_path = Path(args.output).with_suffix('.heatmap.png')
        cv2.imwrite(str(heatmap_path), heatmap)
        print(f"Heatmap saved: {heatmap_path}")
    
    return 0


def process_webcam(args):
    """Process webcam stream."""
    config = load_config(args.config)
    pipeline = create_pipeline(config)
    
    # Add status overlay
    pipeline.add_frame_processor(add_status_overlay)
    
    print(f"Opening camera {args.camera}...")
    print("Controls:")
    print("  q: Quit")
    print("  s: Save screenshot")
    print("  h: Save heatmap")
    print("  r: Reset tracking")
    if args.save:
        print("  Recording enabled")
    
    # Setup output
    output_path = None
    out = None
    
    try:
        with WebcamSource(args.camera) as source:
            props = source.get_properties()
            
            if args.save:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                output_dir = Path(args.output)
                output_dir.mkdir(parents=True, exist_ok=True)
                output_path = output_dir / f"webcam_{timestamp}.mp4"
                
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                out = cv2.VideoWriter(
                    str(output_path), fourcc, 
                    props['fps'], (props['width'], props['height'])
                )
                print(f"Recording to: {output_path}")
            
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
                
                # Write to output
                if out is not None:
                    out.write(vis_frame)
                
                # Show preview
                cv2.imshow('Webcam Tracking', vis_frame)
                
                # Handle keyboard
                key = cv2.waitKey(1) & 0xFF
                
                if key == ord('q'):
                    break
                elif key == ord('s'):
                    # Save screenshot
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    screenshot_path = Path(args.output) / f"screenshot_{timestamp}.png"
                    screenshot_path.parent.mkdir(parents=True, exist_ok=True)
                    cv2.imwrite(str(screenshot_path), vis_frame)
                    print(f"Screenshot saved: {screenshot_path}")
                elif key == ord('h'):
                    # Save heatmap
                    heatmap = pipeline.generate_heatmap((props['height'], props['width']))
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    heatmap_path = Path(args.output) / f"heatmap_{timestamp}.png"
                    cv2.imwrite(str(heatmap_path), heatmap)
                    print(f"Heatmap saved: {heatmap_path}")
                elif key == ord('r'):
                    # Reset tracking
                    pipeline.tracker = ObjectTracker(
                        max_disappeared=config['tracker']['max_disappeared'],
                        max_distance=config['tracker']['max_distance']
                    )
                    print("Tracking reset")
                
                frame_count += 1
                
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1
    finally:
        if out is not None:
            out.release()
        cv2.destroyAllWindows()
    
    print(f"\nTotal frames: {frame_count}")
    return 0


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='Trajector - Object tracking and trajectory visualization'
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Command to run')
    
    # File processing command
    file_parser = subparsers.add_parser('file', help='Process video file')
    file_parser.add_argument('input', help='Input video file')
    file_parser.add_argument('-o', '--output', help='Output video file')
    file_parser.add_argument('-c', '--config', default='config/config.yaml',
                            help='Configuration file')
    file_parser.add_argument('--no-preview', action='store_true',
                            help='Disable preview window')
    file_parser.add_argument('--heatmap', action='store_true',
                            help='Generate trajectory heatmap')
    file_parser.add_argument('--show-status', action='store_true',
                            help='Show status overlay')
    file_parser.add_argument('--quiet', action='store_true',
                            help='Suppress progress output')
    
    # Webcam processing command
    webcam_parser = subparsers.add_parser('webcam', help='Process webcam stream')
    webcam_parser.add_argument('-i', '--camera', type=int, default=0,
                              help='Camera index')
    webcam_parser.add_argument('-c', '--config', default='config/config.yaml',
                              help='Configuration file')
    webcam_parser.add_argument('-s', '--save', action='store_true',
                              help='Save video recording')
    webcam_parser.add_argument('-o', '--output', default='output',
                              help='Output directory')
    
    args = parser.parse_args()
    
    if args.command is None:
        parser.print_help()
        return 1
    
    if args.command == 'file':
        return process_file(args)
    elif args.command == 'webcam':
        return process_webcam(args)


if __name__ == '__main__':
    sys.exit(main())