# Trajector 🎯

Real-time object tracking and trajectory visualization system using YOLOv8 and OpenCV.

## Features

- **Object Detection & Tracking**: Real-time object detection using YOLOv8 with persistent ID tracking
- **Trajectory Visualization**: Display object movement paths with customizable trail effects
- **Heatmap Generation**: Create heatmaps to visualize object presence and movement patterns
- **Multiple Input Sources**: Support for video files and webcam streams
- **Web Interface**: Interactive Streamlit-based web application
- **CLI Tool**: Command-line interface for batch processing

## Installation

### Prerequisites

- Python 3.12+
- [uv](https://github.com/astral-sh/uv) (Python package manager)

### Setup

```bash
# Clone the repository
git clone https://github.com/hisazumi/trajector.git
cd trajector

# Install dependencies using uv
uv sync

# Download YOLOv8 model (automatically downloaded on first run)
# Or manually download to project root:
wget https://github.com/ultralytics/assets/releases/download/v8.2.0/yolov8n.pt
```

## Usage

### Web Interface (Streamlit)

Launch the interactive web application:

```bash
uv run streamlit run src/web_app.py
```

Features:
- **Video Upload**: Process video files with progress tracking
- **Webcam Tracking**: Real-time object tracking with recording capability
- **Interactive Controls**: Adjust tracking parameters in real-time
- **Download Results**: Export processed videos and heatmaps

### Command Line Interface

#### Process Video File

```bash
# Basic usage
uv run python -m src.cli file input_video.mp4

# With options
uv run python -m src.cli file input_video.mp4 \
  --output output_video.mp4 \
  --config config/config.yaml \
  --heatmap \
  --show-status
```

Options:
- `-o, --output`: Output video file path
- `-c, --config`: Configuration file (default: config/config.yaml)
- `--heatmap`: Generate trajectory heatmap
- `--show-status`: Show status overlay on video
- `--no-preview`: Disable preview window
- `--quiet`: Suppress progress output

#### Webcam Tracking

```bash
# Start webcam tracking
uv run python -m src.cli webcam

# With recording
uv run python -m src.cli webcam --save --output output/
```

Keyboard shortcuts:
- `q`: Quit
- `s`: Save screenshot
- `h`: Save heatmap
- `r`: Reset tracking
- `t`: Toggle dynamic heatmap overlay
- `+/-`: Increase/decrease heatmap opacity

## Configuration

Edit `config/config.yaml` to customize:

```yaml
detector:
  model_path: "yolov8n.pt"  # YOLOv8 model file
  device: "cpu"              # cpu or cuda
  confidence: 0.5            # Detection confidence threshold

tracker:
  max_disappeared: 30        # Frames before removing lost object
  max_distance: 50          # Maximum distance for object association

visualizer:
  trajectory_length: 30      # Number of points in trajectory trail
  trajectory_color: [0, 255, 0]  # Trail color (BGR)
  trajectory_thickness: 2    # Trail line thickness
  show_bbox: true           # Show bounding boxes
  show_id: true             # Show object IDs
  show_trajectory: true     # Show trajectory trails
  show_heatmap: false       # Dynamic heatmap overlay
  heatmap_alpha: 0.6        # Heatmap transparency
```

## Project Structure

```
trajector/
├── src/
│   ├── cli.py              # Command-line interface
│   ├── web_app.py          # Streamlit web application
│   ├── core/               # Core tracking components
│   │   ├── detector.py     # YOLO detector wrapper
│   │   ├── tracker.py      # Object tracking logic
│   │   └── visualizer.py   # Visualization utilities
│   ├── sources/            # Input sources
│   │   ├── video.py        # Video file handler
│   │   └── webcam.py       # Webcam handler
│   └── processors/         # Processing pipeline
│       └── pipeline.py     # Main processing pipeline
├── config/
│   └── config.yaml         # Configuration file
├── tests/                  # Unit and integration tests
├── examples/               # Example videos and outputs
└── output/                 # Output directory
```

## Examples

### Process a Video

```bash
# Process with default settings
uv run python -m src.cli file examples/sample_people.mp4

# Generate heatmap and status overlay
uv run python -m src.cli file examples/sample_people.mp4 \
  --heatmap --show-status
```

### Launch Web Interface

```bash
uv run streamlit run src/web_app.py
```

Then open http://localhost:8501 in your browser.

## Testing

Run the test suite:

```bash
# Run all tests
uv run pytest

# With coverage
uv run pytest --cov=src --cov-report=html
```

## Requirements

Main dependencies:
- `opencv-python`: Computer vision operations
- `ultralytics`: YOLOv8 object detection
- `numpy`: Numerical computations
- `streamlit`: Web interface
- `pyyaml`: Configuration management

See `pyproject.toml` for complete dependency list.

## License

MIT License

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## Acknowledgments

- [YOLOv8](https://github.com/ultralytics/ultralytics) for object detection
- [OpenCV](https://opencv.org/) for computer vision operations
- [Streamlit](https://streamlit.io/) for the web interface