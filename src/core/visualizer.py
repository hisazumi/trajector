import cv2
import numpy as np
from typing import List, Dict, Tuple, Optional
from collections import deque


class TrajectoryVisualizer:
    def __init__(self, 
                 trajectory_length: int = 50,
                 trajectory_color: Tuple[int, int, int] = (0, 255, 0),
                 trajectory_thickness: int = 2,
                 show_bbox: bool = True,
                 show_id: bool = True,
                 show_trajectory: bool = True):
        
        self.trajectory_length = trajectory_length
        self.trajectory_color = trajectory_color
        self.trajectory_thickness = trajectory_thickness
        self.show_bbox = show_bbox
        self.show_id = show_id
        self.show_trajectory = show_trajectory
        
        # Color palette for different objects
        self.colors = self._generate_colors(100)
        
    def _generate_colors(self, n: int) -> List[Tuple[int, int, int]]:
        colors = []
        for i in range(n):
            hue = int(180 * i / n)
            color = cv2.cvtColor(np.uint8([[[hue, 255, 255]]]), cv2.COLOR_HSV2BGR)[0][0]
            colors.append(tuple(int(c) for c in color))
        return colors
    
    def draw_frame(self, frame: np.ndarray, tracked_objects: List[Dict]) -> np.ndarray:
        vis_frame = frame.copy()
        
        for obj in tracked_objects:
            obj_id = obj['id']
            color = self.colors[obj_id % len(self.colors)]
            
            # Draw bounding box
            if self.show_bbox:
                bbox = obj['bbox'].astype(int)
                cv2.rectangle(vis_frame, 
                            (bbox[0], bbox[1]), 
                            (bbox[2], bbox[3]), 
                            color, 2)
            
            # Draw ID
            if self.show_id:
                center = obj['center']
                cv2.putText(vis_frame, 
                          f"ID: {obj_id}", 
                          (center[0] - 20, center[1] - 10),
                          cv2.FONT_HERSHEY_SIMPLEX, 
                          0.5, 
                          color, 
                          2)
            
            # Draw trajectory
            if self.show_trajectory and len(obj['trajectory']) > 1:
                trajectory = obj['trajectory'][-self.trajectory_length:]
                for i in range(1, len(trajectory)):
                    # Gradually fade older points
                    thickness = int(self.trajectory_thickness * (i / len(trajectory)))
                    cv2.line(vis_frame, 
                           trajectory[i-1], 
                           trajectory[i], 
                           color, 
                           max(1, thickness))
                
                # Draw points along trajectory
                for i, point in enumerate(trajectory[::5]):  # Every 5th point
                    radius = int(3 * (i / len(trajectory)))
                    cv2.circle(vis_frame, point, max(1, radius), color, -1)
        
        return vis_frame
    
    def create_heatmap(self, frame_shape: Tuple[int, int], 
                      all_trajectories: Dict[int, List[Tuple[int, int]]]) -> np.ndarray:
        heatmap = np.zeros((frame_shape[0], frame_shape[1]), dtype=np.float32)
        
        for track_id, trajectory in all_trajectories.items():
            for point in trajectory:
                x, y = point
                # Add gaussian blob at each point
                cv2.circle(heatmap, (x, y), 20, 1, -1)
        
        # Normalize and apply colormap
        heatmap = cv2.GaussianBlur(heatmap, (21, 21), 0)
        heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + 1e-8)
        heatmap = (heatmap * 255).astype(np.uint8)
        heatmap_colored = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
        
        return heatmap_colored