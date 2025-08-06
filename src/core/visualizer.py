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
                 show_trajectory: bool = True,
                 show_heatmap: bool = False,
                 heatmap_alpha: float = 0.6,
                 heatmap_update_interval: int = 10):
        
        self.trajectory_length = trajectory_length
        self.trajectory_color = trajectory_color
        self.trajectory_thickness = trajectory_thickness
        self.show_bbox = show_bbox
        self.show_id = show_id
        self.show_trajectory = show_trajectory
        self.show_heatmap = show_heatmap
        self.heatmap_alpha = heatmap_alpha
        self.heatmap_update_interval = heatmap_update_interval
        
        # Color palette for different objects
        self.colors = self._generate_colors(100)
        
        # Heatmap cache
        self._heatmap_cache = None
        self._frame_count = 0
        
    def _generate_colors(self, n: int) -> List[Tuple[int, int, int]]:
        colors = []
        for i in range(n):
            hue = int(180 * i / n)
            color = cv2.cvtColor(np.uint8([[[hue, 255, 255]]]), cv2.COLOR_HSV2BGR)[0][0]
            colors.append(tuple(int(c) for c in color))
        return colors
    
    def draw_frame(self, frame: np.ndarray, tracked_objects: List[Dict], 
                  all_trajectories: Optional[Dict[int, List[Tuple[int, int]]]] = None) -> np.ndarray:
        vis_frame = frame.copy()
        
        # Draw dynamic heatmap overlay if enabled
        if self.show_heatmap and all_trajectories:
            self._frame_count += 1
            
            # Update heatmap cache periodically or if cache is empty
            if (self._heatmap_cache is None or 
                self._frame_count % self.heatmap_update_interval == 0):
                self._heatmap_cache = self.create_heatmap(frame.shape[:2], all_trajectories)
            
            # Overlay heatmap on frame
            if self._heatmap_cache is not None:
                # Resize heatmap to match frame if needed
                if self._heatmap_cache.shape[:2] != frame.shape[:2]:
                    self._heatmap_cache = cv2.resize(self._heatmap_cache, 
                                                   (frame.shape[1], frame.shape[0]))
                
                # Blend heatmap with frame
                vis_frame = cv2.addWeighted(vis_frame, 1 - self.heatmap_alpha, 
                                          self._heatmap_cache, self.heatmap_alpha, 0)
        
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
    
    def toggle_heatmap(self):
        """Toggle heatmap display on/off."""
        self.show_heatmap = not self.show_heatmap
        if not self.show_heatmap:
            self._heatmap_cache = None
    
    def set_heatmap_alpha(self, alpha: float):
        """Set heatmap transparency (0.0 = transparent, 1.0 = opaque)."""
        self.heatmap_alpha = max(0.0, min(1.0, alpha))
    
    def reset_heatmap_cache(self):
        """Reset heatmap cache to force regeneration."""
        self._heatmap_cache = None
        self._frame_count = 0
    
    def create_realtime_heatmap(self, frame_shape: Tuple[int, int], 
                               current_trajectories: Dict[int, List[Tuple[int, int]]],
                               decay_factor: float = 0.95) -> np.ndarray:
        """Create a real-time updating heatmap with decay."""
        current_heatmap = np.zeros((frame_shape[0], frame_shape[1]), dtype=np.float32)
        
        # Add current trajectory points
        for track_id, trajectory in current_trajectories.items():
            if len(trajectory) > 0:
                # Weight recent points more heavily
                for i, point in enumerate(trajectory):
                    x, y = point
                    if 0 <= x < frame_shape[1] and 0 <= y < frame_shape[0]:
                        # More recent points get higher weight
                        weight = (i + 1) / len(trajectory)
                        cv2.circle(current_heatmap, (x, y), 15, weight, -1)
        
        # Apply decay to existing heatmap and add new data
        if self._heatmap_cache is not None:
            # Ensure cache is same size as current heatmap
            if self._heatmap_cache.shape[:2] != current_heatmap.shape:
                self._heatmap_cache = cv2.resize(self._heatmap_cache, 
                                               (current_heatmap.shape[1], current_heatmap.shape[0]))
                # Convert back to grayscale if it was colorized
                if len(self._heatmap_cache.shape) == 3:
                    self._heatmap_cache = cv2.cvtColor(self._heatmap_cache, cv2.COLOR_BGR2GRAY)
                    self._heatmap_cache = self._heatmap_cache.astype(np.float32) / 255.0
            
            # Apply decay and combine
            decayed_cache = self._heatmap_cache.astype(np.float32) * decay_factor
            current_heatmap = decayed_cache + current_heatmap
        
        # Blur and normalize
        current_heatmap = cv2.GaussianBlur(current_heatmap, (21, 21), 0)
        if current_heatmap.max() > 0:
            current_heatmap = current_heatmap / current_heatmap.max()
        
        # Store as cache for next frame
        self._heatmap_cache = current_heatmap.copy()
        
        # Convert to color image
        heatmap_uint8 = (current_heatmap * 255).astype(np.uint8)
        heatmap_colored = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
        
        return heatmap_colored