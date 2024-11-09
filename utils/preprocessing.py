"""
LIDAR Preprocessing Utilities for BEV Representation

This module provides functions for converting LIDAR point clouds to Bird's Eye View (BEV)
representations suitable for semantic segmentation tasks in autonomous vehicle perception.

Author: Graduate Student, AI/ML Program, San Jose State University
Date: October 2024
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Union
import cv2
from scipy.spatial.distance import cdist


class LidarBEVProcessor:
    """
    LIDAR to Bird's Eye View (BEV) processor
    
    Converts 3D LIDAR point clouds to 2D BEV representations with
    multi-channel encoding including height statistics, intensity, and density.
    """
    
    def __init__(self, bev_params: Dict):
        """
        Initialize BEV processor with parameters
        
        Args:
            bev_params: Dictionary containing BEV configuration
                - x_range: [min_x, max_x] in meters
                - y_range: [min_y, max_y] in meters  
                - z_range: [min_z, max_z] in meters
                - resolution: Grid resolution in meters per pixel
                - height: BEV image height in pixels
                - width: BEV image width in pixels
                - channels: Number of output channels
        """
        self.params = bev_params
        
        # Extract ranges
        self.x_range = bev_params['x_range']
        self.y_range = bev_params['y_range']
        self.z_range = bev_params['z_range']
        
        # Grid parameters
        self.resolution = bev_params['resolution']
        self.height = bev_params['height']
        self.width = bev_params['width']
        self.channels = bev_params['channels']
        
        # Precompute grid coordinates
        self._setup_grid()
        
    def _setup_grid(self):
        """Setup BEV grid coordinate system"""
        # Calculate actual ranges based on resolution and image size
        self.x_step = (self.x_range[1] - self.x_range[0]) / self.width
        self.y_step = (self.y_range[1] - self.y_range[0]) / self.height
        
        # Grid bounds
        self.x_min, self.x_max = self.x_range
        self.y_min, self.y_max = self.y_range
        self.z_min, self.z_max = self.z_range
        
        print(f"BEV Grid Setup:")
        print(f"  X range: {self.x_range} m, step: {self.x_step:.3f} m/pixel")
        print(f"  Y range: {self.y_range} m, step: {self.y_step:.3f} m/pixel")
        print(f"  Z range: {self.z_range} m")
        print(f"  Output size: {self.height}x{self.width} with {self.channels} channels")
    
    def process_point_cloud(
        self, 
        points: np.ndarray, 
        num_sweeps: int = 1,
        sweep_timestamps: Optional[List[float]] = None
    ) -> np.ndarray:
        """
        Convert point cloud to BEV representation
        
        Args:
            points: Point cloud array [N, 4] with [x, y, z, intensity]
            num_sweeps: Number of sweeps to accumulate (for temporal enhancement)
            sweep_timestamps: Timestamps for each sweep
            
        Returns:
            BEV tensor [channels, height, width]
        """
        
        if len(points.shape) != 2 or points.shape[1] < 4:
            raise ValueError(f"Expected point cloud shape [N, 4], got {points.shape}")
        
        # Filter points within BEV range
        valid_points = self._filter_points_in_range(points)
        
        if len(valid_points) == 0:
            print("Warning: No valid points in BEV range")
            return np.zeros((self.channels, self.height, self.width), dtype=np.float32)
        
        # Convert to pixel coordinates
        pixel_coords = self._world_to_pixel(valid_points)
        
        # Create BEV channels
        bev_tensor = self._create_bev_channels(valid_points, pixel_coords)
        
        # Apply post-processing
        bev_tensor = self._post_process_bev(bev_tensor)
        
        return bev_tensor
    
    def _filter_points_in_range(self, points: np.ndarray) -> np.ndarray:
        """Filter points within BEV spatial and height ranges"""
        
        # Spatial filtering
        x_mask = (points[:, 0] >= self.x_min) & (points[:, 0] <= self.x_max)
        y_mask = (points[:, 1] >= self.y_min) & (points[:, 1] <= self.y_max)
        z_mask = (points[:, 2] >= self.z_min) & (points[:, 2] <= self.z_max)
        
        valid_mask = x_mask & y_mask & z_mask
        
        return points[valid_mask]
    
    def _world_to_pixel(self, points: np.ndarray) -> np.ndarray:
        """Convert world coordinates to pixel coordinates"""
        
        # Convert to pixel coordinates
        pixel_x = ((points[:, 0] - self.x_min) / self.x_step).astype(np.int32)
        pixel_y = ((points[:, 1] - self.y_min) / self.y_step).astype(np.int32)
        
        # Clamp to valid range
        pixel_x = np.clip(pixel_x, 0, self.width - 1)
        pixel_y = np.clip(pixel_y, 0, self.height - 1)
        
        return np.column_stack([pixel_x, pixel_y])
    
    def _create_bev_channels(
        self, 
        points: np.ndarray, 
        pixel_coords: np.ndarray
    ) -> np.ndarray:
        """Create multi-channel BEV representation"""
        
        bev_tensor = np.zeros((self.channels, self.height, self.width), dtype=np.float32)
        
        if len(points) == 0:
            return bev_tensor
        
        # Extract coordinates and features
        x_pixels = pixel_coords[:, 0]
        y_pixels = pixel_coords[:, 1]
        heights = points[:, 2]
        intensities = points[:, 3] if points.shape[1] > 3 else np.ones(len(points))
        
        # Channel 0: Maximum height
        self._fill_channel_max(bev_tensor[0], x_pixels, y_pixels, heights)
        
        # Channel 1: Mean height
        self._fill_channel_mean(bev_tensor[1], x_pixels, y_pixels, heights)
        
        # Channel 2: Maximum intensity
        if self.channels > 2:
            self._fill_channel_max(bev_tensor[2], x_pixels, y_pixels, intensities)
        
        # Channel 3: Point density
        if self.channels > 3:
            self._fill_channel_density(bev_tensor[3], x_pixels, y_pixels)
        
        return bev_tensor
    
    def _fill_channel_max(
        self, 
        channel: np.ndarray, 
        x_pixels: np.ndarray, 
        y_pixels: np.ndarray, 
        values: np.ndarray
    ):
        """Fill channel with maximum values per pixel"""
        
        # Use numpy's maximum reduction
        for i in range(len(x_pixels)):
            x, y = x_pixels[i], y_pixels[i]
            channel[y, x] = max(channel[y, x], values[i])
    
    def _fill_channel_mean(
        self, 
        channel: np.ndarray, 
        x_pixels: np.ndarray, 
        y_pixels: np.ndarray, 
        values: np.ndarray
    ):
        """Fill channel with mean values per pixel"""
        
        # Accumulate values and counts
        value_sum = np.zeros_like(channel)
        counts = np.zeros_like(channel)
        
        for i in range(len(x_pixels)):
            x, y = x_pixels[i], y_pixels[i]
            value_sum[y, x] += values[i]
            counts[y, x] += 1
        
        # Calculate mean where counts > 0
        valid_mask = counts > 0
        channel[valid_mask] = value_sum[valid_mask] / counts[valid_mask]
    
    def _fill_channel_density(
        self, 
        channel: np.ndarray, 
        x_pixels: np.ndarray, 
        y_pixels: np.ndarray
    ):
        """Fill channel with point density per pixel"""
        
        # Count points per pixel
        for i in range(len(x_pixels)):
            x, y = x_pixels[i], y_pixels[i]
            channel[y, x] += 1
        
        # Normalize density (optional)
        max_density = np.max(channel)
        if max_density > 0:
            channel /= max_density
    
    def _post_process_bev(self, bev_tensor: np.ndarray) -> np.ndarray:
        """Apply post-processing to BEV tensor"""
        
        # Normalize height channels
        if self.channels >= 2:
            # Normalize height values to [0, 1] range
            height_range = self.z_max - self.z_min
            bev_tensor[0] = (bev_tensor[0] - self.z_min) / height_range
            bev_tensor[1] = (bev_tensor[1] - self.z_min) / height_range
        
        # Clip intensity values
        if self.channels > 2:
            bev_tensor[2] = np.clip(bev_tensor[2], 0, 1)
        
        # Apply Gaussian smoothing to reduce noise
        for i in range(self.channels):
            if np.max(bev_tensor[i]) > 0:  # Only smooth non-empty channels
                bev_tensor[i] = cv2.GaussianBlur(
                    bev_tensor[i], 
                    (3, 3), 
                    sigmaX=0.5, 
                    sigmaY=0.5
                )
        
        return bev_tensor


def accumulate_multi_sweep_points(
    point_clouds: List[np.ndarray],
    ego_poses: List[np.ndarray],
    timestamps: List[float],
    reference_timestamp: float
) -> np.ndarray:
    """
    Accumulate multiple LIDAR sweeps into a single point cloud
    
    Args:
        point_clouds: List of point clouds [N_i, 4]
        ego_poses: List of ego vehicle poses (4x4 transformation matrices)
        timestamps: List of timestamps for each sweep
        reference_timestamp: Reference timestamp for compensation
        
    Returns:
        Accumulated point cloud [N_total, 4]
    """
    
    if len(point_clouds) != len(ego_poses) or len(point_clouds) != len(timestamps):
        raise ValueError("Mismatch in number of point clouds, poses, and timestamps")
    
    accumulated_points = []
    
    for points, pose, timestamp in zip(point_clouds, ego_poses, timestamps):
        if len(points) == 0:
            continue
            
        # Transform points to reference frame
        transformed_points = transform_points_to_reference(
            points, pose, timestamp, reference_timestamp
        )
        
        accumulated_points.append(transformed_points)
    
    if not accumulated_points:
        return np.empty((0, 4))
    
    return np.vstack(accumulated_points)


def transform_points_to_reference(
    points: np.ndarray,
    ego_pose: np.ndarray,
    timestamp: float,
    reference_timestamp: float
) -> np.ndarray:
    """
    Transform points from sensor frame to reference frame
    
    Args:
        points: Point cloud [N, 4] with [x, y, z, intensity]
        ego_pose: Ego vehicle pose (4x4 transformation matrix)
        timestamp: Timestamp of the point cloud
        reference_timestamp: Reference timestamp
        
    Returns:
        Transformed points [N, 4]
    """
    
    if len(points) == 0:
        return points
    
    # Extract spatial coordinates
    xyz = points[:, :3]
    intensities = points[:, 3:]
    
    # Convert to homogeneous coordinates
    ones = np.ones((len(xyz), 1))
    xyz_homogeneous = np.hstack([xyz, ones])
    
    # Apply transformation
    transformed_xyz = (ego_pose @ xyz_homogeneous.T).T
    
    # Remove homogeneous coordinate
    transformed_xyz = transformed_xyz[:, :3]
    
    # Combine with intensities
    transformed_points = np.hstack([transformed_xyz, intensities])
    
    return transformed_points


def apply_ground_segmentation(
    points: np.ndarray,
    ground_height_threshold: float = -1.5,
    ransac_iterations: int = 100,
    distance_threshold: float = 0.1
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Separate ground and non-ground points using RANSAC plane fitting
    
    Args:
        points: Point cloud [N, 4] with [x, y, z, intensity]
        ground_height_threshold: Maximum height for ground points
        ransac_iterations: Number of RANSAC iterations
        distance_threshold: Distance threshold for inlier detection
        
    Returns:
        Tuple of (ground_points, non_ground_points)
    """
    
    if len(points) < 3:
        return np.empty((0, 4)), points
    
    # Filter points by height for initial ground candidates
    height_mask = points[:, 2] <= ground_height_threshold
    ground_candidates = points[height_mask]
    
    if len(ground_candidates) < 3:
        return np.empty((0, 4)), points
    
    # RANSAC plane fitting
    best_inliers = []
    best_plane = None
    
    for _ in range(ransac_iterations):
        # Sample 3 random points
        sample_indices = np.random.choice(len(ground_candidates), 3, replace=False)
        sample_points = ground_candidates[sample_indices, :3]
        
        # Fit plane through sample points
        plane = fit_plane(sample_points)
        
        if plane is None:
            continue
        
        # Calculate distances to plane
        distances = calculate_point_to_plane_distance(ground_candidates[:, :3], plane)
        
        # Find inliers
        inlier_mask = distances <= distance_threshold
        inliers = np.where(inlier_mask)[0]
        
        # Update best plane if more inliers found
        if len(inliers) > len(best_inliers):
            best_inliers = inliers
            best_plane = plane
    
    if best_plane is None or len(best_inliers) == 0:
        return np.empty((0, 4)), points
    
    # Separate ground and non-ground points
    ground_points = ground_candidates[best_inliers]
    
    # Create mask for all points
    all_ground_mask = np.zeros(len(points), dtype=bool)
    all_ground_mask[height_mask] = False
    all_ground_mask[np.where(height_mask)[0][best_inliers]] = True
    
    non_ground_points = points[~all_ground_mask]
    
    return ground_points, non_ground_points


def fit_plane(points: np.ndarray) -> Optional[np.ndarray]:
    """
    Fit plane to 3D points using least squares
    
    Args:
        points: Points [3, 3] for plane fitting
        
    Returns:
        Plane coefficients [a, b, c, d] for ax + by + cz + d = 0
    """
    
    if len(points) != 3:
        return None
    
    # Calculate two vectors in the plane
    v1 = points[1] - points[0]
    v2 = points[2] - points[0]
    
    # Calculate normal vector
    normal = np.cross(v1, v2)
    
    if np.linalg.norm(normal) < 1e-8:
        return None  # Degenerate case
    
    normal = normal / np.linalg.norm(normal)
    
    # Calculate plane equation: ax + by + cz + d = 0
    a, b, c = normal
    d = -np.dot(normal, points[0])
    
    return np.array([a, b, c, d])


def calculate_point_to_plane_distance(
    points: np.ndarray, 
    plane: np.ndarray
) -> np.ndarray:
    """
    Calculate distances from points to plane
    
    Args:
        points: Points [N, 3]
        plane: Plane coefficients [a, b, c, d]
        
    Returns:
        Distances [N]
    """
    
    a, b, c, d = plane
    distances = np.abs(a * points[:, 0] + b * points[:, 1] + c * points[:, 2] + d)
    distances /= np.sqrt(a**2 + b**2 + c**2)
    
    return distances


def visualize_bev_tensor(
    bev_tensor: np.ndarray, 
    save_path: Optional[str] = None,
    channel_names: Optional[List[str]] = None
) -> np.ndarray:
    """
    Visualize BEV tensor channels
    
    Args:
        bev_tensor: BEV tensor [channels, height, width]
        save_path: Path to save visualization
        channel_names: Names for each channel
        
    Returns:
        Visualization image
    """
    
    channels, height, width = bev_tensor.shape
    
    # Default channel names
    if channel_names is None:
        channel_names = [f'Channel {i}' for i in range(channels)]
    
    # Create visualization grid
    cols = min(channels, 4)
    rows = (channels + cols - 1) // cols
    
    fig_height = rows * height
    fig_width = cols * width
    
    visualization = np.zeros((fig_height, fig_width), dtype=np.float32)
    
    for i in range(channels):
        row = i // cols
        col = i % cols
        
        # Normalize channel to [0, 1]
        channel_data = bev_tensor[i]
        if np.max(channel_data) > np.min(channel_data):
            channel_normalized = (channel_data - np.min(channel_data))
            channel_normalized /= (np.max(channel_data) - np.min(channel_data))
        else:
            channel_normalized = channel_data
        
        # Place in visualization grid
        y_start = row * height
        y_end = (row + 1) * height
        x_start = col * width  
        x_end = (col + 1) * width
        
        visualization[y_start:y_end, x_start:x_end] = channel_normalized
    
    # Convert to 8-bit for saving
    visualization_8bit = (visualization * 255).astype(np.uint8)
    
    if save_path:
        cv2.imwrite(save_path, visualization_8bit)
    
    return visualization_8bit


# Example usage and testing
if __name__ == "__main__":
    # Test BEV processor
    bev_params = {
        'x_range': [-50.0, 50.0],
        'y_range': [-50.0, 50.0], 
        'z_range': [-3.0, 5.0],
        'resolution': 0.1,
        'height': 512,
        'width': 512,
        'channels': 4
    }
    
    processor = LidarBEVProcessor(bev_params)
    
    # Generate test point cloud
    n_points = 10000
    test_points = np.random.randn(n_points, 4).astype(np.float32)
    test_points[:, :3] *= 30  # Scale spatial coordinates
    test_points[:, 3] = np.random.rand(n_points)  # Intensity [0, 1]
    
    print(f"Test point cloud shape: {test_points.shape}")
    print(f"Point cloud range: X: [{np.min(test_points[:, 0]):.1f}, {np.max(test_points[:, 0]):.1f}]")
    print(f"                   Y: [{np.min(test_points[:, 1]):.1f}, {np.max(test_points[:, 1]):.1f}]")
    print(f"                   Z: [{np.min(test_points[:, 2]):.1f}, {np.max(test_points[:, 2]):.1f}]")
    
    # Process to BEV
    bev_tensor = processor.process_point_cloud(test_points)
    
    print(f"BEV tensor shape: {bev_tensor.shape}")
    print(f"BEV tensor range: [{np.min(bev_tensor):.3f}, {np.max(bev_tensor):.3f}]")
    
    # Test visualization
    vis_image = visualize_bev_tensor(
        bev_tensor, 
        channel_names=['Height Max', 'Height Mean', 'Intensity Max', 'Density']
    )
    
    print(f"Visualization image shape: {vis_image.shape}")
    print("BEV preprocessing test completed successfully!") 