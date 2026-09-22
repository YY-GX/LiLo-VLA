"""
pose_augmentation.py

Pose augmentation utilities for training data augmentation.
Supports position and orientation shifts with configurable parameters.
"""

import numpy as np
import torch
import torch.nn as nn
from typing import Optional, Tuple, Union
import random
from prismatic.util import ensure_bfloat16


class PoseAugmentation:
    """
    Pose augmentation for training data.
    
    Applies random shifts to position and orientation components of poses.
    Position shifts are typically larger (2cm), orientation shifts are smaller (5-10°).
    """
    
    def __init__(
        self,
        position_std: float = 0.02,  # 2cm standard deviation
        orientation_std: float = 0.1,  # ~5.7 degrees standard deviation
        position_range: Tuple[float, float] = (-0.05, 0.05),  # ±5cm range
        orientation_range: Tuple[float, float] = (-0.2, 0.2),  # ±11.5 degrees range
        position_dim: int = 3,
        orientation_dim: int = 3,
        enabled: bool = True,
        random_seed: Optional[int] = None,
    ):
        """
        Initialize pose augmentation.
        
        Args:
            position_std: Standard deviation for position shifts (meters)
            orientation_std: Standard deviation for orientation shifts (radians)
            position_range: Min/max range for position shifts (meters)
            orientation_range: Min/max range for orientation shifts (radians)
            position_dim: Number of position dimensions (default: 3 for x,y,z)
            orientation_dim: Number of orientation dimensions (default: 3 for roll,pitch,yaw)
            enabled: Whether augmentation is enabled
            random_seed: Random seed for reproducibility
        """
        self.position_std = position_std
        self.orientation_std = orientation_std
        self.position_range = position_range
        self.orientation_range = orientation_range
        self.position_dim = position_dim
        self.orientation_dim = orientation_dim
        self.enabled = enabled
        
        if random_seed is not None:
            random.seed(random_seed)
            np.random.seed(random_seed)
            torch.manual_seed(random_seed)
    
    def __call__(self, pose: Union[torch.Tensor, np.ndarray]) -> Union[torch.Tensor, np.ndarray]:
        """
        Apply pose augmentation.
        
        Args:
            pose: Input pose tensor/array of shape (pose_dim,) or (batch_size, pose_dim)
                 Expected format: [x, y, z, roll, pitch, yaw] for 6D poses
        
        Returns:
            Augmented pose with same shape and type as input
        """
        if not self.enabled:
            return pose
        
        is_torch = isinstance(pose, torch.Tensor)
        is_batch = len(pose.shape) > 1
        
        if is_torch:
            device = pose.device
            dtype = pose.dtype
            # Convert bfloat16 to float32 for numpy compatibility
            if pose.dtype == torch.bfloat16:
                pose_np = pose.detach().cpu().to(torch.float32).numpy()
            else:
                pose_np = pose.detach().cpu().numpy()
        else:
            pose_np = pose.copy()
        
        if is_batch:
            # Handle batch dimension
            batch_size = pose_np.shape[0]
            augmented_poses = []
            
            for i in range(batch_size):
                augmented_pose = self._augment_single_pose(pose_np[i])
                augmented_poses.append(augmented_pose)
            
            result = np.stack(augmented_poses)
        else:
            # Handle single pose
            result = self._augment_single_pose(pose_np)
        
        if is_torch:
            result = torch.tensor(result, device=device, dtype=dtype)
            # Ensure bfloat16 dtype is preserved
            if dtype == torch.bfloat16:
                result = ensure_bfloat16(result)
        
        return result
    
    def _augment_single_pose(self, pose: np.ndarray) -> np.ndarray:
        """
        Augment a single pose.
        
        Args:
            pose: Single pose array of shape (pose_dim,)
        
        Returns:
            Augmented pose array
        """
        pose_dim = len(pose)
        expected_dim = self.position_dim + self.orientation_dim
        
        if pose_dim != expected_dim:
            raise ValueError(f"Expected pose dimension {expected_dim}, got {pose_dim}")
        
        # Split pose into position and orientation
        position = pose[:self.position_dim]
        orientation = pose[self.position_dim:]
        
        # Apply position augmentation
        position_shift = np.random.normal(0, self.position_std, size=self.position_dim)
        position_shift = np.clip(position_shift, self.position_range[0], self.position_range[1])
        augmented_position = position + position_shift
        
        # Apply orientation augmentation
        orientation_shift = np.random.normal(0, self.orientation_std, size=self.orientation_dim)
        orientation_shift = np.clip(orientation_shift, self.orientation_range[0], self.orientation_range[1])
        augmented_orientation = orientation + orientation_shift
        
        # Normalize orientation to [-π, π] for roll, pitch, yaw
        augmented_orientation = np.mod(augmented_orientation + np.pi, 2 * np.pi) - np.pi
        
        # Combine position and orientation
        augmented_pose = np.concatenate([augmented_position, augmented_orientation])
        
        return augmented_pose
    
    def set_enabled(self, enabled: bool):
        """Enable or disable augmentation."""
        self.enabled = enabled
    
    def get_augmentation_stats(self, num_samples: int = 1000) -> dict:
        """
        Get statistics about the augmentation.
        
        Args:
            num_samples: Number of samples to generate for statistics
        
        Returns:
            Dictionary with augmentation statistics
        """
        if not self.enabled:
            return {"enabled": False}
        
        # Generate random poses for testing
        test_poses = np.random.randn(num_samples, self.position_dim + self.orientation_dim)
        
        # Apply augmentation
        augmented_poses = []
        for pose in test_poses:
            augmented_pose = self._augment_single_pose(pose)
            augmented_poses.append(augmented_pose)
        
        augmented_poses = np.array(augmented_poses)
        
        # Calculate statistics
        position_shifts = augmented_poses[:, :self.position_dim] - test_poses[:, :self.position_dim]
        orientation_shifts = augmented_poses[:, self.position_dim:] - test_poses[:, self.position_dim:]
        
        stats = {
            "enabled": self.enabled,
            "position_std": self.position_std,
            "orientation_std": self.orientation_std,
            "position_shift_mean": np.mean(position_shifts, axis=0).tolist(),
            "position_shift_std": np.std(position_shifts, axis=0).tolist(),
            "position_shift_range": [np.min(position_shifts), np.max(position_shifts)],
            "orientation_shift_mean": np.mean(orientation_shifts, axis=0).tolist(),
            "orientation_shift_std": np.std(orientation_shifts, axis=0).tolist(),
            "orientation_shift_range": [np.min(orientation_shifts), np.max(orientation_shifts)],
        }
        
        return stats


class QuaternionPoseAugmentation(PoseAugmentation):
    """
    Pose augmentation for quaternion-based orientations.
    
    Handles quaternion normalization and proper rotation composition.
    """
    
    def __init__(
        self,
        position_std: float = 0.02,
        orientation_std: float = 0.1,
        position_range: Tuple[float, float] = (-0.05, 0.05),
        orientation_range: Tuple[float, float] = (-0.2, 0.2),
        position_dim: int = 3,
        orientation_dim: int = 4,  # 4 for quaternion
        enabled: bool = True,
        random_seed: Optional[int] = None,
    ):
        """
        Initialize quaternion pose augmentation.
        
        Args:
            position_std: Standard deviation for position shifts (meters)
            orientation_std: Standard deviation for orientation shifts (radians)
            position_range: Min/max range for position shifts (meters)
            orientation_range: Min/max range for orientation shifts (radians)
            position_dim: Number of position dimensions (default: 3 for x,y,z)
            orientation_dim: Number of orientation dimensions (default: 4 for quaternion)
            enabled: Whether augmentation is enabled
            random_seed: Random seed for reproducibility
        """
        super().__init__(
            position_std=position_std,
            orientation_std=orientation_std,
            position_range=position_range,
            orientation_range=orientation_range,
            position_dim=position_dim,
            orientation_dim=orientation_dim,
            enabled=enabled,
            random_seed=random_seed,
        )
        
        if orientation_dim != 4:
            raise ValueError("QuaternionPoseAugmentation requires orientation_dim=4")
    
    def _augment_single_pose(self, pose: np.ndarray) -> np.ndarray:
        """
        Augment a single pose with quaternion orientation.
        
        Args:
            pose: Single pose array of shape (pose_dim,)
                 Expected format: [x, y, z, qw, qx, qy, qz] for 7D poses
        
        Returns:
            Augmented pose array
        """
        pose_dim = len(pose)
        expected_dim = self.position_dim + self.orientation_dim
        
        if pose_dim != expected_dim:
            raise ValueError(f"Expected pose dimension {expected_dim}, got {pose_dim}")
        
        # Split pose into position and orientation
        position = pose[:self.position_dim]
        quaternion = pose[self.position_dim:]  # [qw, qx, qy, qz]
        
        # Apply position augmentation
        position_shift = np.random.normal(0, self.position_std, size=self.position_dim)
        position_shift = np.clip(position_shift, self.position_range[0], self.position_range[1])
        augmented_position = position + position_shift
        
        # Apply orientation augmentation (small rotation around random axis)
        angle = np.random.normal(0, self.orientation_std)
        angle = np.clip(angle, self.orientation_range[0], self.orientation_range[1])
        
        # Create small rotation quaternion
        axis = np.random.randn(3)
        axis = axis / np.linalg.norm(axis)
        
        sin_half_angle = np.sin(angle / 2)
        cos_half_angle = np.cos(angle / 2)
        
        delta_quat = np.array([
            cos_half_angle,
            axis[0] * sin_half_angle,
            axis[1] * sin_half_angle,
            axis[2] * sin_half_angle
        ])
        
        # Compose rotations (quaternion multiplication)
        augmented_quaternion = self._quaternion_multiply(quaternion, delta_quat)
        
        # Normalize quaternion
        augmented_quaternion = augmented_quaternion / np.linalg.norm(augmented_quaternion)
        
        # Combine position and orientation
        augmented_pose = np.concatenate([augmented_position, augmented_quaternion])
        
        return augmented_pose
    
    def _quaternion_multiply(self, q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
        """
        Multiply two quaternions.
        
        Args:
            q1: First quaternion [qw, qx, qy, qz]
            q2: Second quaternion [qw, qx, qy, qz]
        
        Returns:
            Product quaternion [qw, qx, qy, qz]
        """
        w1, x1, y1, z1 = q1
        w2, x2, y2, z2 = q2
        
        w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
        x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
        y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
        z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2
        
        return np.array([w, x, y, z])


def create_pose_augmentation(
    augmentation_type: str = "euler",
    position_std: float = 0.02,
    orientation_std: float = 0.1,
    enabled: bool = True,
    **kwargs
) -> PoseAugmentation:
    """
    Factory function to create pose augmentation.
    
    Args:
        augmentation_type: Type of augmentation ("euler" or "quaternion")
        position_std: Standard deviation for position shifts
        orientation_std: Standard deviation for orientation shifts
        enabled: Whether augmentation is enabled
        **kwargs: Additional arguments passed to augmentation class
    
    Returns:
        PoseAugmentation instance
    """
    if augmentation_type == "euler":
        return PoseAugmentation(
            position_std=position_std,
            orientation_std=orientation_std,
            enabled=enabled,
            **kwargs
        )
    elif augmentation_type == "quaternion":
        return QuaternionPoseAugmentation(
            position_std=position_std,
            orientation_std=orientation_std,
            enabled=enabled,
            **kwargs
        )
    else:
        raise ValueError(f"Unknown augmentation type: {augmentation_type}") 