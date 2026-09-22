"""
pose_heads.py

Implementations of pose prediction heads, following original action head patterns.
Supports GMM (Gaussian Mixture Model) for multi-modal pose distributions.
"""

import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional
from prismatic.util import ensure_bfloat16

from prismatic.models.action_heads import MLPResNet


class GMMPoseHead(nn.Module):
    """
    Gaussian Mixture Model pose head for predicting multi-modal pose distributions.
    Tokenwise: applies a shared MLP to each pose token's hidden state.
    """
    def __init__(
        self,
        input_dim: int = 4096,
        hidden_dim: int = 4096,
        pose_dim: int = 6,  # 3D position + 3D orientation
        num_components: int = 6,  # Changed from 3 to 6
        min_std: float = 0.01,
        max_std: float = 1.0,
        min_epsilon: float = 1e-2,  # Minimum variance to prevent collapse
        entropy_weight: float = 0.1,  # Weight for entropy regularization
    ):
        """
        Initialize GMM pose head.
        
        Args:
            input_dim: Input dimension from LLM hidden states
            hidden_dim: Hidden dimension for MLP
            pose_dim: Dimension of pose (6 for 3D pos + 3D ori)
            num_components: Number of GMM components
            min_std: Minimum standard deviation for numerical stability
            max_std: Maximum standard deviation
            min_epsilon: Minimum variance to prevent component collapse
            entropy_weight: Weight for entropy regularization
        """
        super().__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.pose_dim = pose_dim
        self.num_components = num_components
        self.min_std = min_std
        self.max_std = max_std
        self.min_epsilon = min_epsilon
        self.entropy_weight = entropy_weight
        
        # Calculate output dimensions for GMM parameters
        # For each component: means (pose_dim) + diagonal covariances (pose_dim) + weight (1)
        self.gmm_output_dim = num_components * (pose_dim + pose_dim + 1)
        
        # Shared MLP for all pose tokens
        self.mlp = MLPResNet(
            num_blocks=2,
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            output_dim=self.gmm_output_dim
        )
    
    def forward(self, pose_hidden_states: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            pose_hidden_states: (batch_size, num_pose_tokens, hidden_dim)
        Returns:
            means: (batch_size, num_pose_tokens, num_components, pose_dim)
            covariances: (batch_size, num_pose_tokens, num_components, pose_dim, pose_dim)
            weights: (batch_size, num_pose_tokens, num_components)
        """
        # Ensure input is in bfloat16
        pose_hidden_states = ensure_bfloat16(pose_hidden_states)
        
        batch_size, num_pose_tokens, hidden_dim = pose_hidden_states.shape
        # Apply shared MLP to each token
        gmm_params = self.mlp(pose_hidden_states)  # (batch, num_pose_tokens, gmm_output_dim)
        # Split GMM params for each token
        gmm_params = gmm_params.view(batch_size, num_pose_tokens, self.num_components, -1)
        means = ensure_bfloat16(gmm_params[..., :self.pose_dim])  # (batch, num_pose_tokens, num_components, pose_dim)
        cov_params = ensure_bfloat16(gmm_params[..., self.pose_dim:2*self.pose_dim])  # (batch, num_pose_tokens, num_components, pose_dim)
        
        # Apply variance regularization to prevent collapse
        # Convert to variance and apply minimum threshold
        variances = torch.exp(cov_params)
        variances = torch.clamp(variances, min=self.min_epsilon)
        cov_params = torch.log(variances)
        
        # Diagonal covariance matrices
        covariances = torch.zeros(batch_size, num_pose_tokens, self.num_components, self.pose_dim, self.pose_dim, device=gmm_params.device, dtype=torch.bfloat16)
        diag_indices = torch.arange(self.pose_dim)
        covariances[..., diag_indices, diag_indices] = torch.exp(cov_params)
        
        weights = ensure_bfloat16(gmm_params[..., -1])  # (batch, num_pose_tokens, num_components)
        weights = torch.softmax(weights, dim=-1)
        return means, covariances, weights
    
    def predict_pose_distribution(self, pose_hidden_states: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Predict pose distribution (alias for forward).
        
        Args:
            pose_hidden_states: Hidden states from LLM for pose tokens
        
        Returns:
            Tuple of (means, covariances, weights)
        """
        return self.forward(pose_hidden_states)
    
    def sample_poses(self, pose_hidden_states: torch.Tensor, num_samples: int = 1) -> torch.Tensor:
        """
        Sample poses from the predicted GMM distribution.
        
        Args:
            pose_hidden_states: Hidden states from LLM for pose tokens
            num_samples: Number of samples to generate per batch item
        
        Returns:
            Sampled poses: (batch_size, num_samples, pose_dim)
        """
        means, covariances, weights = self.forward(pose_hidden_states)
        batch_size = means.shape[0]
        
        # Sample from GMM using reparameterization trick
        samples = []
        for b in range(batch_size):
            # Sample component indices based on weights
            component_indices = torch.multinomial(weights[b], num_samples, replacement=True)
            
            # Sample from selected components
            batch_samples = []
            for i in range(num_samples):
                comp_idx = component_indices[i]
                mean = means[b, comp_idx]  # (pose_dim,)
                cov = covariances[b, comp_idx]  # (pose_dim, pose_dim)
                
                # Convert to float32 for MultivariateNormal (Cholesky doesn't support bfloat16)
                mean = mean.float()
                cov = cov.float()
                
                # Sample from multivariate normal
                sample = torch.distributions.MultivariateNormal(mean, cov).sample()
                # Convert back to bfloat16
                sample = ensure_bfloat16(sample)
                batch_samples.append(sample)
            
            batch_samples = torch.stack(batch_samples)  # (num_samples, pose_dim)
            samples.append(batch_samples)
        
        return torch.stack(samples)  # (batch_size, num_samples, pose_dim)
    
    def compute_loss(self, pose_hidden_states: torch.Tensor, target_poses: torch.Tensor) -> torch.Tensor:
        """
        Compute negative log-likelihood loss for GMM with entropy regularization.
        
        Args:
            pose_hidden_states: Hidden states from LLM for pose tokens (batch_size, num_pose_tokens, hidden_dim)
            target_poses: Target poses: (batch_size, pose_dim)
        
        Returns:
            Loss value
        """
        # Ensure all tensors are in bfloat16 to prevent dtype mismatches
        pose_hidden_states = ensure_bfloat16(pose_hidden_states)
        target_poses = ensure_bfloat16(target_poses)
        
        means, covariances, weights = self.forward(pose_hidden_states)
        batch_size, num_pose_tokens = means.shape[:2]
        
        # For now, let's use only the first pose token for loss computation
        # This is a simplified approach - we can extend to multiple tokens later
        means = means[:, 0, :, :]  # (batch_size, num_components, pose_dim)
        covariances = covariances[:, 0, :, :, :]  # (batch_size, num_components, pose_dim, pose_dim)
        weights = weights[:, 0, :]  # (batch_size, num_components)
        
        # Compute log-likelihood for each component
        log_probs = []
        for b in range(batch_size):
            batch_log_probs = []
            for c in range(self.num_components):
                # Convert to float32 for MultivariateNormal (Cholesky doesn't support bfloat16)
                mean = means[b, c].float()  # (pose_dim,)
                cov = covariances[b, c].float()  # (pose_dim, pose_dim)
                target = target_poses[b].float()  # (pose_dim,)
                
                # Compute log probability
                dist = torch.distributions.MultivariateNormal(mean, cov)
                log_prob = dist.log_prob(target)
                # Convert back to bfloat16
                log_prob = ensure_bfloat16(log_prob)
                batch_log_probs.append(log_prob)
            
            batch_log_probs = torch.stack(batch_log_probs)  # (num_components,)
            log_probs.append(batch_log_probs)
        
        log_probs = torch.stack(log_probs)  # (batch_size, num_components)
        
        # Compute weighted log-likelihood
        weighted_log_probs = log_probs + torch.log(ensure_bfloat16(weights) + 1e-8)
        
        # Use logsumexp for numerical stability
        total_log_prob = torch.logsumexp(weighted_log_probs, dim=-1)
        
        # Negative log-likelihood loss
        nll_loss = -total_log_prob.mean()
        
        # Entropy regularization to encourage component diversity
        # Compute entropy of component weights: -sum(w * log(w))
        entropy = -torch.sum(weights * torch.log(weights + 1e-8), dim=-1).mean()
        entropy_loss = -self.entropy_weight * entropy  # Negative because we want to maximize entropy
        
        # Total loss
        total_loss = nll_loss + entropy_loss
        
        return total_loss


class SimplePoseHead(nn.Module):
    """
    Simple pose head for single pose prediction (non-GMM).
    
    Useful for comparison or when multi-modality is not needed.
    """
    
    def __init__(
        self,
        input_dim: int = 4096,
        hidden_dim: int = 4096,
        pose_dim: int = 6,
    ):
        """
        Initialize simple pose head.
        
        Args:
            input_dim: Input dimension from LLM hidden states
            hidden_dim: Hidden dimension for MLP
            pose_dim: Dimension of pose (6 for 3D pos + 3D ori)
        """
        super().__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.pose_dim = pose_dim
        
        # MLP to predict single pose (following L1RegressionActionHead pattern)
        self.model = MLPResNet(
            num_blocks=2,
            input_dim=input_dim * pose_dim,
            hidden_dim=hidden_dim,
            output_dim=pose_dim
        )
    
    def forward(self, pose_hidden_states: torch.Tensor) -> torch.Tensor:
        """
        Forward pass to predict single pose.
        
        Args:
            pose_hidden_states: Hidden states from LLM for pose tokens
                               Shape: (batch_size, pose_dim, hidden_dim)
        
        Returns:
            Predicted poses: (batch_size, pose_dim)
        """
        # Ensure input is in bfloat16
        pose_hidden_states = ensure_bfloat16(pose_hidden_states)
        
        batch_size = pose_hidden_states.shape[0]
        
        # Reshape to match original pattern
        rearranged_states = pose_hidden_states.reshape(batch_size, -1)
        
        # Predict pose
        pose = self.model(rearranged_states)
        
        return ensure_bfloat16(pose)
    
    def predict_pose(self, pose_hidden_states: torch.Tensor) -> torch.Tensor:
        """
        Predict pose (alias for forward).
        
        Args:
            pose_hidden_states: Hidden states from LLM for pose tokens
        
        Returns:
            Predicted poses: (batch_size, pose_dim)
        """
        return self.forward(pose_hidden_states)
    
    def compute_loss(self, pose_hidden_states: torch.Tensor, target_poses: torch.Tensor) -> torch.Tensor:
        """
        Compute L1 loss for pose prediction.
        
        Args:
            pose_hidden_states: Hidden states from LLM for pose tokens
            target_poses: Target poses: (batch_size, pose_dim)
        
        Returns:
            Loss value
        """
        # Ensure all tensors are in bfloat16 to prevent dtype mismatches
        pose_hidden_states = ensure_bfloat16(pose_hidden_states)
        target_poses = ensure_bfloat16(target_poses)
        
        predicted_poses = self.forward(pose_hidden_states)
        return F.l1_loss(predicted_poses, target_poses) 


class HungarianPoseHead(nn.Module):
    """
    Hungarian pose head for direct multi-pose prediction with Hungarian matching.
    
    Predicts N deterministic poses and uses Hungarian algorithm to match with GT poses.
    """
    
    def __init__(
        self,
        input_dim: int = 4096,
        hidden_dim: int = 4096,
        pose_dim: int = 6,
        num_poses: int = 6,  # Number of poses to predict
        hungarian_weight: float = 0.1,  # Weight for Hungarian matching loss
    ):
        """
        Initialize Hungarian pose head.
        
        Args:
            input_dim: Input dimension from LLM hidden states
            hidden_dim: Hidden dimension for MLP
            pose_dim: Dimension of pose (6 for 3D pos + 3D ori)
            num_poses: Number of poses to predict
            hungarian_weight: Weight for Hungarian matching loss
        """
        super().__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.pose_dim = pose_dim
        self.num_poses = num_poses
        self.hungarian_weight = hungarian_weight
        
        # MLP to predict multiple poses
        self.mlp = MLPResNet(
            num_blocks=2,
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            output_dim=num_poses * pose_dim
        )
        
        # Convert MLP to bfloat16 to match input dtype
        self.mlp = self.mlp.to(torch.bfloat16)
    
    def forward(self, pose_hidden_states: torch.Tensor) -> torch.Tensor:
        """
        Forward pass to predict multiple poses.
        
        Args:
            pose_hidden_states: Hidden states from LLM for pose tokens
                               Shape: (batch_size, num_pose_tokens, hidden_dim)
        
        Returns:
            Predicted poses: (batch_size, num_poses, pose_dim)
        """
        # Ensure input is in bfloat16
        pose_hidden_states = ensure_bfloat16(pose_hidden_states)
        
        batch_size, num_pose_tokens, hidden_dim = pose_hidden_states.shape
        
        # Use first pose token for prediction (can be extended to multiple tokens)
        hidden_states = pose_hidden_states[:, 0, :]  # (batch_size, hidden_dim)
        
        # Predict poses (MLP is already in bfloat16)
        poses_flat = self.mlp(hidden_states)  # (batch_size, num_poses * pose_dim)
        poses = poses_flat.view(batch_size, self.num_poses, self.pose_dim)  # (batch_size, num_poses, pose_dim)
        
        return ensure_bfloat16(poses)
    
    def hungarian_matching_loss(self, predicted_poses: torch.Tensor, target_poses: torch.Tensor) -> torch.Tensor:
        """
        Compute Hungarian matching loss between predicted and target poses.
        
        Args:
            predicted_poses: (batch_size, num_poses, pose_dim)
            target_poses: (batch_size, num_poses, pose_dim)
        
        Returns:
            Hungarian matching loss
        """
        batch_size = predicted_poses.shape[0]

        # Ensure tensors are on the same device
        device = predicted_poses.device
        target_poses = target_poses.to(device)

        # Convert to float32 for distance computation (cdist doesn't support bfloat16)
        predicted_poses_float = predicted_poses.float()
        target_poses_float = target_poses.float()

        # We want to compute distances between each predicted pose and each target pose
        # Reshape for broadcasting: (batch_size, num_poses, 1, pose_dim) vs (batch_size, 1, num_poses, pose_dim)
        predicted_expanded = predicted_poses_float.unsqueeze(2)  # (batch_size, num_poses, 1, pose_dim)
        target_expanded = target_poses_float.unsqueeze(1)  # (batch_size, 1, num_poses, pose_dim)

        # Compute L2 distances
        distances = torch.norm(predicted_expanded - target_expanded, dim=-1)  # (batch_size, num_poses, num_poses)

        # For each batch item, find the minimum distance for each predicted pose
        # This is a simple greedy matching (not optimal Hungarian algorithm)
        min_distances = torch.min(distances, dim=-1)[0]  # (batch_size, num_poses)

        # Return the mean of minimum distances
        return min_distances.mean()
    
    def compute_loss(self, pose_hidden_states: torch.Tensor, target_poses: torch.Tensor) -> torch.Tensor:
        """
        Compute Hungarian matching loss.
        
        Args:
            pose_hidden_states: Hidden states from LLM for pose tokens
            target_poses: Target poses (batch_size, 6, pose_dim) from Hungarian dataset
        
        Returns:
            Loss value
        """
        # Ensure all tensors are in bfloat16
        pose_hidden_states = ensure_bfloat16(pose_hidden_states)
        target_poses = ensure_bfloat16(target_poses)
        
        # Predict poses
        predicted_poses = self.forward(pose_hidden_states)  # (batch_size, num_poses, pose_dim)
        
        # Target poses should already be (batch_size, 6, pose_dim) from Hungarian dataset
        # No need to expand since we're using the special Hungarian dataset
        
        # Compute Hungarian matching loss
        hungarian_loss = self.hungarian_matching_loss(predicted_poses, target_poses)
        
        return hungarian_loss 