"""
pose_vlm.py

PoseVLM model for pose prediction from vision and language inputs.
Inherits from PrismaticVLM but removes proprioception and adds pose prediction.
"""

import torch
import torch.nn as nn
from typing import Optional, Dict, Any, Tuple
import logging

from prismatic.models.vlms.prismatic import PrismaticVLM
from prismatic.models.backbones.vision import VisionBackbone
from prismatic.models.backbones.llm import LLMBackbone
from prismatic.models.pose_heads import GMMPoseHead, SimplePoseHead, HungarianPoseHead
from prismatic.util import ensure_bfloat16


logger = logging.getLogger(__name__)


class PoseVLM(PrismaticVLM):
    """
    PoseVLM model for pose prediction from vision and language inputs.
    
    Key differences from PrismaticVLM:
    - Removes proprioception input
    - Replaces action head with pose head
    - Uses pose tokens instead of action tokens
    - Supports both GMM and simple pose prediction
    """
    
    def __init__(
        self,
        model_id: str,
        vision_backbone: VisionBackbone,
        llm_backbone: LLMBackbone,
        pose_head_type: str = "gmm",  # "gmm", "simple", or "hungarian"
        pose_dim: int = 6,  # 3D position + 3D orientation
        num_pose_tokens: int = 6,  # Number of pose tokens (same as original action tokens)
        gmm_num_components: int = 6,  # For GMM pose head (increased from 3 to 6)
        gmm_entropy_weight: float = 0.1,  # Weight for GMM entropy regularization
        gmm_min_epsilon: float = 1e-2,  # Minimum variance for GMM components
        hungarian_weight: float = 0.1,  # Weight for Hungarian matching loss
        enable_mixed_precision_training: bool = True,
        arch_specifier: str = "gelu-mlp",
        **kwargs
    ):
        """
        Initialize PoseVLM model.
        
        Args:
            model_id: Model identifier
            vision_backbone: Vision backbone for image processing
            llm_backbone: Language model backbone
            pose_head_type: Type of pose head ("gmm", "simple", or "hungarian")
            pose_dim: Dimension of pose (6 for 3D pos + 3D ori)
            num_pose_tokens: Number of pose tokens
            gmm_num_components: Number of GMM components
            gmm_entropy_weight: Weight for GMM entropy regularization
            gmm_min_epsilon: Minimum variance for GMM components
            hungarian_weight: Weight for Hungarian matching loss
            enable_mixed_precision_training: Whether to enable mixed precision
            arch_specifier: Architecture specifier for projector
            **kwargs: Additional arguments passed to PrismaticVLM
        """
        # Initialize parent class
        super().__init__(
            model_id=model_id,
            vision_backbone=vision_backbone,
            llm_backbone=llm_backbone,
            enable_mixed_precision_training=enable_mixed_precision_training,
            arch_specifier=arch_specifier,
            **kwargs
        )
        
        self.pose_head_type = pose_head_type
        self.pose_dim = pose_dim
        self.num_pose_tokens = num_pose_tokens
        self.gmm_num_components = gmm_num_components
        self.gmm_entropy_weight = gmm_entropy_weight
        self.gmm_min_epsilon = gmm_min_epsilon
        self.hungarian_weight = hungarian_weight
        
        # Replace action head with pose head
        self._setup_pose_head()
        
        # Move entire model to bfloat16 to prevent dtype mismatches
        self.to(torch.bfloat16)
        
        # Update model info
        self.model_type = "pose_vlm"
        logger.info(f"Initialized PoseVLM with {pose_head_type} pose head")
    
    def _setup_pose_head(self):
        """Setup pose head based on configuration."""
        # Get hidden dimension from LLM
        hidden_dim = self.llm_backbone.embed_dim
        
        if self.pose_head_type == "gmm":
            self.pose_head = GMMPoseHead(
                input_dim=hidden_dim,
                hidden_dim=hidden_dim,
                pose_dim=self.pose_dim,
                num_components=self.gmm_num_components,
                min_epsilon=self.gmm_min_epsilon,
                entropy_weight=self.gmm_entropy_weight,
            )
        elif self.pose_head_type == "simple":
            self.pose_head = SimplePoseHead(
                input_dim=hidden_dim,
                hidden_dim=hidden_dim,
                pose_dim=self.pose_dim,
            )
        elif self.pose_head_type == "hungarian":
            self.pose_head = HungarianPoseHead(
                input_dim=hidden_dim,
                hidden_dim=hidden_dim,
                pose_dim=self.pose_dim,
                num_poses=6,  # Match number of GT poses
                hungarian_weight=self.hungarian_weight,
            )
        else:
            raise ValueError(f"Unknown pose head type: {self.pose_head_type}")
        
        # Ensure pose head is in bfloat16
        self.pose_head = self.pose_head.to(torch.bfloat16)
        
        # Remove original action head if it exists
        if hasattr(self, 'action_head'):
            delattr(self, 'action_head')
    
    def forward(
        self,
        images: torch.Tensor,
        text: torch.Tensor,
        text_attention_mask: torch.Tensor,
        pose_tokens: Optional[torch.Tensor] = None,
        return_dict: bool = True,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass for pose prediction.
        
        Args:
            images: Input images (batch_size, num_images, channels, height, width)
            text: Input text tokens (batch_size, text_length)
            text_attention_mask: Text attention mask (batch_size, text_length)
            pose_tokens: Pose tokens for training (batch_size, num_pose_tokens)
            return_dict: Whether to return dictionary output
        
        Returns:
            Dictionary with model outputs
        """
        # Ensure images are in bfloat16
        images = ensure_bfloat16(images)
        
        batch_size = images.shape[0]
        
        # Process images through vision backbone
        image_features = self.vision_backbone(images)  # (batch_size, num_images, hidden_dim)
        
        # Project image features
        projected_image_features = self.projector(image_features)  # (batch_size, num_images, llm_hidden_dim)
        
        # Prepare LLM inputs
        llm_inputs = self._prepare_llm_inputs(
            text=text,
            text_attention_mask=text_attention_mask,
            projected_image_features=projected_image_features,
            pose_tokens=pose_tokens,
        )
        
        # Forward through LLM
        llm_outputs = self.llm_backbone(**llm_inputs, output_hidden_states=True)
        
        # Check if hidden_states is available
        if llm_outputs.hidden_states is None:
            hidden_states = llm_outputs.last_hidden_state
        else:
            hidden_states = llm_outputs.hidden_states[-1]  # Use last layer
        
        # Extract pose token hidden states
        pose_hidden_states = self._extract_pose_hidden_states(hidden_states, llm_inputs['attention_mask'])
        
        # Predict poses
        if self.pose_head_type == "gmm":
            means, covariances, weights = self.pose_head(pose_hidden_states)
            pose_outputs = {
                'means': means,
                'covariances': covariances,
                'weights': weights,
            }
        else:
            predicted_poses = self.pose_head(pose_hidden_states)
            pose_outputs = {
                'predicted_poses': predicted_poses,
            }
        
        # Prepare output
        outputs = {
            'pose_outputs': pose_outputs,
            'hidden_states': hidden_states,
            'llm_outputs': llm_outputs,
        }
        
        if return_dict:
            return outputs
        else:
            return outputs
    
    def _prepare_llm_inputs(
        self,
        text: torch.Tensor,
        text_attention_mask: torch.Tensor,
        projected_image_features: torch.Tensor,
        pose_tokens: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Prepare inputs for LLM.
        
        Args:
            text: Input text tokens
            text_attention_mask: Text attention mask
            projected_image_features: Projected image features
            pose_tokens: Pose tokens for training
        
        Returns:
            Dictionary with LLM inputs
        """
        batch_size = text.shape[0]
        
        # Create pose tokens if not provided (for inference)
        if pose_tokens is None:
            pose_tokens = torch.zeros(
                batch_size, self.num_pose_tokens,
                device=text.device, dtype=text.dtype
            )
        
        # Prepare input sequence: [text, image_features, pose_tokens]
        input_ids = []
        attention_mask = []
        
        # Add text
        input_ids.append(text)
        attention_mask.append(text_attention_mask)
        
        # Add image features (as special tokens)
        for i in range(projected_image_features.shape[1]):
            # Create placeholder tokens for image features
            img_tokens = torch.full(
                (batch_size, 1),
                self.llm_backbone.config.pad_token_id,
                device=text.device, dtype=text.dtype
            )
            input_ids.append(img_tokens)
            attention_mask.append(torch.ones_like(img_tokens))
        
        # Add pose tokens
        pose_token_ids = torch.full(
            (batch_size, self.num_pose_tokens),
            self.llm_backbone.config.pad_token_id,
            device=text.device, dtype=text.dtype
        )
        input_ids.append(pose_token_ids)
        attention_mask.append(torch.ones_like(pose_token_ids))
        
        # Concatenate all inputs
        input_ids = torch.cat(input_ids, dim=1)
        attention_mask = torch.cat(attention_mask, dim=1)
        
        # Prepare inputs for LLM
        llm_inputs = {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
        }
        
        return llm_inputs
    
    def _extract_pose_hidden_states(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Extract hidden states corresponding to pose tokens.
        Args:
            hidden_states: Hidden states from LLM (batch_size, seq_len, hidden_dim)
            attention_mask: Attention mask (batch_size, seq_len)
        Returns:
            Pose token hidden states (batch_size, num_pose_tokens, hidden_dim)
        """
        # Find pose token positions (last num_pose_tokens positions)
        pose_start_idx = hidden_states.shape[1] - self.num_pose_tokens
        pose_end_idx = hidden_states.shape[1]
        # Extract pose token hidden states
        pose_hidden_states = hidden_states[:, pose_start_idx:pose_end_idx, :]
        return pose_hidden_states
    
    def predict_pose(
        self,
        images: torch.Tensor,
        text: torch.Tensor,
        text_attention_mask: torch.Tensor,
        num_samples: int = 1,
    ) -> Dict[str, torch.Tensor]:
        """
        Predict poses from images and text.
        
        Args:
            images: Input images
            text: Input text tokens
            text_attention_mask: Text attention mask
            num_samples: Number of samples to generate (for GMM)
        
        Returns:
            Dictionary with pose predictions
        """
        with torch.no_grad():
            outputs = self.forward(
                images=images,
                text=text,
                text_attention_mask=text_attention_mask,
                pose_tokens=None,  # No pose tokens for inference
            )
            
            pose_outputs = outputs['pose_outputs']
            
            if self.pose_head_type == "gmm":
                # Sample from GMM
                pose_hidden_states = self._extract_pose_hidden_states(
                    outputs['hidden_states'],
                    torch.ones(images.shape[0], outputs['hidden_states'].shape[1], device=images.device)
                )
                sampled_poses = self.pose_head.sample_poses(pose_hidden_states, num_samples)
                
                return {
                    'sampled_poses': sampled_poses,
                    'means': pose_outputs['means'],
                    'covariances': pose_outputs['covariances'],
                    'weights': pose_outputs['weights'],
                }
            else:
                return {
                    'predicted_poses': pose_outputs['predicted_poses'],
                }
    
    def compute_loss(
        self,
        images: torch.Tensor,
        text: torch.Tensor,
        text_attention_mask: torch.Tensor,
        target_poses: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute loss for pose prediction.
        
        Args:
            images: Input images
            text: Input text tokens
            text_attention_mask: Text attention mask
            target_poses: Target poses (batch_size, pose_dim)
        
        Returns:
            Loss value
        """
        # Ensure all inputs are in bfloat16 to prevent dtype mismatches
        images = ensure_bfloat16(images)
        text = text  # Keep as is (should be long/int)
        text_attention_mask = text_attention_mask  # Keep as is (should be long/int)
        target_poses = ensure_bfloat16(target_poses)
        
        # Forward pass
        outputs = self.forward(
            images=images,
            text=text,
            text_attention_mask=text_attention_mask,
            pose_tokens=torch.zeros_like(target_poses),  # Dummy pose tokens
        )
        
        # Extract pose hidden states
        pose_hidden_states = self._extract_pose_hidden_states(
            outputs['hidden_states'],
            torch.ones(images.shape[0], outputs['hidden_states'].shape[1], device=images.device)
        )
        
        # Compute loss
        loss = self.pose_head.compute_loss(pose_hidden_states, target_poses)
        
        return loss
    
    @classmethod
    def from_pretrained(
        cls,
        model_name_or_path: str,
        pose_head_type: str = "gmm",
        pose_dim: int = 6,
        num_pose_tokens: int = 6,
        gmm_num_components: int = 3,
        **kwargs
    ) -> "PoseVLM":
        """
        Load PoseVLM from pretrained checkpoint.
        
        Args:
            model_name_or_path: Path to pretrained model
            pose_head_type: Type of pose head
            pose_dim: Dimension of pose
            num_pose_tokens: Number of pose tokens
            gmm_num_components: Number of GMM components
            **kwargs: Additional arguments
        
        Returns:
            Loaded PoseVLM model
        """
        # Load base model
        base_model = PrismaticVLM.from_pretrained(model_name_or_path, **kwargs)
        
        # Create PoseVLM with same components
        model = cls(
            model_id=base_model.model_id,
            vision_backbone=base_model.vision_backbone,
            llm_backbone=base_model.llm_backbone,
            pose_head_type=pose_head_type,
            pose_dim=pose_dim,
            num_pose_tokens=num_pose_tokens,
            gmm_num_components=gmm_num_components,
            enable_mixed_precision_training=base_model.enable_mixed_precision_training,
            arch_specifier=base_model.arch_specifier,
        )
        
        # Copy weights from base model (excluding action head)
        model.load_state_dict(base_model.state_dict(), strict=False)
        
        logger.info(f"Loaded PoseVLM from {model_name_or_path}")
        return model 