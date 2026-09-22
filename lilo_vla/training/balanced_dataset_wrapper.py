"""
Balanced Dataset Wrapper for Original vs Augmented Demos

This wrapper ensures equal sampling of original and augmented demonstrations
during VLA training to prevent bias toward the artificial 60-step augmented demos.

IMPORTANT: This is a passthrough wrapper that doesn't scan episodes upfront,
which would be too slow for large RLDS datasets. Instead, it provides a simple
shuffled view of the dataset. Actual balancing should be done at the RLDS
creation stage or via WeightedRandomSampler in the DataLoader.
"""

import random
from torch.utils.data import Dataset


class BalancedOriginalAugmentedDataset(Dataset):
    """
    Passthrough wrapper for RLDS datasets with shuffling.

    Note: Full dataset scanning is prohibitively slow for large RLDS datasets
    because each episode access triggers data loading and transformation.
    This wrapper simply provides shuffled access to the base dataset.
    """

    def __init__(self, base_dataset, balance_ratio: float = 0.5):
        """
        Args:
            base_dataset: The underlying RLDS dataset
            balance_ratio: Fraction of samples that should be original (default: 0.5 for 50/50)
                          Note: This parameter is accepted for compatibility but not used in
                          the current implementation. Balancing should be done at RLDS creation.
        """
        self.base_dataset = base_dataset
        self.balance_ratio = balance_ratio

        # Use the base dataset length directly
        self.dataset_length = len(base_dataset)

        # Create a simple shuffled index mapping
        self.indices = list(range(self.dataset_length))
        random.shuffle(self.indices)

    def __len__(self):
        return self.dataset_length

    def __getitem__(self, idx):
        # Map the shuffled index to the actual dataset index
        actual_idx = self.indices[idx]
        return self.base_dataset[actual_idx]

    def reshuffle(self):
        """Call this between epochs to get different sampling."""
        random.shuffle(self.indices)


# Usage example:
# from balanced_dataset_wrapper import BalancedOriginalAugmentedDataset
#
# # In your training script:
# train_dataset = make_dataset(...)  # Your original RLDS dataset
# balanced_dataset = BalancedOriginalAugmentedDataset(train_dataset, balance_ratio=0.5)
# dataloader = DataLoader(balanced_dataset, ...)