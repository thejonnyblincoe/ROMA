"""Dataset loaders for prompt optimization."""

import dspy
from datasets import load_dataset
import random
from typing import Tuple, List


def load_aimo_datasets(
    train_size: int = 5,
    val_size: int = 5,
    test_size: int = 15,
    seed: int = 0
) -> Tuple[List[dspy.Example], List[dspy.Example], List[dspy.Example]]:
    """
    Load AIMO math datasets with configurable sizes.

    Args:
        train_size: Number of training examples
        val_size: Number of validation examples
        test_size: Number of test examples
        seed: Random seed for reproducibility

    Returns:
        Tuple of (train_set, val_set, test_set)
    """

    # Load training/validation split from AIMO
    train_split = load_dataset("AI-MO/aimo-validation-aime")['train']
    train_split = [
        dspy.Example({
            "goal": x['problem'],
            'solution': x['solution'],
            'answer': x['answer'],
        }).with_inputs("goal")
        for x in train_split
    ]

    # Shuffle with fixed seed
    random.Random(seed).shuffle(train_split)
    tot_num = len(train_split)

    # Load test split from AIME 2025
    test_split = load_dataset("MathArena/aime_2025")['train']
    test_split = [
        dspy.Example({
            "goal": x['problem'],
            'answer': x['answer'],
        }).with_inputs("goal")
        for x in test_split
    ]

    # Split datasets
    train_set = train_split[:train_size]
    val_set = train_split[tot_num // 2:tot_num // 2 + val_size]

    # Repeat test set if needed to reach desired size
    test_set = (test_split * ((test_size // len(test_split)) + 1))[:test_size]

    return train_set, val_set, test_set
