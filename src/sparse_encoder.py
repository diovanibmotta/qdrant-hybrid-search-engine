"""Sparse vector encoding using term frequency with global vocabulary.

Follows the approach from Qdrant Essentials Day 3: uses a global vocabulary
that automatically extends as new texts are processed, mapping each unique
word to a sequential integer index.
"""

import re
from collections import Counter

from qdrant_client import models


# Global vocabulary - automatically extends as new texts are processed
global_vocabulary: dict[str, int] = {}


def create_sparse_vector(text: str) -> models.SparseVector:
    """Create sparse vector from text using term frequency.

    Tokenizes text using regex word boundaries, counts occurrences,
    and maps each unique word to a global vocabulary index. New words
    are automatically added to the vocabulary with the next available index.

    Args:
        text: Input text to encode.

    Returns:
        models.SparseVector with indices from global vocabulary and
        term frequency values. Returns empty vector if no valid tokens.
    """
    # Simple tokenization (same as course reference)
    words = re.findall(r"\b\w+\b", text.lower())
    word_counts = Counter(words)

    if not word_counts:
        return models.SparseVector(indices=[], values=[])

    # Convert to sparse vector format, extending vocabulary as needed
    indices = []
    values = []

    for word, count in word_counts.items():
        if word not in global_vocabulary:
            # Add new word to vocabulary with next available index
            global_vocabulary[word] = len(global_vocabulary)

        indices.append(global_vocabulary[word])
        values.append(float(count))

    return models.SparseVector(indices=indices, values=values)


def reset_vocabulary() -> None:
    """Reset the global vocabulary. Useful for testing or re-indexing."""
    global global_vocabulary
    global_vocabulary.clear()
