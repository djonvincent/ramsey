"""
ramsey:  Probabilistic deep learning using JAX
"""

from ramsey._src.neural_process.attentive_neural_process import ANP, MaskedANP
from ramsey._src.neural_process.doubly_attentive_neural_process import DANP, MaskedDANP
from ramsey._src.neural_process.neural_process import NP
from ramsey._src.neural_process.convolutional_conditional_neural_process import CCNP, ConvGNP
from ramsey._src.neural_process.train_neural_process import train_neural_process, train_masked_np

__version__ = "0.2.1"

__all__ = [
    "ANP",
    "MaskedANP",
    "DANP",
    "MaskedDANP"
    "NP",
    "CCNP",
    "ConvGNP",
    "train_neural_process",
    "train_masked_np"
]
