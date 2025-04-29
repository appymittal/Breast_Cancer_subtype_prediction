# models/__init__.py

# Import core components
from .classifier import MultiOmicsClassifier, SingleOmicsClassifier, SingleOmicsClassifierCNN
from .convnext import MiniConvNeXtMethylation
from .vae import VAEEncoder
from .model_fusion import MultimodalFusion
from .cnn import RNACNNEncoder

# Optional: Define public API
__all__ = [
    'MultiOmicsClassifier',
    'MiniConvNeXtMethylation',
    'VAEEncoder',
    'MultimodalFusion',
    'SingleOmicsClassifier',
    'RNACNNEncoder',
    'SingleOmicsClassifierCNN'
]