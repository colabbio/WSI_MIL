"""
ColabBio Spatial Transcriptomics & Vision Transformers Package.

Módulos:
- FoundationViTLoader: Cargador y descargador de modelos fundacionales (UNI, Virchow 2, Prov-GigaPath, DINOv2).
- SpatialTransMIL: Adaptación de TransMIL con coordenadas espaciales para bolsas de spots.
- SpatialViTDualAttention: (En desarrollo por Lorenzo) Red de predicción de expresión génica in-silico.
"""

from models.FoundationViTLoader import load_foundation_vit, SUPPORTED_ENCODERS

__all__ = [
    "load_foundation_vit",
    "SUPPORTED_ENCODERS",
]
