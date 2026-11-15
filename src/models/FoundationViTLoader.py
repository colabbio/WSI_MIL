"""
FoundationViTLoader: Cargador y descargador unificado de Vision Transformers (ViT)
fundacionales para patología computacional y transcriptómica espacial.

Soporta:
  1. 'dinov2_large': Meta DINOv2 ViT-Large (1024 dims) - Acceso público inmediato sin token.
  2. 'mahmood_uni': Harvard MahmoodLab UNI / UNI2-h (1536 dims) - ViT-Giant DINOv2.
  3. 'paige_virchow2': Paige.ai Virchow 2 (1280 dims) - ViT-Huge 14x14.
  4. 'prov_gigapath': Microsoft / Providence Prov-GigaPath (1536 dims) - ViT-Giant.
  5. 'resnet50': Baseline clásico convolucional (2048 dims).
"""

import os
import sys
import torch
import torch.nn as nn
from typing import Tuple, Optional


SUPPORTED_ENCODERS = {
    "dinov2_large": {
        "repo_id": "facebook/dinov2-large",
        "embed_dim": 1024,
        "requires_token": False,
        "description": "Meta DINOv2 ViT-L/14 (1024 dims) - Acceso público sin gated access."
    },
    "mahmood_uni": {
        "repo_id": "MahmoodLab/UNI2-h",
        "embed_dim": 1536,
        "requires_token": True,
        "description": "Harvard Mahmood Lab UNI2-h (1536 dims) - ViT-Giant 24 layers."
    },
    "paige_virchow2": {
        "repo_id": "paige-ai/Virchow2",
        "embed_dim": 1280,
        "requires_token": True,
        "description": "Paige.ai Virchow 2 (1280 dims) - ViT-Huge entrenado en >3M WSI."
    },
    "prov_gigapath": {
        "repo_id": "prov-gigapath/prov-gigapath",
        "embed_dim": 1536,
        "requires_token": True,
        "description": "Microsoft Prov-GigaPath (1536 dims) - ViT-Giant para patología general."
    },
    "resnet50": {
        "repo_id": "torchvision/resnet50",
        "embed_dim": 2048,
        "requires_token": False,
        "description": "ResNet-50 clásico pre-entrenado en ImageNet (2048 dims)."
    }
}


class PretrainedViTWrapper(nn.Module):
    """
    Wrapper que normaliza la salida de cualquier Vision Transformer a un tensor (B, embed_dim).
    """
    def __init__(self, backbone: nn.Module, embed_dim: int, model_type: str):
        super().__init__()
        self.backbone = backbone
        self.embed_dim = embed_dim
        self.model_type = model_type

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor de parches o spots con shape (B, 3, 224, 224)
        Returns:
            Tensor de embeddings latentes con shape (B, embed_dim)
        """
        if self.model_type == "dinov2_large":
            # DINOv2 de torch.hub devuelve directamente el vector CLS
            return self.backbone(x)
        elif hasattr(self.backbone, "forward_features"):
            feats = self.backbone.forward_features(x)
            if isinstance(feats, dict) and "x_norm_clstoken" in feats:
                return feats["x_norm_clstoken"]
            elif isinstance(feats, torch.Tensor) and feats.dim() == 3:
                return feats[:, 0]  # Extraer token [CLS]
            return feats
        return self.backbone(x)


def load_foundation_vit(
    model_name: str = "dinov2_large",
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    hf_token: Optional[str] = None,
    cache_dir: Optional[str] = None
) -> Tuple[nn.Module, int]:
    """
    Carga un modelo Vision Transformer pre-entrenado garantizando que sus pesos
    estén en caché local o descargándolos automáticamente de HuggingFace/TorchHub.

    Args:
        model_name: Uno de los modelos en SUPPORTED_ENCODERS ('dinov2_large', 'mahmood_uni', etc.)
        device: Dispositivo de ejecución ('cuda' o 'cpu')
        hf_token: Token de Hugging Face (opcional, lee de $HF_TOKEN si existe)
        cache_dir: Directorio de caché (opcional, default /data/vfs/models/foundation_vits o ~/.cache)

    Returns:
        Tuple[nn.Module, int]: (Modelo ViT en modo eval y to(device), dimensión del embedding)
    """
    token = hf_token or os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")

    if model_name not in SUPPORTED_ENCODERS:
        raise ValueError(f"Modelo '{model_name}' no soportado. Opciones: {list(SUPPORTED_ENCODERS.keys())}")

    meta = SUPPORTED_ENCODERS[model_name]
    embed_dim = meta["embed_dim"]

    if meta["requires_token"] and not token:
        print(f"⚠️ AVISO: El modelo '{model_name}' requiere acceso de Hugging Face.")
        print(f"   Por favor exporta HF_TOKEN='hf_...' o solicita acceso en https://huggingface.co/{meta['repo_id']}")
        print(f"   Alternativa inmediata sin token: usa 'dinov2_large' o 'resnet50'.")

    # 1. Caso DINOv2 Large (Público, sin token)
    if model_name == "dinov2_large":
        try:
            raw_model = torch.hub.load("facebookresearch/dinov2", "dinov2_vitl14", pretrained=True)
            model = PretrainedViTWrapper(raw_model, embed_dim, "dinov2_large")
        except Exception as e:
            print(f"Fallo cargando dinov2 de torch.hub ({e}). Usando fallback timm...")
            import timm
            raw_model = timm.create_model("vit_large_patch14_dinov2.lvd142m", pretrained=True, num_classes=0)
            model = PretrainedViTWrapper(raw_model, embed_dim, "dinov2_large")

    # 2. Caso ResNet-50 Clásico
    elif model_name == "resnet50":
        from torchvision import models
        from torchvision.models.resnet import ResNet50_Weights
        res = models.resnet50(weights=ResNet50_Weights.DEFAULT)
        raw_model = nn.Sequential(*list(res.children())[:-1], nn.Flatten())
        model = PretrainedViTWrapper(raw_model, embed_dim, "resnet50")

    # 3. Modelos Gated / Foundation de Patología (UNI, Virchow, Prov-GigaPath)
    else:
        import timm
        try:
            # timm soporta cargar directamente desde Hugging Face Hub con hf-hub:
            raw_model = timm.create_model(
                f"hf-hub:{meta['repo_id']}",
                pretrained=True,
                num_classes=0,
                hf_auth=token
            )
            model = PretrainedViTWrapper(raw_model, embed_dim, model_name)
        except Exception as e:
            raise RuntimeError(
                f"Error descargando {model_name} de {meta['repo_id']}: {e}\n"
                f"Verifica tu HF_TOKEN y que tengas permiso de acceso en Hugging Face."
            )

    model.eval()
    model.to(device)
    for p in model.parameters():
        p.requires_grad = False

    return model, embed_dim


def preload_all_available_models():
    """Descarga los modelos sin token para tenerlos listos en local."""
    print("Iniciando precacheo de Foundation ViTs públicos en ColabBio...")
    load_foundation_vit("dinov2_large", device="cpu")
    load_foundation_vit("resnet50", device="cpu")
    print("✅ Modelos base DINOv2 Large y ResNet-50 precacheados con éxito.")


if __name__ == "__main__":
    preload_all_available_models()
