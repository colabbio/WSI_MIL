"""
ProtoMIL: Prototype-Based Interpretable Multiple Instance Learning for Computational Pathology.

References:
- Chen et al., "This Looks Like That: Deep Learning for Interpretable Image Recognition" (NeurIPS).
- Zhang et al., "ProtoMIL: Interpretable Prototype-Based Multiple Instance Learning" (MICCAI).
- Dov et al., "Morphological Prototyping for Slide Representation" (CVPR / MedIA).

Unlike black-box attention MIL, ProtoMIL learns a dictionary of K visual prototype vectors
in the latent embedding space. Each WSI is mapped to a quantitative "morphological prototype profile"
representing the presence and abundance of each visual tissue archetype (e.g. cribriform punched-out
lumina, fused glands, benign stroma, high-grade solid sheets).
"""

from typing import Dict, Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F


class ProtoMIL(nn.Module):
    """
    Prototype-based Multiple Instance Learning architecture.

    Args:
        input_dim (int): Feature dimension of patch embeddings (e.g. 1536 for H-optimus-0, 1024 for UNI, 2048 for ResNet50).
        hidden_dim (int): Latent projection dimension for patches and prototypes.
        n_prototypes (int): Number of morphological prototypes (e.g. 16 or 32).
        n_isup_classes (int): Number of ISUP grade classes (default: 5).
        gamma (float): Scaling factor / temperature in similarity calculation.
        dropout_rate (float): Dropout probability.
    """
    def __init__(
        self,
        input_dim: int = 1024,
        hidden_dim: int = 256,
        n_prototypes: int = 16,
        n_isup_classes: int = 5,
        gamma: float = 1.0,
        dropout_rate: float = 0.25,
    ):
        super(ProtoMIL, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.n_prototypes = n_prototypes
        self.n_isup_classes = n_isup_classes
        self.gamma = gamma

        # 1. Patch Feature Projection
        self.projector = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
        )

        # 2. Learnable Morphological Prototypes Dictionary [K, hidden_dim]
        self.prototypes = nn.Parameter(torch.randn(n_prototypes, hidden_dim))
        nn.init.orthogonal_(self.prototypes)

        # 3. Downstream Clinical Heads based on Prototype Profiles [K]
        # Head 1: ISUP Grade Group (5 classes)
        self.isup_head = nn.Sequential(
            nn.Linear(n_prototypes, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim // 2, n_isup_classes),
        )

        # Head 2: Biochemical Recurrence (BCR) Risk (Scalar hazard risk)
        self.bcr_head = nn.Linear(n_prototypes, 1)

        # Head 3: PTEN Genomic Loss Logit
        self.pten_head = nn.Linear(n_prototypes, 1)

    def forward(self, bag_features: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Forward pass over a bag of patch embeddings.

        Args:
            bag_features: [N_tiles, input_dim] tensor of patch embeddings for a WSI.

        Returns:
            Dictionary with:
            - isup_logits: [1, n_isup_classes]
            - bcr_risk: [1, 1]
            - pten_logit: [1, 1]
            - slide_prototype_profile: [1, n_prototypes] (proportions / max similarity to each prototype)
            - patch_prototype_similarities: [N_tiles, n_prototypes]
            - nearest_instances_indices: [n_prototypes] index of closest patch to each prototype
        """
        if bag_features.dim() == 3:
            bag_features = bag_features.squeeze(0)

        num_tiles = bag_features.size(0)

        # Project patch features to latent space: [N, hidden_dim]
        h = self.projector(bag_features)

        # Compute pairwise Euclidean squared distances: [N, K]
        # dists[i, k] = ||h[i] - prototypes[k]||^2
        h_norm = (h ** 2).sum(dim=1, keepdim=True)  # [N, 1]
        p_norm = (self.prototypes ** 2).sum(dim=1, keepdim=True).t()  # [1, K]
        dists = h_norm + p_norm - 2.0 * torch.matmul(h, self.prototypes.t())
        dists = torch.clamp(dists, min=0.0)

        # Similarity to prototypes: [N, K]
        # sims in range (0, 1]
        sims = torch.exp(-self.gamma * dists)

        # Slide-level Prototype Profile Aggregation: [1, K]
        # Max-pooling captures the presence of localized morphological patterns (e.g. focal cribriform)
        # Mean-pooling captures total tissue burden
        max_profile = torch.max(sims, dim=0)[0]  # [K]
        mean_profile = torch.mean(sims, dim=0)   # [K]
        slide_prototype_profile = 0.7 * max_profile + 0.3 * mean_profile
        slide_prototype_profile = slide_prototype_profile.unsqueeze(0)  # [1, K]

        # Clinical Predictions
        isup_logits = self.isup_head(slide_prototype_profile)
        bcr_risk = self.bcr_head(slide_prototype_profile)
        pten_logit = self.pten_head(slide_prototype_profile)

        # Identify nearest patch in this slide for each prototype
        # closest_indices: [K]
        closest_indices = torch.argmin(dists, dim=0)

        return {
            "isup_logits": isup_logits,
            "bcr_risk": bcr_risk,
            "pten_logit": pten_logit,
            "slide_prototype_profile": slide_prototype_profile,
            "patch_prototype_similarities": sims,
            "nearest_instances_indices": closest_indices,
            "patch_distances": dists,
        }

    def get_prototype_exemplars(self, bag_features: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Helper method to extract the closest patch embeddings in the bag to each prototype.
        """
        with torch.no_grad():
            out = self.forward(bag_features)
            closest_idx = out["nearest_instances_indices"]
            exemplar_features = bag_features[closest_idx]
            min_distances = torch.min(out["patch_distances"], dim=0)[0]
            return exemplar_features, min_distances


class ProtoMILLoss(nn.Module):
    """
    Compound loss for ProtoMIL combining:
    1. Task classification loss (Cross-Entropy for ISUP).
    2. Prototype separation / diversity loss (keeps prototypes distinct).
    3. Clustering loss (pulls each patch close to at least one prototype).
    """
    def __init__(
        self,
        weight_isup: float = 1.0,
        weight_diversity: float = 0.1,
        weight_cluster: float = 0.05,
    ):
        super(ProtoMILLoss, self).__init__()
        self.weight_isup = weight_isup
        self.weight_diversity = weight_diversity
        self.weight_cluster = weight_cluster
        self.ce_loss = nn.CrossEntropyLoss()

    def forward(
        self,
        outputs: Dict[str, torch.Tensor],
        isup_target: Optional[torch.Tensor] = None,
        prototypes: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        total_loss = torch.tensor(0.0, device=outputs["isup_logits"].device)
        losses_dict = {}

        # 1. Classification Loss
        if isup_target is not None:
            l_isup = self.ce_loss(outputs["isup_logits"], isup_target)
            total_loss = total_loss + self.weight_isup * l_isup
            losses_dict["isup_loss"] = float(l_isup.item())

        # 2. Prototype Clustering Loss: min_k ||h_i - p_k||^2
        if "patch_distances" in outputs:
            min_dist_per_patch = torch.min(outputs["patch_distances"], dim=1)[0]
            l_cluster = torch.mean(min_dist_per_patch)
            total_loss = total_loss + self.weight_cluster * l_cluster
            losses_dict["cluster_loss"] = float(l_cluster.item())

        # 3. Prototype Diversity Loss: penalize high cosine similarity between distinct prototypes
        if prototypes is not None and prototypes.size(0) > 1:
            p_norm = F.normalize(prototypes, p=2, dim=1)
            sim_matrix = torch.matmul(p_norm, p_norm.t())
            # Zero out diagonal
            eye = torch.eye(sim_matrix.size(0), device=sim_matrix.device)
            off_diag = sim_matrix * (1.0 - eye)
            l_diversity = torch.mean(off_diag ** 2)
            total_loss = total_loss + self.weight_diversity * l_diversity
            losses_dict["diversity_loss"] = float(l_diversity.item())

        losses_dict["total_loss"] = float(total_loss.item())
        return total_loss, losses_dict
