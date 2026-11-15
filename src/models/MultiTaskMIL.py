"""
MultiTaskMIL: Multi-task Attention-based Deep Multiple Instance Learning (abMIL).

Built directly on top of GatedAttentionModel (Ilse et al., 2018).
Simultaneously predicts:
  1. ISUP Grade (Multi-class classification: 5 classes, grades 1-5).
  2. HRD / DDR Mutation status (Binary classification: logit output for BCEWithLogitsLoss).
  3. BCR Survival / Recurrence Risk (Scalar continuous risk score for ranking / Cox partial likelihood).

Includes MultiTaskLoss combining:
  L_total = w_isup * L_ce + w_hrd * L_bce + w_bcr * L_cox
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from models.AttentionModel import GatedAttentionModel


class MultiTaskMIL(nn.Module):
    """
    Multi-task Multiple Instance Learning architecture using Gated Attention Pooling.
    
    Args:
        input_dim (int): Feature dimension of patch embeddings (e.g. 2048 for ResNet50, 768 for UNI/Virchow/PLIP).
        hidden_dim1 (int): Projection dimension for patch features.
        hidden_dim2 (int): Latent dimension for gated attention computation.
        dropout_rate (float): Dropout probability.
        n_isup_classes (int): Number of ISUP grade classes (default: 5).
    """
    def __init__(
        self,
        input_dim: int = 2048,
        hidden_dim1: int = 512,
        hidden_dim2: int = 256,
        dropout_rate: float = 0.25,
        n_isup_classes: int = 5,
    ):
        super(MultiTaskMIL, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim1 = hidden_dim1
        self.n_isup_classes = n_isup_classes

        # Shared Gated Attention Backbone
        self.attention_net = GatedAttentionModel(
            input_dim=input_dim,
            hidden_dim1=hidden_dim1,
            hidden_dim2=hidden_dim2,
            dropout_rate=dropout_rate,
            classes=1
        )

        # Multi-Task Classification & Survival Heads
        # Head 1: ISUP Grade (5-class ordinal/categorical logits)
        self.isup_head = nn.Sequential(
            nn.Linear(hidden_dim1, hidden_dim1 // 2),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim1 // 2, n_isup_classes)
        )

        # Head 2: HRD / DDR Mutation Status (Binary logit)
        self.hrd_head = nn.Sequential(
            nn.Linear(hidden_dim1, hidden_dim1 // 2),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim1 // 2, 1)
        )

        # Head 3: BCR Recurrence Risk Score (Scalar log-hazard ratio for Cox / ranking)
        self.bcr_head = nn.Sequential(
            nn.Linear(hidden_dim1, hidden_dim1 // 2),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim1 // 2, 1)
        )

    def forward(self, h: torch.Tensor, return_attention: bool = True):
        """
        Forward pass over a bag of patch embeddings.
        
        Args:
            h (torch.Tensor): Bag of instance embeddings of shape (N, input_dim) or (1, N, input_dim).
            return_attention (bool): Whether to return the attention weights.
            
        Returns:
            dict containing:
                - 'isup_logits': (1, n_isup_classes)
                - 'hrd_logit': (1, 1)
                - 'bcr_risk': (1, 1)
                - 'attention_weights': (N, 1) if return_attention else None
                - 'slide_embedding': (1, hidden_dim1) pooled slide representation
        """
        # Ensure 2D shape (N, input_dim)
        if h.dim() == 3:
            h = h.squeeze(0)

        # A: (N, 1), h_proj: (N, hidden_dim1)
        A, h_proj = self.attention_net(h.float())

        # Gated attention pooling: Softmax across patches
        # A_weights shape: (N, 1)
        A_weights = F.softmax(A, dim=0)

        # Slide-level representation M: (1, hidden_dim1)
        M = torch.mm(A_weights.transpose(0, 1), h_proj)

        # Task-specific predictions
        isup_logits = self.isup_head(M)
        hrd_logit = self.hrd_head(M)
        bcr_risk = self.bcr_head(M)

        out = {
            "isup_logits": isup_logits,
            "hrd_logit": hrd_logit,
            "bcr_risk": bcr_risk,
            "slide_embedding": M,
        }

        if return_attention:
            out["attention_weights"] = A_weights

        return out


class MultiTaskLoss(nn.Module):
    """
    Combined loss for Multi-Task MIL:
      L = w_isup * CrossEntropy(isup) + w_hrd * BCEWithLogits(hrd) + w_bcr * CoxNegativeLogLikelihood(bcr)
    """
    def __init__(
        self,
        weight_isup: float = 1.0,
        weight_hrd: float = 1.0,
        weight_bcr: float = 0.5,
    ):
        super(MultiTaskLoss, self).__init__()
        self.w_isup = weight_isup
        self.w_hrd = weight_hrd
        self.w_bcr = weight_bcr
        self.ce = nn.CrossEntropyLoss()
        self.bce = nn.BCEWithLogitsLoss()

    def forward(
        self,
        preds: dict,
        isup_target: torch.Tensor = None,
        hrd_target: torch.Tensor = None,
        bcr_times: torch.Tensor = None,
        bcr_events: torch.Tensor = None,
    ):
        total_loss = torch.tensor(0.0, device=preds["isup_logits"].device)
        losses = {}

        if isup_target is not None:
            l_isup = self.ce(preds["isup_logits"], isup_target)
            total_loss = total_loss + self.w_isup * l_isup
            losses["loss_isup"] = l_isup.item()

        if hrd_target is not None:
            l_hrd = self.bce(preds["hrd_logit"].view(-1), hrd_target.view(-1).float())
            total_loss = total_loss + self.w_hrd * l_hrd
            losses["loss_hrd"] = l_hrd.item()

        if bcr_times is not None and bcr_events is not None and bcr_events.sum() > 0:
            from models.CoxMIL import cox_negative_partial_log_likelihood
            l_bcr = cox_negative_partial_log_likelihood(
                preds["bcr_risk"].view(-1), bcr_times.view(-1), bcr_events.view(-1)
            )
            total_loss = total_loss + self.w_bcr * l_bcr
            losses["loss_bcr"] = l_bcr.item()

        losses["loss_total"] = total_loss
        return total_loss, losses
