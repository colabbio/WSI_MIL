"""
CoxMIL: Deep Multiple Instance Learning for Survival / Time-to-Event Analysis.

Computes slide-level risk scores from WSI patch embeddings using Attention MIL,
optimized via the Cox Proportional Hazards Negative Partial Log-Likelihood loss
(Breslow approximation for ties).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from models.AttentionModel import GatedAttentionModel


def cox_negative_partial_log_likelihood(
    risk_scores: torch.Tensor,
    times: torch.Tensor,
    events: torch.Tensor,
    eps: float = 1e-7,
) -> torch.Tensor:
    """
    Computes the Negative Partial Log-Likelihood of the Cox Proportional Hazards model
    with Breslow approximation for ties.
    
    Args:
        risk_scores (torch.Tensor): 1D tensor of predicted log-hazard ratios (theta_i = g(X_i)).
        times (torch.Tensor): 1D tensor of survival / follow-up times (T_i).
        events (torch.Tensor): 1D binary tensor of event indicators (E_i = 1 if uncensored / event occurred, 0 if censored).
        eps (float): Numerical stability epsilon.
        
    Returns:
        torch.Tensor: Scalar negative partial log-likelihood.
    """
    if events.sum() == 0:
        return torch.tensor(0.0, device=risk_scores.device, requires_grad=True)

    # Sort descending by survival time
    order = torch.argsort(times, descending=True)
    sorted_risk = risk_scores[order]
    sorted_events = events[order].float()

    # Hazard: exp(risk)
    # Log-sum-exp trick for numerical stability over the risk set R(t_i)
    # R(t_i) contains all patients with T_j >= T_i (all indices <= i in descending order)
    # log_cumsum_exp: log(sum_{j <= i} exp(risk_j)) in reverse, i.e., cumsum backwards
    
    # In descending order, risk set for patient i consists of indices 0 through i
    # log(cumsum(exp(sorted_risk)))
    log_risk_set = torch.logcumsumexp(sorted_risk, dim=0)

    # Partial log-likelihood: sum_i E_i * (risk_i - log(sum_{j in R_i} exp(risk_j)))
    uncensored_likelihood = sorted_events * (sorted_risk - log_risk_set)
    num_events = sorted_events.sum()

    nll = -torch.sum(uncensored_likelihood) / (num_events + eps)
    return nll


class CoxLoss(nn.Module):
    """
    Negative Partial Log-Likelihood Loss for Cox Proportional Hazards Model.
    """
    def __init__(self, eps: float = 1e-7):
        super(CoxLoss, self).__init__()
        self.eps = eps

    def forward(self, risk_scores: torch.Tensor, times: torch.Tensor, events: torch.Tensor):
        return cox_negative_partial_log_likelihood(risk_scores, times, events, self.eps)


class CoxMIL(nn.Module):
    """
    Attention-based Multiple Instance Learning for Survival Analysis (Cox-MIL).
    
    Args:
        input_dim (int): Feature dimension of patch embeddings.
        hidden_dim1 (int): Projection dimension for patch features.
        hidden_dim2 (int): Latent dimension for gated attention computation.
        dropout_rate (float): Dropout probability.
    """
    def __init__(
        self,
        input_dim: int = 2048,
        hidden_dim1: int = 512,
        hidden_dim2: int = 256,
        dropout_rate: float = 0.25,
    ):
        super(CoxMIL, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim1 = hidden_dim1

        # Gated Attention Backbone
        self.attention_net = GatedAttentionModel(
            input_dim=input_dim,
            hidden_dim1=hidden_dim1,
            hidden_dim2=hidden_dim2,
            dropout_rate=dropout_rate,
            classes=1
        )

        # Survival Risk Head (outputs log-hazard ratio theta)
        self.risk_head = nn.Sequential(
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
            return_attention (bool): Whether to return patch attention weights.
            
        Returns:
            dict containing:
                - 'risk_score': (1, 1) scalar log-hazard ratio
                - 'slide_embedding': (1, hidden_dim1) pooled slide representation
                - 'attention_weights': (N, 1) if return_attention else None
        """
        if h.dim() == 3:
            h = h.squeeze(0)

        A, h_proj = self.attention_net(h.float())
        A_weights = F.softmax(A, dim=0)

        # Attention pooling: (1, hidden_dim1)
        M = torch.mm(A_weights.transpose(0, 1), h_proj)

        # Risk score (log hazard ratio)
        risk = self.risk_head(M)

        out = {
            "risk_score": risk,
            "slide_embedding": M,
        }

        if return_attention:
            out["attention_weights"] = A_weights

        return out
