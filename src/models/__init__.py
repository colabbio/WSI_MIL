"""
ColabBio WSI_MIL Models Package.

Architectures:
- AttentionModel: Standard attention pooling (Ilse et al., 2018).
- GatedAttentionModel: Gated attention pooling (Ilse et al., 2018).
- MultiTaskMIL: Multi-task abMIL architecture predicting ISUP grade, HRD status, and BCR recurrence risk.
- CoxMIL: Multiple instance learning for survival / time-to-event analysis with Cox partial likelihood.
- TransMIL: Transformer-based correlated MIL with self-attention and PPEG (Shao et al., 2021).
- MIL_SB: Single-branch CLAM model (lazy-loaded if snorkel is installed).
"""

from models.AttentionModel import AttentionModel, GatedAttentionModel
from models.MultiTaskMIL import MultiTaskMIL, MultiTaskLoss
from models.CoxMIL import CoxMIL, CoxLoss, cox_negative_partial_log_likelihood
from models.TransMIL import TransMIL
from models.ProtoMIL import ProtoMIL, ProtoMILLoss

try:
    from models.MIL_model import MIL_SB
except ImportError:
    MIL_SB = None

__all__ = [
    "AttentionModel",
    "GatedAttentionModel",
    "MultiTaskMIL",
    "MultiTaskLoss",
    "CoxMIL",
    "CoxLoss",
    "cox_negative_partial_log_likelihood",
    "TransMIL",
    "ProtoMIL",
    "ProtoMILLoss",
    "MIL_SB",
]
