import sys
import os
import torch
import pytest

# Ensure WSI_MIL/src is in Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from models import (
    AttentionModel,
    GatedAttentionModel,
    MultiTaskMIL,
    MultiTaskLoss,
    CoxMIL,
    CoxLoss,
    TransMIL,
    ProtoMIL,
    ProtoMILLoss,
)


def test_gated_attention_base():
    """Verify original GatedAttentionModel produces expected attention shape."""
    net = GatedAttentionModel(input_dim=2048, hidden_dim1=512, hidden_dim2=256, classes=1)
    x = torch.randn(100, 2048)  # 100 patches
    A, h = net(x)
    assert A.shape == (100, 1)
    assert h.shape == (100, 512)


def test_multitask_mil_forward_and_backward():
    """Verify MultiTaskMIL forward pass, output shapes, and combined loss backward pass."""
    model = MultiTaskMIL(input_dim=2048, hidden_dim1=512, hidden_dim2=256, n_isup_classes=5)
    loss_fn = MultiTaskLoss(weight_isup=1.0, weight_hrd=1.0, weight_bcr=0.5)

    # Synthetic bag of 50 patches
    bag = torch.randn(50, 2048)
    out = model(bag)

    assert out["isup_logits"].shape == (1, 5)
    assert out["hrd_logit"].shape == (1, 1)
    assert out["bcr_risk"].shape == (1, 1)
    assert out["slide_embedding"].shape == (1, 512)
    assert out["attention_weights"].shape == (50, 1)
    # Attention weights must sum to 1.0
    assert torch.isclose(out["attention_weights"].sum(), torch.tensor(1.0), atol=1e-4)

    # Multi-task loss computation
    isup_target = torch.tensor([3])  # Grade 3
    hrd_target = torch.tensor([1.0])  # Positive HRD mutation
    times = torch.tensor([720.0])  # Follow-up days
    events = torch.tensor([1.0])   # Recurrence observed

    total_loss, losses = loss_fn(
        out,
        isup_target=isup_target,
        hrd_target=hrd_target,
        bcr_times=times,
        bcr_events=events,
    )

    assert total_loss > 0.0
    assert "loss_isup" in losses
    assert "loss_hrd" in losses
    assert "loss_bcr" in losses

    # Check backpropagation
    total_loss.backward()
    for param in model.parameters():
        if param.requires_grad:
            assert param.grad is not None


def test_cox_mil():
    """Verify CoxMIL risk prediction and Cox Negative Partial Log-Likelihood loss."""
    model = CoxMIL(input_dim=2048, hidden_dim1=512, hidden_dim2=256)
    loss_fn = CoxLoss()

    # Batch of 4 synthetic patients
    patient_risks = []
    for _ in range(4):
        bag = torch.randn(30, 2048)
        out = model(bag)
        assert out["risk_score"].shape == (1, 1)
        patient_risks.append(out["risk_score"])

    batch_risks = torch.cat(patient_risks).view(-1)
    times = torch.tensor([300.0, 720.0, 120.0, 450.0])
    events = torch.tensor([0.0, 1.0, 1.0, 0.0])  # 2 censored, 2 events

    loss = loss_fn(batch_risks, times, events)
    assert loss > 0.0
    loss.backward()

    for param in model.parameters():
        if param.requires_grad:
            assert param.grad is not None


def test_transmil():
    """Verify TransMIL Transformer with PPEG and [CLS] classification."""
    model = TransMIL(input_dim=2048, hidden_dim=256, n_classes=5, depth=2, num_heads=4)
    bag = torch.randn(64, 2048)  # 64 patches

    out = model(bag, return_features=True)
    assert out["logits"].shape == (1, 5)
    assert out["slide_embedding"].shape == (1, 256)

    # Backward pass check
    criterion = torch.nn.CrossEntropyLoss()
    target = torch.tensor([2])
    loss = criterion(out["logits"], target)
    loss.backward()

    for param in model.parameters():
        if param.requires_grad:
            assert param.grad is not None


def test_protomil():
    """Verify ProtoMIL prototype learning, profile extraction, and loss backward pass."""
    model = ProtoMIL(
        input_dim=1024,
        hidden_dim=128,
        n_prototypes=16,
        n_isup_classes=5,
        gamma=0.5,
    )
    bag = torch.randn(80, 1024)  # 80 patches with 1024-dim features

    out = model(bag)

    # 1. Output shapes
    assert out["isup_logits"].shape == (1, 5)
    assert out["bcr_risk"].shape == (1, 1)
    assert out["pten_logit"].shape == (1, 1)
    assert out["slide_prototype_profile"].shape == (1, 16)
    assert out["patch_prototype_similarities"].shape == (80, 16)
    assert out["nearest_instances_indices"].shape == (16,)

    # 2. Similarities must be in (0, 1] range
    assert (out["patch_prototype_similarities"] > 0.0).all()
    assert (out["patch_prototype_similarities"] <= 1.0).all()

    # 3. Exemplar patch extraction
    exemplar_feats, min_dists = model.get_prototype_exemplars(bag)
    assert exemplar_feats.shape == (16, 1024)
    assert min_dists.shape == (16,)

    # 4. Compound Loss calculation & backward pass
    loss_fn = ProtoMILLoss(weight_isup=1.0, weight_diversity=0.1, weight_cluster=0.05)
    isup_target = torch.tensor([3])
    loss, loss_dict = loss_fn(out, isup_target=isup_target, prototypes=model.prototypes)

    assert loss > 0.0
    assert "isup_loss" in loss_dict
    assert "cluster_loss" in loss_dict
    assert "diversity_loss" in loss_dict

    loss.backward()

    # Verify gradients reach prototypes
    assert model.prototypes.grad is not None
    assert torch.isfinite(model.prototypes.grad).all()


if __name__ == "__main__":
    test_gated_attention_base()
    test_multitask_mil_forward_and_backward()
    test_cox_mil()
    test_transmil()
    test_protomil()
    print("All MIL model tests passed successfully!")
