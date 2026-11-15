import pandas as pd
import numpy as np
import os
import h5py
import argparse
from tqdm import tqdm
import csv
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
try:
    from snorkel.classification import cross_entropy_with_probs
except ImportError:
    # Fallback to PyTorch native cross entropy if snorkel is not installed
    def cross_entropy_with_probs(logits, target_probs):
        log_probs = torch.log_softmax(logits, dim=-1)
        return -torch.sum(target_probs * log_probs, dim=-1).mean()

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_auc_score
from concurrent.futures import ThreadPoolExecutor, as_completed

 
from models.MIL_model import MIL_SB
from AttentionDataset import AttentionDataset, InstanceDataset, instance_dataloader
from models.AttentionModel import AttentionModel, GatedAttentionModel
from models.MultiTaskMIL import MultiTaskMIL, MultiTaskLoss
from models.CoxMIL import CoxMIL, CoxLoss, cox_negative_partial_log_likelihood
from models.TransMIL import TransMIL
from fold_split import stratified_k_fold_split, stratified_train_test_split

import torch.multiprocessing
if torch.cuda.is_available():
    torch.cuda.manual_seed(42)
torch.manual_seed(42)
torch.backends.cudnn.benchmark = True
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class color:
    reset = '\033[0m'
    BOLD = '\033[01m'
    red = '\033[31m'
    green = '\033[32m'
    yellow = '\033[33m'
    blue = '\033[34m'


def patient_id(row):
    if "patient" in row:
        return row.split("_node")[0]

    elif "TCGA" in row:
       return row[:15]
    else:
        return row


# =============================================================================
# [NOTA PARA LORENZO / PENDIENTE DE REVISIÓN Y CONFIRMACIÓN]:
# Función factoría modular para construir el modelo y su función de pérdida.
# Permite intercambiar de forma limpia:
#   1. 'clam': MIL_SB (tu modelo con clustering de parches y 2-branch instance loss).
#   2. 'abmil': Gated Attention clásico (Ilse et al., 2018) con Cross-Entropy o BCE.
#   3. 'multitask': MultiTaskMIL para predecir ISUP (5 clases) + HRD (binario) + BCR (Cox risk).
#   4. 'transmil': TransMIL con auto-atención Transformer y generador PPEG de contexto espacial.
#   5. 'coxmil': CoxMIL para análisis de supervivencia directa con pérdida de Cox Breslow.
#
# Lorenzo: Revisa si los hidden_dim y dropouts por defecto se adecúan a tu tesis.
# =============================================================================
def build_model_and_criterion(args):
    """
    Construye e inicializa el modelo MIL y la función de pérdida seleccionada por --model_type.
    """
    model_type = getattr(args, "model_type", "clam").lower()
    input_dim = args.input_dim
    hidden_dim1 = args.hidden_dim1
    hidden_dim2 = args.hidden_dim2
    dropout = args.dropout
    n_classes = getattr(args, "n_classes", 2)

    if model_type == "clam":
        instance_loss = cross_entropy_with_probs
        model = MIL_SB(
            instance_loss=instance_loss,
            k=args.k,
            classes=n_classes,
            input_dim=input_dim,
            hidden_dim1=hidden_dim1,
            hidden_dim2=hidden_dim2,
            dropout_rate=dropout,
            k_selection=args.tile_selection
        )
        criterion = nn.CrossEntropyLoss()

    elif model_type == "abmil":
        # abMIL estándar con Gated Attention Pooling y cabezal de clasificación
        class StandardGatedABMIL(nn.Module):
            def __init__(self, in_d, h_d1, h_d2, drop, n_cls):
                super().__init__()
                self.n_classes = n_cls
                self.attention_net = GatedAttentionModel(
                    input_dim=in_d, hidden_dim1=h_d1, hidden_dim2=h_d2,
                    dropout_rate=drop, classes=1
                )
                self.classifier = nn.Linear(h_d1, n_cls)

            def forward(self, h, pos=None, label=None, instance_eval=False, **kwargs):
                if h.dim() == 3:
                    h = h.squeeze(0)
                A, h_proj = self.attention_net(h.float())
                A_weights = torch.softmax(A, dim=0)
                M = torch.mm(A_weights.transpose(0, 1), h_proj)
                logits = self.classifier(M)
                probs = torch.softmax(logits, dim=1)
                y_hat = torch.topk(logits, 1, dim=1)[1]
                # Formato compatible con el bucle: (logits, Y_prob, Y_hat, A_raw, results_dict, h)
                return logits, probs, y_hat, A, {}, h_proj

        model = StandardGatedABMIL(input_dim, hidden_dim1, hidden_dim2, dropout, n_classes)
        criterion = nn.CrossEntropyLoss()

    elif model_type == "multitask":
        # MultiTaskMIL: predice ISUP, HRD y BCR conjuntamente
        model = MultiTaskMIL(
            input_dim=input_dim,
            hidden_dim1=hidden_dim1,
            hidden_dim2=hidden_dim2,
            dropout_rate=dropout,
            n_isup_classes=n_classes
        )
        criterion = MultiTaskLoss(
            weight_isup=getattr(args, "weight_isup", 1.0),
            weight_hrd=getattr(args, "weight_hrd", 0.8),
            weight_bcr=getattr(args, "weight_bcr", 0.5)
        )

    elif model_type == "transmil":
        # TransMIL: Transformer contextual con PPEG
        class TransMILWrapper(nn.Module):
            def __init__(self, in_d, h_d, n_cls, drop):
                super().__init__()
                self.n_classes = n_cls
                self.transmil = TransMIL(
                    input_dim=in_d, hidden_dim=h_d, n_classes=n_cls, dropout=drop
                )

            def forward(self, h, pos=None, label=None, instance_eval=False, **kwargs):
                if h.dim() == 2:
                    h = h.unsqueeze(0)
                out = self.transmil(h, return_features=True)
                logits = out["logits"]
                probs = torch.softmax(logits, dim=1)
                y_hat = torch.topk(logits, 1, dim=1)[1]
                dummy_A = torch.zeros((h.shape[1], 1), device=h.device)
                return logits, probs, y_hat, dummy_A, {}, out.get("slide_embedding", h)

        model = TransMILWrapper(input_dim, hidden_dim1, n_classes, dropout)
        criterion = nn.CrossEntropyLoss()

    elif model_type == "coxmil":
        # CoxMIL: Análisis de supervivencia
        class CoxMILWrapper(nn.Module):
            def __init__(self, in_d, h_d1, h_d2, drop):
                super().__init__()
                self.n_classes = 1
                self.coxmil = CoxMIL(
                    input_dim=in_d, hidden_dim1=h_d1, hidden_dim2=h_d2, dropout_rate=drop
                )

            def forward(self, h, pos=None, label=None, instance_eval=False, **kwargs):
                out = self.coxmil(h, return_attention=True)
                risk = out["risk_score"]
                dummy_prob = torch.sigmoid(risk)
                dummy_hat = (dummy_prob > 0.5).long()
                A = out["attention_weights"] if out["attention_weights"] is not None else torch.zeros((h.shape[0], 1), device=h.device)
                return risk, dummy_prob, dummy_hat, A, {}, out["slide_embedding"]

        model = CoxMILWrapper(input_dim, hidden_dim1, hidden_dim2, dropout)
        criterion = CoxLoss()

    elif model_type == "protomil":
        from models.ProtoMIL import ProtoMIL
        class ProtoMILWrapper(nn.Module):
            def __init__(self, in_d, h_d, n_proto, n_cls, drop):
                super().__init__()
                self.n_classes = n_cls
                self.protomil = ProtoMIL(
                    input_dim=in_d,
                    hidden_dim=h_d,
                    n_prototypes=n_proto,
                    n_isup_classes=n_cls,
                    dropout_rate=drop
                )

            def forward(self, h, pos=None, label=None, instance_eval=False, **kwargs):
                if h.dim() == 3:
                    h = h.squeeze(0)
                out = self.protomil(h)
                logits = out["isup_logits"]
                probs = torch.softmax(logits, dim=1)
                y_hat = torch.topk(logits, 1, dim=1)[1]
                A_raw = out["patch_prototype_similarities"]
                return logits, probs, y_hat, A_raw, out, out["slide_prototype_profile"]

        n_prototypes = getattr(args, "n_prototypes", 16)
        model = ProtoMILWrapper(input_dim, hidden_dim1, n_prototypes, n_classes, dropout)
        criterion = nn.CrossEntropyLoss()

    else:
        raise ValueError(f"Unknown model_type: '{model_type}'. Choose from ['clam', 'abmil', 'multitask', 'transmil', 'coxmil', 'protomil']")

    return model, criterion


class Trainer:
    def __init__(self, criterion, batch_save, model, train_loader, val_loader, save_path, train_dataset, val_dataset,
                 num_epochs=200, patience=20,
                 positional_embed=False, model_type="clam"):
        self.model = model
        self.model_type = model_type.lower()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"device {self.device} | Architecture: {self.model_type}")
        self.model = model.to(self.device)

        self.train_dataset = train_dataset
        self.val_dataset = val_dataset

        self.positional_embed = positional_embed
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.save_path = save_path
        self.num_epochs = num_epochs
        self.patience = patience
        self.batch_save = batch_save
        self.criterion = criterion.to(self.device) if hasattr(criterion, 'to') else criterion
        print(f"Model Device: {next(self.model.parameters()).device}")
        print(f"Criterion Device: {getattr(self.criterion, 'device', 'N/A')}")

        self.optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-4, weight_decay=1e-5)
        self.best_val_loss = float('inf')
        self.paths = {
            "train": os.path.join(self.save_path, "training"),
            "val": os.path.join(self.save_path, "validation"),
            "weights": os.path.join(self.save_path, "weights"),
        }
        self.training_log_path = os.path.join(self.save_path, 'training_log.csv')
        self.patience_counter = 0
        self._setup_directories()
        self.best_weights_path = ""

    def _setup_directories(self):
        # saving outputs
        for path in self.paths.values():
            os.makedirs(path, exist_ok=True)
        with open(self.training_log_path, 'w', newline='') as f:
            pass

    def _save_patient_data(self, dataset, phase):
        phase_path = self.paths[phase]
        print(f"Initializing {phase} directories")
        for batch in tqdm(dataset, total = len(dataset)):
            bags, positional, labels, x, y, tile_paths, scales, original_size, patient_id = batch
            patient_dir = os.path.join(phase_path, patient_id[0])
            patient_file = os.path.join(patient_dir, f"{patient_id[0]}.csv")
            os.makedirs(patient_dir, exist_ok=True)

            if patient_id[0] == "error" or os.path.isfile(patient_file):
                continue
            temp = pd.DataFrame()
            x = np.array(x.squeeze()).flatten()
            y = np.array(y.squeeze()).flatten()
            tile_paths = np.array(tile_paths).flatten()  
            scales = np.repeat(scales, len(x))
            original_size = np.repeat(original_size, len(x)) 
            temp["x"] = x
            temp["y"] = y
            temp["tile_paths"] = tile_paths
            temp["scale"] = scales
            temp["size"] = original_size

            temp.to_csv(patient_file, index=False)
    
            temp.to_csv(patient_file, index=False)
    def _calculate_accuracy(self, outputs, labels):
        preds = torch.argmax(outputs, dim=1)
        correct = (preds == labels).sum().item()
        return correct / len(labels)

    def train(self):
        # train loop
        self._save_patient_data(self.train_dataset, "train")
        self._save_patient_data(self.val_dataset, "val")
        print("Training")
        with open(self.training_log_path, mode='a', newline='') as file:
            writer = csv.writer(file)

            # Write header only if the file is empty
            if file.tell() == 0:
                writer.writerow(['Epoch', 'Train Loss', 'Train Acc', 'Val Loss', 'Val Acc', "Weights Saved"])

            for epoch in range(self.num_epochs):
                train_loss, train_accuracy = self._train_epoch(epoch)
                val_loss, val_accuracy = self._validate_epoch(epoch)

                print(f"Epoch {epoch + 1}/{self.num_epochs} | Train Loss: {train_loss:.4f} | "
                      f"Train Acc: {train_accuracy:.4f} | Val Loss: {val_loss:.4f} | Val Acc: {val_accuracy:.4f}")

                saved = "YES" if self._save_best_model(val_loss, epoch) else None
                if saved == "YES":
                    self.best_weights_path = os.path.join(self.paths["weights"], f"weights_epoch_{epoch + 1}.pth")
                    print(f"Model saved with val_loss: {val_loss:.4f}")

                writer.writerow([epoch + 1, train_loss, train_accuracy, val_loss, val_accuracy, saved])
                file.flush()

                if self.patience_counter >= self.patience:
                    print("Early stopping triggered.")
                    break

    def _train_epoch(self, epoch):
        running_loss, running_correct, total, inst_count, train_inst_loss = 0.0, 0, 0, 0, 0

        for bags, positional, labels, x, y, tile_paths, scales, original_size, patient_id in tqdm(self.train_loader,
                                                                                                  desc=f"Epoch {epoch + 1} [Train]"):
            if patient_id[0] == "error":
                continue
            try:
                bags = bags.to(self.device, non_blocking=True)
                positional = positional.to(self.device, non_blocking=True)
                labels = labels.to(self.device, non_blocking=True)
                self.model.train()

                if labels.dtype != torch.long:
                    labels = labels.long()

                self.optimizer.zero_grad()

                # =====================================================================
                # [NOTA PARA LORENZO / PENDIENTE DE REVISIÓN Y CONFIRMACIÓN]:
                # Invocación unificada del modelo según architecture:
                # - MultiTaskMIL devuelve un diccionario con 'isup_logits', 'hrd_logit', 'bcr_risk'.
                # - CLAM (MIL_SB), abMIL, TransMIL y CoxMIL devuelven la tupla compatible:
                #   (logits, Y_prob, Y_hat, A_raw, results_dict, h).
                # Lorenzo: si en MultiTaskMIL deseas pasar variables adicionales como
                # target HRD o tiempos BCR desde un DataLoader extendido, aquí es el punto.
                # =====================================================================
                if self.model_type == "multitask":
                    preds = self.model(bags, return_attention=True)
                    logits = preds["isup_logits"]
                    A_raw = preds.get("attention_weights", torch.zeros((bags.shape[1] if bags.dim() > 1 else bags.shape[0], 1), device=self.device))
                    h = preds.get("slide_embedding", logits)
                    results_dict = {}

                    # MultiTaskLoss calcula la combinación ponderada
                    loss, losses_dict = self.criterion(preds, isup_target=labels)
                    loss.backward()

                elif self.model_type == "coxmil":
                    risk, Y_prob, Y_hat, A_raw, results_dict, h = self.model(bags)
                    # Cox Loss requiere tiempos y eventos censurados
                    # NOTA LORENZO: por defecto si solo se dispone de 'labels', se asume label como evento binario
                    dummy_times = labels.float() + 1.0
                    loss = self.criterion(risk.view(-1), dummy_times.view(-1), labels.view(-1))
                    logits = risk
                    loss.backward()

                else:
                    # 'clam', 'abmil', 'transmil'
                    pos_arg = positional if self.positional_embed else None
                    logits, Y_prob, Y_hat, A_raw, results_dict, h = self.model(
                        bags, pos=pos_arg, label=labels, instance_eval=(self.model_type == "clam")
                    )

                    n_cls = getattr(self.model, "n_classes", 2)
                    labels_one_hot = nn.functional.one_hot(labels, num_classes=n_cls).float().to(self.device)
                    logits = logits.to(self.device)
                    loss = self.criterion(logits, labels_one_hot)

                    # Si es CLAM con instance loss de clustering
                    if self.model_type == "clam" and "instance_loss" in results_dict:
                        instance_loss = results_dict["instance_loss"].item()
                        if not torch.isnan(torch.tensor(instance_loss)):
                            train_inst_loss += instance_loss
                            c1 = 0.7
                            total_loss = c1 * loss + (1 - c1) * instance_loss
                            total_loss.backward()
                        else:
                            loss.backward()
                    else:
                        loss.backward()

                self.optimizer.step()
                self.optimizer.zero_grad()
                running_loss += loss.item()
                if logits.dim() > 1 and logits.shape[1] > 1:
                    running_correct += (logits.argmax(dim=1) == labels).sum().item()
                else:
                    running_correct += ((torch.sigmoid(logits.view(-1)) > 0.5).long() == labels.view(-1)).sum().item()
                total += labels.size(0)

                # Guardado de mapas de atención periódicos para microscopio e interpretabilidad
                if epoch % self.batch_save == 0 and self.model_type in ["clam", "abmil", "multitask", "coxmil"]:
                    self._save_attention(epoch, A_raw, bags, positional, patient_id, h)
            except Exception as e:
                print(f"[WARN] Error in train batch for {patient_id}: {e}")
                continue

        train_loss = running_loss / max(len(self.train_loader), 1)
        train_accuracy = running_correct / max(total, 1)
        return train_loss, train_accuracy

    def _validate_epoch(self, epoch):
        running_loss, running_correct, total = 0.0, 0, 0
        with torch.inference_mode():
            self.model.eval()
            for bags, positional, labels, x, y, tile_paths, scales, original_size, patient_id in tqdm(self.val_loader,
                                                                                                      desc=f"Epoch {epoch + 1} [Val]"):
                if patient_id[0] == "error":
                    continue
                try:
                    bags = bags.to(self.device, non_blocking=True)
                    positional = positional.to(self.device, non_blocking=True)
                    labels = labels.to(self.device, non_blocking=True)

                    if self.model_type == "multitask":
                        preds = self.model(bags, return_attention=True)
                        logits = preds["isup_logits"]
                        A_raw = preds.get("attention_weights", torch.zeros((bags.shape[1] if bags.dim() > 1 else bags.shape[0], 1), device=self.device))
                        h = preds.get("slide_embedding", logits)
                        loss, _ = self.criterion(preds, isup_target=labels)

                    elif self.model_type == "coxmil":
                        risk, Y_prob, Y_hat, A_raw, results_dict, h = self.model(bags)
                        dummy_times = labels.float() + 1.0
                        loss = self.criterion(risk.view(-1), dummy_times.view(-1), labels.view(-1))
                        logits = risk

                    else:
                        pos_arg = positional if self.positional_embed else None
                        logits, Y_prob, Y_hat, A_raw, results_dict, h = self.model(
                            bags, pos=pos_arg, label=labels, instance_eval=(self.model_type == "clam")
                        )
                        n_cls = getattr(self.model, "n_classes", 2)
                        labels_one_hot = nn.functional.one_hot(labels, num_classes=n_cls).float()
                        loss = self.criterion(logits, labels_one_hot)

                    running_loss += loss.item()
                    if logits.dim() > 1 and logits.shape[1] > 1:
                        running_correct += (logits.argmax(dim=1) == labels).sum().item()
                    else:
                        running_correct += ((torch.sigmoid(logits.view(-1)) > 0.5).long() == labels.view(-1)).sum().item()
                    total += labels.size(0)

                    if epoch % self.batch_save == 0 and self.model_type in ["clam", "abmil", "multitask", "coxmil"]:
                        self._save_attention(epoch, A_raw, bags, positional, patient_id, h, phase="val")
                except Exception as e:
                    print(f"[WARN] Error in val batch for {patient_id}: {e}")
                    continue

        val_loss = running_loss / max(len(self.val_loader), 1)
        val_accuracy = running_correct / max(total, 1)
        return val_loss, val_accuracy

    def _save_best_model(self, val_loss, epoch):
        if val_loss < self.best_val_loss:
            self.best_val_loss = val_loss
            self.patience_counter = 0
            torch.save(self.model.state_dict(), os.path.join(self.paths["weights"], f"weights_epoch_{epoch + 1}.pth"))
            return True
        self.patience_counter += 1
        return False

    def _instance_eval(self, bags, positional):
        if self.model_type != "clam":
            # Para modelos no CLAM (como abMIL o MultiTaskMIL), la cota PAC y atención
            # se calculan directamente a partir del attention pooling
            return [], []

        instances = instance_dataloader(bags)
        instance_labels = []
        instance_probs = []

        self.model.eval()
        if self.positional_embed:
            instance_pos = instance_dataloader(positional)
            with torch.no_grad():
                for instance, position in zip(instances, instance_pos):
                    instance = instance.unsqueeze(1).to(self.device, non_blocking=True)
                    position = position.unsqueeze(1).to(self.device, non_blocking=True)
                    logits_inst, Y_prob_inst, Y_hat_inst, A_raw_inst, dict, h = self.model(instance, pos=position, instance_eval=False)
                    instance_labels.append(Y_hat_inst.cpu().item() if isinstance(Y_hat_inst, torch.Tensor) else Y_hat_inst)
                    instance_probs.append(Y_prob_inst[:, 1].cpu().item() if isinstance(Y_prob_inst, torch.Tensor) else Y_prob_inst[:, 1])
        else:
            with torch.no_grad():
                for instance in instances:
                    instance = instance.unsqueeze(1).to(self.device, non_blocking=True)
                    logits_inst, Y_prob_inst, Y_hat_inst, A_raw_inst, dict, h = self.model(instance, pos=None, instance_eval=False)
                    instance_labels.append(Y_hat_inst.cpu().item() if isinstance(Y_hat_inst, torch.Tensor) else Y_hat_inst)
                    instance_probs.append(Y_prob_inst[:, 1].cpu().item() if isinstance(Y_prob_inst, torch.Tensor) else Y_prob_inst[:, 1])
        return instance_labels, instance_probs

    def _save_attention(self, epoch, A_raw, bags, positional, patient_id, h, phase="train"):
        if phase == "train":
            patient_path = os.path.join(self.paths["train"], patient_id[0], f"{patient_id[0]}.csv")
            cluster_path = os.path.join(self.paths["train"], patient_id[0], f"{patient_id[0]}_cluster.h5")
        else:
            patient_path = os.path.join(self.paths["val"], patient_id[0], f"{patient_id[0]}.csv")
            cluster_path = os.path.join(self.paths["val"], patient_id[0], f"{patient_id[0]}_cluster.h5")
        patient_csv = pd.read_csv(patient_path)
        attentions = A_raw.squeeze(dim=1).squeeze().detach().cpu().numpy()
        if attentions.ndim == 0:
            attentions = np.array([attentions])

        patient_csv[f"attention_{epoch + 1}"] = attentions
        instance_labels, instance_probs = self._instance_eval(bags, positional)

        if len(instance_labels) > 0 and len(instance_probs) > 0:
            instance_labels = [label.cpu().item() if isinstance(label, torch.Tensor) else label for label in
                               instance_labels]
            instance_probs = [prob.cpu().item() if isinstance(prob, torch.Tensor) else prob for prob in instance_probs]

            patient_csv[f"instance_label_{epoch + 1}"] = np.array(instance_labels)
            patient_csv[f"instance_prob_{epoch + 1}"] = np.array(instance_probs)

            # Calculate Polarized Attentional Certainty (PAC): PAC_i = A_i * (2 * P_i - 1)
            pac_scores = attentions * (2 * np.array(instance_probs) - 1)
            patient_csv[f"pac_{epoch + 1}"] = pac_scores

        patient_csv.to_csv(patient_path, index=False)

        # save cluster info
        with h5py.File(cluster_path, "a") as file:
            if f'cluster_epoch_{epoch}' in file.keys():
                del file[f'cluster_epoch_{epoch}']
            file.create_dataset(f'cluster_epoch_{epoch}', data=h.detach().cpu().numpy())


def evaluate(model, dataloader, instance_eval=False, position=False, model_type="clam"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    all_preds, all_probs, all_labels = [], [], []

    with torch.inference_mode():
        for bags, positional, labels, _, _, _, _, _, _ in dataloader:
            try:
                bags = bags.to(device)
                labels = labels.to(device)
                positional = positional.to(device)

                if model_type == "multitask":
                    preds = model(bags, return_attention=False)
                    logits = preds["isup_logits"]
                    if logits.shape[1] > 1:
                        probs = torch.softmax(logits, dim=1)
                        y_hat = torch.topk(logits, 1, dim=1)[1]
                        prob_val = probs[:, 1].cpu().numpy() if probs.shape[1] > 1 else probs[:, 0].cpu().numpy()
                    else:
                        prob_val = torch.sigmoid(logits).cpu().numpy().flatten()
                        y_hat = (prob_val > 0.5).astype(int)

                elif model_type == "coxmil":
                    out = model(bags)
                    risk = out[0] if isinstance(out, tuple) else out["risk_score"]
                    prob_val = torch.sigmoid(risk).cpu().numpy().flatten()
                    y_hat = (prob_val > 0.5).astype(int)

                else:
                    pos_arg = positional if position else None
                    logits, Y_prob, Y_hat, _, _, _ = model(bags, pos=pos_arg, label=labels, instance_eval=instance_eval)
                    if Y_prob.shape[1] > 1:
                        prob_val = Y_prob[:, 1].cpu().numpy()
                    else:
                        prob_val = Y_prob.cpu().numpy().flatten()

                all_probs.extend(prob_val)
                all_preds.extend(Y_hat.cpu().numpy().flatten() if isinstance(Y_hat, torch.Tensor) else y_hat.flatten())
                all_labels.extend(labels.cpu().numpy().flatten())
            except Exception as e:
                continue

    metrics = {
        'accuracy': accuracy_score(all_labels, all_preds) if len(all_labels) > 0 else 0.0,
        'precision': precision_score(all_labels, all_preds, zero_division=0) if len(all_labels) > 0 else 0.0,
        'recall': recall_score(all_labels, all_preds, zero_division=0) if len(all_labels) > 0 else 0.0,
        'f1_score': f1_score(all_labels, all_preds, zero_division=0) if len(all_labels) > 0 else 0.0,
        'roc_auc': roc_auc_score(all_labels, all_probs) if len(np.unique(all_labels)) > 1 else 0.0,
        'confusion_matrix': confusion_matrix(all_labels, all_preds) if len(all_labels) > 0 else None
    }
    return metrics


def get_args():
    parser = argparse.ArgumentParser()
    # =========================================================================
    # [NOTA PARA LORENZO / PENDIENTE DE REVISIÓN Y CONFIRMACIÓN]:
    # Añadimos el selector '--model_type' para permitir entrenar y comparar de forma
    # limpia las diferentes arquitecturas MIL desarrolladas en ColabBio:
    #   - 'clam': MIL_SB con instance clustering (tu modelo base original por defecto).
    #   - 'abmil': Gated Attention clásico (Ilse et al., 2018).
    #   - 'multitask': MultiTaskMIL prediciendo ISUP + HRD/DDR + BCR Survival.
    #   - 'transmil': Transformer con PPEG y correlaciones espaciales (Shao et al., 2021).
    #   - 'coxmil': Análisis de supervivencia directo (Cox Proportional Hazards).
    #
    # Por favor, confirma si los valores por defecto (ej. n_classes=2, lr=1e-4) encajan
    # con tu diseño experimental o si deseas añadir hiperparámetros adicionales.
    # =========================================================================
    parser.add_argument("--model_type", default="clam",
                        choices=["clam", "abmil", "multitask", "transmil", "coxmil", "protomil"],
                        help="Arquitectura de red MIL a entrenar (clam, abmil, multitask, transmil, coxmil, protomil)")
    parser.add_argument("--n_prototypes", default=16, type=int,
                        help="Número de prototipos morfológicos latentes para ProtoMIL (default: 16)")
    parser.add_argument("--n_classes", default=2, type=int,
                        help="Número de clases objetivo (ej. 2 para binario, 5 para ISUP Grade Group)")
    parser.add_argument("--weight_isup", default=1.0, type=float,
                        help="Peso de la pérdida de gradación ISUP (MultiTaskMIL)")
    parser.add_argument("--weight_hrd", default=0.8, type=float,
                        help="Peso de la pérdida de mutación HRD/DDR (MultiTaskMIL)")
    parser.add_argument("--weight_bcr", default=0.5, type=float,
                        help="Peso de la pérdida de supervivencia BCR Cox (MultiTaskMIL)")
    parser.add_argument("--positional_embed", action="store_true")
    parser.add_argument("--testing_csv", default=None)
    parser.add_argument("--split_test", action="store_false",
                        help="flag if test is already pre split under a column called split")
    parser.add_argument("--training_output", default="/mnt/c/Users/loren/Masters/test")
    parser.add_argument("--epochs", default=200, type=int)
    parser.add_argument("--k", default=20, type=int)
    parser.add_argument("--tile_selection", default="CLAM")
    parser.add_argument("--patience", default=10, type=int)
    parser.add_argument("--input_dim", default=2048, type=int)
    parser.add_argument("--hidden_dim1", default=512, type=int)
    parser.add_argument("--hidden_dim2", default=256, type=int)
    parser.add_argument("--metadata_path", default="/mnt/c/Users/loren/Masters/colorectal2/patient_files.csv")
    parser.add_argument("--cv", action="store_true")
    parser.add_argument("--fold_number", default=5, type=int)
    parser.add_argument("--dropout", default=0.35, type=float)
    parser.add_argument("--k_causal", default=20, type=int)
    parser.add_argument("--batch_save", default=5, type=int)
    parser.add_argument("--type_embed", default=None)
    parser.add_argument("--augment", action="store_true")
    return parser.parse_args()


def write_eval_crossval(fold_data, save_path, index):
    evaluation_df = pd.DataFrame(fold_data)
    metrics = {
        'accuracy': np.mean(evaluation_df["accuracy"].to_numpy()),
        'precision': np.mean(evaluation_df["precision"].to_numpy()),
        'recall': np.mean(evaluation_df["recall"].to_numpy()),
        'f1_score': np.mean(evaluation_df["f1_score"].to_numpy()),
        'roc_auc': np.mean(evaluation_df["roc_auc"].to_numpy()),
        'confusion_matrix': None
    }
    evaluation_df.loc[len(evaluation_df)] = metrics
    evaluation_df.index = index
    evaluation_df.to_csv(save_path)
    print(evaluation_df)


def main():
    args = get_args()
    # get the patient labels + ID
    label_data = pd.read_csv(args.metadata_path)
    label_data["sample_id"] = label_data["Patient ID"]
    label_data["patient_id"] = label_data["sample_id"].apply(patient_id)
    # train model with cross val
    # 1. reserve a test set (20%)
    if args.cv:
        train, test = stratified_train_test_split(data=label_data,
                                                  label_column='target',
                                                  patient_id_column='patient_id',
                                                  test_size=0.15,
                                                  random_state=42, split = args.split_test
                                                  )
        # 2. get folds from train CV set
        folds = stratified_k_fold_split(
            data=train,
            label_column='target',
            patient_id_column='patient_id',
            n_splits=args.fold_number,
            random_state=42
        )
        fold_metrics_val = []
        fold_metrics_testing = []
        index = []
        # 3. For fold, train model
        for i, (train_data, val_data) in enumerate(folds):
            print(f"\nFold {i + 1}")
            index.append(f"Folds {i + 1}")
            # get the loaders
            if args.augment:
                train_dataset = AttentionDataset(train_data.reset_index(), type_embed =args.type_embed )
            else:
                train_dataset = AttentionDataset(train_data.reset_index(), type_embed =args.type_embed, mode = "val" )

            val_dataset = AttentionDataset(val_data.reset_index(),type_embed =args.type_embed, mode = "val")
            pin_memory = torch.cuda.is_available()
            # get weightedRandom Sampler 
            class_counts = np.bincount(train_dataset.get_labels())
            class_weights = 1.0 / class_counts
            sample_weights = class_weights[train_dataset.get_labels()]
            sampler = WeightedRandomSampler(weights=sample_weights, num_samples=len(sample_weights), replacement=True)
            train_loader = DataLoader(train_dataset, batch_size=1, sampler = sampler,
                          pin_memory=pin_memory, num_workers=os.cpu_count())
            # train_loader = DataLoader(train_dataset, batch_size=1, sampler = sampler)

            # in order for val_loader
            val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, 
                        pin_memory=pin_memory, num_workers=os.cpu_count())
            # val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)
            # =================================================================
            # [NOTA PARA LORENZO / PENDIENTE DE REVISIÓN Y CONFIRMACIÓN]:
            # Instanciamos el modelo seleccionado mediante build_model_and_criterion(args)
            # en lugar de dejar cableado MIL_SB fijo.
            # =================================================================
            model, criterion = build_model_and_criterion(args)

            save_path = os.path.join(args.training_output, f"fold_{i + 1}")
            os.makedirs(save_path, exist_ok = True)
            # save test-train-val split
            train_data.loc[:,"split"] = "train"
            val_data.loc[:,"split"] = "val"
            test.loc[:,"split"] = "test"
            df_split = pd.concat([train_data, val_data, test])
            df_split.to_csv(os.path.join(save_path,"fold_splits.csv"), index = False)
            trainer = Trainer(criterion, args.batch_save, model, train_loader, val_loader, save_path,  DataLoader(train_dataset, batch_size=1,
                          pin_memory=pin_memory, num_workers=os.cpu_count()),
                              val_loader, num_epochs=args.epochs,
                              patience=args.patience, positional_embed=args.positional_embed,
                              model_type=args.model_type)

            trainer.train()
            print(f"Evaluating fold {i + 1}")
            best_weights = trainer.best_weights_path
            model.load_state_dict(torch.load(best_weights, weights_only=True))

            # now eval
            test_loader = DataLoader(AttentionDataset(test.reset_index(),type_embed = args.type_embed, mode = "test"), batch_size=1)

            results = evaluate(model, test_loader, instance_eval=(args.model_type == "clam"), position=args.positional_embed, model_type=args.model_type)
            fold_metrics_testing.append(results)
            results = evaluate(model, val_loader, instance_eval=(args.model_type == "clam"), position=args.positional_embed, model_type=args.model_type)
            fold_metrics_val.append(results)
            write_eval_crossval([fold_metrics_testing[i], fold_metrics_val[i]],
                                os.path.join(save_path, "test-val_eval.csv"), ["Testing", "Validation", "Average"])

        print("Done with cross-val")

        index.append("Average")
        # get and save average performance for testing
        print("Testing:")
        write_eval_crossval(fold_metrics_testing, os.path.join(args.training_output, "testing_fold_evaluations.csv"),
                            index)
        print("Validation:")
        write_eval_crossval(fold_metrics_val, os.path.join(args.training_output, "validation_fold_evaluations.csv"),
                            index)

    # now do regular train-test-split

    train, test = stratified_train_test_split(data=label_data,
                                              label_column='target',
                                              patient_id_column='patient_id',
                                              test_size=0.2,
                                              random_state=42, split = args.split_test
                                              )
    train, val = stratified_train_test_split(data=train,
                                            label_column='target',
                                            patient_id_column='patient_id',
                                            test_size=0.2,
                                            random_state=42
                                            )
    if args.augment:
                    train_dataset = AttentionDataset(train.reset_index(), type_embed =args.type_embed )
    else:
                    train_dataset = AttentionDataset(train.reset_index(), type_embed =args.type_embed, mode = "val" )  
    
    val_dataset = AttentionDataset(val.reset_index(),type_embed =args.type_embed, mode = "val")

    class_counts = np.bincount(train_dataset.get_labels())
    class_weights = 1.0 / class_counts
    sample_weights = class_weights[train_dataset.get_labels()]
    sampler = WeightedRandomSampler(weights=sample_weights, num_samples=len(sample_weights), replacement=True)
    # train_loader = DataLoader(train_dataset, batch_size=1, sampler = sampler,
    #                       pin_memory=pin_memory, num_workers=os.cpu_count())
    train_loader = DataLoader(train_dataset, batch_size=1, sampler = sampler)

    
    val_loader = DataLoader(val_dataset, batch_size=1)
    # =========================================================================
    # [NOTA PARA LORENZO / PENDIENTE DE REVISIÓN Y CONFIRMACIÓN]:
    # Instanciamos el modelo seleccionado mediante build_model_and_criterion(args)
    # para el entrenamiento estándar train/val/test split.
    # =========================================================================
    model, criterion = build_model_and_criterion(args)

    save_path = os.path.join(args.training_output, f"train-test-split")
    trainer = Trainer(criterion, args.batch_save, model, train_loader, val_loader, save_path, train_dataset,
                      val_dataset, num_epochs=args.epochs,
                      patience=args.patience, positional_embed=args.positional_embed,
                      model_type=args.model_type)
    trainer.train()
    best_weights = trainer.best_weights_path
    model.load_state_dict(torch.load(best_weights, weights_only=True))
    # now eval
    test_loader = DataLoader(AttentionDataset(test.reset_index(), type_embed=args.type_embed, mode="test"))

    results_test = evaluate(model, test_loader, instance_eval=(args.model_type == "clam"), position=args.positional_embed, model_type=args.model_type)
    results_val = evaluate(model, val_loader, instance_eval=(args.model_type == "clam"), position=args.positional_embed, model_type=args.model_type)

    write_eval_crossval([results_test, results_val], os.path.join(save_path, "test-val_eval.csv"),
                        ["Testing", "Validation", "Average"])


if __name__ == "__main__":
    main()
