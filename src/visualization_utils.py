import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

def calculate_pac_scores(attention_scores, instance_probs):
    """
    Calculate the Polarized Attentional Certainty (PAC) scores for a bag of instances.
    
    Formula:
        PAC_i = Attention_i * (2 * Probability_i - 1)
        
    Args:
        attention_scores (array-like): Relative attention weights of the tiles.
        instance_probs (array-like): Predicted instance-level probabilities for the positive class.
        
    Returns:
        np.ndarray: PAC scores in the range [-Attention_i, +Attention_i].
    """
    att = np.array(attention_scores)
    probs = np.array(instance_probs)
    return att * (2 * probs - 1)

def generate_pac_heatmap(patient_csv_path, epoch, output_dir=None, grid_size=None, cmap='RdBu_r'):
    """
    Generates and saves visual heatmaps for:
      1. Attention Weights (relative tile importance)
      2. Instance Probabilities (local class likelihood)
      3. Polarized Attentional Certainty (PAC - the integrated interpretability approach)
      
    Args:
        patient_csv_path (str): Path to the patient's tile coordinates and scores CSV file.
        epoch (int): The epoch number to visualize (1-indexed).
        output_dir (str, optional): Directory where the heatmaps will be saved.
        grid_size (int, optional): Size of the tile in pixels (for plotting square grids).
        cmap (str, optional): Colormap for the PAC score (diverging cmap centered at 0 is recommended).
    """
    df = pd.read_csv(patient_csv_path)
    
    # Check if necessary coordinates are present
    x_col = 'x' if 'x' in df.columns else ('coord_x' if 'coord_x' in df.columns else None)
    y_col = 'y' if 'y' in df.columns else ('coord_y' if 'coord_y' in df.columns else None)
    
    if x_col is None or y_col is None:
        raise ValueError("CSV must contain tile coordinates ('x' and 'y' or 'coord_x' and 'coord_y').")
        
    # Columns for requested epoch
    attn_col = f"attention_{epoch}"
    prob_col = f"instance_prob_{epoch}"
    pac_col = f"pac_{epoch}"
    
    if attn_col not in df.columns:
        raise ValueError(f"Attention scores for epoch {epoch} ({attn_col}) not found in the CSV.")
    if prob_col not in df.columns:
        raise ValueError(f"Instance probabilities for epoch {epoch} ({prob_col}) not found in the CSV.")
        
    # Calculate PAC if not already in df
    if pac_col not in df.columns:
        df[pac_col] = calculate_pac_scores(df[attn_col], df[prob_col])
        df.to_csv(patient_csv_path, index=False)
        
    # Coordinates and scores
    xs = df[x_col].values
    ys = df[y_col].values
    attns = df[attn_col].values
    probs = df[prob_col].values
    pacs = df[pac_col].values
    
    if output_dir is None:
        output_dir = os.path.dirname(patient_csv_path)
    os.makedirs(output_dir, exist_ok=True)
    
    # Create the visualization figure
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    # Normalize attention for visualization
    attn_norm = mcolors.Normalize(vmin=0, vmax=np.max(attns) if np.max(attns) > 0 else 1.0)
    # Probability is naturally [0, 1]
    prob_norm = mcolors.Normalize(vmin=0, vmax=1)
    # PAC is centered at 0, range [-max(attn), max(attn)]
    max_abs_pac = np.max(np.abs(pacs)) if np.max(np.abs(pacs)) > 0 else 1.0
    pac_norm = mcolors.TwoSlopeNorm(vmin=-max_abs_pac, vcenter=0, vmax=max_abs_pac)
    
    # Use tile plotting if grid_size is provided, otherwise scatter plot
    if grid_size is not None:
        # 1. Attention Heatmap
        im0 = axes[0].scatter(xs, ys, c=attns, cmap='YlOrRd', norm=attn_norm, marker='s', s=grid_size)
        axes[0].set_title(f"Attention Weights (Epoch {epoch})", fontsize=12)
        fig.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)
        
        # 2. Instance Probability Heatmap
        im1 = axes[1].scatter(xs, ys, c=probs, cmap='jet', norm=prob_norm, marker='s', s=grid_size)
        axes[1].set_title(f"Instance Probabilities (Epoch {epoch})", fontsize=12)
        fig.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)
        
        # 3. Polarized Attentional Certainty (PAC) Heatmap
        im2 = axes[2].scatter(xs, ys, c=pacs, cmap=cmap, norm=pac_norm, marker='s', s=grid_size)
        axes[2].set_title(f"Polarized Attentional Certainty (Epoch {epoch})", fontsize=12)
        fig.colorbar(im2, ax=axes[2], fraction=0.046, pad=0.04)
    else:
        # Scatter plot fallback
        im0 = axes[0].scatter(xs, ys, c=attns, cmap='YlOrRd', norm=attn_norm, alpha=0.8, edgecolors='none')
        axes[0].set_title(f"Attention Weights (Epoch {epoch})", fontsize=12)
        axes[0].invert_yaxis()
        fig.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)
        
        im1 = axes[1].scatter(xs, ys, c=probs, cmap='jet', norm=prob_norm, alpha=0.8, edgecolors='none')
        axes[1].set_title(f"Instance Probabilities (Epoch {epoch})", fontsize=12)
        axes[1].invert_yaxis()
        fig.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)
        
        im2 = axes[2].scatter(xs, ys, c=pacs, cmap=cmap, norm=pac_norm, alpha=0.8, edgecolors='none')
        axes[2].set_title(f"Polarized Attentional Certainty\nPAC = Attention * (2*Prob - 1)", fontsize=12)
        axes[2].invert_yaxis()
        fig.colorbar(im2, ax=axes[2], fraction=0.046, pad=0.04)
        
    for ax in axes:
        ax.set_aspect('equal')
        ax.set_xlabel('X Coordinate')
        ax.set_ylabel('Y Coordinate')
        
    plt.tight_layout()
    
    # Save output plot
    plot_save_path = os.path.join(output_dir, f"epoch_{epoch}_pac_comparison.png")
    plt.savefig(plot_save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Heatmap comparison saved successfully to: {plot_save_path}")
    return plot_save_path