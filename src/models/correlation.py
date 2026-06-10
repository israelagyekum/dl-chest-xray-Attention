"""
correlation.py — Label-correlation-aware component.

Two complementary pieces:

1.  CorrelationRegularizer  (L_corr)
    --------------------------------
    Penalises prediction patterns that violate the empirical label
    co-occurrence learned from the training set.

    Given:
        P      : (B, 14) sigmoid probabilities
        C_ij   : empirical P(label_j=1 | label_i=1)  — computed from training data

    For each pair (i, j) where co-occurrence is high (C_ij > threshold),
    we want: if P_i is high, P_j should also be high.

    L_corr = mean over (i,j) with C_ij > τ of:
                C_ij · ReLU( P_i - P_j )^2

    Intuition: if label j almost always co-occurs with label i, but our
    model predicts P_i ≫ P_j, penalise that gap.

2.  GCNLabelGraph (stretch / ablation)
    ------------------------------------
    A lightweight 1-layer graph convolution that propagates label embeddings
    through the co-occurrence graph, producing refined label representations.
    This is the ML-GCN-inspired stretch goal mentioned in the architecture doc.
    It is NOT used in the default model but can be swapped in via config.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional


# ---------------------------------------------------------------------------
# Empirical co-occurrence regulariser
# ---------------------------------------------------------------------------

class CorrelationRegularizer(nn.Module):
    """
    Registers the co-occurrence matrix as a buffer (not a parameter)
    so it moves with the model to CUDA automatically.

    Parameters
    ----------
    cooc_matrix : np.ndarray, shape (14, 14)
                  P(j|i) — from splits.compute_cooccurrence_matrix()
    threshold   : float
                  Only consider pairs (i,j) with C_ij > threshold
                  (prunes near-zero co-occurrences).
    """

    def __init__(self,
                 cooc_matrix: np.ndarray,
                 threshold:   float = 0.2):
        super().__init__()
        cooc = torch.tensor(cooc_matrix, dtype=torch.float32)
        # Mask: 1 where co-occurrence is significant (and i ≠ j)
        mask = (cooc > threshold).float()
        mask.fill_diagonal_(0.0)
        self.register_buffer("cooc", cooc)
        self.register_buffer("mask", mask)
        n_pairs = int(mask.sum().item())
        print(f"[correlation] {n_pairs} significant (i,j) pairs "
              f"(threshold={threshold})")

    def forward(self, probs: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        probs : (B, 14) — sigmoid output probabilities

        Returns
        -------
        loss : scalar tensor
        """
        # probs_i: (B, 14, 1)   probs_j: (B, 1, 14)
        p_i = probs.unsqueeze(2)    # (B, 14, 1)
        p_j = probs.unsqueeze(1)    # (B, 1, 14)

        # If P_i is high but P_j is low, penalise
        gap   = F.relu(p_i - p_j)          # (B, 14, 14)
        cost  = self.cooc * gap.pow(2)      # weight by co-occurrence strength
        cost  = cost * self.mask            # zero out low-co-occurrence pairs
        return cost.mean()


# ---------------------------------------------------------------------------
# GCN label graph (stretch goal / ablation)
# ---------------------------------------------------------------------------

class GCNLabelGraph(nn.Module):
    """
    Single-layer Graph Convolution over the label graph (ML-GCN style).

    Takes per-class embeddings (label_emb) and propagates information
    through the co-occurrence adjacency matrix to produce refined
    classifier weights.

    Used as: classifier_weight = gcn(label_emb)

    Parameters
    ----------
    num_classes  : 14
    in_features  : input embedding dimension
    out_features : output dimension (= feature map channels for final GAP output)
    cooc_matrix  : (14, 14) empirical co-occurrence
    p_threshold  : binarise adjacency at this threshold
    """

    def __init__(self,
                 num_classes:  int,
                 in_features:  int,
                 out_features: int,
                 cooc_matrix:  np.ndarray,
                 p_threshold:  float = 0.4):
        super().__init__()
        self.num_classes  = num_classes
        self.out_features = out_features

        # Build symmetric binary adjacency + self-loops
        adj = (cooc_matrix > p_threshold).astype(np.float32)
        adj = np.maximum(adj, adj.T)            # symmetrise
        np.fill_diagonal(adj, 1.0)              # self-loops

        # Normalised Laplacian  D^{-1/2} A D^{-1/2}
        deg  = adj.sum(axis=1)
        dinv = np.diag(1.0 / np.sqrt(np.maximum(deg, 1e-8)))
        adj_norm = dinv @ adj @ dinv

        adj_t = torch.tensor(adj_norm, dtype=torch.float32)
        self.register_buffer("adj", adj_t)

        # Learnable GCN weight matrix
        self.W = nn.Linear(in_features, out_features, bias=False)

        # Label embeddings (initialised from word vectors or random)
        self.label_emb = nn.Parameter(
            torch.randn(num_classes, in_features) * 0.01
        )

    def forward(self) -> torch.Tensor:
        """
        Returns refined label classifier weights: (num_classes, out_features)
        These replace (or augment) the linear classifier head.
        """
        # H = A_norm · label_emb
        H = self.adj @ self.label_emb          # (14, in_features)
        H = F.leaky_relu(self.W(H), 0.2)       # (14, out_features)
        return H   # use as weight matrix for final classification


# ---------------------------------------------------------------------------
# Smoke-test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import numpy as np

    # Random co-occurrence matrix
    rng = np.random.default_rng(0)
    cooc = rng.uniform(0, 1, (14, 14)).astype(np.float32)
    np.fill_diagonal(cooc, 1.0)

    reg = CorrelationRegularizer(cooc, threshold=0.2)

    B = 4
    probs = torch.rand(B, 14)
    loss  = reg(probs)
    print(f"L_corr = {loss.item():.6f}")
    assert loss.item() >= 0, "L_corr must be non-negative"

    gcn = GCNLabelGraph(14, 128, 256, cooc, p_threshold=0.4)
    weights = gcn()
    print(f"GCN label weights shape: {tuple(weights.shape)}")
    print("✓ Correlation module smoke-test passed.")
