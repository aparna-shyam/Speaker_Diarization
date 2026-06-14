"""
model.py
========
GNN-based speaker embedding refiner.

Changes vs previous version
-----------------------------
- ResGCN: adds a linear residual skip from input to output of each GCN block.
  This prevents over-smoothing (all nodes collapsing to the same embedding)
  which is the #1 reason GCN-refined embeddings are *worse* than raw ones.
- Dropout(0.3) after each GCN layer for regularisation.
- BatchNorm1d after each GCN layer for stable training.
- score_pairs_chunked kept for inference; sampled-pair loss used in training.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv


# ---------------------------------------------------------------------------
# Graph construction
# ---------------------------------------------------------------------------

def build_graph(
    embeddings:         np.ndarray,
    threshold:          float = 0.6,
    max_edges_per_node: int   = 32,
) -> Data:
    """
    Sparse cosine-similarity graph.  top-K neighbours per node above threshold.
    Falls back to self-loops if no pair passes the threshold.
    """
    norms    = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms    = np.where(norms == 0, 1e-10, norms)
    emb_norm = embeddings / norms
    sim      = emb_norm @ emb_norm.T
    N        = embeddings.shape[0]
    np.fill_diagonal(sim, -1.0)

    edge_src, edge_dst, edge_w = [], [], []
    for i in range(N):
        row   = sim[i]
        cands = np.where(row > threshold)[0]
        if len(cands) == 0:
            continue
        if len(cands) > max_edges_per_node:
            top_idx = np.argpartition(row[cands], -max_edges_per_node)[-max_edges_per_node:]
            cands   = cands[top_idx]
        for j in cands:
            edge_src.append(i)
            edge_dst.append(int(j))
            edge_w.append(float(sim[i, j]))

    if len(edge_src) == 0:
        edge_src = list(range(N))
        edge_dst = list(range(N))
        edge_w   = [1.0] * N

    return Data(
        x          = torch.tensor(embeddings, dtype=torch.float),
        edge_index = torch.tensor([edge_src, edge_dst], dtype=torch.long),
        edge_attr  = torch.tensor(edge_w, dtype=torch.float),
    )


# ---------------------------------------------------------------------------
# Residual GCN block
# ---------------------------------------------------------------------------

class ResGCNBlock(nn.Module):
    """
    GCNConv  →  BN  →  ELU  →  Dropout  +  linear residual skip.

    The skip connection prevents over-smoothing: without it, stacking
    GCN layers drives all node embeddings toward the same value, which
    destroys the speaker-discriminative structure you need for clustering.
    """
    def __init__(self, in_dim: int, out_dim: int, dropout: float = 0.3):
        super().__init__()
        self.conv  = GCNConv(in_dim, out_dim)
        self.bn    = nn.BatchNorm1d(out_dim)
        self.drop  = nn.Dropout(p=dropout)
        # linear projection for the skip when dims differ
        self.skip  = nn.Linear(in_dim, out_dim, bias=False) \
                     if in_dim != out_dim else nn.Identity()

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        out = self.conv(x, edge_index)
        out = self.bn(out)
        out = F.elu(out)
        out = self.drop(out)
        return out + self.skip(x)          # residual


# ---------------------------------------------------------------------------
# GNNRefiner
# ---------------------------------------------------------------------------

class GNNRefiner(nn.Module):
    """
    Encoder : ResGCNBlock(D→H) → ResGCNBlock(H→Z) → L2-norm
    Scorer  : concat(h_i, h_j) → Linear(2Z,Z) → ELU → Linear(Z,1)

    fc1 / fc2 are public so train_gnn.sampled_pair_loss can access them
    directly without a full N×N forward pass.
    """

    def __init__(
        self,
        input_dim:  int   = 512,
        hidden_dim: int   = 256,
        out_dim:    int   = 128,
        dropout:    float = 0.3,
    ) -> None:
        super().__init__()

        self.block1 = ResGCNBlock(input_dim,  hidden_dim, dropout)
        self.block2 = ResGCNBlock(hidden_dim, out_dim,    dropout)

        self.fc1 = nn.Linear(2 * out_dim, out_dim)
        self.fc2 = nn.Linear(out_dim, 1)
        self._init_weights()

    def _init_weights(self) -> None:
        for m in [self.fc1, self.fc2]:
            nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
            nn.init.zeros_(m.bias)

    def encode(self, data: Data) -> torch.Tensor:
        """Returns L2-normalised node embeddings, shape (N, out_dim)."""
        x, ei = data.x, data.edge_index
        x = self.block1(x, ei)
        x = self.block2(x, ei)
        return F.normalize(x, p=2, dim=-1)

    def score_pairs_chunked(
        self,
        h:             torch.Tensor,
        chunk_size:    int  = 256,
        apply_sigmoid: bool = False,
    ) -> torch.Tensor:
        """Full N×N scoring in row-chunks — used at inference only."""
        N, rows = h.size(0), []
        for start in range(0, N, chunk_size):
            end  = min(start + chunk_size, N)
            hi   = h[start:end].unsqueeze(1).expand(-1, N, -1)
            hj   = h.unsqueeze(0).expand(end - start, N, -1)
            pair = torch.cat([hi, hj], dim=-1)
            out  = F.elu(self.fc1(pair))
            out  = self.fc2(out).squeeze(-1)
            if apply_sigmoid:
                out = torch.sigmoid(out)
            rows.append(out)
        return torch.cat(rows, dim=0)

    def forward(
        self,
        data:       Data,
        chunk_size: int = 256,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        h      = self.encode(data)
        logits = self.score_pairs_chunked(h, chunk_size=chunk_size,
                                          apply_sigmoid=False)
        return h, logits
