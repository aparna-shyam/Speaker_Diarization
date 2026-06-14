"""
train_gnn.py  (optimized for DER < 8%)
========================================
Key changes vs previous version
---------------------------------
OPT #1  Pre-cache graphs + GT adjacency before epoch 1 (no per-epoch I/O).

OPT #2  Sampled-pair loss: sample MAX_PAIRS balanced pos/neg pairs per session
        instead of scoring all N² pairs.  Eliminates the class-imbalance
        problem and removes the O(N²) bottleneck.

OPT #3  Contrastive margin loss IN ADDITION to BCE.
        BCE alone doesn't push same-speaker embeddings together in embedding
        space — it only trains the FC scorer.  The contrastive term directly
        optimises the GCN output h, which is what clustering.py actually uses.
        margin=0.5: same-speaker pairs should be within cosine dist 0.5,
        different-speaker pairs should be farther than 0.5 apart.

OPT #4  Model stays on GPU always.  No device switching.

OPT #5  Mixed precision (AMP) + AdamW + Cosine LR annealing.

OPT #6  Best-model checkpointing: saves the epoch with lowest loss.
"""

from __future__ import annotations

import json
import os
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.amp import GradScaler, autocast

from loss       import bce_loss
from model      import GNNRefiner, build_graph
from groundtruth import _parse_rttm, _dominant_speaker


# ============================================================================
# Paths
# ============================================================================

BASE          = "/DATA/nikhil-data/diarisation_dataset/ami_mixed"
OUT           = "./output_ami_split"
TRAIN_EMB_DIR = f"{OUT}/xvector_embeddings_train"
TRAIN_RTTM    = f"{BASE}/BUT_rttms/train"
MODEL_PATH    = f"{OUT}/gnn_model.pt"

# ============================================================================
# Hyperparameters
# ============================================================================

EPOCHS       = 60
LR           = 5e-4
WEIGHT_DECAY = 1e-4

# Graph
THRESHOLD          = 0.6
MAX_EDGES_PER_NODE = 32

# Model
INPUT_DIM  = 512
HIDDEN_DIM = 256   # reduced from 512 — GCN hidden state is (N, H), N can be 3000+
OUT_DIM    = 128   # reduced from 256
DROPOUT    = 0.3

# Loss
MAX_PAIRS_PER_SESSION = 4096   # sampled pairs per session
CONTRASTIVE_MARGIN    = 0.5    # cosine distance margin
CONTRASTIVE_WEIGHT    = 0.5    # λ · contrastive + (1-λ) · BCE

# Training
USE_AMP        = True
GRAD_CLIP_NORM = 1.0

# Large-session subsampling:
# Sessions with more than this many nodes are randomly subsampled to this
# size before the GCN forward pass.  Keeps GPU memory bounded at O(MAX_NODES).
# 1200 nodes × 256 hidden × 4 bytes × ~8 activations ≈ 1 GB — safe on 8 GB GPU.
# Raise if you have more VRAM; lower if you still see OOM.
MAX_NODES_PER_SESSION = 1200


# ============================================================================
# OOM-safe pair sampling
# ============================================================================
# Root cause of OOM: calling .nonzero() on an (N,N) boolean tensor allocates
# O(N²) GPU memory.  For N=3744 that is ~14M booleans just for the mask,
# before any actual computation.
#
# Fix: keep the GT as a numpy int8 LABEL VECTOR (one label per segment,
# shape (N,)), stored in sess["labels"].  Sample pairs by:
#   1. draw K random anchor indices
#   2. for each anchor, find a positive (same label) and a negative
# This is O(K) memory regardless of N.
# ============================================================================

def _sample_pairs_from_labels(
    labels_np: np.ndarray,   # (N,) int — speaker id per segment, on CPU
    n_pairs:   int,
    device:    torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Returns (idx_i, idx_j, binary_labels) each of length ≤ n_pairs.
    Never allocates an NxN tensor — O(n_pairs) memory only.
    """
    N      = len(labels_np)
    # group segment indices by speaker
    from collections import defaultdict
    spk2segs: dict[int, list[int]] = defaultdict(list)
    for idx, lbl in enumerate(labels_np):
        spk2segs[int(lbl)].append(idx)

    speakers = [s for s, segs in spk2segs.items() if len(segs) >= 2]
    if len(speakers) == 0:
        # degenerate: only one speaker — can't form negatives
        return None, None, None

    half   = n_pairs // 2
    src_i, src_j, lbl_list = [], [], []

    # ── positive pairs ──────────────────────────────────────────────────
    n_pos = 0
    while n_pos < half:
        spk    = speakers[np.random.randint(len(speakers))]
        segs   = spk2segs[spk]
        chosen = np.random.choice(segs, size=2, replace=False)
        src_i.append(chosen[0]); src_j.append(chosen[1]); lbl_list.append(1.0)
        n_pos += 1

    # ── negative pairs ──────────────────────────────────────────────────
    n_neg = 0
    while n_neg < half:
        s1, s2 = np.random.randint(N), np.random.randint(N)
        if labels_np[s1] != labels_np[s2]:
            src_i.append(s1); src_j.append(s2); lbl_list.append(0.0)
            n_neg += 1

    idx_i  = torch.tensor(src_i,   dtype=torch.long,  device=device)
    idx_j  = torch.tensor(src_j,   dtype=torch.long,  device=device)
    labels = torch.tensor(lbl_list, dtype=torch.float32, device=device)
    return idx_i, idx_j, labels


def contrastive_loss(
    h:         torch.Tensor,    # (N, D) L2-normalised, on GPU
    labels_np: np.ndarray,      # (N,) int speaker labels, on CPU numpy
    margin:    float = 0.5,
    n_pairs:   int   = 2048,
) -> torch.Tensor:
    """
    Cosine-distance contrastive loss.
    Operates on h directly so gradients shape the embedding space used
    by clustering — not just the FC scorer.
    Memory: O(n_pairs · D), never O(N²).
    """
    idx_i, idx_j, labels = _sample_pairs_from_labels(
        labels_np, n_pairs, h.device)
    if idx_i is None:
        return torch.tensor(0.0, device=h.device)

    sim      = (h[idx_i] * h[idx_j]).sum(-1)   # cosine sim (L2-normed)
    dist     = 1.0 - sim                        # cosine distance

    pos_mask = labels == 1.0
    neg_mask = ~pos_mask

    loss_pos = dist[pos_mask].pow(2).mean() if pos_mask.any() \
               else torch.tensor(0.0, device=h.device)
    loss_neg = F.relu(margin - dist[neg_mask]).pow(2).mean() if neg_mask.any() \
               else torch.tensor(0.0, device=h.device)
    return loss_pos + loss_neg


def sampled_pair_loss(
    h:         torch.Tensor,    # (N, D) L2-normalised, on GPU
    labels_np: np.ndarray,      # (N,) int speaker labels, on CPU numpy
    model:     GNNRefiner,
    n_pairs:   int = 4096,
) -> torch.Tensor | None:
    """
    BCE loss on sampled pairs.
    Memory: O(n_pairs · D), never O(N²).
    """
    idx_i, idx_j, labels = _sample_pairs_from_labels(
        labels_np, n_pairs, h.device)
    if idx_i is None:
        return None

    pair   = torch.cat([h[idx_i], h[idx_j]], dim=-1)
    out    = F.elu(model.fc1(pair))
    logits = model.fc2(out).squeeze(-1)
    return bce_loss(logits, labels)


# ============================================================================
# Large-session subsampling
# ============================================================================

def subsample_session(
    graph:     "Data",
    labels_np: np.ndarray,
    max_nodes: int,
) -> tuple["Data", np.ndarray]:
    """
    Randomly subsample a session to at most `max_nodes` segments.

    Keeps the subgraph induced by the sampled nodes (edges whose both
    endpoints are in the sample).  Labels are reindexed accordingly.

    This runs on CPU (graph is still on CPU at this point) so it costs
    no GPU memory.
    """
    from torch_geometric.utils import subgraph as pyg_subgraph

    N = graph.x.size(0)
    if N <= max_nodes:
        return graph, labels_np

    idx      = np.random.choice(N, size=max_nodes, replace=False)
    idx_t    = torch.tensor(idx, dtype=torch.long)
    new_x    = graph.x[idx_t]

    # relabel edges to the subgraph
    new_edge_index, new_edge_attr = pyg_subgraph(
        idx_t,
        graph.edge_index,
        graph.edge_attr,
        relabel_nodes=True,
        num_nodes=N,
    )

    from torch_geometric.data import Data
    new_graph     = Data(x=new_x,
                         edge_index=new_edge_index,
                         edge_attr=new_edge_attr)
    new_labels_np = labels_np[idx]
    return new_graph, new_labels_np


# ============================================================================
# Pre-cache sessions  (OPT #1)
# ============================================================================

def preload_sessions(
    embeddings_dir:     str,
    rttm_dir:           str,
    threshold:          float,
    max_edges_per_node: int,
) -> list[dict]:
    npz_files = sorted(Path(embeddings_dir).glob("*_embeddings.npz"))
    sessions, skipped = [], 0
    print("Pre-loading sessions …")

    for i, npz_path in enumerate(npz_files, 1):
        file_id   = npz_path.stem.replace("_embeddings", "")
        json_path = npz_path.parent / f"{file_id}_metadata.json"
        rttm_path = Path(rttm_dir) / f"{file_id}.rttm"

        if not json_path.exists() or not rttm_path.exists():
            skipped += 1
            continue

        data_np  = np.load(str(npz_path))
        emb      = data_np[list(data_np.keys())[0]]
        if emb.shape[0] < 2:
            skipped += 1
            continue

        with open(json_path) as fh:
            metadata = json.load(fh)

        graph = build_graph(emb, threshold=threshold,
                            max_edges_per_node=max_edges_per_node)

        # Store a compact label vector (N,) instead of the full (N,N) GT matrix.
        rttm_segs  = _parse_rttm(str(rttm_path))
        spk_labels = [_dominant_speaker(m["start"], m["end"], rttm_segs)
                      for m in metadata]
        # map string speaker IDs → integers
        uniq  = {s: i for i, s in enumerate(dict.fromkeys(spk_labels))}
        labels_np = np.array([uniq.get(s, -1) for s in spk_labels], dtype=np.int32)

        sessions.append({
            "file_id":   file_id,
            "graph":     graph,
            "labels_np": labels_np,   # (N,) int — replaces gt_np (N,N)
            "N":         emb.shape[0],
        })

        if i % 20 == 0 or i == len(npz_files):
            print(f"  loaded {i}/{len(npz_files)}  (skipped {skipped})")

    print(f"Ready: {len(sessions)} sessions.\n")
    return sessions


# ============================================================================
# Training
# ============================================================================

def train_gnn(
    embeddings_dir:       str,
    rttm_dir:             str,
    model_out:            str,
    epochs:               int   = 60,
    lr:                   float = 5e-4,
    weight_decay:         float = 1e-4,
    threshold:            float = 0.6,
    input_dim:            int   = 512,
    hidden_dim:           int   = 512,
    out_dim:              int   = 256,
    dropout:              float = 0.3,
    max_pairs_per_session:int   = 4096,
    max_edges_per_node:   int   = 32,
    contrastive_margin:   float = 0.5,
    contrastive_weight:   float = 0.5,
    grad_clip_norm:       float = 1.0,
    use_amp:              bool  = True,
    max_nodes_per_session:int   = 1200,
) -> None:

    gpu         = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp_enabled = use_amp and gpu.type == "cuda"

    print(f"Device              : {gpu}")
    print(f"Mixed precision     : {amp_enabled}")
    print(f"Graph threshold     : {threshold}")
    print(f"Pairs / session     : {max_pairs_per_session}")
    print(f"Contrastive weight  : {contrastive_weight}  margin={contrastive_margin}")
    print(f"Hidden / out dim    : {hidden_dim} / {out_dim}  dropout={dropout}")

    sessions = preload_sessions(embeddings_dir, rttm_dir,
                                threshold, max_edges_per_node)
    if not sessions:
        raise FileNotFoundError("No valid sessions found.")

    model     = GNNRefiner(input_dim, hidden_dim, out_dim, dropout).to(gpu)
    optimizer = torch.optim.AdamW(model.parameters(),
                                  lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs, eta_min=lr * 0.01)
    scaler    = GradScaler("cuda", enabled=amp_enabled)

    best_loss  = float("inf")
    best_state = None

    for epoch in range(1, epochs + 1):
        model.train()
        epoch_loss, n_proc = 0.0, 0
        random.shuffle(sessions)

        for sess in sessions:
            # Subsample large sessions to keep GCN forward pass within VRAM.
            # This runs on CPU before the graph is moved to GPU.
            graph, labels_np = subsample_session(
                sess["graph"], sess["labels_np"], max_nodes_per_session)

            graph     = graph.to(gpu)

            optimizer.zero_grad(set_to_none=True)

            try:
                with autocast("cuda", enabled=amp_enabled):
                    h, _ = model(graph, chunk_size=512)

                    bce  = sampled_pair_loss(h, labels_np, model,
                                             n_pairs=max_pairs_per_session)
                    if bce is None:
                        continue

                    cont = contrastive_loss(h, labels_np,
                                            margin=contrastive_margin,
                                            n_pairs=max_pairs_per_session // 2)

                    loss = (1.0 - contrastive_weight) * bce \
                         + contrastive_weight * cont

                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
                scaler.step(optimizer)
                scaler.update()

                epoch_loss += loss.item()
                n_proc     += 1

            except torch.cuda.OutOfMemoryError:
                print(f"  [OOM] {sess['file_id']} ({sess['N']} segs) — skipped")
                torch.cuda.empty_cache()
                continue
            finally:
                del graph
                torch.cuda.empty_cache()

        scheduler.step()
        avg    = epoch_loss / max(n_proc, 1)
        lr_now = scheduler.get_last_lr()[0]
        marker = "  ← best" if avg < best_loss else ""
        print(f"Epoch [{epoch:3d}/{epochs}]  loss={avg:.6f}  "
              f"lr={lr_now:.2e}  sessions={n_proc}{marker}")

        if avg < best_loss:
            best_loss  = avg
            best_state = {k: v.cpu().clone()
                          for k, v in model.state_dict().items()}

    # save best checkpoint
    os.makedirs(os.path.dirname(os.path.abspath(model_out)), exist_ok=True)
    torch.save({
        "model_state":        best_state,
        "input_dim":          input_dim,
        "hidden_dim":         hidden_dim,
        "out_dim":            out_dim,
        "dropout":            dropout,
        "threshold":          threshold,
        "pair_chunk_size":    512,
        "max_edges_per_node": max_edges_per_node,
    }, model_out)
    print(f"\nBest model (loss={best_loss:.6f}) saved → {model_out}")


# ============================================================================
# Entry point
# ============================================================================

if __name__ == "__main__":
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

    print(f"emb_dir  : {TRAIN_EMB_DIR}")
    print(f"rttm_dir : {TRAIN_RTTM}")
    print(f"model_out: {MODEL_PATH}\n")

    train_gnn(
        embeddings_dir        = TRAIN_EMB_DIR,
        rttm_dir              = TRAIN_RTTM,
        model_out             = MODEL_PATH,
        epochs                = EPOCHS,
        lr                    = LR,
        weight_decay          = WEIGHT_DECAY,
        threshold             = THRESHOLD,
        input_dim             = INPUT_DIM,
        hidden_dim            = HIDDEN_DIM,
        out_dim               = OUT_DIM,
        dropout               = DROPOUT,
        max_pairs_per_session = MAX_PAIRS_PER_SESSION,
        max_edges_per_node    = MAX_EDGES_PER_NODE,
        contrastive_margin    = CONTRASTIVE_MARGIN,
        contrastive_weight    = CONTRASTIVE_WEIGHT,
        grad_clip_norm        = GRAD_CLIP_NORM,
        use_amp               = USE_AMP,
        max_nodes_per_session = MAX_NODES_PER_SESSION,
    )
