"""
infer_gnn.py
============
Inference: loads trained GNNRefiner, produces refined embeddings for test set.

Change vs previous: reads dropout from checkpoint (new model has dropout param).
"""

from __future__ import annotations

import os
import shutil
from pathlib import Path

import numpy as np
import torch

from model import GNNRefiner, build_graph


OUT           = "./output_ami_split"
TEST_EMB_DIR  = f"{OUT}/xvector_embeddings_test"
MODEL_PATH    = f"{OUT}/gnn_model.pt"
INFER_OUT_DIR = f"{OUT}/gnn_embeddings"


def refine_embeddings(
    embeddings_dir: str,
    model_path:     str,
    out_dir:        str,
) -> None:

    gpu        = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(model_path, map_location=gpu)

    model = GNNRefiner(
        input_dim  = checkpoint["input_dim"],
        hidden_dim = checkpoint["hidden_dim"],
        out_dim    = checkpoint["out_dim"],
        dropout    = checkpoint.get("dropout", 0.3),
    ).to(gpu)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    threshold          = checkpoint.get("threshold",          0.6)
    max_edges_per_node = checkpoint.get("max_edges_per_node", 32)
    chunk_size         = checkpoint.get("pair_chunk_size",    512)

    print(f"Graph threshold  : {threshold}")
    print(f"Max edges / node : {max_edges_per_node}")

    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    npz_files = sorted(Path(embeddings_dir).glob("*_embeddings.npz"))
    print(f"Refining {len(npz_files)} sessions …\n")

    for npz_path in npz_files:
        file_id   = npz_path.stem.replace("_embeddings", "")
        json_path = npz_path.parent / f"{file_id}_metadata.json"

        if not json_path.exists():
            print(f"  [SKIP] {file_id}: metadata not found.")
            continue

        data_np = np.load(str(npz_path))
        emb     = data_np[list(data_np.keys())[0]]

        if emb.shape[0] < 2:
            np.savez(str(out_path / f"{file_id}_embeddings.npz"), embedding=emb)
            shutil.copy(str(json_path), str(out_path / f"{file_id}_metadata.json"))
            continue

        try:
            graph = build_graph(emb, threshold=threshold,
                                max_edges_per_node=max_edges_per_node).to(gpu)
            with torch.no_grad():
                refined, _ = model(graph, chunk_size=chunk_size)
                refined_np = refined.cpu().numpy()
        except torch.cuda.OutOfMemoryError:
            print(f"  [OOM→CPU] {file_id}")
            torch.cuda.empty_cache()
            cpu   = torch.device("cpu")
            model.to(cpu)
            graph = build_graph(emb, threshold=threshold,
                                max_edges_per_node=max_edges_per_node).to(cpu)
            with torch.no_grad():
                refined, _ = model(graph, chunk_size=chunk_size)
                refined_np = refined.numpy()
            model.to(gpu)

        np.savez(str(out_path / f"{file_id}_embeddings.npz"), embedding=refined_np)
        shutil.copy(str(json_path), str(out_path / f"{file_id}_metadata.json"))
        print(f"  {file_id}: {emb.shape} → {refined_np.shape}")

        del graph
        torch.cuda.empty_cache()

    print(f"\nRefined embeddings written to: {out_dir}")
    print("Next step: python clustering.py --embeddings_dir", out_dir)


if __name__ == "__main__":
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    print(f"emb_dir  : {TEST_EMB_DIR}")
    print(f"model    : {MODEL_PATH}")
    print(f"out_dir  : {INFER_OUT_DIR}\n")
    refine_embeddings(TEST_EMB_DIR, MODEL_PATH, INFER_OUT_DIR)
