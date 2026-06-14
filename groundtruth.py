"""
groundtruth.py
==============
Builds the binary N×N ground-truth adjacency matrix for one session
from an RTTM file.  A[i,j] = 1 iff segment i and segment j share
the same speaker (dominant speaker by overlap).

Used by train_gnn.py only — not needed at inference time.
"""

from __future__ import annotations

import numpy as np


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def build_gt_adjacency(metadata: list[dict], rttm_path: str) -> np.ndarray:
    """
    Parameters
    ----------
    metadata  : list of dicts with keys "start" and "end" (seconds),
                as written by vadfe.py / the JSON sidecar files.
    rttm_path : path to the matching .rttm file.

    Returns
    -------
    adj : float32 ndarray, shape (N, N)
          adj[i,j] = 1.0  if dominant speaker of segment i == segment j
          adj[i,j] = 0.0  otherwise
    """
    rttm_segs = _parse_rttm(rttm_path)
    labels    = [_dominant_speaker(m["start"], m["end"], rttm_segs)
                 for m in metadata]

    N   = len(labels)
    adj = np.zeros((N, N), dtype=np.float32)
    for i in range(N):
        for j in range(N):
            if labels[i] is not None and labels[i] == labels[j]:
                adj[i, j] = 1.0
    return adj


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _parse_rttm(rttm_path: str) -> list[tuple[float, float, str]]:
    """Returns list of (t_start, t_end, speaker_id)."""
    segs = []
    with open(rttm_path) as fh:
        for line in fh:
            parts = line.strip().split()
            if len(parts) >= 8 and parts[0] == "SPEAKER":
                t_start = float(parts[3])
                t_dur   = float(parts[4])
                spk     = parts[7]
                segs.append((t_start, t_start + t_dur, spk))
    return segs


def _dominant_speaker(
    seg_start: float,
    seg_end:   float,
    rttm_segs: list[tuple[float, float, str]],
) -> str | None:
    """Returns the RTTM speaker with the most overlap in [seg_start, seg_end]."""
    overlap: dict[str, float] = {}
    for r_start, r_end, spk in rttm_segs:
        ov = max(0.0, min(seg_end, r_end) - max(seg_start, r_start))
        if ov > 0:
            overlap[spk] = overlap.get(spk, 0.0) + ov
    return max(overlap, key=overlap.get) if overlap else None
