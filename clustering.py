import os
import argparse
import numpy as np
import warnings
warnings.filterwarnings("ignore")

def parse_arguments():
    parser = argparse.ArgumentParser(description="Phase 2: Optimized Spectral Clustering Execution")
    parser.add_argument('--affinity_txt',   type=str,   default="./processed_data/affinity_score_file.txt")
    parser.add_argument('--segments_txt',   type=str,   default="./processed_data/segments_file.txt")
    parser.add_argument('--output_rttm',      type=str,   default="./predicted_output.rttm")
    parser.add_argument('--clustering_script_dir', type=str, default=".")
    
    # Post-processing configurations
    parser.add_argument('--knn_ratio', type=float, default=0.15)
    parser.add_argument('--sym_mode', type=str, default='mean', choices=['mean', 'max', 'cosine'])
    parser.add_argument('--affinity_power', type=float, default=2.0)
    parser.add_argument('--min_seg_duration', type=float, default=0.3)
    return parser.parse_args()

def apply_knn_pruning(matrix: np.ndarray, ratio: float, sym_mode: str = 'mean') -> np.ndarray:
    if ratio >= 1.0: 
        return matrix
    N = matrix.shape[0]
    k = max(1, int(N * ratio))
    refined = np.zeros_like(matrix)
    for i in range(N):
        top_k = np.argsort(matrix[i])[-k:]
        refined[i, top_k] = matrix[i, top_k]

    if sym_mode == 'max':
        return np.maximum(refined, refined.T)
    elif sym_mode == 'cosine':
        row_norms = np.linalg.norm(refined, axis=1, keepdims=True) + 1e-9
        normed = refined / row_norms
        sym = 0.5 * (normed + normed.T)
        mx = sym.max()
        return sym / mx if mx > 0 else sym
    else:  # 'mean'
        return 0.5 * (refined + refined.T)

def sharpen_affinity(matrix: np.ndarray, power: float) -> np.ndarray:
    if power == 1.0: 
        return matrix
    return np.power(np.clip(matrix, 0.0, 1.0), power)

def suppress_short_segment_islands(labels: np.ndarray, intervals: list, min_dur: float) -> np.ndarray:
    labels = labels.copy()
    for i, (start, dur) in enumerate(intervals):
        if dur < min_dur:
            ctx = list(labels[max(0, i - 2): i]) + list(labels[i + 1: min(len(labels), i + 3)])
            if ctx:
                labels[i] = max(set(ctx), key=ctx.count)
    return labels

def load_metadata_maps(segments_path: str) -> dict:
    time_mappings = {}
    with open(segments_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if not parts: 
                continue
            segment_id, session_id = parts[0], parts[1]
            
            # Safer split from the right to protect session IDs containing internal hyphens
            tokens = segment_id.rsplit('-', 2)
            start = float(int(tokens[1]) / 100.0)
            end = float(int(tokens[2]) / 100.0)
            duration = round(end - start, 3)
            
            if session_id not in time_mappings:
                time_mappings[session_id] = []
            time_mappings[session_id].append([start, duration])
    return time_mappings

def main():
    args = parse_arguments()
    time_mappings = load_metadata_maps(args.segments_txt)
    if os.path.exists(args.output_rttm): 
        os.remove(args.output_rttm)

    sessions = []
    with open(args.affinity_txt, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if parts: 
                sessions.append((parts[0], parts[1]))

    import sys
    sys.path.append(args.clustering_script_dir)
    from autotuning_sc import GraphSpectralClusteringClass

    class DummyParams:
        def __init__(self):
            self.threshold = 'NMESC'
            self.spt_est_thres = 'NMESC'
            self.max_speaker = 8
            self.n_sparse_search = 30
            self.sparse_search = True

    params_obj = DummyParams()
    sc_engine = GraphSpectralClusteringClass(params_obj)

    with open(args.output_rttm, 'w') as rttm_out:
        for session_id, npy_path in sessions:
            print(f"Clustering session: {session_id}")
            raw_mat = np.load(npy_path)

            # Pre-processing transformations
            refined_mat = apply_knn_pruning(raw_mat, args.knn_ratio, args.sym_mode)
            sharpened_mat = sharpen_affinity(refined_mat, args.affinity_power)

            # Explicitly providing all required keyword arguments to match your backend signature
            predicted_labels = sc_engine.COSclustering(
                idx=0, 
                key=session_id, 
                mat=sharpened_mat, 
                mat_spkcount=sharpened_mat, 
                param=params_obj
            )

            intervals = time_mappings.get(session_id, [])
            predicted_labels = suppress_short_segment_islands(
                np.array(predicted_labels), intervals, args.min_seg_duration
            ).tolist()

            for i, label in enumerate(predicted_labels):
                if i >= len(intervals): 
                    break
                start, duration = intervals[i]

                if i < len(intervals) - 1:
                    next_start = intervals[i + 1][0]
                    if start + duration > next_start:
                        duration = next_start - start

                if duration <= 0.001: 
                    continue

                rttm_out.write(
                    f"SPEAKER {session_id} 1 {start:.3f} {duration:.3f} "
                    f"<NA> <NA> speaker_{label} <NA>\n"
                )

    print("\nPhase 2 Complete — RTTM predictions written out.")

if __name__ == "__main__":
    main()