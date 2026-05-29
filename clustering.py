import os
import argparse
import numpy as np
import warnings
warnings.filterwarnings("ignore")

def parse_arguments():
    parser = argparse.ArgumentParser(description="Phase 2: Truncated Clean Matrix Cluster Engine")
    parser.add_argument('--affinity_txt', type=str, default="./processed_data/affinity_score_file.txt")
    parser.add_argument('--segments_txt', type=str, default="./processed_data/segments_file.txt")
    parser.add_argument('--output_rttm', type=str, default="./predicted_output.rttm")
    parser.add_argument('--clustering_script_dir', type=str, default=".")
    
    # Clean noise connections: Keep top 12% highest similarity nodes per segment row
    parser.add_argument('--knn_ratio', type=float, default=0.12)
    return parser.parse_args()

def apply_knn_pruning(matrix, ratio):
    if ratio >= 1.0: return matrix
    N = matrix.shape[0]
    k = max(1, int(N * ratio))
    refined = np.zeros_like(matrix)
    for i in range(N):
        top_k = np.argsort(matrix[i])[-k:]
        refined[i, top_k] = matrix[i, top_k]
    return 0.5 * (refined + refined.T)

def load_metadata_maps(segments_path):
    time_mappings = {}
    with open(segments_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if not parts: continue
            segment_id, session_id = parts[0], parts[1]
            tokens = segment_id.split('-')
            start, end = float(int(tokens[1]) / 100.0), float(int(tokens[2]) / 100.0)
            if session_id not in time_mappings: time_mappings[session_id] = []
            time_mappings[session_id].append([start, round(end - start, 3)])
    return time_mappings

def main():
    args = parse_arguments()
    time_mappings = load_metadata_maps(args.segments_txt)
    if os.path.exists(args.output_rttm): os.remove(args.output_rttm)
        
    sessions = []
    with open(args.affinity_txt, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if parts: sessions.append((parts[0], parts[1]))
                
    import sys
    sys.path.append(args.clustering_script_dir)
    from autotuning_sc import GraphSpectralClusteringClass

    class DummyParams:
        def __init__(self):
            self.affinity_score_file = args.affinity_txt
            self.affinity_score_for_spk_count = 'None'
            self.segment_file_input_path = args.segments_txt
            self.segment_spkcount_input_path = 'None'
            self.nmesc_thres_save_path = 'None'
            self.threshold = 'NMESC'
            self.spk_labels_out_path = './dummy_labels.txt'
            self.reco2num_spk = 'None'
            self.score_metric = 'cos'
            self.max_speaker = 4  # Cap cluster searching block to match standard AMI configurations
            
            # CHANGED: Updated from 2.0 to 1.5 to match your required window length spec
            self.xvector_window = 1.5
            
            self.spt_est_thres = 'NMESC'
            self.max_speaker_list = 'None'
            self.n_sparse_search = 20
            self.parallel_threshold = 6000
            self.sparse_search = True

    params_obj = DummyParams()
    sc_engine = GraphSpectralClusteringClass(params_obj)
    
    with open(args.output_rttm, 'w') as rttm_out:
        for session_id, npy_path in sessions:
            print(f"Clustering Session: {session_id}")
            raw_mat = np.load(npy_path)
            
            # Apply KNN affinity cleaning to avoid speaker blending errors
            refined_mat = apply_knn_pruning(raw_mat, args.knn_ratio)
            
            predicted_labels = sc_engine.COSclustering(
                idx=0, key=session_id, mat=refined_mat, mat_spkcount=refined_mat, param=params_obj
            )
            
            intervals = time_mappings.get(session_id, [])
            for i, label in enumerate(predicted_labels):
                start, duration = intervals[i]
                
                # INTEGRATED OPTIMIZATION: Look ahead and truncate overflow bounds
                if i < len(intervals) - 1:
                    next_start = intervals[i + 1][0]
                    if start + duration > next_start:
                        duration = next_start - start
                        
                if duration <= 0.001: continue
                rttm_out.write(f"SPEAKER {session_id} 1 {start:.3f} {duration:.3f} <NA> <NA> speaker_{label} <NA>\n")
    print("\nPhase 2 Complete!")

if __name__ == "__main__":
    main()