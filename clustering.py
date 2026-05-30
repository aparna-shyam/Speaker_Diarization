import argparse
import json
import os
import numpy as np
from pathlib import Path
from collections import Counter
from scipy.linalg import eigh as scipy_eigh
from sklearn.cluster import k_means
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MinMaxScaler

# Check for vector hardware math availability
try:
    import torch
    from torch.linalg import eigh as torch_eigh
    TORCH_EIGH = True
except ImportError:
    TORCH_EIGH = False

scaler = MinMaxScaler(feature_range=(0, 1))

def isGraphFullyConnected(affinity_mat):
    return getTheLargestComponent(affinity_mat, 0).sum() == affinity_mat.shape[0]

def getTheLargestComponent(affinity_mat, seg_index):
    num_of_segments = affinity_mat.shape[0]
    connected_nodes  = np.zeros(num_of_segments, dtype=bool)
    nodes_to_explore = np.zeros(num_of_segments, dtype=bool)
    nodes_to_explore[seg_index] = True

    for _ in range(num_of_segments):
        last_num_component = connected_nodes.sum()
        np.logical_or(connected_nodes, nodes_to_explore, out=connected_nodes)
        if last_num_component >= connected_nodes.sum():
            break
        indices = np.where(nodes_to_explore)[0]
        nodes_to_explore.fill(False)
        for i in indices:
            neighbors = affinity_mat[i]
            np.logical_or(nodes_to_explore, neighbors, out=nodes_to_explore)
    return connected_nodes

def getKneighborsConnections(affinity_mat, p_value):
    binarized = np.zeros_like(affinity_mat)
    for i, line in enumerate(affinity_mat):
        sorted_idx = np.argsort(line)[::-1]
        indices    = sorted_idx[:p_value]
        binarized[indices, i] = 1
    return binarized

def getAffinityGraphMat(affinity_mat_raw, p_value):
    X = getKneighborsConnections(affinity_mat_raw, p_value)
    return 0.5 * (X + X.T)

def getMinimumConnection(mat, max_N, n_list):
    p_value      = 1
    affinity_mat = getAffinityGraphMat(mat, p_value)
    for p_value in n_list:
        fully_connected = isGraphFullyConnected(affinity_mat)
        affinity_mat    = getAffinityGraphMat(mat, p_value)
        if fully_connected or p_value > max_N:
            break
    return affinity_mat, p_value

def getCosAffinityMatrix(emb):
    norms = np.linalg.norm(emb, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1e-10, norms)
    emb_norm = emb / norms
    sim_d = cosine_similarity(emb_norm)
    
    # Reference Scaling Mechanism to avoid multi-speaker clustering confusion
    sim_d = (sim_d - sim_d.min()) / (sim_d.max() - sim_d.min() + 1e-10)
    return sim_d

def getLaplacian(X):
    """Computes the unnormalized graph Laplacian matrix."""
    X = X.copy()
    X[np.diag_indices(X.shape[0])] = 0
    D = np.diag(np.sum(np.abs(X), axis=1))
    return D - X

def eigDecompose(laplacian, cuda=False):
    """
    Robust Multi-Tiered Eigen-Decomposition Routine.
    Prioritizes SciPy's stable LAPACK layout on CPU to prevent Intel MKL shape crashes,
    while utilizing PyTorch only when explicit CUDA hardware is available.
    """
    if cuda and TORCH_EIGH and torch.cuda.is_available():
        try:
            device = "cuda:0"
            lap_t = torch.from_numpy(laplacian).float().to(device)
            lambdas, diffusion_map = torch_eigh(lap_t)
            return lambdas.cpu().numpy(), diffusion_map.cpu().numpy()
        except Exception:
            pass  # Fall through to SciPy if GPU context boundary errors occur

    # Stable calculation path for CPU execution (Bypasses the Intel oneMKL BatchLinearAlgebra crash)
    try:
        lambdas, diffusion_map = scipy_eigh(laplacian)
        return lambdas, diffusion_map
    except Exception:
        # Final emergency fallback to native torch CPU
        lap_t = torch.from_numpy(laplacian).float()
        lambdas, diffusion_map = torch_eigh(lap_t)
        return lambdas.cpu().numpy(), diffusion_map.cpu().numpy()

def getLambdaGapList(lambdas):
    return list(lambdas[1:] - lambdas[:-1])

def estimateNumofSpeakers(affinity_mat, max_num_speaker, is_cuda=False):
    laplacian        = getLaplacian(affinity_mat)
    lambdas, _       = eigDecompose(laplacian, is_cuda)
    lambdas          = np.sort(lambdas)
    lambda_gap_list  = getLambdaGapList(lambdas)
    num_of_spk       = np.argmax(
        lambda_gap_list[: min(max_num_speaker, len(lambda_gap_list))]
    ) + 1
    return num_of_spk, lambdas, lambda_gap_list

def addAnchorEmb(emb, anchor_sample_n, anchor_spk_n, sigma):
    """Injects synthetic anchor coordinates to stabilize data sparsity paths."""
    emb_dim   = emb.shape[1]
    std_org   = np.std(emb, axis=0)
    new_embs  = []
    for _ in range(anchor_spk_n):
        emb_m     = np.tile(np.random.randn(1, emb_dim), (anchor_sample_n, 1))
        emb_noise = np.random.randn(anchor_sample_n, emb_dim).T
        emb_noise = np.dot(np.diag(std_org), emb_noise / np.max(np.abs(emb_noise))).T
        new_embs.append(emb_m + sigma * emb_noise)
    new_embs.append(emb)
    return np.vstack(new_embs)

def getEnhancedSpeakerCount(emb, cuda, random_test_count=5, anchor_spk_n=3, anchor_sample_n=10, sigma=50):
    est_list = []
    for seed in range(random_test_count):
        np.random.seed(seed)
        emb_aug = addAnchorEmb(emb, anchor_sample_n, anchor_spk_n, sigma)
        mat     = getCosAffinityMatrix(emb_aug)
        nmesc   = NMESC(mat, max_num_speaker=emb.shape[0], max_rp_threshold=0.25, sparse_search=True, sparse_search_volume=30, cuda=cuda)
        est_num, _ = nmesc.NMEanalysis()
        est_list.append(est_num)
    ctt = Counter(est_list)
    return max(ctt.most_common(1)[0][0] - anchor_spk_n, 1)

class NMESC:
    def __init__(self, mat, max_num_speaker=10, max_rp_threshold=0.25, sparse_search=True, sparse_search_volume=30, cuda=False):
        self.mat                    = mat
        self.max_num_speaker        = max_num_speaker
        self.max_rp_threshold       = max_rp_threshold
        self.sparse_search          = sparse_search
        self.sparse_search_volume   = sparse_search_volume
        self.cuda                   = cuda
        self.eps                    = 1e-10
        self.max_N                  = None
        self.p_value_list           = []

    def NMEanalysis(self):
        eig_ratio_list  = []
        est_spk_n_dict  = {}
        self.p_value_list = self.getPvalueList()

        for p_value in self.p_value_list:
            est_num_of_spk, g_p = self.getEigRatio(p_value)
            est_spk_n_dict[p_value] = est_num_of_spk
            eig_ratio_list.append(g_p)

        index_nn    = np.argmin(eig_ratio_list)
        rp_p_value  = self.p_value_list[index_nn]
        affinity_mat = getAffinityGraphMat(self.mat, rp_p_value)

        if not isGraphFullyConnected(affinity_mat):
            affinity_mat, rp_p_value = getMinimumConnection(self.mat, self.max_N, self.p_value_list)

        est_num_of_spk = est_spk_n_dict[rp_p_value]
        return est_num_of_spk, rp_p_value

    def getEigRatio(self, p_neighbors):
        affinity_mat = getAffinityGraphMat(self.mat, p_neighbors)
        est_num_of_spk, lambdas, lambda_gap_list = estimateNumofSpeakers(affinity_mat, self.max_num_speaker, self.cuda)
        arg_sorted_idx = np.argsort(lambda_gap_list[: self.max_num_speaker])[::-1]
        max_key     = arg_sorted_idx[0]
        max_eig_gap = lambda_gap_list[max_key] / (max(lambdas) + self.eps)
        g_p         = (p_neighbors / self.mat.shape[0]) / (max_eig_gap + self.eps)
        return est_num_of_spk, g_p

    def getPvalueList(self):
        self.max_N = int(self.mat.shape[0] * self.max_rp_threshold)
        N = min(self.max_N, self.sparse_search_volume)
        return list(np.linspace(1, self.max_N, N, endpoint=True).astype(int))

class SpectralClustering:
    def __init__(self, n_clusters=8, random_state=0, n_init=10, cuda=False):
        self.n_clusters   = n_clusters
        self.random_state = random_state
        self.n_init       = n_init
        self.cuda         = cuda

    def predict(self, X):
        spectral_emb = self._getSpectralEmbeddings(X, n_spks=self.n_clusters, cuda=self.cuda)
        _, labels, _ = k_means(spectral_emb, self.n_clusters, random_state=self.random_state, n_init=self.n_init)
        return labels

    def _getSpectralEmbeddings(self, affinity_mat, n_spks=8, cuda=False):
        laplacian        = getLaplacian(affinity_mat)
        _, diff_         = eigDecompose(laplacian, cuda)
        diffusion_map    = diff_[:, :n_spks]
        embedding        = diffusion_map.T[n_spks::-1]
        return embedding[:n_spks].T

def main():
    parser = argparse.ArgumentParser(description="Auto-Tuning Low-DER Spectral Clustering Execution Backend.")
    parser.add_argument("--embeddings_dir", default="./processed_data")
    parser.add_argument("--output_dir", default="./processed_data/cos+sc")
    parser.add_argument("--max_speaker", type=int, default=25)
    parser.add_argument("--max_rp_threshold", type=float, default=0.12) # 0.12 or 0.10 optimized for 2% targets
    parser.add_argument("--sparse_search_volume", type=int, default=30)
    parser.add_argument("--enhanced_count_thres", type=int, default=80)
    parser.add_argument("--cuda", action="store_true")
    args = parser.parse_args()

    embeddings_dir = Path(args.embeddings_dir)
    output_dir     = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    npz_files = sorted(embeddings_dir.glob("*_embeddings.npz"))
    
    for idx, npz_path in enumerate(npz_files, start=1):
        file_id   = npz_path.stem.replace("_embeddings", "")
        json_path = npz_path.parent / f"{file_id}_metadata.json"

        if not json_path.exists():
            continue

        # Load embedding tensors safely using data key checks
        data = np.load(str(npz_path))
        emb = data[list(data.keys())[0]] if not isinstance(data, np.ndarray) else data
        with open(str(json_path)) as f:
            metadata = json.load(f)

        N = emb.shape[0]
        mat = getCosAffinityMatrix(emb)

        # Dynamic Anchor speaker resolution logic
        est_num_of_spk_enhanced = None
        if N <= max(args.enhanced_count_thres, 6):
            est_num_of_spk_enhanced = getEnhancedSpeakerCount(emb, args.cuda)

        nmesc = NMESC(mat, max_num_speaker=args.max_speaker, max_rp_threshold=args.max_rp_threshold, sparse_search_volume=args.sparse_search_volume, cuda=args.cuda)

        if N > 6:
            est_num_of_spk, p_hat_value = nmesc.NMEanalysis()
            affinity_mat = getAffinityGraphMat(mat, p_hat_value)
        else:
            affinity_mat    = mat
            est_num_of_spk  = 1
            p_hat_value     = 1

        final_num_spk = est_num_of_spk_enhanced if est_num_of_spk_enhanced is not None else est_num_of_spk
        print(f"[{idx:02d}] Processing Recording Track: {file_id} | Solved Auto-Tuning Speaker Count: {final_num_spk}")

        spectral_model = SpectralClustering(n_clusters=final_num_spk, cuda=args.cuda)
        labels         = spectral_model.predict(affinity_mat)

        # Reference-Aligned Contiguous Midpoint Truncation Matrix Resolver
        segs = [[float(s["start"]), float(s["end"]), f"speaker_{l:02d}"] for s, l in zip(metadata, labels)]
        new_segs = []
        if len(segs) > 0:
            for i in range(len(segs) - 1):
                start, end, label = segs[i]
                next_start, _, _ = segs[i+1]
                if end > next_start:
                    avg = (next_start + end) / 2.0
                    segs[i+1][0] = avg
                    new_segs.append([start, avg, label])
                else:
                    new_segs.append([start, end, label])
            new_segs.append(segs[-1])

        # Write final optimized RTTM track map
        rttm_path = output_dir / f"{file_id}.rttm"
        with open(rttm_path, "w") as f:
            for start, end, label in new_segs:
                dur = end - start
                if dur > 0.001:
                    f.write(f"SPEAKER {file_id} 1 {start:.3f} {dur:.3f} <NA> <NA> {label} <NA> <NA>\n")

    print(f"\nMethod: Finished. All low-DER targets exported to: {output_dir}")

if __name__ == "__main__":
    main()