import os
import argparse
import numpy as np
import soundfile as sf
import torch
import torchaudio
from tqdm import tqdm
from speechbrain.inference.classifiers import EncoderClassifier

def parse_arguments():
    parser = argparse.ArgumentParser(description="Phase 1: Robust Context Feature Extraction")
    parser.add_argument('--audio_dir', type=str, default="/DATA/nikhil-data/diarisation_dataset/ami_mixed/split_audio/test")
    parser.add_argument('--rttm_dir', type=str, default="/DATA/nikhil-data/diarisation_dataset/ami_mixed/BUT_rttms/test")
    parser.add_argument('--output_dir', type=str, default="./processed_data")
    parser.add_argument('--window_len', type=float, default=3.0)
    parser.add_argument('--hop_len', type=float, default=1.5)
    parser.add_argument('--min_len', type=float, default=1.0)
    return parser.parse_args()

def load_oracle_vad_segments(rttm_path):
    raw_intervals = []
    if not os.path.exists(rttm_path): return []
    with open(rttm_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 5 and parts[0] == "SPEAKER":
                start, duration = float(parts[3]), float(parts[4])
                raw_intervals.append({"start": start, "end": start + duration})
    if not raw_intervals: return []
    sorted_segs = sorted(raw_intervals, key=lambda x: x["start"])
    merged = []
    curr_start, curr_end = sorted_segs[0]["start"], sorted_segs[0]["end"]
    for seg in sorted_segs[1:]:
        if seg["start"] <= curr_end:
            curr_end = max(curr_end, seg["end"])
        else:
            merged.append((curr_start, curr_end))
            curr_start, curr_end = seg["start"], seg["end"]
    merged.append((curr_start, curr_end))
    return merged

def compute_cosine_affinity(embeddings):
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1e-10, norms)
    normalized = embeddings / norms
    sim = np.dot(normalized, normalized.T)
    sim = (sim - sim.min()) / (sim.max() - sim.min() + 1e-10)
    return sim

def test_cuda_environment():
    """Returns True if CUDA can cleanly compute a cuDNN layer, False otherwise."""
    if not torch.cuda.is_available():
        return False
    try:
        # Run a quick 1D dummy convolution to verify cuDNN backend initialization
        t = torch.randn(1, 1, 16).cuda()
        m = torch.nn.Conv1d(1, 1, 3).cuda()
        _ = m(t)
        return True
    except Exception:
        return False

def main():
    args = parse_arguments()
    os.makedirs(args.output_dir, exist_ok=True)
    torch.set_num_threads(8)
    torch.set_num_interop_threads(8)
    
    # Run proactive baseline environmental validation
    if test_cuda_environment():
        device = "cuda:0"
        print("CUDA environment checks passed! Running on GPU device context.")
    else:
        device = "cpu"
        print("[WARNING] GPU environment check or cuDNN initialization failed. Safely falling back to CPU mode...")
    
    classifier = EncoderClassifier.from_hparams(source="speechbrain/spkrec-ecapa-voxceleb", run_opts={"device": device})
        
    affinity_score_lines, segment_file_lines, embedding_scp_lines = [], [], []
    audio_files = [f for f in os.listdir(args.audio_dir) if f.endswith('.wav') or f.endswith('.flac')]
    
    for audio_file in tqdm(audio_files, desc="Extracting Audio Speaker Profiles"):
        session_id = os.path.splitext(audio_file)[0]
        audio_path = os.path.join(args.audio_dir, audio_file)
        rttm_path = os.path.join(args.rttm_dir, f"{session_id}.rttm")
        if not os.path.exists(rttm_path): continue
            
        speech_intervals = load_oracle_vad_segments(rttm_path)
        if not speech_intervals: continue
            
        signal, fs = torchaudio.load(audio_path)
        if signal.shape[0] > 1: signal = signal.mean(dim=0, keepdim=True)
        if fs != 16000:
            signal = torchaudio.transforms.Resample(orig_freq=fs, new_freq=16000)(signal)
            fs = 16000
        audio = signal.squeeze(0).numpy()
        
        session_embeddings = []
        for start_bound, end_bound in speech_intervals:
            current_pos = start_bound
            while current_pos + args.window_len <= end_bound:
                current_win_end = current_pos + args.window_len
                chunk = audio[int(current_pos * fs):int(current_win_end * fs)]
                if len(chunk) < int(0.1 * fs):
                    current_pos += args.hop_len
                    continue
                
                with torch.no_grad():
                    tensor = torch.tensor(chunk, dtype=torch.float32).unsqueeze(0).to(device)
                    emb = classifier.encode_batch(tensor).squeeze(0).squeeze(0).cpu().numpy()
                
                session_embeddings.append(emb)
                segment_id = f"{session_id}-{int(round(current_pos * 100))}-{int(round(current_win_end * 100))}"
                segment_file_lines.append(f"{segment_id} {session_id}\n")
                embedding_scp_lines.append(f"{segment_id} dummy_not_used\n")
                current_pos += args.hop_len
                
            if current_pos < end_bound and (end_bound - current_pos) >= args.min_len:
                chunk = audio[int(current_pos * fs):int(end_bound * fs)]
                if len(chunk) >= int(0.1 * fs):
                    with torch.no_grad():
                        tensor = torch.tensor(chunk, dtype=torch.float32).unsqueeze(0).to(device)
                        emb = classifier.encode_batch(tensor).squeeze(0).squeeze(0).cpu().numpy()
                    session_embeddings.append(emb)
                    segment_id = f"{session_id}-{int(round(current_pos * 100))}-{int(round(end_bound * 100))}"
                    segment_file_lines.append(f"{segment_id} {session_id}\n")
                    embedding_scp_lines.append(f"{segment_id} dummy_not_used\n")

        if len(session_embeddings) == 0: continue
        session_embeddings = np.vstack(session_embeddings)
        
        np.save(os.path.join(args.output_dir, f"{session_id}_embeddings.npy"), session_embeddings)
        out_npy_path = os.path.join(args.output_dir, f"{session_id}_affinity.npy")
        np.save(os.path.abspath(out_npy_path), compute_cosine_affinity(session_embeddings))
        affinity_score_lines.append(f"{session_id} {os.path.abspath(out_npy_path)}\n")

    with open(os.path.join(args.output_dir, "affinity_score_file.txt"), "w") as f: f.writelines(affinity_score_lines)
    with open(os.path.join(args.output_dir, "segments_file.txt"), "w") as f: f.writelines(segment_file_lines)
    with open(os.path.join(args.output_dir, "embedding_segments.scp"), "w") as f: f.writelines(embedding_scp_lines)
    print("\nPhase 1 Extraction Complete!")

if __name__ == "__main__":
    main()