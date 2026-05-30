import os
import json
import numpy as np
import torch
import torchaudio
import argparse
from speechbrain.inference.classifiers import EncoderClassifier
from tqdm import tqdm

def parse_rttm(rttm_path):
    """Parses RTTM file and returns a list of dictionaries with segment details."""
    segments = []
    file_id = os.path.splitext(os.path.basename(rttm_path))[0]
    with open(rttm_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 5 and parts[0] == "SPEAKER":
                start = float(parts[3])
                duration = float(parts[4])
                segments.append({
                    "file_id": file_id,
                    "start": start,
                    "duration": duration,
                    "end": start + duration
                })
    return segments

def get_sliding_windows(segment_start, segment_end, max_len=3.0, hop_size=1.5, min_len=1.0):
    """Yields (start, end) tuples for sliding windows within a segment."""
    windows = []
    current_start = segment_start
    while current_start + max_len <= segment_end:
        windows.append((current_start, current_start + max_len))
        current_start += hop_size
    
    if current_start < segment_end:
        remaining_len = segment_end - current_start
        if remaining_len >= min_len:
            windows.append((current_start, segment_end))
    return windows

def merge_vad_segments(segs):
    """Merges all segments into continuous speech regions (Oracle VAD), handling overlaps."""
    if not segs:
        return []
    sorted_segs = sorted(segs, key=lambda x: x["start"])
    merged = []
    current_start = sorted_segs[0]["start"]
    current_end = sorted_segs[0]["end"]
    
    for seg in sorted_segs[1:]:
        if seg["start"] <= current_end:
            current_end = max(current_end, seg["end"])
        else:
            merged.append({"start": current_start, "end": current_end})
            current_start = seg["start"]
            current_end = seg["end"]
            
    merged.append({"start": current_start, "end": current_end})
    return merged

def test_cuda_environment():
    """Safety validation check to prevent dynamic driver-level cuDNN init errors."""
    if not torch.cuda.is_available():
        return False
    try:
        t = torch.randn(1, 1, 16).cuda()
        m = torch.nn.Conv1d(1, 1, 3).cuda()
        _ = m(t)
        return True
    except Exception:
        return False

def extract_features(args):
    torch.set_num_threads(8)
    torch.set_num_interop_threads(8)
    
    if test_cuda_environment():
        device = "cuda:0"
        print("CUDA environment check passed. Processing vectors on GPU.")
    else:
        device = "cpu"
        print("[WARNING] Hardware environment initialization failed. Processing on CPU.")
        
    print(f"Loading ECAPA-TDNN embedding model on device context: {device}...")
    classifier = EncoderClassifier.from_hparams(
        source=args.model_dir, 
        run_opts={"device": device}
    )
    
    os.makedirs(args.out_dir, exist_ok=True)
    
    audio_files = {os.path.splitext(f)[0]: os.path.join(args.audio_dir, f) 
                   for f in os.listdir(args.audio_dir) if f.endswith('.wav') or f.endswith('.flac')}
    rttm_files = [f for f in os.listdir(args.rttm_dir) if f.endswith(".rttm")]
    
    # Manifest line logs requested by your previous configuration
    affinity_score_lines, segment_file_lines, embedding_scp_lines = [], [], []
    
    for rttm_file in rttm_files:
        rttm_path = os.path.join(args.rttm_dir, rttm_file)
        segments = parse_rttm(rttm_path)
        if not segments:
            continue
            
        file_id = segments[0]["file_id"]
        if file_id not in audio_files:
            print(f"Warning: Audio file for {file_id} not found. Skipping.")
            continue
            
        audio_path = audio_files[file_id]
        signal, fs = torchaudio.load(audio_path)
        
        if signal.shape[0] > 1:
            signal = signal.mean(dim=0, keepdim=True)
        if fs != 16000:
            resampler = torchaudio.transforms.Resample(orig_freq=fs, new_freq=16000)
            signal = resampler(signal)
            fs = 16000
            
        embeddings_list = []
        vad_segs = merge_vad_segments(segments)
        
        for seg in vad_segs:
            windows = get_sliding_windows(seg["start"], seg["end"], max_len=args.max_seg_len, hop_size=args.hop_size, min_len=args.min_seg_len)
            for w_start, w_end in windows:
                start_sample = int(w_start * fs)
                end_sample = int(w_end * fs)
                audio_chunk = signal[:, start_sample:end_sample].to(device)
                
                if audio_chunk.shape[1] < int(0.1 * fs):
                    continue
                    
                with torch.no_grad():
                    emb = classifier.encode_batch(audio_chunk)
                    # Squeezing only batch/time frames to protect raw shape dimension stability
                    emb = emb.squeeze(0).squeeze(0).cpu().numpy()
                    
                    embeddings_list.append({
                        "file_id": file_id,
                        "start": w_start,
                        "end": w_end,
                        "embedding": emb
                    })
                    
                    # Generate requested manifest string tokens
                    start_str, end_str = f"{int(round(w_start * 100))}", f"{int(round(w_end * 100))}"
                    segment_id = f"{file_id}-{start_str}-{end_str}"
                    segment_file_lines.append(f"{segment_id} {file_id}\n")
                    embedding_scp_lines.append(f"{segment_id} dummy_not_used\n")
        
        if embeddings_list:
            embeddings = np.array([item["embedding"] for item in embeddings_list])
            metadata = []
            for i, item in enumerate(embeddings_list):
                metadata.append({
                    "segment_id": f"{file_id}_{i:04d}",
                    "file_id": item["file_id"],
                    "start": item["start"],
                    "end": item["end"]
                })
                
            out_npz_path = os.path.join(args.out_dir, f"{file_id}_embeddings.npz")
            np.savez(out_npz_path, embedding=embeddings)
            
            # Additionally store the standard pure .npy file requested by your output configurations
            np.save(os.path.join(args.out_dir, f"{file_id}_embeddings.npy"), embeddings)
            
            out_json_path = os.path.join(args.out_dir, f"{file_id}_metadata.json")
            with open(out_json_path, "w") as jf:
                json.dump(metadata, jf, indent=4)
                
            affinity_score_lines.append(f"{file_id} {os.path.abspath(out_npz_path)}\n")

    # Save tracking files inside your data root layout
    with open(os.path.join(args.out_dir, "affinity_score_file.txt"), "w") as f: f.writelines(affinity_score_lines)
    with open(os.path.join(args.out_dir, "segments_file.txt"), "w") as f: f.writelines(segment_file_lines)
    with open(os.path.join(args.out_dir, "embedding_segments.scp"), "w") as f: f.writelines(embedding_scp_lines)
    print("\nPhase 1 Feature Extraction Complete! Saved all raw arrays and JSON metadata tracks.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract ECAPA-TDNN embeddings matching reference guidelines.")
    parser.add_argument("--model_dir", type=str, default="speechbrain/spkrec-ecapa-voxceleb")
    parser.add_argument("--audio_dir", type=str, default="/DATA/nikhil-data/diarisation_dataset/ami_mixed/split_audio/test")
    parser.add_argument("--rttm_dir", type=str, default="/DATA/nikhil-data/diarisation_dataset/ami_mixed/BUT_rttms/test")
    parser.add_argument("--out_dir", type=str, default="./processed_data")
    
    # 2% DER Target Settings: Optimized Fine-Grained Windows
    parser.add_argument("--max_seg_len", type=float, default=2.0)
    parser.add_argument("--min_seg_len", type=float, default=0.5)
    parser.add_argument("--hop_size", type=float, default=0.5)
    
    args = parser.parse_args()
    extract_features(args)