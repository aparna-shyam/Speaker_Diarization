import os
import json
import numpy as np
import torch
import torchaudio
import argparse
from speechbrain.pretrained import EncoderClassifier
from tqdm import tqdm

def parse_rttm(rttm_path):
    """
    Parses RTTM file and returns a list of dictionaries with segment details.
    """
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

def get_sliding_windows(segment_start, segment_end, max_len=1.5, hop_size=0.75, min_len=0.5):
    """
    Yields (start, end) tuples for sliding windows within a segment.
    """
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
    """
    Merges all segments into a single list of continuous speech regions (Oracle VAD),
    ignoring speaker IDs and handling overlaps.
    """
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

def extract_xvector(classifier, audio_chunk):
    """
    Extracts a 512-dim x-vector from an audio chunk.
    SpeechBrain's spkrec-xvect-voxceleb returns shape (1, 1, 512).
    We squeeze to (512,) for storage.
    """
    with torch.no_grad():
        emb = classifier.encode_batch(audio_chunk)   # (1, 1, 512)
        emb = emb.squeeze(0).squeeze(0).cpu().numpy()  # (512,)

    assert emb.shape[0] == 512, (
        f"Expected 512-dim x-vector, got shape {emb.shape}. "
        "Ensure --model_dir points to 'speechbrain/spkrec-xvect-voxceleb'."
    )
    return emb

def extract_features(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading X-vector model from '{args.model_dir}' on {device}...")
    print("Expected output dimensionality: 512-dim x-vectors.")

    # X-vector model: speechbrain/spkrec-xvect-voxceleb
    classifier = EncoderClassifier.from_hparams(
        source=args.model_dir,
        savedir=args.model_dir,
        run_opts={"device": device}
    )
    
    os.makedirs(args.out_dir, exist_ok=True)
    
    audio_files = {os.path.splitext(f)[0]: os.path.join(args.audio_dir, f) 
                   for f in os.listdir(args.audio_dir) if f.endswith('.wav') or f.endswith('.flac')}
    
    rttm_files = [f for f in os.listdir(args.rttm_dir) if f.endswith(".rttm")]
    
    for rttm_file in rttm_files:
        rttm_path = os.path.join(args.rttm_dir, rttm_file)
        segments = parse_rttm(rttm_path)
        
        file_segments = {}
        for seg in segments:
            file_id = seg["file_id"]
            if file_id not in file_segments:
                file_segments[file_id] = []
            file_segments[file_id].append(seg)
            
        print(f"\nProcessing {rttm_file}...")
        for file_id, segs in tqdm(file_segments.items(), desc="File IDs"):
            if file_id not in audio_files:
                print(f"Warning: Audio file for {file_id} not found in {args.audio_dir}. Skipping.")
                continue
                
            audio_path = audio_files[file_id]
            signal, fs = torchaudio.load(audio_path)
            
            if signal.shape[0] > 1:
                signal = signal.mean(dim=0, keepdim=True)
            
            # X-vector model also expects 16 kHz input
            if fs != 16000:
                resampler = torchaudio.transforms.Resample(orig_freq=fs, new_freq=16000)
                signal = resampler(signal)
                fs = 16000
                
            embeddings_list = []
            vad_segs = merge_vad_segments(segs)
            
            for seg in vad_segs:
                windows = get_sliding_windows(
                    seg["start"], seg["end"],
                    max_len=args.max_seg_len,
                    hop_size=args.hop_size,
                    min_len=args.min_seg_len
                )
                
                for w_start, w_end in windows:
                    start_sample = int(w_start * fs)
                    end_sample   = int(w_end * fs)
                    audio_chunk  = signal[:, start_sample:end_sample].to(device)

                    # Skip chunks that are too short (< 100 ms)
                    if audio_chunk.shape[1] < int(0.1 * fs):
                        continue

                    # Extract 512-dim x-vector
                    emb = extract_xvector(classifier, audio_chunk)

                    embeddings_list.append({
                        "file_id": file_id,
                        "start":   w_start,
                        "end":     w_end,
                        "embedding": emb       # shape: (512,)
                    })
            
            if embeddings_list:
                embeddings = np.array([item["embedding"] for item in embeddings_list])

                # Sanity check: must be (N, 512)
                assert embeddings.ndim == 2 and embeddings.shape[1] == 512, (
                    f"Unexpected embedding matrix shape for {file_id}: {embeddings.shape}"
                )

                metadata = [
                    {
                        "segment_id":    f"{file_id}_{i:04d}",
                        "file_id":       item["file_id"],
                        "start":         item["start"],
                        "end":           item["end"],
                        "embedding_dim": 512
                    }
                    for i, item in enumerate(embeddings_list)
                ]

                out_npz_path = os.path.join(args.out_dir, f"{file_id}_embeddings.npz")
                np.savez(out_npz_path, embedding=embeddings)
                
                out_json_path = os.path.join(args.out_dir, f"{file_id}_metadata.json")
                with open(out_json_path, "w") as jf:
                    json.dump(metadata, jf, indent=4)

                print(f"  Saved {len(embeddings_list)} x-vectors {embeddings.shape} -> {out_npz_path}")

if __name__ == "__main__":
    # ── Change SPLIT here to switch between train and test ──────────────────
    SPLIT = "test"   # "train" or "test"
    # ────────────────────────────────────────────────────────────────────────

    BASE = "/DATA/nikhil-data/diarisation_dataset/ami_mixed"
    OUT  = "./output_ami_split"

    parser = argparse.ArgumentParser(description="Extract X-vector (512-dim) embeddings using RTTM files.")
    parser.add_argument("--model_dir",   type=str,   default="./xvec")
    parser.add_argument("--audio_dir",   type=str,   default=f"{BASE}/split_audio/{SPLIT}")
    parser.add_argument("--rttm_dir",    type=str,   default=f"{BASE}/BUT_rttms/{SPLIT}")
    parser.add_argument("--out_dir",     type=str,   default=f"{OUT}/xvector_embeddings_{SPLIT}")
    parser.add_argument("--max_seg_len", type=float, default=1.5)
    parser.add_argument("--min_seg_len", type=float, default=0.5)
    parser.add_argument("--hop_size",    type=float, default=0.75)
    parser.add_argument("--batch_size",  type=int,   default=64)

    args = parser.parse_args()
    print(f"Split    : {SPLIT}")
    print(f"audio_dir: {args.audio_dir}")
    print(f"rttm_dir : {args.rttm_dir}")
    print(f"out_dir  : {args.out_dir}\n")
    extract_features(args)
