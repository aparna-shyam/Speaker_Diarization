import os
import json
import numpy as np
import torch
import torchaudio
import argparse
# FIXED: Updated deprecated import from speechbrain.pretrained to speechbrain.inference
from speechbrain.inference import EncoderClassifier
from tqdm import tqdm

# FIXED: Completely disable cuDNN to fix the CUDNN_STATUS_NOT_INITIALIZED driver crash.
# This forces PyTorch to use standard, stable CUDA kernels instead of the broken system cuDNN.
torch.backends.cudnn.enabled = False
torch.backends.cudnn.benchmark = False

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

def get_sliding_windows(segment_start, segment_end, max_len=1.5, hop_size=0.75, min_len=0.5):
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
    """Merges all segments into a single list of continuous speech regions (Oracle VAD)."""
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

def extract_features(args):
    # FIXED: Device formatting is now exactly what SpeechBrain's string splitter expects
    device_str = "cuda:0" if torch.cuda.is_available() else "cpu"
    print(f"Loading ECAPA-TDNN model from Hugging Face (saving to {args.model_dir}) on {device_str}...")
    
    classifier = EncoderClassifier.from_hparams(
        source="speechbrain/spkrec-ecapa-voxceleb", 
        savedir=args.model_dir, 
        run_opts={"device": device_str}
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
        for file_id, segs in tqdm(file_segments.items(), desc=f"File IDs"):
            if file_id not in audio_files:
                print(f"Warning: Audio file for {file_id} not found in {args.audio_dir}. Skipping.")
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
            vad_segs = merge_vad_segments(segs)
            
            for seg in vad_segs:
                windows = get_sliding_windows(seg["start"], seg["end"], max_len=args.max_seg_len, hop_size=args.hop_size, min_len=args.min_seg_len)
                
                for w_start, w_end in windows:
                    start_sample = int(w_start * fs)
                    end_sample = int(w_end * fs)
                    
                    audio_chunk = signal[:, start_sample:end_sample].to(device_str)
                    
                    with torch.no_grad():
                        emb = classifier.encode_batch(audio_chunk)
                        emb = emb.squeeze(1).cpu().numpy()
                        
                        embeddings_list.append({
                            "file_id": file_id,
                            "start": w_start,
                            "end": w_end,
                            "embedding": emb.squeeze()
                        })
            
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
                
                out_json_path = os.path.join(args.out_dir, f"{file_id}_metadata.json")
                with open(out_json_path, "w") as jf:
                    json.dump(metadata, jf, indent=4)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract ECAPA-TDNN embeddings using RTTM files.")
    parser.add_argument("--model_dir", type=str, default="./pretrained_models/ecapa", help="Path to ECAPA model directory.")
    parser.add_argument("--audio_dir", type=str, default="/DATA/nikhil-data/diarisation_dataset/ami_mixed/split_audio/test", help="Directory containing audio files.")
    parser.add_argument("--rttm_dir", type=str, default="/DATA/nikhil-data/diarisation_dataset/ami_mixed/BUT_rttms/test", help="Directory containing .rttm files.")
    parser.add_argument("--out_dir", type=str, default="./output_ami_split/embeddings_raw", help="Output directory for embeddings.")
    
    parser.add_argument("--max_seg_len", type=float, default=3.0, help="Maximum segment length in seconds.")
    parser.add_argument("--min_seg_len", type=float, default=1.0, help="Minimum segment length in seconds.")
    parser.add_argument("--hop_size", type=float, default=1.5, help="Hop size in seconds for sliding window.")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size for GPU extraction.")
    
    args = parser.parse_args()
    extract_features(args)