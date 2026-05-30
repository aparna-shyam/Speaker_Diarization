import os
import argparse
import numpy as np
import soundfile as sf
import torch
from tqdm import tqdm
from speechbrain.inference.classifiers import EncoderClassifier

def parse_arguments():
    parser = argparse.ArgumentParser(description="Phase 1: Robust Context Feature Extraction")
    parser.add_argument('--audio_dir', type=str, default="/DATA/nikhil-data/diarisation_dataset/ami_mixed/split_audio/test")
    parser.add_argument('--rttm_dir', type=str, default="/DATA/nikhil-data/diarisation_dataset/ami_mixed/BUT_rttms/test")
    parser.add_argument('--output_dir', type=str, default="./processed_data")
    
    # 3.0s window balanced context extraction (Matching reference baseline defaults)
    parser.add_argument('--window_len', type=float, default=3.0)
    parser.add_argument('--hop_len', type=float, default=1.5)
    parser.add_argument('--min_len', type=float, default=1.0)
    return parser.parse_args()

def load_oracle_vad_segments(rttm_path):
    """
    Merges segments into a single list of continuous speech regions (Oracle VAD),
    ignoring speaker IDs and handling overlaps cleanly to match reference structure.
    """
    raw_intervals = []
    if not os.path.exists(rttm_path): 
        return []
    with open(rttm_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 5 and parts[0] == "SPEAKER":
                start = float(parts[3])
                duration = float(parts[4])
                raw_intervals.append({"start": start, "end": start + duration})
                
    if not raw_intervals: 
        return []
    
    # Sort segments by start time
    sorted_segs = sorted(raw_intervals, key=lambda x: x["start"])
    
    merged = []
    current_start = sorted_segs[0]["start"]
    current_end = sorted_segs[0]["end"]
    
    for seg in sorted_segs[1:]:
        if seg["start"] <= current_end:
            # Overlapping or adjacent, extend the current end region
            current_end = max(current_end, seg["end"])
        else:
            # Gap found, commit the completed speech block
            merged.append((current_start, current_end))
            current_start = seg["start"]
            current_end = seg["end"]
            
    merged.append((current_start, current_end))
    return merged

def compute_cosine_affinity(embeddings):
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms[norms == 0] = 1e-10  
    normalized_embeddings = embeddings / norms
    return np.dot(normalized_embeddings, normalized_embeddings.T)

def main():
    args = parse_arguments()
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Core CPU thread safety assignments
    torch.set_num_threads(8)
    torch.set_num_interop_threads(8)
    
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    print(f"Attempting to load model on device: {device}...")
    
    try:
        classifier = EncoderClassifier.from_hparams(source="speechbrain/spkrec-ecapa-voxceleb", run_opts={"device": device})
    except Exception as e:
        print(f"GPU initialization failed ({e}). Falling back to CPU mode safely...")
        device = "cpu"
        classifier = EncoderClassifier.from_hparams(source="speechbrain/spkrec-ecapa-voxceleb", run_opts={"device": device})
    
    affinity_score_lines, segment_file_lines, embedding_scp_lines = [], [], []
    audio_files = [f for f in os.listdir(args.audio_dir) if f.endswith('.wav') or f.endswith('.flac')]
    
    for audio_file in tqdm(audio_files, desc="Extracting Audio Speaker Profiles"):
        session_id = os.path.splitext(audio_file)[0]
        audio_path = os.path.join(args.audio_dir, audio_file)
        rttm_path = os.path.join(args.rttm_dir, f"{session_id}.rttm")
        if not os.path.exists(rttm_path): 
            continue
            
        speech_intervals = load_oracle_vad_segments(rttm_path)
        if not speech_intervals: 
            continue
            
        audio, sample_rate = sf.read(audio_path)
        if len(audio.shape) > 1: 
            audio = audio.mean(axis=1)  
            
        session_embeddings = []
        
        for start_bound, end_bound in speech_intervals:
            current_pos = start_bound
            
            while current_pos + args.window_len <= end_bound:
                current_win_end = current_pos + args.window_len
                audio_chunk = audio[int(current_pos * sample_rate):int(current_win_end * sample_rate)]
                
                if len(audio_chunk) < int(0.1 * sample_rate):
                    current_pos += args.hop_len
                    continue
                
                try:
                    with torch.no_grad():
                        audio_tensor = torch.tensor(audio_chunk, dtype=torch.float32).unsqueeze(0).to(device)
                        embedding = classifier.encode_batch(audio_tensor).squeeze(0).squeeze(0).cpu().numpy()
                except RuntimeError as e:
                    if "cuDNN" in str(e) and device != "cpu":
                        print("\n[VADFE] cuDNN failure caught mid-run. Switching dynamic pipeline context to CPU...")
                        device = "cpu"
                        classifier = EncoderClassifier.from_hparams(source="speechbrain/spkrec-ecapa-voxceleb", run_opts={"device": device})
                        with torch.no_grad():
                            audio_tensor = torch.tensor(audio_chunk, dtype=torch.float32).unsqueeze(0).to(device)
                            embedding = classifier.encode_batch(audio_tensor).squeeze(0).squeeze(0).cpu().numpy()
                    else:
                        raise e
                
                session_embeddings.append(embedding)
                start_str = f"{int(round(current_pos * 100))}"
                end_str = f"{int(round(current_win_end * 100))}"
                segment_id = f"{session_id}-{start_str}-{end_str}"
                
                segment_file_lines.append(f"{segment_id} {session_id}\n")
                embedding_scp_lines.append(f"{segment_id} dummy_not_used\n")
                
                current_pos += args.hop_len
                
            if current_pos < end_bound and (end_bound - current_pos) >= args.min_len:
                audio_chunk = audio[int(current_pos * sample_rate):int(end_bound * sample_rate)]
                
                if len(audio_chunk) < int(0.1 * sample_rate):
                    continue
                    
                try:
                    with torch.no_grad():
                        audio_tensor = torch.tensor(audio_chunk, dtype=torch.float32).unsqueeze(0).to(device)
                        embedding = classifier.encode_batch(audio_tensor).squeeze(0).squeeze(0).cpu().numpy()
                except RuntimeError as e:
                    if "cuDNN" in str(e) and device != "cpu":
                        print("\n[VADFE] cuDNN failure caught mid-run. Switching dynamic pipeline context to CPU...")
                        device = "cpu"
                        classifier = EncoderClassifier.from_hparams(source="speechbrain/spkrec-ecapa-voxceleb", run_opts={"device": device})
                        with torch.no_grad():
                            audio_tensor = torch.tensor(audio_chunk, dtype=torch.float32).unsqueeze(0).to(device)
                            embedding = classifier.encode_batch(audio_tensor).squeeze(0).squeeze(0).cpu().numpy()
                    else:
                        raise e
                    
                session_embeddings.append(embedding)
                start_str = f"{int(round(current_pos * 100))}"
                end_str = f"{int(round(end_bound * 100))}"
                segment_id = f"{session_id}-{start_str}-{end_str}"
                
                segment_file_lines.append(f"{segment_id} {session_id}\n")
                embedding_scp_lines.append(f"{segment_id} dummy_not_used\n")

        if len(session_embeddings) == 0: 
            continue
            
        session_embeddings = np.vstack(session_embeddings)
        out_npy_path = os.path.join(args.output_dir, f"{session_id}_affinity.npy")
        np.save(os.path.abspath(out_npy_path), compute_cosine_affinity(session_embeddings))
        affinity_score_lines.append(f"{session_id} {os.path.abspath(out_npy_path)}\n")

    with open(os.path.join(args.output_dir, "affinity_score_file.txt"), "w") as f: f.writelines(affinity_score_lines)
    with open(os.path.join(args.output_dir, "segments_file.txt"), "w") as f: f.writelines(segment_file_lines)
    with open(os.path.join(args.output_dir, "embedding_segments.scp"), "w") as f: f.writelines(embedding_scp_lines)
    print("\nPhase 1 Extraction Complete!")

if __name__ == "__main__":
    main()