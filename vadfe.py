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
    
    # 2.0s window balanced context extraction
    parser.add_argument('--window_len', type=float, default=2.0)
    parser.add_argument('--hop_len', type=float, default=1.0)
    parser.add_argument('--min_len', type=float, default=0.5)
    return parser.parse_args()

def load_strict_single_speaker_vad(rttm_path):
    raw_intervals = []
    if not os.path.exists(rttm_path): return []
    with open(rttm_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 5 or parts[0] != "SPEAKER": continue
            raw_intervals.append((float(parts[3]), float(parts[3]) + float(parts[4])))
    if not raw_intervals: return []
    
    max_time = max(end for start, end in raw_intervals) + 1.0
    time_step = 0.01  
    timeline_slots = np.zeros(int(max_time / time_step) + 1, dtype=int)
    
    for start, end in raw_intervals:
        timeline_slots[int(round(start / time_step)):int(round(end / time_step))] += 1

    single_speaker_intervals = []
    in_speech_block = False
    start_time = 0.0
    for idx, speaker_count in enumerate(timeline_slots):
        current_time = idx * time_step
        if speaker_count == 1:
            if not in_speech_block:
                start_time = current_time
                in_speech_block = True
        else:
            if in_speech_block:
                if (current_time - start_time) >= 0.01:
                    single_speaker_intervals.append((start_time, current_time))
                in_speech_block = False
    return single_speaker_intervals

def compute_cosine_affinity(embeddings):
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms[norms == 0] = 1e-10  
    normalized_embeddings = embeddings / norms
    return np.dot(normalized_embeddings, normalized_embeddings.T)

def main():
    args = parse_arguments()
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Force Multi-threading CPU optimizations to prevent frozen extraction runs
    torch.set_num_threads(8)
    torch.set_num_interop_threads(8)
    
    device = "cpu"
    if torch.cuda.is_available():
        try:
            t = torch.randn(1, 1, 16).cuda()
            m = torch.nn.Conv1d(1, 1, 3).cuda()
            _ = m(t)
            device = "cuda"
        except: pass
        
    print(f"Loading model on device: {device}...")
    classifier = EncoderClassifier.from_hparams(source="speechbrain/spkrec-ecapa-voxceleb", run_opts={"device": device})
    
    affinity_score_lines, segment_file_lines, embedding_scp_lines = [], [], []
    audio_files = [f for f in os.listdir(args.audio_dir) if f.endswith('.wav')]
    
    for audio_file in tqdm(audio_files, desc="Extracting Audio Speaker Profiles"):
        session_id = os.path.splitext(audio_file)[0]
        audio_path = os.path.join(args.audio_dir, audio_file)
        rttm_path = os.path.join(args.rttm_dir, f"{session_id}.rttm")
        if not os.path.exists(rttm_path): continue
            
        speech_intervals = load_strict_single_speaker_vad(rttm_path)
        if not speech_intervals: continue
            
        audio, sample_rate = sf.read(audio_path)
        if len(audio.shape) > 1: audio = audio[:, 0]  
        session_embeddings = []
        
        for start_bound, end_bound in speech_intervals:
            current_pos = start_bound
            while current_pos + args.min_len <= end_bound:
                current_win_end = min(current_pos + args.window_len, end_bound)
                if (current_win_end - current_pos) < args.min_len: break
                
                audio_chunk = audio[int(current_pos * sample_rate):int(current_win_end * sample_rate)]
                with torch.no_grad():
                    audio_tensor = torch.tensor(audio_chunk, dtype=torch.float32).unsqueeze(0).to(device)
                    embedding = classifier.encode_batch(audio_tensor).squeeze().cpu().numpy()
                
                session_embeddings.append(embedding)
                start_str, end_str = f"{int(round(current_pos * 100))}", f"{int(round(current_win_end * 100))}"
                segment_id = f"{session_id}-{start_str}-{end_str}"
                
                segment_file_lines.append(f"{segment_id} {session_id}\n")
                embedding_scp_lines.append(f"{segment_id} dummy_not_used\n")
                
                if current_pos + args.window_len >= end_bound: break
                current_pos += args.hop_len
        
        if len(session_embeddings) == 0: continue
        session_embeddings = np.vstack(session_embeddings)
        np.save(os.path.abspath(os.path.join(args.output_dir, f"{session_id}_affinity.npy")), compute_cosine_affinity(session_embeddings))
        affinity_score_lines.append(f"{session_id} {os.path.abspath(os.path.join(args.output_dir, f'{session_id}_affinity.npy'))}\n")

    with open(os.path.join(args.output_dir, "affinity_score_file.txt"), "w") as f: f.writelines(affinity_score_lines)
    with open(os.path.join(args.output_dir, "segments_file.txt"), "w") as f: f.writelines(segment_file_lines)
    with open(os.path.join(args.output_dir, "embedding_segments.scp"), "w") as f: f.writelines(embedding_scp_lines)
    print("\nPhase 1 Extraction Complete!")

if __name__ == "__main__":
    main()