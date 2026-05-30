import os
import numpy as np
from pyannote.core import Annotation, Segment
from pyannote.metrics.diarization import DiarizationErrorRate

GROUND_TRUTH_DIR = "/DATA/nikhil-data/diarisation_dataset/ami_mixed/BUT_rttms/test"
PREDICTED_RTTM = "./predicted_output.rttm"

def load_rttm(path, target_session=None):
    annotation = Annotation()
    if not os.path.exists(path): 
        return annotation
    with open(path, "r") as f:
        for line in f:
            parts = line.strip().split()
            if not parts or parts[0] != "SPEAKER": 
                continue
            session_id = parts[1]
            if target_session and session_id != target_session: 
                continue
            start, duration, speaker = float(parts[3]), float(parts[4]), parts[7]
            annotation[Segment(start, start + duration)] = speaker
    return annotation

def main():
    if not os.path.exists(PREDICTED_RTTM):
        print(f"[ERROR] Output path {PREDICTED_RTTM} missing.")
        return

    predicted_sessions = set()
    with open(PREDICTED_RTTM, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if parts: 
                predicted_sessions.add(parts[1])
            
    # Explicitly matching target reference parameters: 250ms collar, skip evaluated overlap
    metric = DiarizationErrorRate(skip_overlap=True, collar=0.25)
    der_scores = []
    
    print("\nParsing and grading individual meeting file tracks...")
    print("-" * 65)
    for session in sorted(list(predicted_sessions)):
        gt_file = os.path.join(GROUND_TRUTH_DIR, f"{session}.rttm")
        if not os.path.exists(gt_file): 
            print(f"[WARNING] Reference file not found for {session}. Skipping...")
            continue
            
        ref = load_rttm(gt_file, target_session=session)
        hyp = load_rttm(PREDICTED_RTTM, target_session=session)
        
        if len(ref) == 0 or len(hyp) == 0:
            print(f"[WARNING] Found empty track timeline mappings for {session}.")
            continue
            
        session_der = metric(ref, hyp, file_index=session)
        der_percentage = abs(session_der) * 100
        der_scores.append(der_percentage)
        print(f" -> Meeting ID: {session:10} | Diarization Error Rate (DER): {der_percentage:.2f}%")
        
    global_average_der = np.mean(der_scores) if der_scores else 0.0
    
    print("\n" + "="*65)
    print(" BATCH DATASET EVALUATION COMPLETE")
    print(f" FINAL AVERAGE DIARIZATION ERROR RATE (DER): {global_average_der:.2f}%")
    print("="*65 + "\n")

if __name__ == "__main__":
    main()