import os
import glob
from pyannote.database.util import load_rttm
from pyannote.metrics.diarization import DiarizationErrorRate

class Config:
    REF_RTTM_DIR = "/DATA/nikhil-data/diarisation_dataset/ami_mixed/BUT_rttms/test"
    HYP_RTTM_DIR = "/home/teaching/spkdia/output_ami_split/cos+sc"
    OUTPUT_REPORT = "./output_ami_split/final_performance_report.txt"

def evaluate_rttms():   
    print(f"Loading Reference RTTMs from: {Config.REF_RTTM_DIR}")
    print(f"Loading Hypothesis RTTMs from: {Config.HYP_RTTM_DIR}\n")
    
    ref_files = glob.glob(os.path.join(Config.REF_RTTM_DIR, "*.rttm"))
    if not ref_files:
        print("[ERROR] No reference RTTM targets found.")
        return
        
    metric = DiarizationErrorRate(skip_overlap=True, collar=0.25)
    evaluated_files = 0
    
    for ref_file in sorted(ref_files):
        basename = os.path.basename(ref_file)
        stem = os.path.splitext(basename)[0]
        hyp_file = os.path.join(Config.HYP_RTTM_DIR, basename)
        
        if not os.path.exists(hyp_file):
            continue
            
        try:
            ref_dict = load_rttm(ref_file)
            hyp_dict = load_rttm(hyp_file)
            
            if not ref_dict or not hyp_dict:
                continue
                
            ref_uri = list(ref_dict.keys())[0]
            hyp_uri = list(hyp_dict.keys())[0]
            
            file_der = metric(ref_dict[ref_uri], hyp_dict[hyp_uri])
            print(f" -> URI Session ID: {stem:12} | Measured File DER: {abs(file_der):.2%}")
            evaluated_files += 1
        except Exception as e:
            print(f"[ERROR] Track skipped {stem}: {str(e)}")
            
    if evaluated_files == 0:
        print("[ERROR] No valid overlapping matching pairs found.")
        return
        
    print("\n" + "="*60)
    print(" COMPREHENSIVE SYSTEM PERFORMANCE EVALUATION")
    print("="*60)
    
    overall_der = abs(metric)
    print(f" GLOBAL AVG DIARIZATION ERROR RATE (DER): {overall_der:.2%}\n")
    metric.report(display=True)
    
    if Config.OUTPUT_REPORT:
        os.makedirs(os.path.dirname(Config.OUTPUT_REPORT), exist_ok=True)
        with open(Config.OUTPUT_REPORT, "w") as f:
            f.write(f"GLOBAL AVG DIARIZATION ERROR RATE (DER): {overall_der:.2%}\n\n")
            f.write(metric.report().to_string())

if __name__ == "__main__":
    evaluate_rttms()