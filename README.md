# Speaker_Diarization

This project implements speaker diarization using 512-dim X-vectors (SpeechBrain) followed by session-level speaker embedding refinement with Graph Neural Networks (GCN), based on the paper "Speaker Diarization with Session-Level Speaker Embedding Refinement Using Graph Neural Networks" (Wang et al., 2020). Refined embeddings are then clustered with auto-tuning spectral clustering (NME-SC).
