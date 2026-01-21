#  Speech-to-Text Translation BTP

This is the official repository for our B.Tech Final Year Project titled  
**"Zero-shot and Few-shot Speech-to-Text Translation for Low-Resource Indian Languages using Audio LLMs."**

##  Team Members
- **Prabhkirat** – 2022ucs0102(IIT JAMMU)
- **Sangeet** – 2022ucs0107(IIT JAMMU)

## Guided By 
Prof. Mrinmoy Bhattacharya

# End-to-End Punjabi to English Speech Translation

This repository contains the complete codebase and documentation for fine-tuning OpenAI’s Whisper Small model for **Punjabi to English speech translation**.

Despite Punjabi–English being a low-resource language pair, the fine-tuned model achieves a **BLEU score of 30.00**, significantly outperforming the zero-shot Whisper baseline and surpassing official benchmarks for comparable Indic languages.

---

## Project Overview

The objective of this project is to build an **end-to-end Automatic Speech Translation (AST)** system that translates spoken Punjabi directly into English text.

---

## Key Challenges

### Resource Scarcity
- Limited availability of large-scale Punjabi speech-to-English parallel datasets.
- Existing datasets contain alignment noise and transcription errors.

### Hardware Constraints
- Consumer-grade GPU with limited VRAM.
- Training whisper-medium or whisper-large was not feasible.

---

## Our Solution

### Synthetic Data Pipeline
- Generated English translations for Punjabi audio using Meta’s SeamlessM4T model.
- Created large-scale parallel speech–translation pairs.

### Manual Quality Control
- Manually reviewed and corrected **15,000+ audio–text pairs**.
- Approximately **11 hours** of annotation effort.
- Produced a high-quality “gold standard” dataset.

### Efficient Fine-Tuning
- Base model: `openai/whisper-small` (244M parameters).
- FP16 mixed-precision training.
- Gradient checkpointing enabled for VRAM efficiency.
- Carefully tuned batch size and accumulation strategy.

---

## Results

Evaluation was performed on a **held-out test set of approximately 2 hours of unseen audio**.

| Model | BLEU Score | Notes |
|-----|-----------|------|
| Whisper Small (Zero-shot) | 19.00 | Pre-trained model |
| Whisper Small (Fine-Tuned) | **30.00** | Final model |
| Official Benchmark (Bengali–English) | 10.08 | Indic reference |

**Key takeaway**  
A data-centric approach resulted in a **+11 BLEU improvement** without increasing model size.

---

## Installation

### Clone the repository
```
git clone https://github.com/yourusername/whisper-punjabi-translation.git
cd whisper-punjabi-translation
```

### Install dependencies
```
pip install torch transformers datasets evaluate jiwer soundfile librosa pandas
```

### Install ffmpeg
```
sudo apt-get install ffmpeg
```

---

## Dataset Structure

The training script expects the following local directory structure:

```
punjabi/
├── train/
│   ├── audio/
│   │   ├── sample1.wav
│   │   ├── sample2.wav
│   └── english_transcripts.txt
└── valid/
    ├── audio/
    │   ├── sampleA.wav
│   └── english_transcripts.txt
```

Transcript format:
```
filename.wav<TAB>English translation
```

### Data Sources
- Source audio: Kathbath Dataset / Spoken-Tutorial (AI4Bharat)
- English labels: Generated via SeamlessM4T and manually corrected

---

## Training

### Script
```
finetune_whisper_local.py
```

### Key Training Parameters
- Model: openai/whisper-small
- Batch size: 4
- Effective batch size: 16 (gradient accumulation)
- Learning rate: 5e-6
- Epochs: 10
- Precision: FP16
- Gradient checkpointing: Enabled

### Run training
```
nohup python -u finetune_whisper_local.py > training.log 2>&1 &
```

---

## Model Architecture

Whisper uses a Transformer encoder–decoder architecture.

### Encoder
- Converts raw audio to log-Mel spectrograms.
- Processes 30-second chunks.
- Uses convolution layers followed by self-attention blocks.

### Decoder
- Autoregressively generates English text tokens.
- Uses cross-attention over encoded audio features.

### Task Control
- Translation enforced using special tokens:
  - Target language token `<|en|>`
  - Task token `<|translate|>`

---

## Evaluation

- Metric: SacreBLEU
- Evaluation performed only after training completion.
- BLEU score does not affect training or checkpoint selection.

---

## Limitations

- Whisper Small limits representational capacity.
- Dataset size remains modest for low-resource speech translation.
- Further improvements require larger curated datasets or multi-GPU training.

---

## Future Work

- Expand manually verified dataset.
- Explore domain-specific fine-tuning.
- Train whisper-medium using distributed setup.
- Release trained checkpoints on Hugging Face.

---

## Acknowledgements

- OpenAI for the Whisper model.
- AI4Bharat for Indian language datasets.
- Meta AI for the SeamlessM4T model.






