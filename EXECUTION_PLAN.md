# RobustAV: Degradation-Aware Cross-Modal Consistency Learning for Robust Audio-Visual Deepfake Detection

## Detailed Execution Plan

---

## 1. Problem Statement & Motivation

### 1.1 Core Objective

> To enhance the robustness of deepfake detection models against compression artifacts, noise, resolution variations, and unseen deepfake generation methods through effective audio-visual feature learning.

### 1.2 Why the Previous Plan Failed

The original "Beyond Accuracy" benchmarking plan required:

- 13 models × 12 attacks × 5 ε values × compression variants = **thousands of experiments**
- ~360 GPU-hours on Kaggle (30-40 sessions at 12h each)
- Phase 0 baselines were broken (AUC ~0.50-0.58, near random chance)
- Benchmarking papers lack the **methodological novelty** demanded by CVPR/NeurIPS/ICML

### 1.3 The Research Gap We Exploit

Four concurrent 2025-2026 trends have **not yet been combined**:

* TrendPaperWhat It DoesWhat It MissesParameter-efficient foundation model tuning for deepfake detectionGenD (WACV 2026)LN-tuning of DINOv2/CLIP achieves SOTA cross-dataset generalization**Image-only**. No audio-visual.LMMs for AV deepfake detectionAV-LMMDetect (ICASSP 2026)Fine-tuned Qwen 2.5 Omni for AV detectionMassive model, no degradation robustness studySelf-supervised AV consistencySAVe / X-AVDT (2026)Cross-modal alignment from authentic videoNo explicit degradation-adaptation mechanismTest-time adaptation for deepfakesT²A (IJCAI 2025), OST (NeurIPS 2025)TTA via entropy minimization at inference**Image-only**. Never applied to AV detection.

**Our unique position**: First work combining foundation-model cross-modal consistency for AV deepfake detection **with** test-time degradation adaptation.

---

## 2. Proposed Solution: RobustAV

### 2.1 High-Level Idea

Rather than benchmarking how existing detectors break, we **build a new detector** that:

1. Uses **frozen foundation models** (DINOv2 + Whisper) as feature extractors — inheriting their powerful, generator-agnostic representations
2. Learns **cross-modal consistency** between audio and visual streams via a lightweight attention module — because AV synchronization is a universal forgery signal that transcends specific generators
3. **Adapts at test time** to unknown degradations (compression, noise, resolution) by updating only LayerNorm parameters — requiring zero labels and negligible compute

### 2.2 Why This Approach

**Why foundation models instead of training from scratch:**

- DINOv2 was trained on 142M images with self-supervised learning. Its features capture rich visual semantics that transfer to deepfake detection without generator-specific overfitting.
- Whisper was trained on 680K hours of multilingual speech. Its audio features capture spectral and temporal patterns far richer than any mel-spectrogram CNN trained on ASVspoof alone.
- GenD (WACV 2026) proved that simply tuning LayerNorm params of DINOv2 achieves SOTA deepfake detection — beating complex purpose-built architectures. We extend this to audio-visual.

**Why cross-modal consistency:**

- Deepfakes that manipulate only video or only audio create a **synchronization gap** between modalities. This gap is a universal signal — it exists regardless of which generator was used, which compression was applied, or what resolution the video is in.
- Cross-modal attention naturally captures lip-audio alignment, emotion consistency, and temporal co-occurrence patterns.

**Why test-time adaptation:**

- Real-world deepfakes go through unknown post-processing pipelines (social media compression, re-encoding, screenshot-to-video, etc.). A model trained on clean or mildly compressed data will encounter distribution shifts at deployment.
- TTA via entropy minimization on pseudo-degraded copies of each test sample allows the model to "calibrate" its LayerNorm statistics to the specific degradation present — without needing labels.

---

## 3. Architecture

### 3.1 Architecture Diagram

```
INPUT
  │
  ├── Video Frames (T × 224 × 224 × 3)
  │       │
  │       ▼
  │   ┌─────────────────────────────┐
  │   │  Frozen DINOv2-Small        │
  │   │  (ViT-S/14, ~22M params)   │
  │   │  Only LayerNorm is tunable  │
  │   └──────────┬──────────────────┘
  │              │
  │              ▼
  │   Visual Features: (T × D_v)     D_v = 384
  │              │
  │              ▼
  │   ┌──────────────────────┐
  │   │ Temporal Pooling      │
  │   │ (Learnable [CLS] +   │
  │   │  1-layer Transformer) │
  │   └──────────┬───────────┘
  │              │
  │              ▼
  │   Pooled Visual: (N_seg × D)     D = 256
  │              │
  │              │
  ├── Audio Waveform (16kHz, ≤ 10s)
  │       │
  │       ▼
  │   ┌─────────────────────────────┐
  │   │  Frozen Whisper-Small       │
  │   │  (Encoder only, ~244M      │
  │   │   params but all frozen)   │
  │   │  Only LayerNorm is tunable  │
  │   └──────────┬──────────────────┘
  │              │
  │              ▼
  │   Audio Features: (T_a × D_a)    D_a = 768
  │              │
  │              ▼
  │   ┌──────────────────────┐
  │   │ Linear Projection    │
  │   │ D_a → D (768 → 256)  │
  │   └──────────┬───────────┘
  │              │
  │              ▼
  │   Pooled Audio: (N_seg × D)
  │              │
  └──────────────┤
                 │
                 ▼
  ┌──────────────────────────────────────┐
  │  Cross-Modal Consistency Module      │
  │  (CMCM)                             │
  │                                      │
  │  Layer 1: CrossAttn(Q=V, K=A, V=A)  │
  │  Layer 2: CrossAttn(Q=A, K=V, V=V)  │
  │  + Residual connections + LayerNorm  │
  │                                      │
  │  Output: fused (N_seg × D)           │
  └──────────────┬───────────────────────┘
                 │
                 ▼
  ┌──────────────────────────────────────┐
  │  Classification Head                 │
  │                                      │
  │  Global Average Pool → D             │
  │  Linear(D, 128) → ReLU → Dropout    │
  │  Linear(128, 1) → Sigmoid           │
  │                                      │
  │  Output: P(fake) ∈ [0, 1]           │
  └──────────────────────────────────────┘


  ═══════════════════════════════════════
  AT TEST TIME (TTDA Module):
  ═══════════════════════════════════════

  Test Sample x
       │
       ├──► Generate K=5 pseudo-degraded copies:
       │    x_jpeg50, x_jpeg75, x_resize05, x_noise002, x_mp3_128k
       │
       ├──► Forward pass all K+1 through model
       │
       ├──► Compute entropy: H = -Σ p_k log(p_k)
       │
       ├──► 1-step gradient descent on LN params
       │    to minimize H (prediction consistency)
       │
       └──► Final prediction with adapted LN params
```

### 3.2 Component-by-Component Rationale

#### 3.2.1 Visual Encoder — DINOv2-Small (ViT-S/14)

| Property       | Value                      | Why                                                  |
| :------------- | :------------------------- | :--------------------------------------------------- |
| Architecture   | ViT-S/14                   | Small enough for Kaggle T4 (16GB VRAM)               |
| Parameters     | ~22M (frozen)              | No gradient memory for backbone                      |
| Tunable params | ~18K (LayerNorm only)      | 0.08% of total — prevents overfitting               |
| Patch size     | 14×14                     | 16×16 patches from 224×224 → 256 tokens per frame |
| Feature dim    | 384                        | Rich enough for cross-modal fusion                   |
| Pre-training   | LVD-142M (self-supervised) | Generator-agnostic features                          |

**Why not CLIP?** GenD showed DINOv2 slightly outperforms CLIP for deepfake detection because CLIP's contrastive objective creates text-aligned features that may not capture fine-grained visual artifacts as well as DINOv2's self-supervised features.

**Why not DINOv2-Base or Large?** Memory. DINOv2-B is 86M params, DINOv2-L is 300M. With video frames (T=16 per clip), even frozen forward passes consume significant VRAM. ViT-S/14 is the sweet spot for Kaggle T4.

#### 3.2.2 Audio Encoder — Whisper-Small (Encoder Only)

| Property       | Value                                            | Why                                      |
| :------------- | :----------------------------------------------- | :--------------------------------------- |
| Architecture   | Transformer encoder (12 layers)                  | Proven audio representation              |
| Parameters     | ~244M (all frozen)                               | Only encoder used, no decoder            |
| Tunable params | ~14K (LayerNorm only)                            | Same LN-tuning strategy                  |
| Input          | 16kHz waveform → log-mel spectrogram (internal) | Whisper handles preprocessing internally |
| Feature dim    | 768                                              | Projected down to 256 for fusion         |
| Pre-training   | 680K hours of labeled speech                     | Extremely rich audio features            |

**Why not Wav2Vec2 or HuBERT?** Whisper was trained on more diverse data (multilingual, noisy conditions). For deepfake audio that may contain compression artifacts, Whisper's robustness to noisy inputs is an advantage.

**Why encoder only?** We need feature representations, not transcriptions. The encoder provides per-frame audio features; the decoder is unnecessary.

#### 3.2.3 Cross-Modal Consistency Module (CMCM)

```python
# Pseudocode
class CMCM(nn.Module):
    def __init__(self, d_model=256, nhead=4, num_layers=2):
        # Layer 1: Video attends to Audio
        self.v2a_attn = nn.MultiheadAttention(d_model, nhead)
        self.v2a_norm = nn.LayerNorm(d_model)
        self.v2a_ffn  = FFN(d_model, d_model*4)

        # Layer 2: Audio attends to Video
        self.a2v_attn = nn.MultiheadAttention(d_model, nhead)
        self.a2v_norm = nn.LayerNorm(d_model)
        self.a2v_ffn  = FFN(d_model, d_model*4)

    def forward(self, v_feat, a_feat):
        # Video-to-Audio cross attention
        v_enhanced = self.v2a_attn(query=v_feat, key=a_feat, value=a_feat)
        v_enhanced = self.v2a_norm(v_feat + v_enhanced)
        v_enhanced = self.v2a_ffn(v_enhanced)

        # Audio-to-Video cross attention
        a_enhanced = self.a2v_attn(query=a_feat, key=v_feat, value=v_feat)
        a_enhanced = self.a2v_norm(a_feat + a_enhanced)
        a_enhanced = self.a2v_ffn(a_enhanced)

        # Concatenate and return
        fused = torch.cat([v_enhanced, a_enhanced], dim=-1)  # (N_seg × 2D)
        return self.projection(fused)  # (N_seg × D)
```

**Why cross-attention (not self-attention on concatenated features)?**

- Cross-attention explicitly models the **relationship** between modalities. If audio says "hello" but lips show "goodbye," cross-attention Q-K dot products will produce low alignment scores — a direct forgery signal.
- Self-attention on concatenated features treats audio and video tokens equally, losing the explicit cross-modal alignment signal.

**Why only 2 layers?**

- The foundation model features are already very high-quality. The CMCM needs only to learn the *consistency mapping*, not rebuild the features. 2 layers (~5M params) is sufficient and prevents overfitting on small AV datasets.

#### 3.2.4 Test-Time Degradation Adaptation (TTDA) — The Key Novel Component

**The core insight**: If a model is truly robust, its prediction should be **invariant** to quality degradations. At test time, we can enforce this invariance by minimizing the entropy of predictions across synthetically degraded versions of the input.

**Algorithm (detailed)**:

```
Algorithm: TTDA (Test-Time Degradation Adaptation)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Input:  Test sample (video V, audio A)
        Trained model M with parameters θ
        LN parameters θ_LN ⊂ θ
        Degradation set D = {jpeg50, jpeg75, resize_half, gauss_noise_002, mp3_128k}
        Learning rate η = 1e-4
        Number of adaptation steps S = 1

1. Create augmented batch:
   B = {(V, A)}  ∪  {(d_v(V), d_a(A)) for d in D}
   where d_v applies visual degradation, d_a applies audio degradation

2. Forward pass all samples through M:
   P = {M(v, a; θ) for (v, a) in B}       # K+1 predictions

3. Compute consistency loss:
   p_mean = mean(P)
   L_consist = -[p_mean · log(p_mean) + (1-p_mean) · log(1-p_mean)]
   (This is the entropy of the mean prediction)

4. Update only LN parameters:
   θ_LN ← θ_LN - η · ∇_{θ_LN} L_consist

5. Final prediction:
   y_hat = M(V, A; θ_updated)

6. Restore θ_LN to original (for next test sample)
```

**The 5 degradation transforms**:

| Degradation            | Visual               | Audio              | Why                                        |
| :--------------------- | :------------------- | :----------------- | :----------------------------------------- |
| JPEG QF=50             | ✓ (save/reload)     | —                 | Heavy compression, common on social media  |
| JPEG QF=75             | ✓ (save/reload)     | —                 | Moderate compression, standard web         |
| Resize 0.5× then back | ✓ (bicubic down/up) | —                 | Resolution loss, common in forwarded media |
| Gaussian noise σ=0.02 | ✓ (additive)        | ✓ (additive)      | Sensor/channel noise                       |
| MP3 128kbps            | —                   | ✓ (encode/decode) | Audio compression, standard on platforms   |

**Why only 1 gradient step?**

- We're updating ~32K parameters (LN params from both encoders + CMCM). One step is enough to shift the batch normalization statistics to the current degradation type.
- More steps risk overfitting to the specific test sample (the test sample could be an adversarial input designed to manipulate TTA).
- T²A (IJCAI 2025) showed that 1-step TTA outperforms multi-step in deepfake detection due to this overfitting risk.

**Why entropy minimization on the mean prediction?**

- If the model predicts [0.8, 0.3, 0.9, 0.2, 0.7, 0.6] across degraded versions, the mean is ~0.58 with high entropy — the model is uncertain and inconsistent.
- After TTA, the model should predict [0.8, 0.78, 0.82, 0.79, 0.81, 0.80] — consistent and confident.
- We minimize entropy of the *mean* prediction, which pushes all individual predictions toward agreement.

### 3.3 Training Procedure

#### 3.3.1 Loss Function

```
L_total = L_bce + λ_con · L_contrastive + λ_deg · L_degradation

Where:
  L_bce = Binary Cross-Entropy(P(fake), y_true)
  L_contrastive = Contrastive loss on cross-modal features
  L_degradation = Prediction consistency under training-time augmentations
```

**L_bce (Binary Cross-Entropy)**: Standard classification loss. Weight: 1.0.

**L_contrastive**: For matched (same video) AV pairs, pull features together. For mismatched (different video) AV pairs, push apart. This teaches the CMCM what "consistent" vs "inconsistent" looks like.

```
L_contrastive = -log(exp(sim(v_i, a_i)/τ) / Σ_j exp(sim(v_i, a_j)/τ))
```

where sim() is cosine similarity, τ=0.07. Weight λ_con = 0.5.

**L_degradation**: During training, randomly degrade each sample and enforce prediction consistency between clean and degraded versions. This is a soft version of TTDA applied during training.

```
L_degradation = KL(P_clean || P_degraded) + KL(P_degraded || P_clean)
```

Weight λ_deg = 0.3.

#### 3.3.2 Training Hyperparameters

| Hyperparameter              | Value                                   | Rationale                                    |
| :-------------------------- | :-------------------------------------- | :------------------------------------------- |
| Optimizer                   | AdamW                                   | Standard for transformer fine-tuning         |
| Learning rate (LN params)   | 1e-4                                    | Small LR for pre-trained LN params           |
| Learning rate (CMCM + head) | 5e-4                                    | Larger LR for randomly initialized modules   |
| Weight decay                | 0.01                                    | Prevent overfitting                          |
| Batch size                  | 8                                       | Kaggle T4 memory constraint                  |
| Epochs                      | 30                                      | With early stopping (patience=5)             |
| Scheduler                   | Cosine annealing with warmup (3 epochs) | Standard practice                            |
| Video frames per clip       | 16 (uniformly sampled)                  | Balance between temporal coverage and memory |
| Audio length                | 10 seconds (pad/trim)                   | Covers most FakeAVCeleb clips                |
| Gradient accumulation       | 4 steps (effective batch = 32)          | Compensate for small physical batch          |

### 3.4 Parameter Count Summary

| Component                 | Total Params    | Trainable Params | % Trainable    |
| :------------------------ | :-------------- | :--------------- | :------------- |
| DINOv2-Small              | 22M             | ~18K (LN only)   | 0.08%          |
| Whisper-Small encoder     | 244M            | ~14K (LN only)   | 0.006%         |
| Visual temporal pooling   | ~0.4M           | 0.4M             | 100%           |
| Audio linear projection   | ~0.2M           | 0.2M             | 100%           |
| CMCM (2-layer cross-attn) | ~5M             | 5M               | 100%           |
| Classification head       | ~0.04M          | 0.04M            | 100%           |
| **Total**           | **~271M** | **~5.7M**  | **2.1%** |

> Only 5.7M parameters are trained. The rest are frozen foundation model weights loaded from checkpoints. This means training is fast, memory-efficient, and resistant to overfitting.

---

## 4. Datasets & Resources

### 4.1 Primary Dataset: FakeAVCeleb

1. PropertyDetail**Full name**FakeAVCeleb: A Novel Audio-Video Multimodal Deepfake Dataset**Paper**Khalid et al., NeurIPS 2021 Datasets Track**Total videos**~500 real + ~19,500 fake videos**Manipulation types**Face swap (FaceSwap, FSGAN), face reenactment, lip-sync (Wav2Lip, SV2TTS)**Audio types**Real audio, TTS-generated audio (SV2TTS)**Categories**RealVideo-RealAudio, FakeVideo-RealAudio, RealVideo-FakeAudio, FakeVideo-FakeAudio**Resolution**224×224 face-cropped (pre-processed)**Duration**3-10 seconds per clip**Kaggle availability**✓ Available as Kaggle dataset

**Why FakeAVCeleb as primary:**

1. It is the **standard be****nchmark** for audio-visual deepfake detection (used by AV-LMMDetect, BA-TFD, and virtually every AV detection paper since 2022).
2. It covers **all four AV manipulation categories** — essential for testing cross-modal consistency.
3. Manageable size (~500 videos for training after splits) — fits within Kaggle constraints.
4. Pre-cropped faces — no need for face detection/alignment preprocessing.

**Proposed split:**

| Split      | Real | Fake  | Total | Usage                                 |
| :--------- | :--- | :---- | :---- | :------------------------------------ |
| Train      | 350  | 3,500 | 3,850 | Model training                        |
| Validation | 75   | 750   | 825   | Hyperparameter tuning, early stopping |
| Test       | 75   | 750   | 825   | Final evaluation                      |

> We follow the standard FakeAVCeleb protocol with race/gender-balanced splits to ensure fair evaluation.

### 4.2 Cross-Dataset Generalization: LAV-DF

| Property                      | Detail                                                        |
| :---------------------------- | :------------------------------------------------------------ |
| **Full name**           | Localized Audio-Visual DeepFake Dataset                       |
| **Paper**               | Cai et al., 2022                                              |
| **Total videos**        | ~36K video segments                                           |
| **Manipulation**        | Localized audio-visual manipulations with temporal boundaries |
| **Key feature**         | Temporal segment annotations (start/end of fake region)       |
| **Resolution**          | Variable (720p/1080p source)                                  |
| **Kaggle availability** | ✓ Available as Kaggle dataset                                |

**Why LAV-DF for cross-dataset:**

- Different generation pipeline from FakeAVCeleb — tests true generalization.
- Temporal localization annotations allow us to evaluate if our model's cross-modal attention aligns with actual manipulation boundaries (qualitative analysis).
- Already available in your Kaggle workspace from previous work.

**Usage**: Test split only (~2,000 clips). We do NOT train on LAV-DF. It is purely for cross-dataset generalization evaluation.

### 4.3 Degraded Test Sets (Generated from FakeAVCeleb Test Split)

We generate degraded versions of the FakeAVCeleb test set to evaluate robustness:

| Degradation                 | Parameters                       | How Generated                                 | What It Tests                             |
| :-------------------------- | :------------------------------- | :-------------------------------------------- | :---------------------------------------- |
| JPEG compression            | QF ∈ {50, 75, 90}               | `PIL.Image.save(quality=QF)` per frame      | Social media upload compression           |
| Resolution reduction        | Scale ∈ {0.25×, 0.5×, 0.75×} | Bicubic downscale → upscale back to 224×224 | Forwarded/re-shared media resolution loss |
| Gaussian noise (visual)     | σ ∈ {0.01, 0.02, 0.05}         | Additive N(0, σ²) per pixel                 | Sensor noise, re-capture artifacts        |
| Gaussian noise (audio)      | SNR ∈ {20dB, 30dB, 40dB}        | Additive white Gaussian noise                 | Background noise, channel noise           |
| MP3 compression (audio)     | Bitrate ∈ {64, 128, 192} kbps   | Encode WAV→MP3→WAV via pydub                | Audio platform compression                |
| H.264 re-encoding (video)   | CRF ∈ {23, 28, 35}              | ffmpeg re-encode                              | Video platform re-encoding                |
| Combined "social media sim" | JPEG75 + resize0.75 + MP3-128k   | Sequential pipeline                           | Realistic social media pipeline           |

**Total degraded test sets**: 7 degradation types × 3 severity levels + 1 combined = **22 test conditions** (plus 1 clean baseline = 23 total).

### 4.4 Software & Library Requirements

```
# Core
torch>=2.0
torchvision>=0.15
torchaudio>=2.0

# Foundation models
transformers>=4.35       # Whisper-Small via HuggingFace
timm>=0.9               # DINOv2-Small via timm or torch.hub

# Adversarial attacks (for robustness evaluation)
torchattacks>=3.4        # PGD, FGSM

# Audio processing
librosa>=0.10
soundfile>=0.12
pydub>=0.25

# Video/Image processing
opencv-python>=4.8
Pillow>=10.0
ffmpeg-python>=0.2

# Metrics
scikit-learn>=1.3        # AUC, EER, F1
scipy>=1.11              # Confidence intervals

# Visualization
matplotlib>=3.8
seaborn>=0.13

# Utilities
pandas>=2.1
numpy>=1.24
tqdm
```

### 4.5 Hardware Requirements

| Resource        | Minimum                    | Recommended         |
| :-------------- | :------------------------- | :------------------ |
| GPU             | Kaggle T4 (16GB)           | Kaggle T4×2 (32GB) |
| RAM             | 13GB (Kaggle default)      | 13GB sufficient     |
| Disk            | 20GB (for cached features) | 50GB                |
| GPU hours total | ~50                        | ~60 (with buffer)   |

---

## 5. Feature Caching Strategy

### 5.1 Why Cache Features?

Feature extraction through frozen DINOv2 and Whisper is **deterministic** — the same input always produces the same output. Since these encoders are frozen, we extract features **once** and save them. All subsequent training and evaluation runs load cached features directly, eliminating the most expensive computation.

**Time savings:**

- DINOv2 forward pass for 16 frames: ~0.3s per clip on T4
- Whisper encoder forward pass for 10s audio: ~0.2s per clip on T4
- For 5,500 clips × 0.5s = ~46 minutes for full extraction
- But training for 30 epochs would repeat this 30× = ~23 hours wasted without caching

With caching, feature extraction is a **one-time 1-hour cost**.

### 5.2 Cache Format

```
/kaggle/working/robustav_cache/
├── features/
│   ├── visual/
│   │   ├── train/
│   │   │   ├── {clip_id}.pt     # Tensor: (T=16, D_v=384)
│   │   │   └── ...
│   │   ├── val/
│   │   └── test/
│   ├── audio/
│   │   ├── train/
│   │   │   ├── {clip_id}.pt     # Tensor: (T_a, D_a=768)
│   │   │   └── ...
│   │   ├── val/
│   │   └── test/
│   └── degraded_test/
│       ├── jpeg50_visual/       # Degraded visual features
│       ├── jpeg75_visual/
│       ├── mp3_128k_audio/
│       └── ...
├── manifests/
│   ├── train.csv                # clip_id, label, av_category
│   ├── val.csv
│   └── test.csv
└── metadata.json                # Extraction config, model versions, timestamps
```

### 5.3 Caching Execution (Kaggle Session 1)

```python
# Pseudocode for feature extraction notebook
# This runs ONCE and produces a Kaggle dataset for all future sessions

import torch
from transformers import WhisperModel, WhisperProcessor
import timm

# Load frozen encoders
dinov2 = timm.create_model('vit_small_patch14_dinov2', pretrained=True)
dinov2.eval()

whisper = WhisperModel.from_pretrained('openai/whisper-small')
whisper_encoder = whisper.encoder
whisper_encoder.eval()

processor = WhisperProcessor.from_pretrained('openai/whisper-small')

# Extract and save
for clip_id, video_path, audio_path, label in manifest:
    # Visual features
    frames = sample_frames(video_path, n=16)  # (16, 3, 224, 224)
    with torch.no_grad():
        v_feat = dinov2.forward_features(frames)  # (16, 384)
    torch.save(v_feat.cpu(), f'cache/visual/train/{clip_id}.pt')

    # Audio features
    waveform = load_audio(audio_path, sr=16000, max_len=10)
    mel = processor(waveform, return_tensors='pt').input_features
    with torch.no_grad():
        a_feat = whisper_encoder(mel).last_hidden_state  # (1, T_a, 768)
    torch.save(a_feat.squeeze(0).cpu(), f'cache/audio/train/{clip_id}.pt')

# Upload cache as Kaggle dataset for reuse
```

**Estimated time**: ~1 hour for full FakeAVCeleb + degraded variants.

**Upload as Kaggle dataset**: Once extracted, upload the cache directory as a private Kaggle dataset. All subsequent training/evaluation notebooks simply add this dataset as input — no re-extraction ever.

---

## 6. Experiments & Baselines

### 6.1 Baseline Models

We compare RobustAV against 4 baselines spanning different detection paradigms:

#### Baseline 1: BA-TFD (Boundary-Aware Temporal Fusion Detection)

- **Source**: ControlNet/LAV-DF GitHub
- **Type**: Purpose-built AV deepfake detector with boundary-aware temporal fusion
- **Why**: The current SOTA audio-visual deepfake detector with public weights. This is the primary baseline to beat.
- **Setup**: Load pre-trained checkpoint from LAV-DF repo. Evaluate on FakeAVCeleb and LAV-DF.
- **Kaggle effort**: ~2 hours setup + evaluation.

#### Baseline 2: Late-Fusion (XceptionNet + AASIST)

- **Type**: Score-level fusion of best unimodal detectors
- **Why**: Tests whether naive fusion of strong unimodal detectors outperforms learned cross-modal approaches.
- **Setup**: Run XceptionNet on video frames → P_visual. Run AASIST on audio → P_audio. Final score = 0.5 × P_visual + 0.5 × P_audio.
- **Kaggle effort**: ~3 hours (use pre-trained weights from DeepfakeBench / AASIST repos).

#### Baseline 3: RobustAV-NoTTDA (Ablation)

- **Type**: Our full architecture but without test-time degradation adaptation
- **Why**: Isolates the contribution of TTDA. If RobustAV-NoTTDA already performs well on clean data but degrades on compressed data, and RobustAV with TTDA recovers, this proves TTDA's value.
- **Setup**: Same trained model, just skip the TTA step at inference.
- **Kaggle effort**: Zero — same model, different inference path.

#### Baseline 4: RobustAV-VisualOnly (Ablation)

- **Type**: DINOv2 + classification head, no audio, no cross-modal module
- **Why**: Isolates the contribution of audio-visual fusion. Shows that cross-modal consistency is essential for robust detection, not just better visual features.
- **Setup**: Train a simplified version with only the visual encoder + head.
- **Kaggle effort**: ~4 hours training.

### 6.2 Experiment Matrix

| Experiment ID | Test Set                                      | Models Evaluated | What It Measures                |
| :------------ | :-------------------------------------------- | :--------------- | :------------------------------ |
| **E1**  | FakeAVCeleb clean                             | All 4 + RobustAV | In-domain baseline accuracy     |
| **E2**  | FakeAVCeleb + JPEG {50,75,90}                 | All 4 + RobustAV | Visual compression robustness   |
| **E3**  | FakeAVCeleb + resize {0.25,0.5,0.75}          | All 4 + RobustAV | Resolution robustness           |
| **E4**  | FakeAVCeleb + Gaussian noise {0.01,0.02,0.05} | All 4 + RobustAV | Noise robustness                |
| **E5**  | FakeAVCeleb + MP3 {64,128,192}kbps            | All 4 + RobustAV | Audio compression robustness    |
| **E6**  | FakeAVCeleb + H.264 {CRF 23,28,35}            | All 4 + RobustAV | Video re-encoding robustness    |
| **E7**  | FakeAVCeleb + social media sim                | All 4 + RobustAV | Combined realistic degradation  |
| **E8**  | LAV-DF test (clean)                           | All 4 + RobustAV | Cross-dataset generalization    |
| **E9**  | LAV-DF test + social media sim                | All 4 + RobustAV | Cross-dataset + degradation     |
| **E10** | FakeAVCeleb + PGD (visual)                    | All 4 + RobustAV | Adversarial robustness (visual) |
| **E11** | FakeAVCeleb + PGD (audio)                     | All 4 + RobustAV | Adversarial robustness (audio)  |
| **E12** | FakeAVCeleb per-category                      | RobustAV only    | Per-manipulation-type analysis  |

### 6.3 Adversarial Robustness Evaluation (E10, E11)

**Visual adversarial attack (E10):**

- Attack: PGD-20 (20 iterations, step size α=2/255)
- Perturbation budget: ε ∈ {2/255, 4/255, 8/255}
- Constraint: L∞ norm
- Target: Visual encoder input (raw frames before DINOv2)
- Imperceptibility gate: SSIM ≥ 0.95

**Audio adversarial attack (E11):**

- Attack: PGD-20 on raw waveform
- Perturbation budget: SNR ∈ {20dB, 30dB, 40dB}
- Target: Audio waveform before Whisper preprocessing
- Imperceptibility gate: PESQ ≥ 3.5

**Important note on adversarial attacks with frozen encoders:**
Since our encoders are frozen and we only tune LN params, adversarial attacks must be crafted end-to-end through the full model. The gradient flows through the frozen encoder (which is differentiable) to the input. This is a standard white-box attack setup.

### 6.4 Per-Category Analysis (E12)

FakeAVCeleb has four manipulation categories. We break down RobustAV's performance per category:

| Category            | What It Tests                                                                 |
| :------------------ | :---------------------------------------------------------------------------- |
| FakeVideo-RealAudio | Does the model detect visual-only manipulation via cross-modal inconsistency? |
| RealVideo-FakeAudio | Does the model detect audio-only manipulation (TTS)?                          |
| FakeVideo-FakeAudio | Can the model detect when both modalities are fake but "consistent"?          |
| RealVideo-RealAudio | False positive rate — does the model correctly identify real content?        |

This analysis reveals which manipulation types benefit most from cross-modal consistency learning.

---

## 7. Ablation Studies

### 7.1 Ablation Matrix

| Ablation ID   | What We Remove/Change                              | What It Proves                                |
| :------------ | :------------------------------------------------- | :-------------------------------------------- |
| **A1**  | Remove TTDA (= Baseline 3)                         | Value of test-time adaptation                 |
| **A2**  | Remove audio (= Baseline 4)                        | Value of cross-modal fusion                   |
| **A3**  | Remove contrastive loss (train with BCE only)      | Value of explicit consistency learning        |
| **A4**  | Remove degradation loss (no L_deg during training) | Value of degradation-aware training           |
| **A5**  | Replace DINOv2 with CLIP-ViT-B/16                  | DINOv2 vs CLIP for deepfake features          |
| **A6**  | Replace Whisper with Wav2Vec2-base                 | Whisper vs Wav2Vec2 for audio features        |
| **A7**  | TTDA with {1, 3, 5, 10} gradient steps             | Optimal TTA adaptation steps                  |
| **A8**  | TTDA with different degradation subsets            | Which degradations are most important for TTA |
| **A9**  | CMCM with {1, 2, 4} layers                         | Optimal cross-attention depth                 |
| **A10** | Full fine-tune vs LN-only tuning                   | Value of parameter-efficient approach         |

### 7.2 Ablation Details

#### A1: TTDA Impact (Most Important Ablation)

This is the **central claim** of the paper. We evaluate:

```
RobustAV (with TTDA) vs RobustAV (without TTDA)
```

On all 23 test conditions (clean + 22 degraded). The expected result:

- On **clean** data: similar performance (TTDA shouldn't hurt clean accuracy)
- On **degraded** data: TTDA should recover 5-15% AUC that degrades without it
- The **more severe** the degradation, the **larger** the TTDA benefit

This produces the paper's **main figure**: a line chart showing AUC vs degradation severity, with two lines (with/without TTDA) diverging as degradation increases.

#### A5: DINOv2 vs CLIP

GenD (WACV 2026) showed DINOv2 slightly outperforms CLIP for image-only deepfake detection. We verify this holds for AV detection:

- Replace `vit_small_patch14_dinov2` with `clip-vit-base-patch16` (via HuggingFace)
- Same LN-tuning strategy, same CMCM, same training
- Compare AUC on clean and degraded test sets

#### A7: Number of TTA Steps

| Steps | Expected behavior                                          |
| :---- | :--------------------------------------------------------- |
| 0     | No adaptation — baseline                                  |
| 1     | Sweet spot — enough to shift LN statistics                |
| 3     | Marginal improvement over 1                                |
| 5     | Diminishing returns                                        |
| 10    | Risk of overfitting to test sample — may hurt performance |

#### A10: Full Fine-Tune vs LN-Only

This validates the GenD insight for AV detection:

- **LN-only** (ours): Tune ~32K params. Fast training, no overfitting.
- **Full fine-tune**: Tune all 266M params. Requires much larger learning rate reduction, longer training, and risks overfitting on small FakeAVCeleb dataset.

Expected: LN-only matches or slightly outperforms full fine-tune on cross-dataset (LAV-DF), while full fine-tune overfits to FakeAVCeleb.

---

## 8. Evaluation Metrics

### 8.1 Primary Metrics

| Metric            | Formula                           | What It Measures                     | Range                   |
| :---------------- | :-------------------------------- | :----------------------------------- | :---------------------- |
| **AUC-ROC** | Area under ROC curve              | Overall discrimination ability       | [0, 1], higher = better |
| **EER**     | FPR at FPR=FNR                    | Operating point where errors balance | [0, 1], lower = better  |
| **AP**      | Area under Precision-Recall curve | Performance under class imbalance    | [0, 1], higher = better |

### 8.2 Robustness-Specific Metrics

| Metric                                    | Formula                               | What It Measures                                                      |
| :---------------------------------------- | :------------------------------------ | :-------------------------------------------------------------------- |
| **RAR (Robustness-Accuracy Ratio)** | `AUC_degraded / AUC_clean`          | How much performance survives degradation. RAR=1 is perfectly robust. |
| **Δ-AUC**                          | `AUC_clean - AUC_degraded`          | Absolute performance drop under degradation.                          |
| **TTDA-Gain**                       | `AUC_with_TTA - AUC_without_TTA`    | How much TTDA improves performance (per degradation condition).       |
| **Cross-Dataset Gap**               | `AUC_in_domain - AUC_cross_dataset` | How well the model generalizes. Smaller = better.                     |

### 8.3 Statistical Rigor

Every metric is reported with **95% confidence intervals** from 3 runs with different random seeds:

```
AUC = mean ± 1.96 × (std / √n),  where n = 3
```

Significance testing between RobustAV and baselines uses the **paired bootstrap test** (10,000 resamples) with p < 0.05 threshold.

### 8.4 Per-Category Metrics (for E12)

For each FakeAVCeleb manipulation category, we report:

- **AUC** (per category)
- **Sensitivity** (true positive rate for that manipulation type)
- **Specificity** (true negative rate on real samples)

### 8.5 Qualitative Evaluation

1. **Cross-modal attention maps**: Visualize which temporal segments the CMCM attends to. For FakeVideo-RealAudio, attention should highlight lip-sync mismatches.
2. **TTDA prediction shift**: Show examples where TTDA changes the prediction from incorrect to correct on degraded samples.
3. **Feature t-SNE**: Visualize feature space separation between real/fake before and after TTDA adaptation.

---

## 9. Expected Results

### 9.1 Expected Main Results Table (E1: Clean Accuracy)

| Model                         | AUC ↑              | EER ↓              | AP ↑               |
| :---------------------------- | :------------------ | :------------------ | :------------------ |
| Late-Fusion (Xception+AASIST) | 0.82-0.86           | 0.18-0.22           | 0.80-0.85           |
| BA-TFD                        | 0.88-0.92           | 0.12-0.16           | 0.87-0.91           |
| RobustAV-VisualOnly           | 0.85-0.89           | 0.14-0.18           | 0.84-0.88           |
| RobustAV-NoTTDA               | 0.90-0.94           | 0.10-0.14           | 0.89-0.93           |
| **RobustAV (ours)**     | **0.91-0.95** | **0.09-0.13** | **0.90-0.94** |

**Rationale**: On clean data, RobustAV with TTDA should perform slightly better or equal to RobustAV-NoTTDA. The main advantage of TTDA appears under degradation. Foundation model features (DINOv2 + Whisper) should provide a meaningful boost over BA-TFD's custom architecture.

### 9.2 Expected Degradation Robustness (E2-E7: RAR Values)

| Degradation      | BA-TFD RAR | Late-Fusion RAR | RobustAV-NoTTDA RAR | RobustAV RAR        |
| :--------------- | :--------- | :-------------- | :------------------ | :------------------ |
| JPEG QF=75       | 0.85-0.90  | 0.75-0.82       | 0.90-0.95           | **0.94-0.98** |
| JPEG QF=50       | 0.70-0.80  | 0.60-0.70       | 0.80-0.88           | **0.88-0.95** |
| Resize 0.5×     | 0.75-0.85  | 0.65-0.75       | 0.82-0.90           | **0.90-0.96** |
| Resize 0.25×    | 0.55-0.70  | 0.50-0.60       | 0.65-0.78           | **0.78-0.88** |
| Noise σ=0.02    | 0.80-0.88  | 0.70-0.80       | 0.85-0.92           | **0.92-0.97** |
| MP3 128kbps      | 0.88-0.93  | 0.78-0.85       | 0.92-0.96           | **0.95-0.98** |
| Social media sim | 0.60-0.72  | 0.50-0.62       | 0.70-0.82           | **0.82-0.92** |

**Key expected findings:**

1. **TTDA provides the largest gains under severe degradation** (JPEG QF=50, resize 0.25×, social media sim) — where other models drop 20-40%, RobustAV with TTDA drops only 5-15%.
2. **Foundation model features are inherently more robust** than purpose-built architectures (RobustAV-NoTTDA already outperforms BA-TFD on degraded data).
3. **Late-fusion is the weakest** because unimodal detectors independently fail under degradation, and score averaging amplifies errors.

### 9.3 Expected Cross-Dataset Generalization (E8-E9)

| Model              | LAV-DF Clean AUC    | LAV-DF + Social Media AUC | Cross-Dataset Gap   |
| :----------------- | :------------------ | :------------------------ | :------------------ |
| BA-TFD             | 0.72-0.78           | 0.55-0.65                 | 0.15-0.20           |
| Late-Fusion        | 0.65-0.72           | 0.48-0.58                 | 0.18-0.25           |
| RobustAV-NoTTDA    | 0.78-0.85           | 0.65-0.75                 | 0.10-0.15           |
| **RobustAV** | **0.80-0.87** | **0.72-0.82**       | **0.08-0.13** |

**Key expected finding**: TTDA is especially powerful for cross-dataset evaluation because it adapts to the specific degradation characteristics of the new dataset at test time.

### 9.4 Expected Adversarial Robustness (E10-E11)

| Attack     | ε / SNR | BA-TFD AUC | RobustAV AUC |
| :--------- | :------- | :--------- | :----------- |
| PGD visual | ε=2/255 | 0.55-0.65  | 0.70-0.80    |
| PGD visual | ε=4/255 | 0.40-0.55  | 0.55-0.68    |
| PGD visual | ε=8/255 | 0.25-0.40  | 0.40-0.55    |
| PGD audio  | SNR=30dB | 0.50-0.62  | 0.65-0.75    |
| PGD audio  | SNR=20dB | 0.35-0.48  | 0.50-0.62    |

**Key expected finding**: RobustAV should be more adversarially robust than BA-TFD because:

1. Foundation model features are more robust to input perturbations than purpose-built features
2. Cross-modal fusion provides redundancy — attacking one modality still leaves the other modality's signal

### 9.5 Expected Ablation Results

| Ablation                   | Clean AUC | Degraded AUC (avg) | Takeaway                             |
| :------------------------- | :-------- | :----------------- | :----------------------------------- |
| Full RobustAV              | 0.93      | 0.88               | Best overall                         |
| A1: No TTDA                | 0.93      | 0.80               | TTDA adds ~8% under degradation      |
| A2: Visual-only            | 0.87      | 0.75               | Cross-modal adds ~13%                |
| A3: No contrastive loss    | 0.91      | 0.83               | Contrastive loss adds ~5%            |
| A4: No degradation loss    | 0.93      | 0.84               | Deg loss adds ~4%                    |
| A5: CLIP instead of DINOv2 | 0.91      | 0.85               | DINOv2 slightly better               |
| A10: Full fine-tune        | 0.94      | 0.78               | Overfits; LN-only generalizes better |

---

## 10. Kaggle Session Plan

### Session 1: Feature Extraction & Caching (~4-5 hours)

**Notebook: `01_feature_extraction.ipynb`**

```
Tasks:
1. Install dependencies
2. Load DINOv2-Small and Whisper-Small
3. Generate train/val/test manifests from FakeAVCeleb
4. Extract visual features for all clips (train+val+test)
5. Extract audio features for all clips
6. Generate degraded test versions (all 22 conditions)
7. Extract features for degraded test sets
8. Save everything as structured .pt files
9. Create metadata.json

Output: Upload as private Kaggle dataset "robustav-features-v1"
Estimated time: 3-4 hours
```

### Session 2: Model Training (~8-10 hours)

**Notebook: `02_train_robustav.ipynb`**

```
Tasks:
1. Load cached features from Session 1 dataset
2. Build CachedAVDataset (loads .pt files, applies degradation augmentation)
3. Initialize CMCM, temporal pooling, classification head
4. Initialize LN-tuning parameters for DINOv2 and Whisper
5. Train for 30 epochs with early stopping
6. Log training/val loss and AUC per epoch
7. Save best checkpoint

Output: Upload best model checkpoint as Kaggle dataset "robustav-checkpoint-v1"
Estimated time: 6-8 hours
Key risk: OOM — monitor memory, reduce batch size if needed
```

### Session 3: Main Evaluation (E1-E9) (~4-6 hours)

**Notebook: `03_evaluate_clean_degraded.ipynb`**

```
Tasks:
1. Load trained RobustAV checkpoint
2. Evaluate on FakeAVCeleb clean test set (E1)
3. Evaluate on all 22 degraded test conditions (E2-E7)
4. For each condition, run with and without TTDA
5. Evaluate on LAV-DF test set clean + degraded (E8-E9)
6. Compute AUC, EER, AP, RAR, TTDA-Gain for every condition
7. Save all results as JSON/CSV

Output: Results JSON + upload as dataset "robustav-results-v1"
Estimated time: 4-5 hours (TTDA adds ~30% overhead per sample)
```

### Session 4: Baselines Evaluation (~6-8 hours)

**Notebook: `04_evaluate_baselines.ipynb`**

```
Tasks:
1. Set up BA-TFD with pre-trained checkpoint
2. Set up Late-Fusion (XceptionNet + AASIST)
3. Evaluate both on FakeAVCeleb clean + all degraded conditions
4. Evaluate on LAV-DF clean + degraded
5. Compute same metrics as Session 3

Output: Baseline results JSON
Estimated time: 5-7 hours
```

### Session 5: Ablation Studies (A1-A4, A9) (~6-8 hours)

**Notebook: `05_ablations_architecture.ipynb`**

```
Tasks:
1. A1: Already done (Session 3 ran with/without TTDA)
2. A2: Train RobustAV-VisualOnly (no audio branch)
3. A3: Retrain with BCE-only loss (no contrastive)
4. A4: Retrain without degradation loss
5. A9: Retrain with 1-layer and 4-layer CMCM variants
6. Evaluate each ablation on clean + key degraded test sets

Output: Ablation results JSON
Estimated time: 6-8 hours
```

### Session 6: Ablation Studies (A5-A8, A10) (~6-8 hours)

**Notebook: `06_ablations_encoders.ipynb`**

```
Tasks:
1. A5: Extract CLIP features, retrain with CLIP encoder
2. A6: Extract Wav2Vec2 features, retrain with Wav2Vec2 encoder
3. A7: Evaluate TTDA with {1,3,5,10} steps
4. A8: Evaluate TTDA with degradation subsets
5. A10: Full fine-tune training + evaluation

Output: Encoder ablation results JSON
Estimated time: 6-8 hours
```

### Session 7: Adversarial Robustness (E10-E11) (~4-6 hours)

**Notebook: `07_adversarial_evaluation.ipynb`**

```
Tasks:
1. Implement PGD-20 for visual stream (end-to-end through frozen DINOv2)
2. Implement PGD-20 for audio stream (end-to-end through frozen Whisper)
3. Generate adversarial examples at ε={2,4,8}/255 and SNR={20,30,40}dB
4. Verify imperceptibility (SSIM/PESQ gates)
5. Evaluate RobustAV + baselines on adversarial test sets
6. Compute adversarial RAR

Output: Adversarial results JSON
Estimated time: 4-5 hours
```

### Session 8: Figures & Visualization (~2-3 hours)

**Notebook: `08_figures.ipynb`**

```
Tasks:
1. Generate all paper figures (see Section 11)
2. Cross-modal attention map visualizations
3. t-SNE feature space visualization
4. TTDA prediction shift examples
5. Export as high-resolution PNG/PDF

Output: All paper figures
Estimated time: 2-3 hours
```

### Total: 8 Kaggle sessions, ~45-55 GPU-hours

---

## 11. Paper Figures (6 Key Figures)

### Figure 1: Architecture Diagram

- **Content**: Full RobustAV architecture with TTDA loop
- **Tool**: TikZ or draw.io, exported as PDF
- **Location**: Section 3 (Method)

### Figure 2: TTDA Effect — The Paper's Main Figure

- **Content**: Line chart. X-axis = degradation severity (clean → mild → severe). Y-axis = AUC. Lines: RobustAV (with TTDA), RobustAV (without TTDA), BA-TFD, Late-Fusion.
- **Key visual**: The gap between with/without TTDA lines widens as degradation increases.
- **Location**: Section 4 (Experiments), front and center.

### Figure 3: RAR Heatmap

- **Content**: Heatmap. Rows = models (RobustAV, BA-TFD, Late-Fusion, ablations). Columns = degradation conditions. Color = RAR value (green=robust, red=fragile).
- **Location**: Section 4.

### Figure 4: Cross-Dataset Generalization Bar Chart

- **Content**: Grouped bar chart. Groups = {FakeAVCeleb clean, FakeAVCeleb degraded, LAV-DF clean, LAV-DF degraded}. Bars = models. Height = AUC.
- **Location**: Section 4.

### Figure 5: Cross-Modal Attention Maps

- **Content**: Visualization of CMCM attention weights on example clips. Show that attention focuses on lip-sync regions for FakeVideo-RealAudio and on spectral anomalies for RealVideo-FakeAudio.
- **Location**: Section 5 (Analysis/Discussion).

### Figure 6: Ablation Summary

- **Content**: Bar chart showing AUC for each ablation variant on clean and degraded data (averaged). Clear visual of each component's contribution.
- **Location**: Section 4 (Ablations).

---

## 12. Timeline

| Week               | Sessions | Activities                                  | Deliverable                    |
| :----------------- | :------- | :------------------------------------------ | :----------------------------- |
| **Week 1**   | S1       | Feature extraction, caching, dataset upload | Cached features Kaggle dataset |
| **Week 1-2** | S2       | Model training, hyperparameter tuning       | Trained checkpoint             |
| **Week 2**   | S3, S4   | Main evaluation + baselines                 | Results for E1-E9              |
| **Week 3**   | S5, S6   | All ablation studies                        | Ablation results               |
| **Week 3**   | S7       | Adversarial robustness evaluation           | Adversarial results            |
| **Week 4**   | S8       | Figure generation, visualization            | All paper figures              |
| **Week 4-5** | —       | Paper writing (no GPU needed)               | Draft manuscript               |
| **Week 5-6** | —       | Revision, polish, supplementary             | Camera-ready                   |

**Total active compute time: ~3 weeks. Total project time: ~5-6 weeks.**

---

## 13. Risk Mitigation

| Risk                                           | Likelihood | Impact | Mitigation                                                                                                                                 |
| :--------------------------------------------- | :--------- | :----- | :----------------------------------------------------------------------------------------------------------------------------------------- |
| DINOv2-Small + Whisper-Small OOM on T4         | Medium     | High   | Use gradient checkpointing; reduce batch to 4; split visual/audio forward passes                                                           |
| TTDA hurts clean accuracy                      | Low        | High   | Use very small LR (1e-5) for TTA step; cap at 1 step; add a "no-adapt" fallback if entropy is already low                                  |
| RobustAV doesn't beat BA-TFD on clean data     | Medium     | Medium | Acceptable — the story is about*robustness*, not clean accuracy. If clean is equal but degraded is better, that's still a strong paper. |
| FakeAVCeleb too small for training             | Medium     | Medium | Use heavy augmentation (our degradation pipeline IS augmentation); use contrastive learning which works well with small datasets           |
| Whisper features don't help for audio deepfake | Low        | High   | Ablation A6 compares with Wav2Vec2; worst case, use Wav2Vec2 instead                                                                       |
| Kaggle session timeouts mid-training           | Medium     | Medium | Checkpoint every 2 epochs; load from checkpoint on resume; all data is cached so restart is fast                                           |
| LAV-DF cross-dataset performance is poor       | Medium     | Low    | Expected — cross-dataset is always lower. As long as RobustAV is*relatively* better than baselines, the story holds                     |

---

## 14. Contributions Summary (for the Paper)

The paper will claim the following contributions:

1. **RobustAV**: A novel audio-visual deepfake detection framework that combines parameter-efficient foundation model adaptation with cross-modal consistency learning, achieving competitive detection accuracy with only 2.1% trainable parameters.
2. **Test-Time Degradation Adaptation (TTDA)**: A novel inference-time mechanism that adapts model parameters to unknown degradation conditions using entropy minimization over pseudo-degraded copies, requiring no labels and negligible compute overhead.
3. **Comprehensive robustness evaluation**: Systematic evaluation across 23 degradation conditions, 2 datasets, and adversarial attacks, demonstrating that RobustAV with TTDA maintains robust performance where existing detectors fail.
4. **Ablation-backed design**: Thorough ablation study (10 variants) validating each architectural choice and demonstrating the complementary benefits of foundation model features, cross-modal fusion, and test-time adaptation.

---

*End of Execution Plan. All code, data, and experiments are designed to be fully executable on Kaggle with T4 GPU within the stated session count.*
