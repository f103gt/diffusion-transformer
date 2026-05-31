Diffusion Transformers (DiT-B)
========

This repository implements **DiT-B (Diffusion Transformer Base)** for class-conditional image generation on LandscapesHQ. It provides:

* **VAE**: Three interchangeable modes — train from scratch (`model/vae/vae.py`), fine-tune the pre-trained SD VAE, or use it frozen — all controlled by a single `vae_mode` key in the config
* **Conditional DiT Training**: Class-conditional DiT-B with **Classifier-Free Guidance (CFG)** for 7 mountain categories
* **Interactive Testing**: Gradio web interface for real-time image generation with adjustable upscaling
* **Latent Caching**: Fast VAE latent caching to accelerate DiT training

**Key Implementation Details:**
* **DiT-B Architecture**: 12 transformer layers, 768 hidden dimensions, 12 attention heads
* **Classifier-Free Guidance**: Randomly drops class labels during training (10% dropout) to enable CFG during inference for better control over generation quality vs diversity
* **Conditional Generation**: 7 mountain landscape classes (canyon, cliff, glacier, mountain, rocky coast, snowy mountain, volcano)
* **CLIP-based Dataset Classification**: Used CLIP vision model to automatically classify Kaggle LandscapesHQ dataset into mountain categories
* **Fixed Variance**: Variance fixed during diffusion training (like original DDPM, not learned)
* **No EMA**: Simplified training without exponential moving average of model weights
* **Efficient Training**: Gradient accumulation support and optional VAE latent pre-computation


## Setup

### Using Virtual Environment (venv)
1. Create and activate a virtual environment with Python 3.10
   ```bash
   python -m venv dit_env
   # On Windows
   dit_env\Scripts\activate
   # On macOS/Linux
   source dit_env/bin/activate
   ```

2. Install dependencies
   ```bash
   pip install -r requirements.txt
   ```

3. Download LPIPS weights (required for VAE training)
   - Open this link in your browser (do not use curl/wget): https://github.com/richzhang/PerceptualSimilarity/raw/master/lpips/weights/v0.1/vgg.pth
   - Download the raw file and place it at: `model/vae/weights/vgg.pth`

## Data Preparation

### LandscapesHQ Dataset
1. Download the images from [Kaggle - LHQ-1024 dataset](https://www.kaggle.com/datasets/dimensi0n/lhq-1024)
2. Place them in the following directory structure:
   ```
   DiT-PyTorch/
   └── data/
       └── LandscapesHQ/
           └── *.jpg
   ```

### Dataset Classification with CLIP
The dataset is automatically classified into 7 mountain categories using CLIP vision model:
- **Categories**: Canyon, Cliff, Glacier, Mountain, Rocky Coast, Snowy Mountain, Volcano

Classification can be performed using CLIP:
```python
from model.vae.clip_classifier import CLIPClassifier

classifier = CLIPClassifier(model_name="ViT-B/32")
labels = classifier.classify_directory(
    image_dir="data/LandscapesHQ",
    categories=["canyon", "cliff", "glacier", "mountain", "rocky coast", "snowy mountain", "volcano"],
    batch_size=32
)

# Save labels to JSON
import json
with open("data/mountain-labels.json", "w") as f:
    json.dump(labels, f, indent=2)
```

After classification, verify labels with:
```bash
python -m tools.create_labels --mode validate --input data/mountain-labels.json --image-dir data/LandscapesHQ
```
## Configuration

Configuration files control all training and model parameters. Primary config: `config/landscapeshq.yaml`

### Important Parameters

**Diffusion Configuration:**
- `num_timesteps`: Number of diffusion steps (typically 1000)
- `beta_start`, `beta_end`: Noise schedule bounds

**DiT-B Model Configuration:**
- `num_layers`: 12 (DiT-B has 12 transformer layers)
- `hidden_size`: 768 (embedding dimension)
- `num_heads`: 12 (number of attention heads)
- `head_dim`: 64 (dimension per head)
- `patch_size`: 2 (patches extracted from latent space)
- `num_classes`: 7 (mountain landscape categories)
- `class_dropout_prob`: 0.1 (**Classifier-Free Guidance** - drops labels 10% of the time during training)

**Training Configuration:**
- `vae_mode`: VAE experiment mode — `scratch`, `finetune`, or `pretrained` (see below)
- `pretrained_vae_model`: HuggingFace model ID used in `finetune`/`pretrained` modes (default: `stabilityai/sd-vae-ft-mse`)
- `autoencoder_epochs`: Training epochs for `scratch` mode
- `finetune_epochs`: Training epochs for `finetune` mode (fewer needed — weights are pretrained)
- `save_latents`: Enable to pre-compute VAE latents (recommended for faster training)
- `dit_batch_size`: Batch size for DiT training
- `dit_epochs`: Number of training epochs
- `dit_acc_steps`: Gradient accumulation steps for larger effective batch sizes
- `dit_lr`: Learning rate (default: 1e-5)
## Training Workflow

### 1. VAE Setup

Choose a VAE experiment mode by setting `vae_mode` in `config/landscapeshq.yaml`:

| Mode | Description | Requires checkpoint? |
|---|---|---|
| `pretrained` | Frozen SD VAE from HuggingFace — no training needed | No |
| `finetune` | Start from SD VAE, fine-tune on your dataset | Yes (saved by train script) |
| `scratch` | Train a custom VAE from zero using `model/vae/vae.py` | Yes (saved by train script) |

```yaml
train_params:
  vae_mode: 'pretrained'          # scratch | finetune | pretrained
  pretrained_vae_model: 'stabilityai/sd-vae-ft-mse'  # used by finetune/pretrained
  finetune_epochs: 1              # epochs for finetune mode (1-3 is sufficient)
  autoencoder_epochs: 50          # epochs for scratch mode
```

**`pretrained` mode** — fastest path, good baseline before deciding to fine-tune. The SD VAE weights (~330 MB) download automatically on first use and are cached at `~/.cache/huggingface/hub/`. No training script needed; skip straight to latent caching.

**`finetune` mode** — recommended if you see domain-specific reconstruction artefacts with the frozen VAE. Fine-tunes the SD VAE on your dataset for `finetune_epochs` epochs:

```bash
python -m tools.train_vae --config config/landscapeshq.yaml
```

**`scratch` mode** — trains `model/vae/vae.py` from random initialisation. Requires more epochs and data than fine-tuning, but gives full control over the architecture:

```bash
python -m tools.train_vae --config config/landscapeshq.yaml
```

**Outputs (finetune / scratch):**
- Checkpoint: `landscapeshq/vae_autoencoder_ckpt.pth`
- Sample reconstructions: `landscapeshq/vae_autoencoder_samples/`

### 2. Generate and Cache VAE Latents (Optional but Recommended)

Pre-compute VAE latents for the entire dataset to significantly speed up DiT training:

```bash
python -m tools.infer_vae --config config/landscapeshq.yaml
```

**Outputs:**
- Cached latents: `landscapeshq/vae_latents/`
- This enables faster DiT training iterations

### 3. Train Conditional DiT-B with Classifier-Free Guidance

Train the conditional Diffusion Transformer with class-conditioning and CFG:

```bash
python -m tools.train_vae_dit --config config/landscapeshq.yaml
```

**Key Features:**
- **Class Conditioning**: Generates images specific to the 7 mountain categories
- **Classifier-Free Guidance**: 10% of training samples use the "null" token (no class info)
  - During inference, CFG scale can be adjusted to trade-off diversity for quality
- **Checkpointing**: Saves latest model as `landscapeshq/dit_ckpt.pth`
- **Resumable Training**: Supports resuming from checkpoints via `landscapeshq/dit_training_state.pth`

**Outputs:**
- DiT checkpoint: `landscapeshq/dit_ckpt.pth`
- Training state: `landscapeshq/dit_training_state.pth` (for resuming)


## Testing & Inference

### Running with Pre-trained Weights (skip training)

If you have trained checkpoints (or downloaded them from the repo), you can skip all training steps and run inference immediately.

1. Place checkpoint files in `landscapeshq/` (tracked by git — `.pth` files there are not gitignored):
   - `landscapeshq/dit_ckpt.pth` — trained DiT-B model (always required)
   - `landscapeshq/vae_autoencoder_ckpt.pth` — VAE weights (only for `scratch` or `finetune` mode)

2. Set `vae_mode` in `config/landscapeshq.yaml` to match the weights you have:

   | Weights available | Set `vae_mode` to |
   |---|---|
   | DiT only | `pretrained` (uses frozen SD VAE, no extra weights needed) |
   | DiT + fine-tuned VAE | `finetune` |
   | DiT + custom VAE trained from scratch | `scratch` |

3. Launch directly:
   ```bash
   python gradio_app.py --config config/landscapeshq.yaml
   ```

### Interactive Generation with Gradio

Launch the web interface for real-time image generation:

```bash
python gradio_app.py --config config/landscapeshq.yaml --server-name 0.0.0.0 --server-port 7860
```

**Features:**
- **Class Selection**: Choose from 7 mountain categories (canyon, cliff, glacier, mountain, rocky coast, snowy mountain, volcano)
- **Random Seed**: Each generation is unique (random seed for variety)
- **Upscaling**: 1-4× upscaling
- **Live Progress**: Real-time sampling progress bar

**Access:**
- Local: `http://localhost:7860`
- Remote (with `--share`): Generates a public link

### Batch Sampling

Generate multiple images and save denoising progression:

```bash
python -m tools.sample_dit --config config/landscapeshq.yaml
```

**Output Files:**
- Sampled images at each timestep: `landscapeshq/samples/x0_*.png`
- `x0_0.png`: Final decoded image (timestep T=0, clean image)
- `x0_999.png` to `x0_1.png`: Intermediate denoising steps (for visualization)
  - Shows the progressive denoising from pure noise (T=999) to clean image (T=0)

## Output Structure

All outputs are saved to the `task_name` directory (default: `landscapeshq/`)

```
landscapeshq/
├── vae_autoencoder_ckpt.pth              # VAE checkpoint (scratch or finetune mode only)
├── dit_ckpt.pth                          # Trained DiT-B checkpoint
├── dit_training_state.pth                # Training state (for resuming)
├── vae_autoencoder_samples/              # VAE reconstruction samples during training
├── vae_latents/                          # Pre-computed VAE latents (if save_latents=True)
└── samples/                              # Generated images during sampling
    ├── x0_0.png                          # Final generated image
    ├── x0_1.png
    ├── ...
    └── x0_999.png                        # Earliest denoising step (most noise)
```

## Architecture Details

### DiT-B (Diffusion Transformer Base)

**Model Specifications:**
- **Transformer Layers**: 12
- **Hidden Dimension**: 768
- **Attention Heads**: 12
- **Head Dimension**: 64
- **Patch Size**: 2×2 (applied to 32×32 latent space)
- **Number of Patches**: 16×16 = 256 patches
- **Sequence Length**: 257 tokens (256 patches + 1 class token)

**Key Components:**

1. **Patch Embedding** (`model/patch_embed.py`)
   - Converts 32×32×4 latent into 256 patch embeddings of dimension 768

2. **Class Embedding** (`model/transformer.py::LabelEmbedder`)
   - Embeds 7 mountain classes into 768-dim vectors
   - Implements **Classifier-Free Guidance**: Randomly drops class labels during training (10% probability)
   - Null token class: `num_classes + 1 = 8`

3. **Time Embedding** (`model/blocks/time_embedding.py`)
   - Sinusoidal positional encoding for diffusion timesteps (0-999)
   - Projected to 768 dimensions

4. **Transformer Blocks** (`model/transformer_layer.py`)
   - Multi-head self-attention with 12 heads
   - Feed-forward networks (MLP) with SiLU activation
   - Adaptive Layer Normalization (AdaLN) for time/class conditioning
   - Pre-normalization architecture

5. **Output Projection**
   - Projects back to 32 channels (4 channels × 4 patches per position)
   - Predicts noise to subtract from noisy latents

### VAE

Three modes are available, selected via `vae_mode` in the config.

**`pretrained` / `finetune` — `stabilityai/sd-vae-ft-mse`** (HuggingFace diffusers `AutoencoderKL`):
- Input: 128×128×3 → Output: 32×32×4 (16× compression)
- Latents scaled by `0.18215` before DiT training (standard SD practice)
- In `finetune` mode, the saved state dict is loaded on top of the pretrained weights

**`scratch` — `model/vae/vae.py`** (custom `VAE`):
- Same spatial compression (128×128 → 32×32×4) via configurable encoder/decoder blocks
- No latent scaling — DiT trains directly on raw latents
- Architecture controlled by `autoencoder_params` in the config (channels, layers, attention)

**Training losses (scratch and finetune modes):**
- Reconstruction (MSE) — pixel-level fidelity
- KL divergence — keeps the latent space well-structured
- Perceptual (LPIPS, VGG) — preserves semantic and texture quality
- For `finetune`: 1–2 epochs is sufficient given the strong pretrained initialisation

## Classifier-Free Guidance (CFG)

Classifier-Free Guidance allows control over generation quality during inference:

```python
# During sampling (in tools/sample_vae_dit.py or gradio_app.py):
# CFG formula: x_pred = x_uncond + cfg_scale * (x_cond - x_uncond)

cfg_scale = 7.5  # Typical range: 1.0 - 15.0
# Higher cfg_scale: More adherence to class, less diversity
# Lower cfg_scale: More diversity, weaker class control
```

**Training Implementation:**
- 10% of training batches: Class labels replaced with null token
- Model learns both conditional and unconditional generation
- Enables CFG during inference without additional fine-tuning