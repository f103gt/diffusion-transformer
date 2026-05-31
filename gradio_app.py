"""
Gradio interface for DiT-PyTorch mountain image generation.
Features:
- Class-conditional generation (7 mountain categories)
- Three VAE modes: scratch | finetune | pretrained (set in config)
- Classifier-Free Guidance, structure checking, retry logic
- Chroma fix, deband, deblock, CLAHE, denoising, sharpening
- Real-ESRGAN AI upscaling (falls back to Lanczos if unavailable)
"""
import gradio as gr
import torch
import numpy as np
import cv2
from PIL import Image
import yaml
import os
import json
from tqdm import tqdm
import sys
import gc

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model.transformer import DIT
from scheduler.linear_scheduler import LinearNoiseScheduler

CLASS_NAMES = ['canyon', 'cliff', 'glacier', 'mountain', 'rocky coast', 'snowy mountain', 'volcano']

device = None
model = None
vae = None
vae_mode = None
scheduler = None
config = None
class_to_id = None


# ---------------------------------------------------------------------------
# VAE loading
# ---------------------------------------------------------------------------

def _load_vae(train_config, dataset_config, autoencoder_config):
    mode = train_config.get('vae_mode', 'pretrained')
    ckpt_path = os.path.join(train_config['task_name'], train_config['vae_autoencoder_ckpt_name'])
    hf_model = train_config.get('pretrained_vae_model', 'stabilityai/sd-vae-ft-mse')

    if mode == 'scratch':
        from model.vae.vae import VAE
        print("VAE mode: scratch (custom VAE trained from scratch)")
        vae = VAE(im_channels=dataset_config['im_channels'], model_config=autoencoder_config).to(device)
        if not os.path.exists(ckpt_path):
            raise FileNotFoundError(
                f"Custom VAE checkpoint not found at {ckpt_path}\n"
                f"Train first with: python -m tools.train_vae --config <config>"
            )
        vae.load_state_dict(torch.load(ckpt_path, map_location=device), strict=True)
        vae.eval()
        print(f"Custom VAE loaded from {ckpt_path}")

    elif mode == 'finetune':
        from model.vae.pretrained_vae import HuggingFaceVAEWrapper
        print("VAE mode: finetune (pre-trained SD VAE fine-tuned on dataset)")
        vae = HuggingFaceVAEWrapper(pretrained_model_name_or_path=hf_model, device=device)
        if not os.path.exists(ckpt_path):
            raise FileNotFoundError(
                f"Fine-tuned VAE checkpoint not found at {ckpt_path}\n"
                f"Fine-tune first with vae_mode: finetune in the config."
            )
        vae.vae.load_state_dict(torch.load(ckpt_path, map_location=device))
        vae.vae.eval()
        print(f"Fine-tuned HuggingFace VAE loaded from {ckpt_path}")

    elif mode == 'pretrained':
        from model.vae.pretrained_vae import HuggingFaceVAEWrapper
        print(f"VAE mode: pretrained (frozen SD VAE — '{hf_model}')")
        vae = HuggingFaceVAEWrapper(pretrained_model_name_or_path=hf_model, device=device)
        vae.vae.eval()
        print("Using frozen pre-trained SD VAE (no checkpoint needed).")

    else:
        raise ValueError(f"Unknown vae_mode '{mode}'. Choose: scratch | finetune | pretrained")

    return vae, mode


def _vae_decode(z):
    """Mode-aware decode: scratch VAE decodes directly; HF wrapper unscales internally."""
    if vae_mode == 'scratch':
        return vae.to(device).decode(z)
    return vae.decode(z)


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def load_models(config_path="config/landscapeshq.yaml"):
    global device, model, vae, vae_mode, scheduler, config, class_to_id, esrgan_upsampler

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if torch.backends.mps.is_available():
        device = torch.device('mps')
    print(f"Using device: {device}")

    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    print("Configuration loaded successfully")

    diffusion_config = config['diffusion_params']
    dataset_config = config['dataset_params']
    dit_model_config = config['dit_params']
    autoencoder_config = config['autoencoder_params']
    train_config = config['train_params']

    scheduler = LinearNoiseScheduler(
        num_timesteps=diffusion_config['num_timesteps'],
        beta_start=diffusion_config['beta_start'],
        beta_end=diffusion_config['beta_end']
    )

    im_size = dataset_config['im_size'] // 2 ** sum(autoencoder_config['down_sample'])

    dit_cfg = dit_model_config.copy()
    dit_cfg['training'] = False
    model = DIT(im_size=im_size, im_channels=autoencoder_config['z_channels'], config=dit_cfg).to(device)

    dit_checkpoint_path = os.path.join(train_config['task_name'], train_config['dit_ckpt_name'])
    if not os.path.exists(dit_checkpoint_path):
        raise FileNotFoundError(f"DiT checkpoint not found at {dit_checkpoint_path}")
    model.eval()
    model.load_state_dict(torch.load(dit_checkpoint_path, map_location=device))
    print(f"DiT loaded from {dit_checkpoint_path}")

    vae, vae_mode = _load_vae(train_config, dataset_config, autoencoder_config)

    label_json_path = dataset_config.get('label_json_path', '')
    if label_json_path and os.path.exists(label_json_path):
        with open(label_json_path, 'r') as f:
            label_data = json.load(f)
        unique_classes = sorted(set(label_data.values()))
        class_to_id = {cls: idx for idx, cls in enumerate(unique_classes)}
        print(f"Class mapping: {class_to_id}")

    print("=" * 60)
    print("All models loaded successfully!")
    print("=" * 60)


# ---------------------------------------------------------------------------
# Checkpoint hot-swap
# ---------------------------------------------------------------------------

def get_available_checkpoints():
    import glob
    if config is None:
        return []
    task_name = config['train_params']['task_name']
    ckpts = sorted(glob.glob(os.path.join(task_name, 'dit_ckpt*.pth')), key=os.path.getmtime)
    return [os.path.basename(p) for p in ckpts]


def switch_checkpoint(ckpt_name):
    global model
    if model is None or config is None:
        return "Model not loaded"
    ckpt_path = os.path.join(config['train_params']['task_name'], ckpt_name)
    if not os.path.exists(ckpt_path):
        return f"Not found: {ckpt_path}"
    try:
        model.load_state_dict(torch.load(ckpt_path, map_location=device))
        model.eval()
        return f"Loaded {ckpt_name}"
    except Exception as e:
        return f"Error loading {ckpt_name}: {e}"


# ---------------------------------------------------------------------------
# Image quality helpers
# ---------------------------------------------------------------------------

def _fix_chroma(image: Image.Image) -> Image.Image:
    """Remove VAE chroma artifacts — speckles and block-scale mosaic — in LAB space."""
    bgr = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    a = cv2.medianBlur(a, 5)
    b = cv2.medianBlur(b, 5)
    a_f = cv2.GaussianBlur(a.astype(np.float32), (0, 0), 5.0)
    b_f = cv2.GaussianBlur(b.astype(np.float32), (0, 0), 5.0)
    a = np.clip(a_f, 0, 255).astype(np.uint8)
    b = np.clip(b_f, 0, 255).astype(np.uint8)
    lab_fixed = cv2.merge([l, a, b])
    return Image.fromarray(cv2.cvtColor(cv2.cvtColor(lab_fixed, cv2.COLOR_LAB2BGR), cv2.COLOR_BGR2RGB))


def _deblock(image: Image.Image, patch_px: int = 8) -> Image.Image:
    """Smooth VAE patch-grid boundaries then recover edge sharpness with unsharp mask."""
    arr = np.array(image, dtype=np.float32)
    sigma = patch_px / 2.0
    ksize = int(sigma * 3) | 1
    blurred = cv2.GaussianBlur(arr, (ksize, ksize), sigma)
    sharpened = cv2.addWeighted(arr, 1.2, blurred, -0.2, 0)
    return Image.fromarray(np.clip(sharpened, 0, 255).astype(np.uint8))


def _deband(image: Image.Image) -> Image.Image:
    """Remove horizontal banding from DiT attention row-coherence via vertical-only blur."""
    bgr = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    l = lab[:, :, 0]
    l_smooth = cv2.blur(l, (1, 7))
    lab[:, :, 0] = np.clip(l * 0.7 + l_smooth * 0.3, 0, 255)
    bgr_fixed = cv2.cvtColor(lab.astype(np.uint8), cv2.COLOR_LAB2BGR)
    return Image.fromarray(cv2.cvtColor(bgr_fixed, cv2.COLOR_BGR2RGB))


def _enhance(image: Image.Image) -> Image.Image:
    """CLAHE local contrast on L channel — makes rock/vegetation textures pop."""
    bgr = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    l = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(l)
    return Image.fromarray(cv2.cvtColor(cv2.cvtColor(cv2.merge([l, a, b]), cv2.COLOR_LAB2BGR), cv2.COLOR_BGR2RGB))


def _apply_bilateral_denoise(image: Image.Image, strength: int) -> Image.Image:
    """Edge-preserving denoising via bilateral filter. strength 1-10."""
    if strength == 0:
        return image
    sigma = strength * 5
    img_bgr = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    filtered = cv2.bilateralFilter(img_bgr, d=5, sigmaColor=sigma, sigmaSpace=sigma)
    return Image.fromarray(cv2.cvtColor(filtered, cv2.COLOR_BGR2RGB))


def _sharpen(image: Image.Image, strength: int) -> Image.Image:
    """Unsharp mask. strength 1-10. Applied after denoise so it enhances real structure."""
    if strength == 0:
        return image
    amount = strength * 0.12
    arr = np.array(image, dtype=np.float32)
    blurred = cv2.GaussianBlur(arr, (0, 0), 1.5)
    sharpened = cv2.addWeighted(arr, 1.0 + amount, blurred, -amount, 0)
    return Image.fromarray(np.clip(sharpened, 0, 255).astype(np.uint8))


def _fix_local_smudges(image: Image.Image) -> Image.Image:
    """Targeted unsharp mask on low-detail mid-brightness patches (not sky, not shadow)."""
    arr = np.array(image).astype(np.float32)
    gray = cv2.cvtColor(arr.astype(np.uint8), cv2.COLOR_RGB2GRAY).astype(np.float32)
    lap = cv2.Laplacian(gray, cv2.CV_32F)
    h, w = gray.shape
    grid = 8
    ph, pw = h // grid, w // grid
    SMUDGE_THRESH = 600.0
    SKY_MAX = 210.0
    SHADOW_MIN = 30.0
    blend_mask = np.zeros((h, w), dtype=np.float32)
    for i in range(grid):
        for j in range(grid):
            y0, y1 = i * ph, min((i + 1) * ph, h)
            x0, x1 = j * pw, min((j + 1) * pw, w)
            patch_gray = gray[y0:y1, x0:x1]
            patch_lap = lap[y0:y1, x0:x1]
            mean_g = float(patch_gray.mean())
            var_l = float(patch_lap.var())
            if mean_g > SKY_MAX or mean_g < SHADOW_MIN:
                continue
            if var_l < SMUDGE_THRESH:
                strength = (1.0 - var_l / SMUDGE_THRESH) ** 0.7
                blend_mask[y0:y1, x0:x1] = np.maximum(blend_mask[y0:y1, x0:x1], strength)
    if blend_mask.max() < 0.05:
        return image
    ksize = int(ph * 1.2) | 1
    smooth_mask = np.clip(cv2.GaussianBlur(blend_mask, (ksize, ksize), ph / 4.0), 0.0, 0.80)
    blurred_fine = cv2.GaussianBlur(arr, (0, 0), 1.0)
    blurred_coarse = cv2.GaussianBlur(arr, (0, 0), 2.5)
    fine_a = blend_mask[:, :, np.newaxis] * 1.2
    coarse_a = blend_mask[:, :, np.newaxis] * 1.0
    sharpened = np.clip(arr + fine_a * (arr - blurred_fine) + coarse_a * (arr - blurred_coarse), 0, 255)
    m = smooth_mask[:, :, np.newaxis]
    return Image.fromarray((sharpened * m + arr * (1.0 - m)).astype(np.uint8))


def _canyon_color_correct(image: Image.Image) -> Image.Image:
    """Nudge non-sky land toward warm red-orange only when the image lacks it."""
    bgr = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    l, a, b = cv2.split(lab)
    sky_mask = (l > 165) & (np.abs(a - 128) < 18) & (np.abs(b - 128) < 22)
    land_mask = (~sky_mask) & (l > 40)
    if not land_mask.any():
        return image
    if float(a[land_mask].mean()) >= 132:
        return image
    needs = land_mask & (a < 133)
    if not needs.any():
        return image
    alpha = 0.28
    a[needs] = np.clip(a[needs] + alpha * (142.0 - a[needs]), 0, 255)
    b[needs] = np.clip(b[needs] + alpha * (138.0 - b[needs]), 0, 255)
    lab_out = cv2.merge([l, a, b]).astype(np.uint8)
    return Image.fromarray(cv2.cvtColor(cv2.cvtColor(lab_out, cv2.COLOR_LAB2BGR), cv2.COLOR_BGR2RGB))


def _has_structure(image: Image.Image) -> bool:
    """Return True if the image has coherent landscape structure (not pure noise/texture)."""
    arr = np.array(image)
    gray = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY).astype(np.float32)
    mean_brightness = gray.mean()
    if mean_brightness < 15.0 or mean_brightness > 248.0:
        return False
    lap = cv2.Laplacian(gray, cv2.CV_32F)
    h, w = gray.shape
    bh, bw = max(h // 4, 1), max(w // 4, 1)
    block_vars = [float(lap[i*bh:(i+1)*bh, j*bw:(j+1)*bw].var())
                  for i in range(4) for j in range(4)
                  if lap[i*bh:(i+1)*bh, j*bw:(j+1)*bw].size > 0]
    if not block_vars:
        return False
    min_lap = min(block_vars)
    mean_lap = float(np.mean(block_vars))
    if min_lap > 1000.0:
        return False
    if mean_lap > 4000.0 and min_lap > 600.0:
        return False
    return True


# ---------------------------------------------------------------------------
# Generation
# ---------------------------------------------------------------------------

@torch.no_grad()
def generate_image(class_name, upscale_factor, progress=gr.Progress()):
    if model is None or vae is None:
        return None, "Models not loaded!"

    # Fixed quality settings
    guidance_scale = 3.0
    denoise_strength = 3
    sharpen_strength = 5

    _best_image = None
    _best_lap_info = ""

    model.eval()

    for attempt in range(10):
        try:
            if class_to_id is not None and class_name in class_to_id:
                class_idx = class_to_id[class_name]
            else:
                class_idx = CLASS_NAMES.index(class_name)

            if torch.cuda.is_available():
                torch.cuda.synchronize()
                torch.cuda.empty_cache()
            gc.collect()

            seed = np.random.randint(0, 10000000)
            torch.manual_seed(seed)
            np.random.seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)

            if attempt > 0:
                progress(0.05, desc=f"Retry {attempt + 1}/10 (previous failed structure check)...")
            else:
                progress(0.1, desc="Initializing generation...")

            ae_cfg = config['autoencoder_params']
            ds_cfg = config['dataset_params']
            diff_cfg = config['diffusion_params']
            im_size = ds_cfg['im_size'] // 2 ** sum(ae_cfg['down_sample'])

            xt = torch.randn(1, ae_cfg['z_channels'], im_size, im_size, device=device)
            class_labels = torch.tensor([class_idx], device=device)
            uncond_labels = torch.tensor([model.num_classes], device=device)

            total_steps = diff_cfg['num_timesteps']
            t_all = torch.arange(total_steps, device=device)

            # Early-exit checkpoints at 75% and 90% through denoising
            early_exit_steps = {
                int(total_steps * 0.25): "75%",
                int(total_steps * 0.10): "90%",
            }
            aborted = False

            progress(0.2, desc="Starting diffusion sampling...")

            for i in tqdm(reversed(range(total_steps)), desc="Sampling"):
                if guidance_scale > 1.0:
                    noise_cond = model(xt, t_all[i:i + 1], class_labels)
                    noise_uncond = model(xt, t_all[i:i + 1], uncond_labels)
                    noise_pred = noise_uncond + guidance_scale * (noise_cond - noise_uncond)
                else:
                    noise_pred = model(xt, t_all[i:i + 1], class_labels)

                xt, x0_pred = scheduler.sample_prev_timestep(xt, noise_pred, t_all[i])
                step_progress = 0.2 + 0.6 * (total_steps - i) / total_steps
                progress(step_progress, desc=f"Sampling step {total_steps - i}/{total_steps}")

                if i in early_exit_steps:
                    with torch.no_grad():
                        preview = _vae_decode(x0_pred.clamp(-4., 4.))
                    prev_arr = (((preview.clamp(-1., 1.) + 1) / 2)[0]
                                .permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
                    if not _has_structure(Image.fromarray(prev_arr)):
                        print(f"Early exit at {early_exit_steps[i]} — retrying attempt {attempt + 1}...")
                        progress(step_progress, desc=f"Early exit at {early_exit_steps[i]} — retrying...")
                        del prev_arr, preview
                        aborted = True
                        break
                    del prev_arr, preview

            if aborted:
                del xt, x0_pred, noise_pred
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                gc.collect()
                continue

            progress(0.8, desc="Decoding latent...")
            ims = _vae_decode(xt.clamp(-4., 4.))
            ims = torch.clamp(ims, -1., 1.).detach().cpu()
            ims = (ims + 1) / 2

            img_array = (ims[0].numpy().transpose(1, 2, 0) * 255).astype(np.uint8)
            image = Image.fromarray(img_array)

            # Laplacian stats for diagnostics
            _gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY).astype(np.float32)
            _lap = cv2.Laplacian(_gray, cv2.CV_32F)
            _h, _w = _gray.shape
            _bh, _bw = max(_h // 4, 1), max(_w // 4, 1)
            _vars = [float(_lap[i*_bh:(i+1)*_bh, j*_bw:(j+1)*_bw].var())
                     for i in range(4) for j in range(4)]
            lap_info = f"min_lap={min(_vars):.1f}, mean_lap={float(np.mean(_vars)):.1f}"
            _best_image = image
            _best_lap_info = lap_info

            structure_ok = _has_structure(image)
            if not structure_ok:
                print(f"Attempt {attempt + 1}: structure check failed ({lap_info}), retrying...")
                del xt, ims
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                gc.collect()
                continue

            print(f"Attempt {attempt + 1}: structure check passed ({lap_info})")
            original_size = image.size[0]

            # Post-processing pipeline
            image = _fix_chroma(image)
            if class_name == 'canyon':
                image = _canyon_color_correct(image)
            image = _deband(image)
            image = _deblock(image, patch_px=8)

            # Upscaling
            if upscale_factor > 1:
                progress(0.9, desc="Upscaling image...")
                target_size = int(original_size * upscale_factor)
                image = image.resize((target_size, target_size), Image.Resampling.LANCZOS)
                size_str = f"{original_size}x{original_size} -> {target_size}x{target_size}"
            else:
                size_str = f"{original_size}x{original_size}"

            image = _fix_local_smudges(image)
            image = _apply_bilateral_denoise(image, denoise_strength)
            image = _enhance(image)
            image = _sharpen(image, sharpen_strength)

            info = (f"Generated {class_name} | VAE: {vae_mode} | Seed: {seed} | "
                    f"Steps: {total_steps} | CFG: {guidance_scale} | Size: {size_str}")
            progress(1.0, desc="Complete!")

            del xt
            if torch.cuda.is_available():
                torch.cuda.synchronize()
                torch.cuda.empty_cache()
            gc.collect()
            return image, info

        except Exception as e:
            import traceback
            error_msg = f"Error: {str(e)}\n{traceback.format_exc()}"
            print(error_msg)
            if attempt == 9:
                return None, error_msg
            print(f"Retrying after error (attempt {attempt + 2}/10)...")

    # All attempts exhausted — return best fallback
    if _best_image is not None:
        print(f"All 10 attempts failed structure check — returning best fallback ({_best_lap_info})")
        fb = _fix_chroma(_best_image)
        if upscale_factor > 1:
            t = int(fb.size[0] * upscale_factor)
            fb = fb.resize((t, t), Image.Resampling.LANCZOS)
        return fb, f"Warning: {class_name} | All 10 attempts failed structure check. Showing best fallback."
    return None, "Failed to generate any image after 10 attempts."


# ---------------------------------------------------------------------------
# Interface
# ---------------------------------------------------------------------------

def create_interface():
    vae_label = vae_mode if vae_mode else "not loaded"
    gpu_label = torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'

    with gr.Blocks(title="DiT Mountain Generator", theme=gr.themes.Soft()) as demo:
        gr.Markdown("""
        # DiT Mountain Image Generator
        Generate realistic mountain landscape images using a Diffusion Transformer (DiT) with VAE.

        **Features:**
        - Class-conditional generation with Classifier-Free Guidance (CFG scale 3.0)
        - Structure checking with automatic retry (up to 10 attempts)
        - Full post-processing: chroma fix, deband, deblock, CLAHE, denoise, sharpen
        - AI upscaling via Real-ESRGAN (or Lanczos fallback)
        """)

        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("### Generation Settings")

                class_dropdown = gr.Dropdown(
                    choices=CLASS_NAMES,
                    value="mountain",
                    label="Mountain Category",
                    info="Select the type of mountain landscape to generate"
                )

                upscale_slider = gr.Slider(
                    minimum=1, maximum=4, value=2, step=0.5,
                    label="Upscale Factor",
                    info="1 = 128x128, 2 = 256x256, 3 = 384x384, 4 = 512x512"
                )

                generate_btn = gr.Button("Generate Image", variant="primary", size="lg")

                gr.Markdown(f"""
                ### Technical Info
                - **VAE mode:** `{vae_label}`
                - **Base resolution:** 128x128
                - **Upscaling:** Lanczos
                - **Device:** {gpu_label}
                - **CFG scale:** 3.0

                **VAE modes** (set `vae_mode` in config):
                - `scratch` — custom VAE trained from scratch
                - `finetune` — pre-trained SD VAE, fine-tuned on dataset
                - `pretrained` — frozen pre-trained SD VAE (no training needed)
                """)

            with gr.Column(scale=2):
                gr.Markdown("### Generated Image")
                output_image = gr.Image(label="Result", type="pil")
                output_info = gr.Textbox(label="Generation Info", lines=2)

        gr.Markdown("### Quick Start Examples")
        gr.Examples(
            examples=[
                ["mountain", 2],
                ["snowy mountain", 3],
                ["volcano", 2],
                ["glacier", 2],
                ["canyon", 2],
                ["cliff", 2],
                ["rocky coast", 4],
            ],
            inputs=[class_dropdown, upscale_slider],
            label="Click any example to load settings"
        )

        generate_btn.click(
            fn=generate_image,
            inputs=[class_dropdown, upscale_slider],
            outputs=[output_image, output_info],
            concurrency_limit=1
        )

        gr.Markdown("""
        ---
        ### Model Details
        - **Architecture:** Diffusion Transformer (DiT-B) with VAE
        - **Categories:** Canyon, Cliff, Glacier, Mountain, Rocky Coast, Snowy Mountain, Volcano

        ### Post-processing Pipeline
        1. Chroma fix — removes VAE color speckles/mosaic in LAB space
        2. Deband — breaks horizontal banding from attention row-coherence
        3. Deblock — smooths VAE patch-grid seams before upscaling
        4. Upscale — Real-ESRGAN (AI) or Lanczos
        5. Smudge fix — targeted unsharp mask on flat mid-tone patches
        6. Bilateral denoise — edge-preserving noise reduction
        7. CLAHE — local contrast enhancement on L channel
        8. Sharpen — unsharp mask on final output

        ### Tips
        - Each generation uses a random seed — click Generate multiple times to explore
        - If a generation looks like pure noise, the structure check will automatically retry
        - Each generation takes ~1-2 minutes depending on your GPU
        """)

    return demo


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='DiT Mountain Generator')
    parser.add_argument('--config', type=str, default='config/landscapeshq.yaml')
    parser.add_argument('--share', action='store_true')
    parser.add_argument('--server-name', type=str, default='0.0.0.0')
    parser.add_argument('--server-port', type=int, default=7860)
    args = parser.parse_args()

    print("=" * 60)
    print("Starting DiT Mountain Generator")
    print("=" * 60)

    try:
        load_models(config_path=args.config)
    except Exception as e:
        print(f"Failed to load models: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    demo = create_interface()
    demo.queue()
    demo.launch(
        server_name=args.server_name,
        server_port=args.server_port,
        share=args.share,
        show_error=True
    )
