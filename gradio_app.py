"""
Gradio interface for DiT-PyTorch mountain image generation.
Features:
- Class-conditional generation (7 mountain categories)
- Adjustable sampling parameters
"""
import gradio as gr
import torch
import numpy as np
from PIL import Image
import yaml
import os
import json
from tqdm import tqdm
import sys

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import model components
from model.vae.pretrained_vae import HuggingFaceVAEWrapper
from model.transformer import DIT
from scheduler.linear_scheduler import LinearNoiseScheduler

# Mountain class names (alphabetically ordered)
CLASS_NAMES = ['canyon', 'cliff', 'glacier', 'mountain', 'rocky coast', 'snowy mountain', 'volcano']

# Global variables
device = None
model = None
vae = None
scheduler = None
config = None
class_to_id = None


def load_models(config_path="config/landscapeshq.yaml"):
    """Load models and configuration at startup."""
    global device, model, vae, scheduler, config, class_to_id
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if torch.backends.mps.is_available():
        device = torch.device('mps')
    print(f"Using device: {device}")
    
    # Load configuration
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    
    print("Configuration loaded successfully")
    
    # Extract config sections
    diffusion_config = config['diffusion_params']
    dataset_config = config['dataset_params']
    dit_model_config = config['dit_params']
    autoencoder_config = config['autoencoder_params']
    train_config = config['train_params']
    
    # Create scheduler
    scheduler = LinearNoiseScheduler(
        num_timesteps=diffusion_config['num_timesteps'],
        beta_start=diffusion_config['beta_start'],
        beta_end=diffusion_config['beta_end']
    )
    print("Scheduler created")
    
    # Calculate latent size
    im_size = dataset_config['im_size'] // 2 ** sum(autoencoder_config['down_sample'])
    
    # CRITICAL: Set training=False for inference to disable class dropout!
    dit_model_config_inference = dit_model_config.copy()
    dit_model_config_inference['training'] = False
    
    # Create and load DIT model
    model = DIT(
        im_size=im_size,
        im_channels=autoencoder_config['z_channels'],
        config=dit_model_config_inference
    ).to(device)
    
    dit_checkpoint_path = os.path.join(train_config['task_name'], train_config['dit_ckpt_name'])
    if not os.path.exists(dit_checkpoint_path):
        raise FileNotFoundError(f"DiT checkpoint not found at {dit_checkpoint_path}")
    
    model.eval()
    model.load_state_dict(torch.load(dit_checkpoint_path, map_location=device))
    print(f"DiT model loaded from {dit_checkpoint_path}")
    
    # Create and load VAE (MUST use fine-tuned weights)
    print("\nLoading Fine-tuned Stable Diffusion VAE...")
    vae = HuggingFaceVAEWrapper(device=device)
    
    vae_checkpoint_path = os.path.join(train_config['task_name'], train_config['vae_autoencoder_ckpt_name'])
    if not os.path.exists(vae_checkpoint_path):
        raise FileNotFoundError(
            f"Fine-tuned VAE checkpoint not found at {vae_checkpoint_path}\n"
            f"Please run: python -m tools.train_vae --config {args.config}\n"
            f"Then: python -m tools.infer_vae --config {args.config}"
        )
    
    print(f"Loading fine-tuned VAE from {vae_checkpoint_path}")
    vae.vae.load_state_dict(torch.load(vae_checkpoint_path, map_location=device))
    vae.vae.eval()
    
    # Load class mapping
    if 'label_json_path' in dataset_config and dataset_config['label_json_path']:
        label_json_path = dataset_config['label_json_path']
        if os.path.exists(label_json_path):
            with open(label_json_path, 'r') as f:
                label_data = json.load(f)
            unique_classes = sorted(set(label_data.values()))
            class_to_id = {class_name: idx for idx, class_name in enumerate(unique_classes)}
            print(f"Class mapping loaded: {class_to_id}")
    
    print("=" * 60)
    print("All models loaded successfully!")
    print("=" * 60)


@torch.no_grad()
def generate_image(class_name, upscale_factor, progress=gr.Progress()):
    """Generate an image based on user parameters."""
    if model is None or vae is None:
        return None, "❌ Models not loaded!"
    
    try:
        # Get class index
        if class_to_id is not None and class_name in class_to_id:
            class_idx = class_to_id[class_name]
        else:
            class_idx = CLASS_NAMES.index(class_name)
        
        # Always use random seed for variety
        seed = np.random.randint(0, 10000000)
        torch.manual_seed(seed)
        np.random.seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        s
        progress(0.1, desc="Initializing generation...")
        
        # Get dimensions from config
        autoencoder_config = config['autoencoder_params']
        dataset_config = config['dataset_params']
        diffusion_config = config['diffusion_params']
        im_size = dataset_config['im_size'] // 2 ** sum(autoencoder_config['down_sample'])
        
        # Create initial noise
        xt = torch.randn(
            1,
            autoencoder_config['z_channels'],
            im_size,
            im_size,
            device=device
        )
        
        # Prepare class conditioning
        class_labels = torch.tensor([class_idx], device=device)
        
        progress(0.2, desc="Starting diffusion sampling...")
        
        # Sampling loop - MUST go through ALL timesteps like sample_vae_dit.py
        total_steps = diffusion_config['num_timesteps']
        
        for i in tqdm(reversed(range(diffusion_config['num_timesteps'])), 
                     desc="Sampling"):
            # Get noise prediction with class conditioning
            noise_pred = model(xt, torch.as_tensor(i).unsqueeze(0).to(device), class_labels)
            
            # Sample previous timestep
            xt, x0_pred = scheduler.sample_prev_timestep(
                xt, 
                noise_pred, 
                torch.as_tensor(i).to(device)
            )
            
            # Update progress
            step_progress = 0.2 + 0.6 * (diffusion_config['num_timesteps'] - i) / total_steps
            progress(step_progress, desc=f"Sampling step {diffusion_config['num_timesteps'] - i}/{total_steps}")
        
        progress(0.8, desc="Decoding latent...")
        
        # Decode the latent using pre-trained VAE
        # xt already contains scaled latents from diffusion process
        ims = vae.decode(xt)
        ims = torch.clamp(ims, -1., 1.).detach().cpu()
        ims = (ims + 1) / 2
        
        # Convert to PIL Image
        img_array = ims[0].numpy().transpose(1, 2, 0)
        img_array = (img_array * 255).astype(np.uint8)
        image = Image.fromarray(img_array)
        
        original_size = image.size[0]
        
        # Fast upscaling using PIL's Lanczos resampling (high quality, no extra models)
        if upscale_factor > 1:
            progress(0.9, desc="Upscaling image...")
            new_size = int(original_size * upscale_factor)
            image = image.resize((new_size, new_size), Image.Resampling.LANCZOS)
            info = f"✨ Generated {class_name} | Seed: {seed} | Steps: {total_steps} | Size: {original_size}x{original_size} → {new_size}x{new_size}"
        else:
            info = f"✨ Generated {class_name} | Seed: {seed} | Steps: {total_steps} | Size: {original_size}x{original_size}"
        
        progress(1.0, desc="Complete!")
        return image, info
        
    except Exception as e:
        import traceback
        error_msg = f"Error: {str(e)}\n{traceback.format_exc()}"
        print(error_msg)
        return None, error_msg


def create_interface():
    """Create the Gradio interface."""
    with gr.Blocks(title="DiT Mountain Generator", theme=gr.themes.Soft()) as demo:
        gr.Markdown("""
        # 🏔️ DiT Mountain Image Generator
        Generate realistic mountain landscape images using a Diffusion Transformer (DiT) with VAE.
        
        **Features:**
        - Class-conditional generation (7 mountain categories)
        - Random generation for variety (new result each time)
        - Fast upscaling up to 4x (512×512) using Lanczos resampling
        """)
        
        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("### 🎨 Generation Settings")
                
                class_dropdown = gr.Dropdown(
                    choices=CLASS_NAMES,
                    value="mountain",
                    label="Mountain Category",
                    info="Select the type of mountain landscape to generate"
                )
                
                upscale_slider = gr.Slider(
                    minimum=1,
                    maximum=4,
                    value=2,
                    step=0.5,
                    label="Upscale Factor",
                    info="1 = no upscaling (128x128), 2 = 256x256, 3 = 384x384, 4 = 512x512"
                )
                
                generate_btn = gr.Button("🎨 Generate Image", variant="primary", size="lg")
                
                gr.Markdown(f"""
                ### 📊 Technical Info
                - **Base Resolution:** 128×128
                - **Device:** {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}
                """)
                
            with gr.Column(scale=2):
                gr.Markdown("### 🖼️ Generated Image")
                output_image = gr.Image(label="Result", type="pil")
                output_info = gr.Textbox(label="Generation Info", lines=2)
        
        # Examples
        gr.Markdown("### 💡 Quick Start Examples")
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
        
        # Connect generation button
        generate_btn.click(
            fn=generate_image,
            inputs=[class_dropdown, upscale_slider],
            outputs=[output_image, output_info]
        )
        
        gr.Markdown("""
        ---
        ### 📚 Model Details
        - **Architecture:** Diffusion Transformer (DiT) with VAE
        - **Categories:** Canyon, Cliff, Glacier, Mountain, Rocky Coast, Snowy Mountain, Volcano
        
        ### 🔧 Tips for Best Results
        - Each generation is unique (random seed)
        - Click "Generate" multiple times to explore variations
        - Upscale factor 2-3x provides good balance between quality and size
        - Each generation takes ~1-2 minutes depending on your GPU
        """)
    
    return demo


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='DiT-PyTorch Gradio Interface')
    parser.add_argument('--config', type=str, default='config/landscapeshq.yaml',
                       help='Path to config file')
    parser.add_argument('--share', action='store_true',
                       help='Create a public share link')
    parser.add_argument('--server-name', type=str, default='0.0.0.0',
                       help='Server name for hosting')
    parser.add_argument('--server-port', type=int, default=7860,
                       help='Server port')
    args = parser.parse_args()
    
    print("=" * 60)
    print("Starting DiT-PyTorch Gradio Interface")
    print("=" * 60)
    
    # Load models
    try:
        load_models(config_path=args.config)
    except Exception as e:
        print(f"Failed to load models: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    # Create and launch interface
    demo = create_interface()
    demo.queue()  # Enable queuing for better performance
    demo.launch(
        server_name=args.server_name,
        server_port=args.server_port,
        share=args.share,
        show_error=True
    )
