import yaml
import argparse
import torch
import random
import torchvision
import os
import numpy as np
from tqdm import tqdm
from model.vae.vae import VAE
from model.vae.lpips import LPIPS
from model.vae.discriminator import Discriminator
from torch.utils.data.dataloader import DataLoader
from dataset.landscapes_dataset import LandscapesDataset
from torch.optim import Adam
from torchvision.utils import make_grid

# Global Device Configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if torch.backends.mps.is_available():
    device = torch.device('mps')
    print('Using mps')


# ==========================================
# Main Training Loop
# ==========================================

def train(args):
    """
    Main training pipeline for the Variational Autoencoder (VAE) with a GAN component.

    Unlike standard VAEs that only use Mean Squared Error (MSE) and KL Divergence, 
    this script implements a "Perceptual VAE" (often used in Latent Diffusion Models). 
    It trains two networks simultaneously:
    1. The VAE (Generator): Learns to compress images into a low-dimensional latent 
       space and reconstruct them, while trying to fool the discriminator.
    2. The Discriminator: A separate network trained to distinguish between real 
       dataset images and the VAE's reconstructed images.

    Training is phased: the VAE trains alone for a "warmup" period to establish 
    basic reconstructions before the Discriminator is activated to refine high-frequency 
    details and textures.

    :param args: Parsed command-line arguments containing the `config_path`.
    """
    # 1. Setup Environment
    config = setup_environment(args.config_path)
    if not config: return
    
    dataset_config = config['dataset_params']
    train_config = config['train_params']

    # 2. Initialize Models & Dataset
    model = VAE(im_channels=dataset_config['im_channels'], model_config=config['autoencoder_params']).to(device)
    lpips_model = LPIPS().eval().to(device)
    discriminator = Discriminator(im_channels=dataset_config['im_channels']).to(device)

    im_dataset = LandscapesDataset(
        split='train',
        im_path=dataset_config['im_path'],
        im_size=dataset_config['im_size'],
        im_channels=dataset_config['im_channels'],
        label_json_path=dataset_config.get('label_json_path', None)
    )
    data_loader = DataLoader(im_dataset, batch_size=train_config['autoencoder_batch_size'], shuffle=True)

    # 3. Setup Optimizers & Loss Functions
    optimizer_g = Adam(model.parameters(), lr=train_config['autoencoder_lr'], betas=(0.5, 0.999))
    optimizer_d = Adam(discriminator.parameters(), lr=train_config['autoencoder_lr'], betas=(0.5, 0.999))
    
    recon_criterion = torch.nn.MSELoss()
    disc_criterion = torch.nn.MSELoss()

    # 4. Load Resumption States
    start_epoch, step_count = load_checkpoints_and_state(model, discriminator, optimizer_g, optimizer_d, train_config)

    # 5. Loop Tracking Variables
    acc_steps = train_config['autoencoder_acc_steps']
    img_save_count = step_count // train_config['autoencoder_img_save_steps']

    # 6. Execute Epochs
    for epoch_idx in range(start_epoch, train_config['autoencoder_epochs']):
        metrics = {'recon': [], 'lpips': [], 'gen': [], 'disc': []}

        optimizer_g.zero_grad()
        optimizer_d.zero_grad()

        for batch in tqdm(data_loader, desc=f"Epoch {epoch_idx + 1}"):
            step_count += 1
            im = batch[0].float().to(device) if isinstance(batch, (tuple, list)) else batch.float().to(device)

            # --- Forward Pass ---
            output, encoder_output = model(im)

            # --- Save Samples ---
            if step_count % train_config['autoencoder_img_save_steps'] == 0 or step_count == 1:
                save_reconstruction_samples(im, output, train_config, img_save_count)
                img_save_count += 1

            # --- Optimize Generator (VAE) ---
            g_loss, r_loss, p_loss, adv_g_loss = compute_generator_loss(
                im, output, encoder_output, discriminator, lpips_model, 
                train_config, recon_criterion, step_count, acc_steps
            )
            g_loss.backward()
            
            metrics['recon'].append(r_loss)
            metrics['lpips'].append(p_loss)
            if adv_g_loss > 0: metrics['gen'].append(adv_g_loss)

            # --- Optimize Discriminator (GAN) ---
            if step_count > train_config['disc_start']:
                d_loss_scaled, d_loss_item = compute_discriminator_loss(
                    im, output, discriminator, disc_criterion, train_config, acc_steps
                )
                d_loss_scaled.backward()
                metrics['disc'].append(d_loss_item)
                
                if step_count % acc_steps == 0:
                    optimizer_d.step()
                    optimizer_d.zero_grad()

            # --- Step Generator Optimizer ---
            if step_count % acc_steps == 0:
                optimizer_g.step()
                optimizer_g.zero_grad()

        # Handle edge case where epoch ends on a non-accumulated step
        optimizer_d.step()
        optimizer_d.zero_grad()
        optimizer_g.step()
        optimizer_g.zero_grad()

        # --- Logging & Checkpointing ---
        log_str = f"Finished epoch: {epoch_idx + 1} | Recon Loss : {np.mean(metrics['recon']):.4f} | Perceptual Loss : {np.mean(metrics['lpips']):.4f}"
        if len(metrics['disc']) > 0:
            log_str += f" | G Loss : {np.mean(metrics['gen']):.4f} | D Loss {np.mean(metrics['disc']):.4f}"
        print(log_str)

        task_dir = train_config['task_name']
        torch.save(model.state_dict(), os.path.join(task_dir, train_config['vae_autoencoder_ckpt_name']))
        torch.save(discriminator.state_dict(), os.path.join(task_dir, train_config['vae_discriminator_ckpt_name']))
        torch.save({
            'epoch': epoch_idx,
            'step_count': step_count,
            'optimizer_g': optimizer_g.state_dict(),
            'optimizer_d': optimizer_d.state_dict()
        }, os.path.join(task_dir, 'training_state.pth'))
        
    print('Done Training...')



# Helper Functions
def setup_environment(config_path):
    """
    Parses the configuration file, enforces reproducibility via static random seeds, 
    and provisions the output directories for saving checkpoints and samples.

    :param config_path: String. Path to the YAML configuration file.
    :return: Dictionary containing the loaded config parameters, or None if it fails.
    """
    """Loads config, sets random seeds, and creates output directories."""
    with open(config_path, 'r') as file:
        try:
            config = yaml.safe_load(file)
        except yaml.YAMLError as exc:
            print(exc)
            return None
            
    train_config = config['train_params']
    seed = train_config['seed']
    
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if device.type == 'cuda':
        torch.cuda.manual_seed_all(seed)

    if not os.path.exists(train_config['task_name']):
        os.mkdir(train_config['task_name'])
        
    return config

def load_checkpoints_and_state(model, discriminator, optimizer_g, optimizer_d, train_config):
    """
    Detects and loads existing model weights and optimizer states to seamlessly 
    resume an interrupted training run.

    :param model: The instantiated VAE network.
    :param discriminator: The instantiated Discriminator network.
    :param optimizer_g: The Adam optimizer for the VAE.
    :param optimizer_d: The Adam optimizer for the Discriminator.
    :param train_config: Dictionary containing file paths and names.
    :return: A tuple `(start_epoch, step_count)` indicating where the loop should resume.
    """
    start_epoch = 0
    step_count = 0
    task_dir = train_config['task_name']

    # Load Model Weights
    vae_path = os.path.join(task_dir, train_config['vae_autoencoder_ckpt_name'])
    if os.path.exists(vae_path):
        model.load_state_dict(torch.load(vae_path, map_location=device))
        print('Loaded autoencoder from checkpoint')

    disc_path = os.path.join(task_dir, train_config['vae_discriminator_ckpt_name'])
    if os.path.exists(disc_path):
        discriminator.load_state_dict(torch.load(disc_path, map_location=device))
        print('Loaded discriminator from checkpoint')

    # Load Training State
    state_path = os.path.join(task_dir, 'training_state.pth')
    if os.path.exists(state_path):
        training_state = torch.load(state_path, map_location=device)
        start_epoch = training_state['epoch'] + 1
        step_count = training_state['step_count']
        optimizer_g.load_state_dict(training_state['optimizer_g'])
        optimizer_d.load_state_dict(training_state['optimizer_d'])
        print(f'Resuming training from epoch {start_epoch}, step {step_count}')

    return start_epoch, step_count

def save_reconstruction_samples(im, output, train_config, img_save_count):
    """
    Visualizes the current performance of the VAE by saving a side-by-side grid 
    of the original real images and their VAE reconstructions.

    It handles denormalizing the tensor values from the [-1, 1] neural network range 
    back to the [0, 1] range required for saving valid image files.

    :param im: Float tensor of original images.
    :param output: Float tensor of VAE reconstructed images.
    :param train_config: Dictionary containing output directory info.
    :param img_save_count: Integer tracking how many image grids have been saved so far.
    """
    sample_size = min(8, im.shape[0])
    save_output = torch.clamp(output[:sample_size], -1., 1.).detach().cpu()
    save_output = ((save_output + 1) / 2)
    save_input = ((im[:sample_size] + 1) / 2).detach().cpu()

    grid = make_grid(torch.cat([save_input, save_output], dim=0), nrow=sample_size)
    img = torchvision.transforms.ToPILImage()(grid)
    
    sample_dir = os.path.join(train_config['task_name'], 'vae_autoencoder_samples')
    if not os.path.exists(sample_dir):
        os.mkdir(sample_dir)
        
    img_path = os.path.join(sample_dir, f'current_autoencoder_sample_{img_save_count}.png')
    img.save(img_path)
    img.close()

def compute_generator_loss(im, output, encoder_output, discriminator, lpips_model, 
                           train_config, recon_criterion, step_count, acc_steps):
    r"""
    Aggregates the four distinct loss objectives for training the VAE (Generator):

    1. **Reconstruction Loss**: Standard Pixel-wise Mean Squared Error (MSE) comparing 
       the output to the input image. Ensures spatial accuracy.
    2. **KL Divergence**: Regularizes the latent space. Forces the distribution of the 
       encoded latents to approximate a standard normal distribution, which is required 
       for generative sampling later.
    3. **Perceptual (LPIPS) Loss**: Uses a pre-trained feature extractor (like VGG) to 
       compare the structural and semantic similarity between images, rather than just 
       pixel intensity. Stops reconstructions from looking blurry.
    4. **Adversarial Loss**: Forces the VAE to generate reconstructions realistic enough 
       to trick the discriminator into classifying them as "real".

    :param im: Original input tensor.
    :param output: Reconstructed output tensor from the VAE.
    :param encoder_output: The pre-sampled parameters [mean, logvar] from the VAE bottleneck.
    :param discriminator: The Discriminator network.
    :param lpips_model: The LPIPS perceptual metric network.
    :param train_config: Dictionary of loss weights and scheduling steps.
    :param recon_criterion: MSE loss function.
    :param step_count: Current global training step.
    :param acc_steps: Number of gradient accumulation steps (used to scale the loss).
    :return: Tuple containing (Total Generator Loss, Raw Recon Loss, Raw LPIPS Loss, Raw Adversarial Loss).
    """
    # 1. Reconstruction Loss
    recon_loss = recon_criterion(output, im)
    
    # 2. KL Divergence
    mean, logvar = torch.chunk(encoder_output, 2, dim=1)
    kl_loss = torch.mean(0.5 * torch.sum(torch.exp(logvar) + mean ** 2 - 1 - logvar, dim=[1, 2, 3]))
    
    g_loss = (recon_loss / acc_steps) + (train_config['kl_weight'] * kl_loss / acc_steps)

    disc_fake_loss_val = 0.0
    # 3. Adversarial Loss (Only after warmup period)
    if step_count > train_config['disc_start']:
        disc_fake_pred = discriminator(output)
        disc_fake_loss = torch.nn.MSELoss()(disc_fake_pred, torch.ones_like(disc_fake_pred))
        disc_fake_loss_val = train_config['disc_weight'] * disc_fake_loss.item()
        g_loss += train_config['disc_weight'] * disc_fake_loss / acc_steps
        
    # 4. Perceptual (LPIPS) Loss
    lpips_loss = torch.mean(lpips_model(output, im))
    lpips_loss_val = train_config['perceptual_weight'] * lpips_loss.item()
    g_loss += train_config['perceptual_weight'] * lpips_loss / acc_steps
    
    return g_loss, recon_loss.item(), lpips_loss_val, disc_fake_loss_val

def compute_discriminator_loss(im, output, discriminator, disc_criterion, train_config, acc_steps):
    """
    Calculates the standard GAN objective for the Discriminator network.

    The Discriminator receives a batch of real images and a batch of reconstructed 
    (fake) images from the VAE. It is penalized via MSE if it fails to output 1 for 
    the real images, or 0 for the reconstructed images.

    :param im: Tensor of real ground-truth images.
    :param output: Tensor of reconstructed images from the VAE.
    :param discriminator: The Discriminator network.
    :param disc_criterion: MSE loss function.
    :param train_config: Dictionary containing loss weights.
    :param acc_steps: Number of gradient accumulation steps.
    :return: Tuple containing the scaled tensor loss (for backward pass) and the raw scalar item.
    """
    disc_fake_pred = discriminator(output.detach())
    disc_real_pred = discriminator(im)
    
    disc_fake_loss = disc_criterion(disc_fake_pred, torch.zeros_like(disc_fake_pred))
    disc_real_loss = disc_criterion(disc_real_pred, torch.ones_like(disc_real_pred))
    
    disc_loss = train_config['disc_weight'] * (disc_fake_loss + disc_real_loss) / 2
    disc_loss_scaled = disc_loss / acc_steps
    
    return disc_loss_scaled, disc_loss.item()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Arguments for vae training')
    parser.add_argument('--config', dest='config_path', default='config/landscapeshq.yaml', type=str)
    args = parser.parse_args()
    train(args)