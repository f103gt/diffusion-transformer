import yaml
import argparse
import torch
import random
import torchvision
import os
import numpy as np
from tqdm import tqdm
from model.vae.pretrained_vae import HuggingFaceVAEWrapper
from model.vae.lpips import LPIPS
from torch.utils.data.dataloader import DataLoader
from dataset.landscapes_dataset import LandscapesDataset
from torch.optim import Adam
from torchvision.utils import make_grid

# Global Device Configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if torch.backends.mps.is_available():
    device = torch.device('mps')
    print('Using mps')


def train(args):
    """
    Fine-tuning pipeline for the pre-trained Stable Diffusion VAE.

    When finetune_vae is False: loads the frozen VAE, saves reconstruction
    samples so you can verify quality, then exits — no training needed.

    When finetune_vae is True: fine-tunes the VAE for finetune_epochs epochs
    using reconstruction (MSE), KL divergence, and perceptual (LPIPS) losses.

    :param args: Parsed command-line arguments containing the `config_path`.
    """
    config = setup_environment(args.config_path)
    if not config:
        return

    dataset_config = config['dataset_params']
    train_config = config['train_params']
    is_finetuning = train_config.get('finetune_vae', False)

    print("\n" + "="*60)
    print("Loading Pre-trained Stable Diffusion VAE...")
    print("="*60 + "\n")

    model = HuggingFaceVAEWrapper(device=device)

    # Load fine-tuned weights if a previous run produced them
    vae_ckpt = os.path.join(train_config['task_name'], train_config['vae_autoencoder_ckpt_name'])
    if os.path.exists(vae_ckpt):
        model.vae.load_state_dict(torch.load(vae_ckpt, map_location=device))
        print(f'Loaded fine-tuned VAE weights from {vae_ckpt}')

    if not is_finetuning:
        print("VAE fine-tuning DISABLED — using frozen pre-trained weights.")
        print("  Set 'finetune_vae: True' in config to enable fine-tuning.\n")
        print("  Use 'python -m tools.infer_vae' to inspect reconstruction quality.")
        return

    model.enable_finetuning()

    im_dataset = LandscapesDataset(
        split='train',
        im_path=dataset_config['im_path'],
        im_size=dataset_config['im_size'],
        im_channels=dataset_config['im_channels'],
        label_json_path=dataset_config.get('label_json_path', None)
    )
    data_loader = DataLoader(im_dataset, batch_size=train_config['autoencoder_batch_size'], shuffle=True)

    lpips_model = LPIPS().eval().to(device)
    optimizer = Adam(model.parameters(), lr=train_config['autoencoder_lr'], betas=(0.5, 0.999))
    recon_criterion = torch.nn.MSELoss()

    start_epoch, step_count = load_checkpoint(optimizer, train_config)
    acc_steps = train_config.get('autoencoder_acc_steps', 1)
    img_save_count = step_count // train_config['autoencoder_img_save_steps']
    total_epochs = train_config.get('finetune_epochs', 1)

    print("="*60)
    print(f"Fine-tuning for {total_epochs} epoch(s)")
    print("="*60 + "\n")

    for epoch_idx in range(start_epoch, total_epochs):
        recon_losses, lpips_losses = [], []
        optimizer.zero_grad()

        for batch in tqdm(data_loader, desc=f"Epoch {epoch_idx + 1}/{total_epochs}"):
            step_count += 1
            im = batch[0].float().to(device) if isinstance(batch, (tuple, list)) else batch.float().to(device)

            output, encoder_output = model(im)

            if step_count % train_config['autoencoder_img_save_steps'] == 0 or step_count == 1:
                save_reconstruction_samples(im, output, train_config, img_save_count)
                img_save_count += 1

            loss, r_loss, p_loss = compute_loss(
                im, output, encoder_output, lpips_model,
                train_config, recon_criterion, acc_steps
            )
            loss.backward()
            recon_losses.append(r_loss)
            lpips_losses.append(p_loss)

            if step_count % acc_steps == 0:
                optimizer.step()
                optimizer.zero_grad()

        # Flush any remaining accumulated gradients at epoch end
        optimizer.step()
        optimizer.zero_grad()

        print(f"Epoch {epoch_idx + 1} | Recon: {np.mean(recon_losses):.4f} | LPIPS: {np.mean(lpips_losses):.4f}")

        task_dir = train_config['task_name']
        torch.save(model.vae.state_dict(), os.path.join(task_dir, train_config['vae_autoencoder_ckpt_name']))
        torch.save({
            'epoch': epoch_idx,
            'step_count': step_count,
            'optimizer': optimizer.state_dict(),
        }, os.path.join(task_dir, 'training_state.pth'))

    print('\nFine-tuning Complete!')


# Helper Functions

def setup_environment(config_path):
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


def load_checkpoint(optimizer, train_config):
    start_epoch = 0
    step_count = 0
    state_path = os.path.join(train_config['task_name'], 'training_state.pth')

    if os.path.exists(state_path):
        state = torch.load(state_path, map_location=device)
        start_epoch = state['epoch'] + 1
        step_count = state['step_count']
        if 'optimizer' in state and state['optimizer'] is not None:
            optimizer.load_state_dict(state['optimizer'])
        print(f'Resuming from epoch {start_epoch}, step {step_count}')

    return start_epoch, step_count


def save_reconstruction_samples(im, output, train_config, img_save_count):
    sample_size = min(8, im.shape[0])
    save_output = torch.clamp(output[:sample_size], -1., 1.).detach().cpu()
    save_output = ((save_output + 1) / 2)
    save_input = ((im[:sample_size] + 1) / 2).detach().cpu()

    grid = make_grid(torch.cat([save_input, save_output], dim=0), nrow=sample_size)
    img = torchvision.transforms.ToPILImage()(grid)

    sample_dir = os.path.join(train_config['task_name'], 'vae_autoencoder_samples')
    if not os.path.exists(sample_dir):
        os.mkdir(sample_dir)

    img.save(os.path.join(sample_dir, f'sample_{img_save_count}.png'))
    img.close()


def compute_loss(im, output, encoder_output, lpips_model, train_config, recon_criterion, acc_steps):
    """
    Computes VAE fine-tuning loss:
      1. Reconstruction (MSE)
      2. KL divergence — regularises the latent space
      3. Perceptual (LPIPS) — preserves semantic/texture quality

    Returns total loss tensor plus scalar values for logging.
    """
    recon_loss = recon_criterion(output, im)

    mean, logvar = torch.chunk(encoder_output, 2, dim=1)
    kl_loss = torch.mean(0.5 * torch.sum(torch.exp(logvar) + mean ** 2 - 1 - logvar, dim=[1, 2, 3]))

    lpips_loss = torch.mean(lpips_model(output, im))

    total = (
        recon_loss / acc_steps
        + train_config['kl_weight'] * kl_loss / acc_steps
        + train_config['perceptual_weight'] * lpips_loss / acc_steps
    )

    return total, recon_loss.item(), (train_config['perceptual_weight'] * lpips_loss).item()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Fine-tune pre-trained Stable Diffusion VAE')
    parser.add_argument('--config', dest='config_path', default='config/landscapeshq.yaml', type=str)
    args = parser.parse_args()
    train(args)
