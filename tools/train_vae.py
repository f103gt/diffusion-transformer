"""
VAE training entry point.

Dispatches to one of three modes set by `vae_mode` in the config:

  scratch    Train model/vae/vae.py from random init with MSE + KL + LPIPS + GAN losses.
  finetune   Fine-tune the pre-trained Stable Diffusion VAE (diffusers AutoencoderKL).
  pretrained Use the frozen SD VAE as-is; saves reconstruction samples and exits.
"""
import yaml
import argparse
import torch
import random
import torchvision
import os
import numpy as np
from tqdm import tqdm
from torch.utils.data.dataloader import DataLoader
from torch.optim import Adam
from torchvision.utils import make_grid

from model.vae.lpips import LPIPS
from dataset.landscapes_dataset import LandscapesDataset

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if torch.backends.mps.is_available():
    device = torch.device('mps')
    print('Using mps')


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def train(args):
    config = _setup_environment(args.config_path)
    if not config:
        return

    mode = config['train_params'].get('vae_mode', 'scratch')
    print(f"\n{'='*60}")
    print(f"VAE mode: {mode}")
    print(f"{'='*60}\n")

    if mode == 'scratch':
        _train_scratch(config)
    elif mode == 'finetune':
        _train_finetune(config)
    elif mode == 'pretrained':
        _run_pretrained(config)
    else:
        raise ValueError(f"Unknown vae_mode '{mode}'. Choose: scratch | finetune | pretrained")


# ---------------------------------------------------------------------------
# scratch — custom VAE + GAN discriminator
# ---------------------------------------------------------------------------

def _train_scratch(config):
    from model.vae.vae import VAE
    from model.vae.discriminator import Discriminator

    dataset_config = config['dataset_params']
    train_config = config['train_params']
    ae_config = config['autoencoder_params']

    model = VAE(im_channels=dataset_config['im_channels'], model_config=ae_config).to(device)
    discriminator = Discriminator(im_channels=dataset_config['im_channels']).to(device)
    lpips_model = LPIPS().eval().to(device)

    data_loader = _make_dataloader(dataset_config, train_config)

    optimizer_g = Adam(model.parameters(), lr=train_config['autoencoder_lr'], betas=(0.5, 0.999))
    optimizer_d = Adam(discriminator.parameters(), lr=train_config['autoencoder_lr'], betas=(0.5, 0.999))
    recon_criterion = torch.nn.MSELoss()

    start_epoch, step_count = _load_scratch_checkpoint(model, discriminator, optimizer_g, optimizer_d, train_config)

    acc_steps = train_config['autoencoder_acc_steps']
    img_save_count = step_count // train_config['autoencoder_img_save_steps']
    total_epochs = train_config.get('autoencoder_epochs', 50)

    for epoch_idx in range(start_epoch, total_epochs):
        metrics = {'recon': [], 'lpips': [], 'gen': [], 'disc': []}
        optimizer_g.zero_grad()
        optimizer_d.zero_grad()

        for batch in tqdm(data_loader, desc=f"Epoch {epoch_idx + 1}/{total_epochs}"):
            step_count += 1
            im = batch[0].float().to(device) if isinstance(batch, (tuple, list)) else batch.float().to(device)

            output, encoder_output = model(im)

            if step_count % train_config['autoencoder_img_save_steps'] == 0 or step_count == 1:
                _save_samples(im, output, train_config, img_save_count)
                img_save_count += 1

            g_loss, r_loss, p_loss, adv_g_loss = _generator_loss(
                im, output, encoder_output, discriminator, lpips_model,
                train_config, recon_criterion, step_count, acc_steps
            )
            g_loss.backward()
            metrics['recon'].append(r_loss)
            metrics['lpips'].append(p_loss)
            if adv_g_loss > 0:
                metrics['gen'].append(adv_g_loss)

            if step_count > train_config['disc_start']:
                d_loss_scaled, d_loss_item = _discriminator_loss(
                    im, output, discriminator, train_config, acc_steps
                )
                d_loss_scaled.backward()
                metrics['disc'].append(d_loss_item)
                if step_count % acc_steps == 0:
                    optimizer_d.step()
                    optimizer_d.zero_grad()

            if step_count % acc_steps == 0:
                optimizer_g.step()
                optimizer_g.zero_grad()

        optimizer_d.step()
        optimizer_d.zero_grad()
        optimizer_g.step()
        optimizer_g.zero_grad()

        log = (f"Epoch {epoch_idx + 1} | Recon: {np.mean(metrics['recon']):.4f} | "
               f"LPIPS: {np.mean(metrics['lpips']):.4f}")
        if metrics['disc']:
            log += f" | G: {np.mean(metrics['gen']):.4f} | D: {np.mean(metrics['disc']):.4f}"
        print(log)

        task_dir = train_config['task_name']
        torch.save(model.state_dict(),
                   os.path.join(task_dir, train_config['vae_autoencoder_ckpt_name']))
        torch.save(discriminator.state_dict(),
                   os.path.join(task_dir, train_config['vae_discriminator_ckpt_name']))
        torch.save({
            'epoch': epoch_idx,
            'step_count': step_count,
            'optimizer_g': optimizer_g.state_dict(),
            'optimizer_d': optimizer_d.state_dict(),
        }, os.path.join(task_dir, 'vae_training_state.pth'))

    print('Scratch VAE training complete.')


# ---------------------------------------------------------------------------
# finetune — start from SD VAE, fine-tune on dataset
# ---------------------------------------------------------------------------

def _train_finetune(config):
    from model.vae.pretrained_vae import HuggingFaceVAEWrapper

    dataset_config = config['dataset_params']
    train_config = config['train_params']
    hf_model = train_config.get('pretrained_vae_model', 'stabilityai/sd-vae-ft-mse')

    model = HuggingFaceVAEWrapper(pretrained_model_name_or_path=hf_model, device=device)

    vae_ckpt = os.path.join(train_config['task_name'], train_config['vae_autoencoder_ckpt_name'])
    if os.path.exists(vae_ckpt):
        model.vae.load_state_dict(torch.load(vae_ckpt, map_location=device))
        print(f'Loaded fine-tuned weights from {vae_ckpt}')

    model.enable_finetuning()

    data_loader = _make_dataloader(dataset_config, train_config)
    lpips_model = LPIPS().eval().to(device)
    optimizer = Adam(model.parameters(), lr=train_config['autoencoder_lr'], betas=(0.5, 0.999))
    recon_criterion = torch.nn.MSELoss()

    start_epoch, step_count = _load_finetune_checkpoint(optimizer, train_config)
    acc_steps = train_config.get('autoencoder_acc_steps', 1)
    img_save_count = step_count // train_config['autoencoder_img_save_steps']
    total_epochs = train_config.get('finetune_epochs', 1)

    print(f"Fine-tuning for {total_epochs} epoch(s)")

    for epoch_idx in range(start_epoch, total_epochs):
        recon_losses, lpips_losses = [], []
        optimizer.zero_grad()

        for batch in tqdm(data_loader, desc=f"Epoch {epoch_idx + 1}/{total_epochs}"):
            step_count += 1
            im = batch[0].float().to(device) if isinstance(batch, (tuple, list)) else batch.float().to(device)

            output, encoder_output = model(im)

            if step_count % train_config['autoencoder_img_save_steps'] == 0 or step_count == 1:
                _save_samples(im, output, train_config, img_save_count)
                img_save_count += 1

            loss, r_loss, p_loss = _finetune_loss(
                im, output, encoder_output, lpips_model, train_config, recon_criterion, acc_steps
            )
            loss.backward()
            recon_losses.append(r_loss)
            lpips_losses.append(p_loss)

            if step_count % acc_steps == 0:
                optimizer.step()
                optimizer.zero_grad()

        optimizer.step()
        optimizer.zero_grad()

        print(f"Epoch {epoch_idx + 1} | Recon: {np.mean(recon_losses):.4f} | LPIPS: {np.mean(lpips_losses):.4f}")

        task_dir = train_config['task_name']
        torch.save(model.vae.state_dict(),
                   os.path.join(task_dir, train_config['vae_autoencoder_ckpt_name']))
        torch.save({
            'epoch': epoch_idx,
            'step_count': step_count,
            'optimizer': optimizer.state_dict(),
        }, os.path.join(task_dir, 'vae_training_state.pth'))

    print('Fine-tuning complete.')


# ---------------------------------------------------------------------------
# pretrained — frozen SD VAE, just verify reconstruction quality
# ---------------------------------------------------------------------------

def _run_pretrained(config):
    from model.vae.pretrained_vae import HuggingFaceVAEWrapper

    dataset_config = config['dataset_params']
    train_config = config['train_params']
    hf_model = train_config.get('pretrained_vae_model', 'stabilityai/sd-vae-ft-mse')

    print(f"Loading frozen pre-trained VAE '{hf_model}'...")
    model = HuggingFaceVAEWrapper(pretrained_model_name_or_path=hf_model, device=device)
    model.vae.eval()

    data_loader = _make_dataloader(dataset_config, train_config)
    img_save_count = 0

    print("Saving reconstruction samples (no training)...")
    with torch.no_grad():
        for batch in data_loader:
            im = batch[0].float().to(device) if isinstance(batch, (tuple, list)) else batch.float().to(device)
            output, _ = model(im)
            _save_samples(im, output, train_config, img_save_count)
            img_save_count += 1
            if img_save_count >= 4:
                break

    print(f"Saved {img_save_count} reconstruction grids to "
          f"{os.path.join(train_config['task_name'], 'vae_autoencoder_samples')}/")
    print("No training performed. Run tools.infer_vae to inspect latent quality.")


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _make_dataloader(dataset_config, train_config):
    dataset = LandscapesDataset(
        split='train',
        im_path=dataset_config['im_path'],
        im_size=dataset_config['im_size'],
        im_channels=dataset_config['im_channels'],
        label_json_path=dataset_config.get('label_json_path', None)
    )
    return DataLoader(dataset, batch_size=train_config['autoencoder_batch_size'], shuffle=True)


def _save_samples(im, output, train_config, img_save_count):
    n = min(8, im.shape[0])
    save_out = torch.clamp(output[:n], -1., 1.).detach().cpu()
    save_out = (save_out + 1) / 2
    save_in = ((im[:n] + 1) / 2).detach().cpu()

    grid = make_grid(torch.cat([save_in, save_out], dim=0), nrow=n)
    img = torchvision.transforms.ToPILImage()(grid)

    sample_dir = os.path.join(train_config['task_name'], 'vae_autoencoder_samples')
    os.makedirs(sample_dir, exist_ok=True)
    img.save(os.path.join(sample_dir, f'sample_{img_save_count}.png'))
    img.close()


def _generator_loss(im, output, encoder_output, discriminator, lpips_model,
                    train_config, recon_criterion, step_count, acc_steps):
    recon_loss = recon_criterion(output, im)
    mean, logvar = torch.chunk(encoder_output, 2, dim=1)
    kl_loss = torch.mean(0.5 * torch.sum(torch.exp(logvar) + mean ** 2 - 1 - logvar, dim=[1, 2, 3]))

    g_loss = recon_loss / acc_steps + train_config['kl_weight'] * kl_loss / acc_steps

    adv_val = 0.0
    if step_count > train_config['disc_start']:
        fake_pred = discriminator(output)
        adv = torch.nn.MSELoss()(fake_pred, torch.ones_like(fake_pred))
        adv_val = train_config['disc_weight'] * adv.item()
        g_loss = g_loss + train_config['disc_weight'] * adv / acc_steps

    lpips_loss = torch.mean(lpips_model(output, im))
    g_loss = g_loss + train_config['perceptual_weight'] * lpips_loss / acc_steps

    return g_loss, recon_loss.item(), (train_config['perceptual_weight'] * lpips_loss).item(), adv_val


def _discriminator_loss(im, output, discriminator, train_config, acc_steps):
    fake_pred = discriminator(output.detach())
    real_pred = discriminator(im)
    criterion = torch.nn.MSELoss()
    d_loss = train_config['disc_weight'] * (
        criterion(fake_pred, torch.zeros_like(fake_pred)) +
        criterion(real_pred, torch.ones_like(real_pred))
    ) / 2
    return d_loss / acc_steps, d_loss.item()


def _finetune_loss(im, output, encoder_output, lpips_model, train_config, recon_criterion, acc_steps):
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


def _load_scratch_checkpoint(model, discriminator, optimizer_g, optimizer_d, train_config):
    start_epoch, step_count = 0, 0
    task_dir = train_config['task_name']

    vae_path = os.path.join(task_dir, train_config['vae_autoencoder_ckpt_name'])
    if os.path.exists(vae_path):
        model.load_state_dict(torch.load(vae_path, map_location=device))
        print(f'Loaded VAE from {vae_path}')

    disc_path = os.path.join(task_dir, train_config['vae_discriminator_ckpt_name'])
    if os.path.exists(disc_path):
        discriminator.load_state_dict(torch.load(disc_path, map_location=device))
        print(f'Loaded discriminator from {disc_path}')

    state_path = os.path.join(task_dir, 'vae_training_state.pth')
    if os.path.exists(state_path):
        state = torch.load(state_path, map_location=device)
        start_epoch = state['epoch'] + 1
        step_count = state['step_count']
        optimizer_g.load_state_dict(state['optimizer_g'])
        optimizer_d.load_state_dict(state['optimizer_d'])
        print(f'Resuming from epoch {start_epoch}, step {step_count}')

    return start_epoch, step_count


def _load_finetune_checkpoint(optimizer, train_config):
    start_epoch, step_count = 0, 0
    state_path = os.path.join(train_config['task_name'], 'vae_training_state.pth')
    if os.path.exists(state_path):
        state = torch.load(state_path, map_location=device)
        start_epoch = state['epoch'] + 1
        step_count = state['step_count']
        if 'optimizer' in state:
            optimizer.load_state_dict(state['optimizer'])
        print(f'Resuming from epoch {start_epoch}, step {step_count}')
    return start_epoch, step_count


def _setup_environment(config_path):
    with open(config_path, 'r') as f:
        try:
            config = yaml.safe_load(f)
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

    os.makedirs(train_config['task_name'], exist_ok=True)
    return config


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='VAE training (scratch | finetune | pretrained)')
    parser.add_argument('--config', dest='config_path', default='config/landscapeshq.yaml', type=str)
    args = parser.parse_args()
    train(args)
