import argparse
import glob
import os
import pickle

import torch
import torchvision
import yaml
from torch.utils.data.dataloader import DataLoader
from torchvision.utils import make_grid
from tqdm import tqdm

from dataset.landscapes_dataset import LandscapesDataset
from model.vae.pretrained_vae import HuggingFaceVAEWrapper

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if torch.backends.mps.is_available():
    device = torch.device('mps')
    print('Using mps')


def infer(args):
    """
    Runs inference on the pre-trained Stable Diffusion VAE.

    This function serves two distinct purposes based on the configuration:
    1. Visual Evaluation: It randomly samples a batch of images from the dataset, 
       passes them through the VAE encoder and decoder, and saves image grids 
       comparing the original inputs and the final reconstructions. This helps 
       visually verify reconstruction quality.
       
    2. Latent Extraction: If 'save_latents' is enabled in the config, it iterates 
       through the entire dataset, encodes every image into its latent representation, 
       and saves these latents to disk in chunked pickle files. These saved latents 
       are then used to train the downstream Diffusion Transformer (DiT), drastically 
       speeding up training since the VAE encoder doesn't need to be run dynamically.

    The pre-trained VAE automatically handles scaling of latents for optimal
    diffusion model training.

    :param args: Parsed command-line arguments containing `config_path`.
    """
    with open(args.config_path, 'r') as file:
        try:
            config = yaml.safe_load(file)
        except yaml.YAMLError as exc:
            print(exc)
    print(config)

    dataset_config = config['dataset_params']
    train_config = config['train_params']

    im_dataset = LandscapesDataset(split='train',
                                im_path=dataset_config['im_path'],
                                im_size=dataset_config['im_size'],
                                im_channels=dataset_config['im_channels'],
                                label_json_path=dataset_config.get('label_json_path', None))

    # Used for saving latents
    data_loader = DataLoader(im_dataset,
                             batch_size=1,
                             shuffle=False)

    num_images = train_config['num_samples']
    num_grid_rows = train_config['num_grid_rows']

    idxs = torch.randint(0, len(im_dataset) - 1, (num_images,))
    ims = torch.cat([im_dataset[idx][0][None, :] for idx in idxs]).float()
    ims = ims.to(device)

    print("\n" + "="*60)
    print("Loading Pre-trained Stable Diffusion VAE...")
    print("="*60 + "\n")
    
    model = HuggingFaceVAEWrapper(device=device)
    
    # Optionally load fine-tuned weights if they exist
    vae_ckpt_path = os.path.join(train_config['task_name'], train_config['vae_autoencoder_ckpt_name'])
    if os.path.exists(vae_ckpt_path):
        print(f"Loading fine-tuned VAE from {vae_ckpt_path}")
        model.vae.load_state_dict(torch.load(vae_ckpt_path, map_location=device))
    else:
        print("Using pre-trained weights (no fine-tuning found)")
    
    model.vae.eval()

    with torch.no_grad():

        # Encode and decode for visualization
        encoded_latents, _ = model.encode(ims)
        decoded_output = model.decode(encoded_latents)
        
        # Clamp and normalize for visualization
        decoded_output = torch.clamp(decoded_output, -1., 1.)
        decoded_output = (decoded_output + 1) / 2
        ims_norm = (ims + 1) / 2

        decoder_grid = make_grid(decoded_output.cpu(), nrow=num_grid_rows)
        input_grid = make_grid(ims_norm.cpu(), nrow=num_grid_rows)
        decoder_grid = torchvision.transforms.ToPILImage()(decoder_grid)
        input_grid = torchvision.transforms.ToPILImage()(input_grid)

        input_grid.save(os.path.join(train_config['task_name'], 'input_samples.png'))
        decoder_grid.save(os.path.join(train_config['task_name'], 'reconstructed_samples.png'))
        
        print("Saved visualization samples:")
        print(f"  - input_samples.png: Original images")
        print(f"  - reconstructed_samples.png: Reconstructed images\n")

        if train_config['save_latents']:
            # Save Latents for DiT training
            latent_path = os.path.join(train_config['task_name'], train_config['vae_latent_dir_name'])
            latent_fnames = glob.glob(os.path.join(latent_path, '*.pkl'))
            assert len(latent_fnames) == 0, 'Latents already present. Delete all latent files and re-run'
            
            if not os.path.exists(latent_path):
                os.mkdir(latent_path)
            
            print(f"\nSaving Latents for {dataset_config['im_path']}...")

            fname_latent_map = {}
            part_count = 0
            count = 0
            
            for idx, batch in enumerate(tqdm(data_loader)):

                # Unpack batch - dataset returns (image, class_label) or just image
                if isinstance(batch, (tuple, list)):
                    im = batch[0].float().to(device)
                else:
                    im = batch.float().to(device)
                
                # Encode to latent space (returns scaled latents)
                z_scaled, _ = model.encode(im)
                
                fname_latent_map[im_dataset.images[idx]] = z_scaled.cpu()
                
                # Save latents every 1000 images
                if (count + 1) % 1000 == 0:
                    pickle.dump(fname_latent_map, open(os.path.join(latent_path,
                                                                    f'{part_count}.pkl'), 'wb'))
                    part_count += 1
                    fname_latent_map = {}
                count += 1
            
            # Save remaining latents
            if len(fname_latent_map) > 0:
                pickle.dump(fname_latent_map, open(os.path.join(latent_path,
                                                                f'{part_count}.pkl'), 'wb'))
            
            print(f"Saved {count} latent vectors to {latent_path}/")
            print(f"  - Latent shape: (1, 4, 32, 32) per image")
            print(f"  - Total files: {part_count + 1}")
            print(f"\nLatents are now ready for DiT training!")
            print(f"Set 'use_latents: True' in your config to use these.")



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Arguments for vae inference')
    parser.add_argument('--config', dest='config_path',
                        default='config/landscapeshq.yaml', type=str)
    args = parser.parse_args()
    infer(args)
