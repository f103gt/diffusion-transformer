import torch
import torchvision
import argparse
import yaml
import os
from torchvision.utils import make_grid
from PIL import Image
from tqdm import tqdm
from model.vae.vae import VAE
from model.transformer import DIT
from scheduler.linear_scheduler import LinearNoiseScheduler

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if torch.backends.mps.is_available():
    device = torch.device('mps')
    print('Using mps')


def sample(model, scheduler, train_config, dit_model_config,
           autoencoder_model_config, diffusion_config, dataset_config, vae, class_to_id=None):
    """
    Executes the reverse diffusion process to generate novel images from pure noise.

    This function performs the core ancestral sampling loop. It starts by sampling pure 
    Gaussian noise in the latent space and iteratively denoises it using the trained 
    Diffusion Transformer (DiT). It supports conditional generation (targeting specific 
    classes) or unconditional/random generation. 

    To save computational overhead, intermediate steps are visualized directly in 
    latent space (taking the first 3 channels), and the full VAE decoder is only 
    triggered on the final timestep to produce the actual high-resolution image.

    :param model: The trained DIT model.
    :param scheduler: The noise scheduler handling variance schedules (e.g., LinearNoiseScheduler).
    :param train_config: Dictionary of training and sampling hyperparameters.
    :param dit_model_config: Dictionary of DiT architecture parameters.
    :param autoencoder_model_config: Dictionary of VAE architecture parameters.
    :param diffusion_config: Dictionary of diffusion process parameters (e.g., num_timesteps).
    :param dataset_config: Dictionary containing dataset properties.
    :param vae: The trained VAE model used to decode the final latent representations.
    :param class_to_id: Optional dictionary mapping string class names to integer IDs.
    """
    im_size = dataset_config['im_size'] // 2 ** sum(autoencoder_model_config['down_sample'])
    xt = torch.randn((train_config['num_samples'],
                      autoencoder_model_config['z_channels'],
                      im_size,
                      im_size)).to(device)
    
    # Generate class labels for conditional sampling
    # You can specify desired class by name or ID
    if 'sample_classes' in train_config and train_config['sample_classes'] is not None:
        sample_classes = train_config['sample_classes']
        class_labels = []
        
        for cls in sample_classes:
            if isinstance(cls, str) and class_to_id is not None:
                # Convert class name to ID
                if cls in class_to_id:
                    class_labels.append(class_to_id[cls])
                else:
                    print(f'Warning: Class "{cls}" not found, using 0')
                    class_labels.append(0)
            else:
                # Use numeric ID directly
                class_labels.append(int(cls))
        
        class_labels = torch.tensor(class_labels).to(device)
    else:
        # Sample random classes
        num_classes = dit_model_config['num_classes']
        class_labels = torch.randint(0, num_classes, (train_config['num_samples'],)).to(device)
    
    # Print class info
    if class_to_id is not None:
        id_to_class = {v: k for k, v in class_to_id.items()}
        class_names = [id_to_class.get(idx.item(), f'ID_{idx.item()}') for idx in class_labels]
        print(f'Generating images for classes: {class_names}')
    else:
        print(f'Generating images for class IDs: {class_labels.cpu().tolist()}')

    for i in tqdm(reversed(range(diffusion_config['num_timesteps']))):
        # Get prediction of noise with class conditioning
        noise_pred = model(xt, torch.as_tensor(i).unsqueeze(0).to(device), class_labels)

        # Use scheduler to get x0 and xt-1
        xt, x0_pred = scheduler.sample_prev_timestep(xt, noise_pred, torch.as_tensor(i).to(device))

        if i == 0:
            # Decode ONLY the final image to save time
            ims = vae.to(device).decode(xt)
        else:
            # For intermediate steps, visualize the noisy latent (drop last channel for RGB display)
            ims = xt[:, :-1, :, :]

        ims = torch.clamp(ims, -1., 1.).detach().cpu()
        ims = (ims + 1) / 2

        grid = make_grid(ims, nrow=train_config['num_grid_rows'])
        img = torchvision.transforms.ToPILImage()(grid)

        if not os.path.exists(os.path.join(train_config['task_name'], 'samples')):
            os.mkdir(os.path.join(train_config['task_name'], 'samples'))
        img.save(os.path.join(train_config['task_name'], 'samples', 'x0_{}.png'.format(i)))
        img.close()


def infer(args):
    """
    Initializes models, loads checkpoints, and triggers the sampling generation.

    This function reads the main configuration file, sets up the Diffusion Transformer, 
    the Variational Autoencoder, and the linear noise scheduler. It verifies that 
    pre-trained checkpoints exist, pushes models to the target device, and manages 
    the directory setup for the output samples before initiating the `sample` loop.

    :param args: Parsed command-line arguments containing `config_path`.
    """
    # Read the config file #
    with open(args.config_path, 'r') as file:
        try:
            config = yaml.safe_load(file)
        except yaml.YAMLError as exc:
            print(exc)
    print(config)
    ########################

    diffusion_config = config['diffusion_params']
    dataset_config = config['dataset_params']
    dit_model_config = config['dit_params']
    autoencoder_model_config = config['autoencoder_params']
    train_config = config['train_params']

    # Create the noise scheduler
    scheduler = LinearNoiseScheduler(num_timesteps=diffusion_config['num_timesteps'],
                                     beta_start=diffusion_config['beta_start'],
                                     beta_end=diffusion_config['beta_end'])

    # Get latent image size
    im_size = dataset_config['im_size'] // 2 ** sum(autoencoder_model_config['down_sample'])
    model = DIT(im_size=im_size,
                im_channels=autoencoder_model_config['z_channels'],
                config=dit_model_config).to(device)

    model.eval()

    assert os.path.exists(os.path.join(train_config['task_name'],
                                       train_config['dit_ckpt_name'])), "Train DiT first"

    model.load_state_dict(torch.load(os.path.join(train_config['task_name'],
                                                  train_config['dit_ckpt_name']),
                                     map_location=device))
    print('Loaded dit checkpoint')

    # Create output directories
    if not os.path.exists(train_config['task_name']):
        os.mkdir(train_config['task_name'])

    vae = VAE(im_channels=dataset_config['im_channels'],
              model_config=autoencoder_model_config)
    vae.eval()

    # Load vae if found
    assert os.path.exists(os.path.join(train_config['task_name'], train_config['vae_autoencoder_ckpt_name'])), \
        "VAE checkpoint not present. Train VAE first."
    vae.load_state_dict(torch.load(os.path.join(train_config['task_name'],
                                                train_config['vae_autoencoder_ckpt_name']),
                                   map_location=device), strict=True)
    print('Loaded vae checkpoint')

    # Load class mapping if available
    class_to_id = None
    if 'label_json_path' in dataset_config and dataset_config['label_json_path']:
        import json
        label_json_path = dataset_config['label_json_path']
        if os.path.exists(label_json_path):
            with open(label_json_path, 'r') as f:
                label_data = json.load(f)
            unique_classes = sorted(set(label_data.values()))
            class_to_id = {class_name: idx for idx, class_name in enumerate(unique_classes)}
            print(f'Loaded class mapping: {class_to_id}')

    with torch.no_grad():
        sample(model, scheduler, train_config, dit_model_config,
               autoencoder_model_config, diffusion_config, dataset_config, vae, class_to_id)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Arguments for dit image generation')
    parser.add_argument('--config', dest='config_path',
                        default='config/landscapeshq.yaml', type=str)
    args = parser.parse_args()
    infer(args)
