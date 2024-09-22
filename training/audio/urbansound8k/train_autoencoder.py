"""Train a variational autoencoder for UrbanSound8k.

This is a simple VAE to help reduce the dimensionality of the UrbanSound8k
dataset from 1x128x256 to 4x10x32.
"""

from accelerate import Accelerator, DataLoaderConfiguration
import argparse
from einops import rearrange
import math
import os
import torch
from torch.optim import Adam
from torch.utils.data import DataLoader
from torchinfo import summary
from torchvision import transforms
from torchvision import utils as torch_utils
from torchvision.datasets import MNIST
from tqdm import tqdm

from xdiffusion.datasets.urbansound8k import UrbanSound8k
from xdiffusion.utils import cycle, load_yaml, save_mel_spectrogram_audio
from xdiffusion.layers.audio import (
    mel_to_logmel,
    logmel_to_mel,
)
from xdiffusion.autoencoders.kl import AutoencoderKL

OUTPUT_NAME = "output/audio/urbansound8k/autoencoder_kl"


def train_autoencoder(num_training_steps: int, batch_size: int, config_path: str):
    # Ensure the output directories exist
    os.makedirs(OUTPUT_NAME, exist_ok=True)

    # Load the MNIST dataset. This is a supervised dataset so
    # it contains both images and class labels. We will ignore the class
    # labels for now.
    dataset = UrbanSound8k(
        ".",
    )

    # Create the dataloader for the MNIST dataset
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4)

    # Open the model configuration
    config = load_yaml(config_path)

    # Create the autoencoder we will train.
    vae = AutoencoderKL(config)
    summary(
        vae,
        [(128, 1, 128, 256)],
    )

    # The accelerate library will handle of the GPU device management for us.
    accelerator = Accelerator(
        dataloader_config=DataLoaderConfiguration(split_batches=False),
        mixed_precision="no",
    )
    device = accelerator.device

    # Prepare the dataset with the accelerator. This makes sure all of the
    # dataset items are placed onto the correct device.
    dataloader = accelerator.prepare(dataloader)

    # We are going to train for a fixed number of steps, so set the dataloader
    # to repeat indefinitely over the entire dataset.
    dataloader = cycle(dataloader)

    # Configure the optimizers for training.
    opts, _ = vae.configure_optimizers(learning_rate=4.5e-6)

    # Move the model and the optimizer to the accelerator as well.
    vae = accelerator.prepare(vae)
    optimizers = []
    for opt in opts:
        optimizers.append(accelerator.prepare(opt))

    # Step counter to keep track of training
    step = 0

    # We will sample the autoencoder every N steps, to monitor
    # training and see how it improves over time.
    save_and_sample_every_n = 1000

    # Clip the graidents for smoother training.
    max_grad_norm = 1.0
    average_losses = [0.0 for _ in optimizers]
    average_losses_cumulative = [0.0 for _ in optimizers]
    average_posterior_mean = 0.0
    average_posterior_mean_cumulative = 0.0
    average_posterior_std = 0.0
    average_posterior_std_cumulative = 0.0

    with tqdm(initial=step, total=num_training_steps) as progress_bar:
        # Perform gradient descent for the given number of training steps.
        while step < num_training_steps:
            # The dataset has images and classes. Let's use the classes,
            # and convert them into a fixed embedding space.
            images, labels = next(dataloader)
            images = images[:, None, ...]
            batch = {"image": mel_to_logmel(images), "label": labels}

            # Calculate the loss on the batch of training data.
            current_loss = []
            for optimizer_idx, optimizer in enumerate(optimizers):
                loss, reconstructions, posterior = vae.training_step(
                    batch, -1, optimizer_idx, step
                )

                # Calculate the gradients at each step in the network.
                accelerator.backward(loss)

                # On a multi-gpu machine or cluster, wait for all of the workers
                # to finish.
                accelerator.wait_for_everyone()

                # Clip the gradients.
                accelerator.clip_grad_norm_(vae.parameters(), max_grad_norm)

                # Perform the gradient descent step using the optimizer.
                optimizer.step()

                # Resent the gradients for the next step.
                optimizer.zero_grad()

                average_losses_cumulative[optimizer_idx] += loss.item()
                current_loss.append(loss.item())
            average_posterior_mean_cumulative += posterior.mean.detach().mean()
            average_posterior_std_cumulative += posterior.std.detach().mean()

            x = reconstructions

            # Show the current loss in the progress bar.
            progress_bar.set_description(
                f"loss: {[f'{val:.4f}' for idx, val in enumerate(current_loss)]} avg_loss: {[f'{val:.4f}' for idx, val in enumerate(average_losses)]} KL: {posterior.kl().detach().mean():.4f} posterior_mean: {average_posterior_mean:.4f} posterior_std: {average_posterior_std:.4f}"
            )

            # To help visualize training, periodically sample from the
            # autoencoder to see how well its doing.
            if step % save_and_sample_every_n == 0:
                average_losses = [
                    loss / save_and_sample_every_n for loss in average_losses_cumulative
                ]
                average_losses_cumulative = [0.0 for _ in optimizers]
                average_posterior_mean = (
                    average_posterior_mean_cumulative / save_and_sample_every_n
                )
                average_posterior_std = (
                    average_posterior_std_cumulative / save_and_sample_every_n
                )
                average_posterior_mean_cumulative = 0.0
                average_posterior_std_cumulative = 0.0

                # Save the reconstructed samples into an image grid
                base_output_path = str(f"{OUTPUT_NAME}/sample-{step}/")
                os.makedirs(base_output_path, exist_ok=True)
                output_path_spec = (
                    base_output_path + f"reconstructed-sample-{step}" + "-{idx}.wav"
                )
                save_mel_spectrogram_audio(
                    logmel_to_mel(x.detach()),
                    output_path_spec,
                )
                output_path_spec = (
                    base_output_path + f"original-sample-{step}" + "-{idx}.wav"
                )
                save_mel_spectrogram_audio(
                    images.detach(),
                    output_path_spec,
                )

                # Save a corresponding model checkpoint.
                torch.save(
                    {
                        "step": step,
                        "model_state_dict": vae.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "loss": loss,
                    },
                    f"{OUTPUT_NAME}/vae-{step}.pt",
                )

            # Update the current step.
            step += 1

            # Update the training progress bar in the console.
            progress_bar.update(1)

    # Save the last samples
    base_output_path = str(f"{OUTPUT_NAME}/sample-{step}/")
    os.makedirs(base_output_path, exist_ok=True)
    output_path_spec = base_output_path + f"reconstructed-sample-{step}" + "-{idx}.wav"
    save_mel_spectrogram_audio(
        logmel_to_mel(x.detach()),
        output_path_spec,
    )
    output_path_spec = base_output_path + f"original-sample-{step}" + "-{idx}.wav"
    save_mel_spectrogram_audio(
        images.detach(),
        output_path_spec,
    )


def main(override=None):
    """
    Main entrypoint for the standalone version of this package.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_training_steps", type=int, default=30000)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--config_path", type=str, required=True)
    args = parser.parse_args()

    train_autoencoder(
        num_training_steps=args.num_training_steps,
        batch_size=args.batch_size,
        config_path=args.config_path,
    )


if __name__ == "__main__":
    main()
