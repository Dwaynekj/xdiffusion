"""Creates a pre-tokenized, pre-embedded version of MNIST to improve training time.

The text captioning scheme for MNIST is based on the label, e.g. the class label
1 gets converted to to the string "one" or "1". So for each sample of MNIST,
we will create two additional samples with the captions.
"""

import argparse

import numpy as np
import os
import torch
from torchvision.datasets import MNIST
from torchvision.transforms import v2
from tqdm import tqdm
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
)


def create_mnist_embedded(width: int = 32, height: int = 32):
    xforms = [
        # To make the math work out easier, resize the MNIST
        # images from (28,28) to (32, 32).
        v2.Resize(size=(height, width)),
        # Conversion to tensor scales the data from (0,255)
        # to (0,1).
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
    ]

    dataset = MNIST(
        ".",
        train=True,
        transform=v2.Compose(xforms),
        download=True,
    )

    text_labels = [
        ("zero", "0"),
        ("one", "1"),
        ("two", "2"),
        ("three", "3"),
        ("four", "4"),
        ("five", "5"),
        ("six", "6"),
        ("seven", "7"),
        ("eight", "8"),
        ("nine", "9"),
    ]

    image_data = []
    class_labels = []
    caption_embeddings = []
    caption_embedding_attention_masks = []

    text_encoder_model_name = "google/gemma-2-2b-it"
    max_length = 300
    tokenizer = AutoTokenizer.from_pretrained(text_encoder_model_name)
    tokenizer.padding_side = "right"
    text_encoder = AutoModelForCausalLM.from_pretrained(
        text_encoder_model_name, device_map="auto", torch_dtype=torch.float32
    ).get_decoder()
    text_encoder = text_encoder.eval().requires_grad_(False)

    num_entries_per_shard = 10
    shard_idx = 0
    shard_file_path = "MNISTEmbeddedGemma2/mnist_embedded_gemma_2_{name}_{idx:03d}.npy"
    os.makedirs("MNISTEmbeddedGemma2", exist_ok=True)

    dataset_length = len(dataset)
    for dataset_idx in tqdm(range(dataset_length)):
        image, label = dataset[dataset_idx]

        for caption in text_labels[label]:
            # Embed the caption
            chat = [{"role": "user", "content": caption}]
            prompts = tokenizer.apply_chat_template(
                chat, tokenize=False, add_generation_prompt=True
            )

            # Tokenize the prompts
            txt_tokens = tokenizer(
                prompts,
                padding="max_length",
                max_length=max_length,
                truncation=True,
                return_tensors="pt",
                return_length=False,
                return_overflowing_tokens=False,
            )

            # first bos and end N-1
            select_index = [0] + list(range(-max_length + 1, 0))

            input_ids = txt_tokens.input_ids.to(text_encoder.device)
            attention_mask = txt_tokens.attention_mask.to(text_encoder.device)

            y = text_encoder(input_ids, attention_mask=attention_mask)[0][:, None][
                :, :, select_index
            ]

            y_mask = txt_tokens.attention_mask[:, None, None][:, :, :, select_index]

            image_data.append(image.to("cpu").numpy())
            class_labels.append(label)
            caption_embeddings.append(y[0].to("cpu").numpy())
            caption_embedding_attention_masks.append(y_mask[0].to("cpu").numpy())

        if len(image_data) == num_entries_per_shard:
            np.save(
                shard_file_path.format(name="image_data", idx=shard_idx),
                allow_pickle=False,
                arr=image_data,
            )
            np.save(
                shard_file_path.format(name="class_labels", idx=shard_idx),
                allow_pickle=False,
                arr=class_labels,
            )
            np.save(
                shard_file_path.format(name="caption_embeddings", idx=shard_idx),
                allow_pickle=False,
                arr=caption_embeddings,
            )
            np.save(
                shard_file_path.format(
                    name="caption_embedding_attention_masks", idx=shard_idx
                ),
                allow_pickle=False,
                arr=caption_embedding_attention_masks,
            )

            shard_idx += 1
            image_data.clear()
            class_labels.clear()
            caption_embeddings.clear()
            caption_embedding_attention_masks.clear()

    # Write the last entries
    batch_size = len(image_data)

    if batch_size > 0:
        np.save(
            shard_file_path.format(name="image_data", idx=shard_idx),
            allow_pickle=False,
            arr=image_data,
        )
        np.save(
            shard_file_path.format(name="class_labels", idx=shard_idx),
            allow_pickle=False,
            arr=class_labels,
        )
        np.save(
            shard_file_path.format(name="caption_embeddings", idx=shard_idx),
            allow_pickle=False,
            arr=caption_embeddings,
        )
        np.save(
            shard_file_path.format(
                name="caption_embedding_attention_masks", idx=shard_idx
            ),
            allow_pickle=False,
            arr=caption_embedding_attention_masks,
        )


def main(override=None):
    """
    Main entrypoint for the standalone version of this package.
    """
    parser = argparse.ArgumentParser()
    args = parser.parse_args()

    create_mnist_embedded()


if __name__ == "__main__":
    main()
