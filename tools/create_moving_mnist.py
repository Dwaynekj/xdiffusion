"""Generate Moving MNIST dataset with labels."""

from PIL import Image
import sys
import os
import math
import numpy as np
import torch
import codecs
from tqdm import tqdm


# helper functions
def arr_from_img(im, mean=0, std=1):
    """
    Args:
        im: Image
        shift: Mean to subtract
        std: Standard Deviation to subtract
    Returns:
        Image in np.float32 format, in width height channel format. With values in range 0,1
        Shift means subtract by certain value. Could be used for mean subtraction.
    """
    width, height = im.size
    arr = im.getdata()
    c = int(np.prod(arr.size) / (width * height))

    return (
        np.asarray(arr, dtype=np.float32).reshape((height, width, c)).transpose(2, 1, 0)
        / 255.0
        - mean
    ) / std


def get_image_from_array(X, index, mean=0, std=1):
    """
    Args:
        X: Dataset of shape N x C x W x H
        index: Index of image we want to fetch
        mean: Mean to add
        std: Standard Deviation to add
    Returns:
        Image with dimensions H x W x C or H x W if it's a single channel image
    """
    ch, w, h = X.shape[1], X.shape[2], X.shape[3]
    ret = (
        (((X[index] + mean) * 255.0) * std)
        .reshape(ch, w, h)
        .transpose(2, 1, 0)
        .clip(0, 255)
        .astype(np.uint8)
    )
    if ch == 1:
        ret = ret.reshape(h, w)
    return ret


TRAIN_IMAGES = "train-images-idx3-ubyte.gz"
TRAIN_LABELS = "train-labels-idx1-ubyte.gz"


# loads mnist from web on demand
def load_dataset():
    from urllib.request import urlretrieve

    def download(filename, source="https://ossci-datasets.s3.amazonaws.com/mnist/"):
        print("Downloading %s" % filename)
        urlretrieve(source + filename, filename)

    import gzip

    def load_mnist_images(filename):
        if not os.path.exists(filename):
            download(filename)
        with gzip.open(filename, "rb") as f:
            data = np.frombuffer(f.read(), np.uint8, offset=16)
        data = data.reshape(-1, 1, 28, 28)
        return data / np.float32(255)

    def load_mnist_labels(filename):
        if not os.path.exists(filename):
            download(filename)
        return read_label_file(filename).numpy()

    def read_label_file(path: str) -> torch.Tensor:
        x = read_sn3_pascalvincent_tensor(path, strict=False)
        if x.dtype != torch.uint8:
            raise TypeError(f"x should be of dtype torch.uint8 instead of {x.dtype}")
        if x.ndimension() != 1:
            raise ValueError(f"x should have 1 dimension instead of {x.ndimension()}")
        return x.long()

    def read_image_file(path: str) -> torch.Tensor:
        x = read_sn3_pascalvincent_tensor(path, strict=False)
        if x.dtype != torch.uint8:
            raise TypeError(f"x should be of dtype torch.uint8 instead of {x.dtype}")
        if x.ndimension() != 3:
            raise ValueError(f"x should have 3 dimension instead of {x.ndimension()}")
        return x

    return load_mnist_images(TRAIN_IMAGES), load_mnist_labels(TRAIN_LABELS)


SN3_PASCALVINCENT_TYPEMAP = {
    8: torch.uint8,
    9: torch.int8,
    11: torch.int16,
    12: torch.int32,
    13: torch.float32,
    14: torch.float64,
}


def get_int(b: bytes) -> int:
    return int(codecs.encode(b, "hex"), 16)


def _flip_byte_order(t: torch.Tensor) -> torch.Tensor:
    return (
        t.contiguous()
        .view(torch.uint8)
        .view(*t.shape, t.element_size())
        .flip(-1)
        .view(*t.shape[:-1], -1)
        .view(t.dtype)
    )


def read_sn3_pascalvincent_tensor(path: str, strict: bool = True) -> torch.Tensor:
    """Read a SN3 file in "Pascal Vincent" format (Lush file 'libidx/idx-io.lsh').
    Argument may be a filename, compressed filename, or file object.
    """
    # read
    import gzip

    with gzip.open(path, "rb") as f:
        data = f.read()

    # parse
    if sys.byteorder == "little":
        magic = get_int(data[0:4])
        nd = magic % 256
        ty = magic // 256
    else:
        nd = get_int(data[0:1])
        ty = (
            get_int(data[1:2])
            + get_int(data[2:3]) * 256
            + get_int(data[3:4]) * 256 * 256
        )

    assert 1 <= nd <= 3
    assert 8 <= ty <= 14
    torch_type = SN3_PASCALVINCENT_TYPEMAP[ty]
    s = [get_int(data[4 * (i + 1) : 4 * (i + 2)]) for i in range(nd)]

    if sys.byteorder == "big":
        for i in range(len(s)):
            s[i] = int.from_bytes(
                s[i].to_bytes(4, byteorder="little"), byteorder="big", signed=False
            )

    parsed = torch.frombuffer(bytearray(data), dtype=torch_type, offset=(4 * (nd + 1)))

    # The MNIST format uses the big endian byte order, while `torch.frombuffer` uses whatever the system uses. In case
    # that is little endian and the dtype has more than one byte, we need to flip them.
    if sys.byteorder == "little" and parsed.element_size() > 1:
        parsed = _flip_byte_order(parsed)

    assert parsed.shape[0] == np.prod(s) or not strict
    return parsed.view(*s)


def generate_moving_mnist(
    shape=(64, 64),
    num_frames=30,
    num_images=100,
    original_size=28,
    nums_per_image=2,
):
    """
    Args:
        training: Boolean, used to decide if downloading/generating train set or test set
        shape: Shape we want for our moving images (new_width and new_height)
        num_frames: Number of frames in a particular movement/animation/gif
        num_images: Number of movement/animations/gif to generate
        original_size: Real size of the images (eg: MNIST is 28x28)
        nums_per_image: Digits per movement/animation/gif.
    Returns:
        Dataset of np.uint8 type with dimensions num_frames * num_images x 1 x new_width x new_height
    """
    mnist, mnist_labels = load_dataset()
    width, height = shape

    # Get how many pixels can we move around a single image
    lims = (x_lim, y_lim) = width - original_size, height - original_size

    # Create a dataset of shape of num_frames * num_images x 1 x new_width x new_height
    # Eg : 3000000 x 1 x 64 x 64
    dataset = np.empty((num_frames * num_images, 1, width, height), dtype=np.uint8)
    labels = np.empty((num_frames * num_images, nums_per_image), dtype=np.uint8)

    for img_idx in tqdm(range(num_images)):
        # Randomly generate direction, speed and velocity for both images
        direcs = np.pi * (np.random.rand(nums_per_image) * 2 - 1)
        speeds = np.random.randint(5, size=nums_per_image) + 2
        veloc = np.asarray(
            [
                (speed * math.cos(direc), speed * math.sin(direc))
                for direc, speed in zip(direcs, speeds)
            ]
        )

        # Get a list containing two PIL images randomly sampled from the database
        mnist_images_and_labels = [
            (
                Image.fromarray(get_image_from_array(mnist, r, mean=0)).resize(
                    (original_size, original_size), resample=Image.BICUBIC
                ),
                mnist_labels[r],
                r,
            )
            for r in np.random.randint(0, mnist.shape[0], nums_per_image)
        ]
        selected_mnist_images = [v[0] for v in mnist_images_and_labels]
        selected_mnist_labels = [v[1] for v in mnist_images_and_labels]
        r = [v[2] for v in mnist_images_and_labels]

        # Generate tuples of (x,y) i.e initial positions for nums_per_image (default : 2)
        positions = np.asarray(
            [
                (np.random.rand() * x_lim, np.random.rand() * y_lim)
                for _ in range(nums_per_image)
            ]
        )

        # Generate new frames for the entire num_framesgth
        for frame_idx in range(num_frames):

            canvases = [Image.new("L", (width, height)) for _ in range(nums_per_image)]
            canvas = np.zeros((1, width, height), dtype=np.float32)

            # In canv (i.e Image object) place the image at the respective positions
            # Super impose both images on the canvas (i.e empty np array)
            for i, canv in enumerate(canvases):
                canv.paste(selected_mnist_images[i], tuple(positions[i].astype(int)))
                canvas += arr_from_img(canv, mean=0)

            # Get the next position by adding velocity
            next_pos = positions + veloc

            # Iterate over velocity and see if we hit the wall
            # If we do then change the  (change direction)
            for i, pos in enumerate(next_pos):
                for j, coord in enumerate(pos):
                    if coord < -2 or coord > lims[j] + 2:
                        veloc[i] = list(
                            list(veloc[i][:j])
                            + [-1 * veloc[i][j]]
                            + list(veloc[i][j + 1 :])
                        )

            # Make the permanent change to position by adding updated velocity
            positions = positions + veloc

            # Add the canvas to the dataset array
            dataset[img_idx * num_frames + frame_idx] = (
                (canvas * 255).clip(0, 255).astype(np.uint8)
            )
            labels[img_idx * num_frames + frame_idx] = selected_mnist_labels

    return dataset, labels


def label_list(labels, idx) -> str:
    num_labels = labels.shape[1]
    output = f"{labels[idx][0]}"

    for j in range(num_labels - 1):
        output = output + f"_{labels[idx][j+1]}"
    return output


def main(
    dest,
    filetype="npz",
    frame_size=64,
    num_frames=30,
    num_images=100,
    original_size=28,
    nums_per_image=2,
):
    image_data, labels = generate_moving_mnist(
        shape=(frame_size, frame_size),
        num_frames=num_frames,
        num_images=num_images,
        original_size=original_size,
        nums_per_image=nums_per_image,
    )

    if filetype == "npz":
        np.savez(dest, image_data)
        np.savez(f"{dest}_labels", labels)
    elif filetype == "jpg":
        image_idx = 0
        frame_idx = 0

        os.makedirs(dest, exist_ok=True)
        for i in range(image_data.shape[0]):
            if i != 0 and i % num_frames == 0:
                image_idx += 1
                frame_idx = 0

            Image.fromarray(image_data[i][0]).save(
                os.path.join(
                    dest, f"{image_idx}_{frame_idx}_{label_list(labels, i)}.jpg"
                )
            )
            frame_idx += 1


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Command line options")
    parser.add_argument("--dest", type=str, dest="dest", default="movingmnistdata")
    parser.add_argument("--filetype", type=str, dest="filetype", default="npz")
    parser.add_argument("--frame_size", type=int, dest="frame_size", default=64)
    parser.add_argument(
        "--num_frames", type=int, dest="num_frames", default=30
    )  # length of each sequence
    parser.add_argument(
        "--num_images", type=int, dest="num_images", default=20000
    )  # number of sequences to generate
    parser.add_argument(
        "--original_size", type=int, dest="original_size", default=28
    )  # size of mnist digit within frame
    parser.add_argument(
        "--nums_per_image", type=int, dest="nums_per_image", default=2
    )  # number of digits in each frame
    args = parser.parse_args(sys.argv[1:])
    main(**{k: v for (k, v) in vars(args).items() if v is not None})
