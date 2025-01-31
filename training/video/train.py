import argparse

from xdiffusion.training.video.train import train


def main(override=None):
    """
    Main entrypoint for the standalone version of this package.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_training_steps", type=int, default=-1)
    parser.add_argument("--batch_size", type=int, default=-1)
    parser.add_argument("--config_path", type=str, required=True)
    parser.add_argument("--save_and_sample_every_n", type=int, default=1000)
    parser.add_argument("--load_model_weights_from_checkpoint", type=str, default="")
    parser.add_argument("--load_vae_weights_from_checkpoint", type=str, default="")
    parser.add_argument("--resume_from", type=str, default="")
    parser.add_argument("--joint_image_video_training_step", type=int, default=-1)
    parser.add_argument("--dataset_name", type=str, default="")
    parser.add_argument("--mixed_precision", type=str, default="")
    parser.add_argument("--force_cpu", action="store_true")
    args = parser.parse_args()

    train(
        num_training_steps=args.num_training_steps,
        batch_size=args.batch_size,
        config_path=args.config_path,
        save_and_sample_every_n=args.save_and_sample_every_n,
        load_model_weights_from_checkpoint=args.load_model_weights_from_checkpoint,
        load_vae_weights_from_checkpoint=args.load_vae_weights_from_checkpoint,
        resume_from=args.resume_from,
        joint_image_video_training_step=args.joint_image_video_training_step,
        dataset_name=args.dataset_name,
        output_path="output",
        mixed_precision=args.mixed_precision,
        force_cpu=args.force_cpu,
    )


if __name__ == "__main__":
    main()
