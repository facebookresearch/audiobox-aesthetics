import requests
import os
import argparse
import torch

from audiobox_aesthetics.inference import AudioBoxAesthetics

if __name__ == "__main__":
    # Set up argument parser
    parser = argparse.ArgumentParser(
        description="Download and test AudioBox Aesthetics model"
    )
    parser.add_argument(
        "--checkpoint-url",
        default="https://dl.fbaipublicfiles.com/audiobox-aesthetics/checkpoint.pt",
        help="URL for the base checkpoint",
    )
    parser.add_argument(
        "--model-name",
        default="audiobox-aesthetics",
        help="Name to save/load the pretrained model",
    )
    parser.add_argument(
        "--push-to-hub",
        action="store_true",
        help="Push the model to the Hugging Face Hub",
    )
    args = parser.parse_args()

    checkpoint_local_path = "base_checkpoint.pth"

    if not os.path.exists(checkpoint_local_path):
        print("Downloading base checkpoint")
        response = requests.get(args.checkpoint_url)
        with open(checkpoint_local_path, "wb") as f:
            f.write(response.content)

    # get model config from the base checkpoint
    checkpoint = torch.load(
        checkpoint_local_path, map_location="cpu", weights_only=True
    )
    model_cfg = checkpoint["model_cfg"]

    # extract normalization params from the base checkpoint
    target_transform = checkpoint["target_transform"]

    target_transform = {
        axis: {
            "mean": checkpoint["target_transform"][axis]["mean"],
            "std": checkpoint["target_transform"][axis]["std"],
        }
        for axis in target_transform.keys()
    }
    # force precision to be bfloat16 to match infer class
    model_cfg["precision"] = "bf16"

    model = AudioBoxAesthetics(
        sample_rate=16_000, target_transform=target_transform, **model_cfg
    )

    model._load_base_checkpoint(checkpoint_local_path)
    print("✅ Loaded model from base checkpoint")

    model.save_pretrained(args.model_name, push_to_hub=args.push_to_hub)
    print(f"✅ Saved model to {args.model_name}")
    if args.push_to_hub:
        model.push_to_hub(args.model_name)
        print(f"✅ Pushed model to Hub under {args.model_name}")

    # test load from pretrained
    model = AudioBoxAesthetics.from_pretrained(args.model_name)
    model.eval()
    print(f"✅ Loaded model from pretrained {args.model_name}")

    # test inference
    wav = model.load_audio("sample_audio/libritts_spk-84.wav")
    predictions = model.predict_from_wavs(wav)
    print(predictions)
    print("✅ Inference test passed")
