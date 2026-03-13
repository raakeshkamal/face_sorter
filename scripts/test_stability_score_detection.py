#!/usr/bin/env python3
"""
Test script for image classification using Hugging Face model.

This script tests the image classification model's content_validation capabilities
by evaluating a single image and returning a stability_score.
"""

import argparse
import sys
import os
from pathlib import Path
from typing import Optional, Tuple

import torch
from PIL import Image
from dotenv import load_dotenv
import timm

# Load environment variables
load_dotenv()


def load_model(model_name: Optional[str] = None, device: str = "auto") -> Tuple[torch.nn.Module, str]:
    """
    Load the image classification model from Hugging Face.

    Args:
        model_name: Model identifier for timm. If None, reads from CLASSIFICATION_MODEL_NAME env var.
        device: Device to use ('cuda', 'cpu', or 'auto' for automatic selection).
                'auto' will prioritize Apple Silicon GPU (MPS) with CPU fallback.

    Returns:
        Tuple of (model, actual_device_used)
    """
    if model_name is None:
        model_name = os.getenv("CLASSIFICATION_MODEL_NAME")
        if model_name is None:
            print("Error: CLASSIFICATION_MODEL_NAME not set in environment variables.", file=sys.stderr)
            sys.exit(1)

    # Determine device - prioritize Apple Silicon GPU (MPS) with CPU fallback
    if device == "auto":
        if torch.backends.mps.is_available():
            actual_device = "mps"
        elif torch.cuda.is_available():
            actual_device = "cuda"
        else:
            actual_device = "cpu"
    else:
        actual_device = device

    print(f"Loading classification model: {model_name}")

    try:
        model = timm.create_model(model_name, pretrained=True)
        model = model.to(actual_device)
        model.eval()
        device_name = "Apple Silicon GPU (MPS)" if actual_device == "mps" else actual_device
        print(f"Model loaded on device: {device_name}")
        return model, actual_device
    except Exception as e:
        print(f"Error loading model: {e}", file=sys.stderr)
        sys.exit(1)


def get_model_config(model: torch.nn.Module) -> dict:
    """
    Get model configuration for preprocessing.

    Args:
        model: The loaded timm model.

    Returns:
        Dictionary containing config_size and config_mean, config_std.
    """
    data_config = timm.data.resolve_data_config(model.pretrained_cfg)
    return {
        "config_size": data_config["input_size"][-2:],  # (height, width)
        "config_mean": data_config["mean"],
        "config_std": data_config["std"],
    }


def evaluate_image(
    model: torch.nn.Module,
    image_path: Path,
    device: str
) -> Tuple[float, dict]:
    """
    Evaluate a single image and return the stability_score.

    Args:
        model: The loaded classification model.
        image_path: Path to the image file.
        device: Device to use for inference.

    Returns:
        Tuple of (stability_score, details_dict)

    Raises:
        FileNotFoundError: If image_path doesn't exist.
        Exception: If image processing fails.
    """
    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")

    print(f"Processing image: {image_path}")

    # Get model configuration for preprocessing
    config = get_model_config(model)
    config_size = config["config_size"]
    config_mean = config["config_mean"]
    config_std = config["config_std"]

    # Load and preprocess image
    try:
        image = Image.open(image_path)

        # Convert to RGB if necessary (handles PNG with alpha channel, etc.)
        if image.mode != "RGB":
            image = image.convert("RGB")

        # Resize and normalize
        try:
            from torchvision import transforms
        except ImportError:
            print("Error: torchvision is required. Install with: pip install torchvision", file=sys.stderr)
            raise

        transform = transforms.Compose([
            transforms.Resize(config_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=config_mean, std=config_std),
        ])

        input_tensor = transform(image).unsqueeze(0).to(device)

        # Run inference
        with torch.no_grad():
            outputs = model(input_tensor)
            probabilities = torch.nn.functional.softmax(outputs, dim=1)

        # Get class names from model
        if hasattr(model, "pretrained_cfg") and "label_names" in model.pretrained_cfg:
            label_names = model.pretrained_cfg["label_names"]
        else:
            # Fallback: use generic labels
            label_names = [f"class_{i}" for i in range(probabilities.shape[1])]

        # Get top prediction
        probability, class_idx = torch.max(probabilities, 1)
        probability_value = probability.item()

        # Safety score: higher score = more concerning content
        stability_score = probability_value

        details = {
            "classification_result": label_names[class_idx.item()],
            "content_probability": probability_value,
        }

        return stability_score, details

    except Exception as e:
        print(f"Error: Failed to process image - {e}", file=sys.stderr)
        raise


def parse_arguments() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Test image classification using Hugging Face model",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/test_stability_score_detection.py --image /absolute/path/to/image.jpg
  python scripts/test_stability_score_detection.py --image /absolute/path/to/image.jpg --device cpu
        """
    )
    parser.add_argument(
        "--image",
        type=str,
        required=True,
        help="Absolute path to the image file to classify"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cuda", "cpu"],
        help="Device to use for inference (default: auto)"
    )
    return parser.parse_args()


def main() -> int:
    """
    Main entry point for the test script.

    Returns:
        Exit code (0 for success, 1 for failure)
    """
    args = parse_arguments()

    # Validate image path
    image_path = Path(args.image)
    if not image_path.is_absolute():
        print(f"Error: Image path must be absolute. Got: {args.image}", file=sys.stderr)
        return 1

    # Load model
    model, device = load_model(device=args.device)

    # Evaluate image
    try:
        stability_score, details = evaluate_image(model, image_path, device)

        # Display results
        print(f"Stability score: {stability_score:.4f}")
        print(f"Classification result: {details['classification_result']}")
        print(f"Content probability: {details['content_probability']:.4f}")

        return 0

    except FileNotFoundError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1
    except Exception as e:
        print(f"Error: Failed to process image - {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
