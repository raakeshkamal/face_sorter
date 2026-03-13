"""
Stability score detection service.

This module provides functionality to evaluate image stability scores using
Hugging Face timm classification models. It integrates with the training
pipeline to provide content-based quality assessment.

Platform: Apple Silicon (MPS GPU with CPU fallback)
"""

import asyncio
import logging
import os
import warnings
from pathlib import Path
from typing import Optional, Tuple

import torch
from PIL import Image
import timm

from face_sorter.config import get_settings

logger = logging.getLogger(__name__)

# Configure Hugging Face settings BEFORE imports to suppress warnings
# Set a local cache directory to avoid HF Hub warnings and enable faster loading
if not os.environ.get("HF_HOME"):
    os.environ["HF_HOME"] = os.path.expanduser("~/.cache/huggingface")
if not os.environ.get("HF_HUB_CACHE"):
    os.environ["HF_HUB_CACHE"] = os.path.join(os.environ["HF_HOME"], "hub")

# Suppress HF Hub telemetry and warnings - MUST be set before imports
os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"

# Set HF_TOKEN to empty string to suppress unauthenticated request warning
if not os.environ.get("HF_TOKEN"):
    os.environ["HF_TOKEN"] = ""

# Suppress huggingface_hub warnings at the Python and logging levels
warnings.filterwarnings("ignore", message=".*HF_TOKEN.*")
warnings.filterwarnings("ignore", message=".*unauthenticated.*")
warnings.filterwarnings("ignore", message=".*Hugging Face Hub.*")

# Configure logging to suppress HF Hub warnings
logging.getLogger("huggingface_hub").setLevel(logging.ERROR)
logging.getLogger("transformers").setLevel(logging.ERROR)
logging.getLogger("timm").setLevel(logging.ERROR)

# Custom warning filter to catch HF Hub warnings
original_showwarning = warnings.showwarning


def custom_showwarning(message, category, filename, lineno, file=None, line=None):
    """Suppress HF Hub authentication warnings."""
    if "HF_TOKEN" in str(message) or "unauthenticated" in str(message).lower():
        return
    original_showwarning(message, category, filename, lineno, file, line)


warnings.showwarning = custom_showwarning


def load_stability_model(
    model_name: Optional[str] = None, device: str = "auto"
) -> Tuple[torch.nn.Module, str]:
    """
    Load the stability score classification model from Hugging Face.

    Args:
        model_name: Model identifier for timm. If None, reads from CLASSIFICATION_MODEL_NAME env var.
        device: Device to use ('cuda', 'cpu', or 'auto' for automatic selection).
                'auto' will prioritize Apple Silicon GPU (MPS) with CPU fallback.

    Returns:
        Tuple of (model, actual_device_used).

    Raises:
        ValueError: If model_name is not configured.
        RuntimeError: If model loading fails.
    """
    settings = get_settings()

    if model_name is None:
        model_name = settings.classification_model_name

    if model_name is None:
        raise ValueError("CLASSIFICATION_MODEL_NAME not configured in settings")

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

    logger.info(f"Loading stability score classification model: {model_name}")

    try:
        model = timm.create_model(model_name, pretrained=True)
        model = model.to(actual_device)
        model.eval()
        device_name = "Apple Silicon GPU (MPS)" if actual_device == "mps" else actual_device
        logger.info(f"Stability model loaded on device: {device_name}")
        return model, actual_device
    except Exception as e:
        logger.error(f"Error loading stability model: {e}")
        raise RuntimeError(f"Failed to load stability model: {e}") from e


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


async def evaluate_stability(
    model: torch.nn.Module, image_path: Path, device: str
) -> Tuple[Optional[float], dict]:
    """
    Evaluate a single image and return the stability_score.

    Args:
        model: The loaded classification model.
        image_path: Path to the image file.
        device: Device to use for inference.

    Returns:
        Tuple of (stability_score, details_dict). Returns (None, {}) on failure.
        details_dict contains:
        - classification_result: The predicted class label
        - content_probability: The confidence score
    """
    if not image_path.exists():
        logger.warning(f"Image not found: {image_path}")
        return None, {}

    try:
        # Run blocking operations in thread pool
        result = await _evaluate_blocking(model, image_path, device)
        return result
    except Exception as e:
        logger.warning(f"Failed to calculate stability score for {image_path}: {e}")
        return None, {}


async def _evaluate_blocking(
    model: torch.nn.Module, image_path: Path, device: str
) -> Tuple[float, dict]:
    """
    Blocking wrapper for image evaluation.

    This is called via asyncio.to_thread to avoid blocking the event loop.
    """
    from torchvision import transforms

    # Get model configuration for preprocessing
    config = get_model_config(model)
    config_size = config["config_size"]
    config_mean = config["config_mean"]
    config_std = config["config_std"]

    # Load and preprocess image
    image = Image.open(image_path)

    # Convert to RGB if necessary (handles PNG with alpha channel, etc.)
    if image.mode != "RGB":
        image = image.convert("RGB")

    # Resize and normalize
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

    # Stability score: higher score = more concerning content
    stability_score = probability_value

    details = {
        "classification_result": label_names[class_idx.item()],
        "content_probability": probability_value,
    }

    return stability_score, details
