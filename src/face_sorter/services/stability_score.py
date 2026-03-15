"""
Stability score detection service using ONNX Runtime.

This module provides functionality to evaluate image stability scores using
Vision Transformers with ONNX Runtime. It integrates with the training
pipeline to provide content-based quality assessment.

Platform: Optimized for macOS with CoreML support, falls back to CPU.
"""

import asyncio
import logging
import os
import sys
import warnings
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
from PIL import Image
from face_sorter.config import get_settings

logger = logging.getLogger(__name__)

# Configure Hugging Face settings to prioritize local/cache and suppress warnings
# Use /tmp for storing and loading models to avoid disk permission issues in some environments
# but default to ~/.cache/huggingface if not specified
if not os.environ.get("HF_HOME"):
    os.environ["HF_HOME"] = os.path.expanduser("~/.cache/huggingface")
if not os.environ.get("HF_HUB_CACHE"):
    os.environ["HF_HUB_CACHE"] = os.path.join(os.environ["HF_HOME"], "hub")

os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
if not os.environ.get("HF_TOKEN"):
    os.environ["HF_TOKEN"] = ""

# Suppress warnings
warnings.filterwarnings("ignore", message=".*HF_TOKEN.*")
logging.getLogger("huggingface_hub").setLevel(logging.ERROR)
logging.getLogger("transformers").setLevel(logging.ERROR)

# Import ONNX Runtime and Transformers only when needed or here if they are core dependencies
try:
    import onnxruntime as ort
    from transformers import AutoImageProcessor
except ImportError as e:
    logger.error(
        f"Required libraries missing: {e}. Install with: pip install onnxruntime transformers"
    )
    # We don't exit here to allow the rest of the app to function if stability scoring is disabled


def softmax(x: np.ndarray) -> np.ndarray:
    """Compute softmax of array x."""
    e_x = np.exp(x - np.max(x, axis=1, keepdims=True))
    return e_x / e_x.sum(axis=1, keepdims=True)


async def load_stability_model(
    model_path: Optional[str] = None,
    model_name: Optional[str] = None,
    providers: Optional[list] = None,
) -> Tuple[Optional["ort.InferenceSession"], Optional["AutoImageProcessor"], str]:
    """
    Load the ONNX model and image processor.

    Returns:
        Tuple of (session, processor, model_description)
    """
    settings = get_settings()

    if not model_name:
        model_name = settings.stability_model_name

    if not model_name:
        logger.warning("No stability model name configured. Stability detection will be limited.")
        return None, None, "No model configured"

    if not model_path:
        model_path = settings.stability_onnx_model_path

    # If model_path is still None, use a default /tmp path and prepare for auto-download
    was_model_path_provided = bool(model_path)
    if not model_path:
        model_slug = model_name.replace("/", "--")
        model_dir = Path("/tmp/huggingface_onnx") / model_slug
        model_path = str(model_dir / "model_int8.onnx")

    path_obj = Path(model_path).resolve()
    model_path = str(path_obj)
    if not path_obj.exists():
        if was_model_path_provided:
            logger.warning(
                f"Specified model path {model_path} does not exist. Attempting auto-download..."
            )

        # Attempt to auto-download and export using optimum
        logger.info(f"Auto-downloading and exporting '{model_name}' to ONNX format...")
        try:
            from optimum.onnxruntime import ORTModelForImageClassification
            from onnxruntime.quantization import quantize_dynamic, QuantType

            export_dir = path_obj.parent
            export_dir.mkdir(parents=True, exist_ok=True)

            # Use thread pool for blocking download/export
            ort_model = await asyncio.to_thread(
                ORTModelForImageClassification.from_pretrained, model_name, export=True
            )
            await asyncio.to_thread(ort_model.save_pretrained, export_dir)

            exported_model = export_dir / "model.onnx"

            if exported_model.exists():
                if str(exported_model) == str(path_obj):
                    target_path = exported_model.with_name(exported_model.stem + "_int8.onnx")
                else:
                    target_path = path_obj

                logger.info("Quantizing model for faster inference...")
                await asyncio.to_thread(
                    quantize_dynamic,
                    model_input=str(exported_model),
                    model_output=str(target_path),
                    weight_type=QuantType.QUInt8,
                )

                model_path = str(target_path)
                path_obj = target_path

            logger.info(f"Successfully exported and quantized {model_name} to {model_path}")
        except ImportError:
            logger.error(
                "optimum is required for auto-download. Install with: pip install optimum[exporters]"
            )
            return None, None, "Failed (missing optimum)"
        except Exception as e:
            logger.error(f"Error during auto-download/export: {e}")
            return None, None, f"Failed: {e}"

    # Auto-detect providers if not specified
    if providers is None:
        available_providers = ort.get_available_providers()
        providers = []
        if "CoreMLExecutionProvider" in available_providers:
            providers.append("CoreMLExecutionProvider")
        if "CUDAExecutionProvider" in available_providers:
            providers.append("CUDAExecutionProvider")
        providers.append("CPUExecutionProvider")

    # CoreML has known issues with dynamic int8 quantized ViT models, producing NaNs.
    if "CoreMLExecutionProvider" in providers and "_int8" in model_path:
        logger.warning(
            "Disabling CoreMLExecutionProvider because it produces NaNs with int8 models."
        )
        providers.remove("CoreMLExecutionProvider")

    try:
        opts = ort.SessionOptions()
        opts.log_severity_level = 3  # Error level
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

        # Optimize threading for Apple Silicon (usually 4-8 high performance cores)
        opts.intra_op_num_threads = max(1, os.cpu_count() or 4)
        opts.inter_op_num_threads = 1

        # CoreML specific provider options
        provider_options = []
        for p in providers:
            if p == "CoreMLExecutionProvider":
                provider_options.append({"MLComputeUnits": "ALL"})
            else:
                provider_options.append({})

        session = ort.InferenceSession(
            model_path, opts, providers, provider_options=provider_options
        )

        # Load image processor
        try:
            processor = await asyncio.to_thread(
                AutoImageProcessor.from_pretrained, model_name, local_files_only=True
            )
        except Exception:
            processor = await asyncio.to_thread(AutoImageProcessor.from_pretrained, model_name)

        model_description = f"{model_name} (ONNX)"
        logger.info(f"Stability model loaded: {model_description} on {providers[0]}")
        return session, processor, model_description

    except Exception as e:
        logger.error(f"Error loading ONNX model: {e}")
        return None, None, f"Error: {e}"


async def evaluate_stability(
    session: "ort.InferenceSession", processor: "AutoImageProcessor", image_path: Path
) -> Tuple[Optional[float], dict]:
    """
    Evaluate a single image and return the stability_score.
    """
    if not image_path.exists():
        logger.warning(f"Image not found: {image_path}")
        return None, {}

    try:
        return await asyncio.to_thread(_evaluate_blocking, session, processor, image_path)
    except Exception as e:
        logger.warning(f"Failed to calculate stability score for {image_path}: {e}")
        return None, {}


def _evaluate_blocking(
    session: "ort.InferenceSession", processor: "AutoImageProcessor", image_path: Path
) -> Tuple[float, dict]:
    """
    Blocking ONNX inference.
    """
    # Load and preprocess image
    img_original = Image.open(image_path)
    image = img_original.convert("RGB")

    # Process image with ViT processor
    inputs = processor(images=image, return_tensors="np")

    # Prepare ONNX inputs
    onnx_inputs = {session.get_inputs()[0].name: inputs["pixel_values"]}

    # Run inference
    onnx_outputs = session.run(None, onnx_inputs)
    logits = onnx_outputs[0]

    # Apply softmax to get probabilities
    probabilities = softmax(logits)[0]

    # Most models use: 0 -> stable, 1 -> unstable
    # (Checking labels like stable/unstable or normal)
    # label 0: normal (stable), label 1: unstable
    stable_score = float(probabilities[0])
    stability_score = float(probabilities[1])

    details = {
        "classification_result": "unstable" if stability_score > 0.5 else "stable",
        "content_probability": stability_score,
        "probabilities": probabilities.tolist(),
    }

    image.close()
    img_original.close()
    return stability_score, details
