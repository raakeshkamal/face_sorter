#!/usr/bin/env python3
"""
Test script for stability score detection using Vision Transformers with ONNX Runtime.

This script tests stability score detection by evaluating a single image and returning
stability_score (probability of being unstable) and stable_score (probability of being stable).
"""

import argparse
import sys
import os
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
from PIL import Image
from dotenv import load_dotenv

# Try to import ONNX Runtime and Transformers
try:
    import onnxruntime as ort
except ImportError:
    print("Error: onnxruntime is required. Install with: pip install onnxruntime", file=sys.stderr)
    sys.exit(1)

try:
    from transformers import AutoImageProcessor
except ImportError:
    print("Error: transformers is required. Install with: pip install transformers", file=sys.stderr)
    sys.exit(1)

# Load environment variables
load_dotenv()

# Configure Hugging Face settings to prioritize local/cache and suppress warnings
# Use /tmp for storing and loading models
os.environ["HF_HOME"] = "/tmp/huggingface"
os.environ["HF_HUB_CACHE"] = "/tmp/huggingface/hub"
os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
if not os.environ.get("HF_TOKEN"):
    os.environ["HF_TOKEN"] = ""

import warnings
warnings.filterwarnings("ignore", message=".*HF_TOKEN.*")


def load_model(
    model_path: Optional[str] = None,
    model_name: Optional[str] = None,
    providers: Optional[list] = None
) -> Tuple[ort.InferenceSession, AutoImageProcessor, str]:
    """
    Load the ONNX model and image processor for stability score detection.
    If the model_path is not found, attempts to auto-download and export using Optimum.

    Args:
        model_path: Path to the ONNX model file. If None, resolves from environment or auto-downloads.
        model_name: Hugging Face model ID. Default: Resolves from environment.
        providers: ONNX Runtime providers.

    Returns:
        Tuple of (inference_session, image_processor, model_description)
    """
    if not model_name:
        model_name = os.getenv("STABILITY_MODEL_NAME")

    if not model_path:
        model_path = os.getenv("STABILITY_ONNX_MODEL_PATH")

    # If model_path is still None, or the file doesn't exist, we'll try to use a default /tmp path
    # and auto-download if necessary.
    was_model_path_provided = bool(model_path)
    if not model_path:
        model_slug = model_name.replace("/", "--")
        model_dir = Path("/tmp/huggingface_onnx") / model_slug
        model_path = str(model_dir / "model_int8.onnx")

    path_obj = Path(model_path).resolve()
    model_path = str(path_obj)
    if not path_obj.exists():
        if was_model_path_provided:
            print(f"Warning: Specified model path {model_path} does not exist. Attempting auto-download...", file=sys.stderr)

        # Attempt to auto-download and export using optimum
        print(f"Auto-downloading and exporting '{model_name}' to ONNX format...")
        try:
            from optimum.onnxruntime import ORTModelForImageClassification
            from onnxruntime.quantization import quantize_dynamic, QuantType
            
            # Export to the directory containing the requested model_path
            export_dir = path_obj.parent
            export_dir.mkdir(parents=True, exist_ok=True)
            
            print(f"Exporting model to {export_dir} (this may take a minute)...")
            # We use from_pretrained with export=True to get the ONNX version
            ort_model = ORTModelForImageClassification.from_pretrained(model_name, export=True)
            ort_model.save_pretrained(export_dir)
            
            exported_model = export_dir / "model.onnx"
            
            if exported_model.exists():
                if str(exported_model) == str(path_obj):
                    target_path = exported_model.with_name(exported_model.stem + "_int8.onnx")
                else:
                    target_path = path_obj
                    
                print("Quantizing model for faster inference...")
                quantize_dynamic(
                    model_input=str(exported_model),
                    model_output=str(target_path),
                    weight_type=QuantType.QUInt8
                )
                
                model_path = str(target_path)
                path_obj = target_path
                
            print(f"Successfully exported and quantized {model_name} to {model_path}")
        except ImportError:
            print("Error: optimum is required for auto-download. Install with: pip install optimum[exporters]", file=sys.stderr)
            sys.exit(1)
        except Exception as e:
            print(f"Error during auto-download/export: {e}", file=sys.stderr)
            sys.exit(1)

    # Auto-detect providers if not specified
    if providers is None:
        available_providers = ort.get_available_providers()
        # Prefer CoreML if available on macOS, then CUDA, then CPU
        providers = []
        if 'CoreMLExecutionProvider' in available_providers:
            providers.append('CoreMLExecutionProvider')
        if 'CUDAExecutionProvider' in available_providers:
            providers.append('CUDAExecutionProvider')
        providers.append('CPUExecutionProvider')

    print(f"Loading stability detection model from: {model_path}")
    print(f"Using image processor from: {model_name}")

    # CoreML has known issues with dynamic int8 quantized ViT models, producing NaNs.
    if 'CoreMLExecutionProvider' in providers and '_int8' in model_path:
        print("Warning: Disabling CoreMLExecutionProvider because it produces NaNs with int8 models.", file=sys.stderr)
        providers.remove('CoreMLExecutionProvider')

    print(f"ONNX Runtime providers: {providers}")

    try:
        # Load ONNX model
        # Set session options to suppress INFO logs
        opts = ort.SessionOptions()
        opts.log_severity_level = 3  # Error level
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        
        # Optimize threading for Apple Silicon (usually 4-8 high performance cores)
        opts.intra_op_num_threads = max(1, os.cpu_count() or 4)
        opts.inter_op_num_threads = 1
        
        # CoreML specific provider options
        provider_options = []
        for p in providers:
            if p == 'CoreMLExecutionProvider':
                provider_options.append({'MLComputeUnits': 'ALL'})
            else:
                provider_options.append({})

        session = ort.InferenceSession(model_path, opts, providers, provider_options=provider_options)

        # Load image processor - try local first if model_name looks like a path or if requested
        try:
            processor = AutoImageProcessor.from_pretrained(model_name, local_files_only=True)
        except Exception:
            # Fallback to normal loading if local_files_only fails (e.g. first time)
            processor = AutoImageProcessor.from_pretrained(model_name)

        model_description = f"{model_name} (via ONNX Runtime)"
        print(f"Model loaded: {model_description}")
        print(f"Input shape: {session.get_inputs()[0].shape}")

        return session, processor, model_description

    except Exception as e:
        print(f"Error loading model: {e}", file=sys.stderr)
        sys.exit(1)


def softmax(x: np.ndarray) -> np.ndarray:
    """
    Compute softmax of array x.

    Args:
        x: Input array.

    Returns:
        Softmax probabilities.
    """
    e_x = np.exp(x - np.max(x, axis=1, keepdims=True))
    return e_x / e_x.sum(axis=1, keepdims=True)


def evaluate_image(
    session: ort.InferenceSession,
    processor: AutoImageProcessor,
    image_path: Path
) -> Tuple[float, float, dict]:
    """
    Evaluate a single image and return stability and stable scores.

    Args:
        session: The ONNX Runtime inference session.
        processor: The image processor.
        image_path: Path to the image file.

    Returns:
        Tuple of (stability_score, stable_score, details_dict)

    Raises:
        FileNotFoundError: If image_path doesn't exist.
        Exception: If image processing fails.
    """
    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")

    print(f"Processing image: {image_path}")

    try:
        # Load and preprocess image
        image = Image.open(image_path).convert("RGB")

        # Process image with ViT processor
        inputs = processor(images=image, return_tensors="np")

        # Prepare ONNX inputs
        onnx_inputs = {session.get_inputs()[0].name: inputs['pixel_values']}

        # Run inference
        onnx_outputs = session.run(None, onnx_inputs)
        logits = onnx_outputs[0]

        # Apply softmax to get probabilities
        probabilities = softmax(logits)[0]

        # Most models use: 0 -> stable, 1 -> unstable
        stable_score = float(probabilities[0])
        stability_score = float(probabilities[1])

        details = {
            "logits": logits[0].tolist(),
            "probabilities": probabilities.tolist(),
            "predicted_class": int(np.argmax(probabilities)),
        }

        return stability_score, stable_score, details

    except Exception as e:
        print(f"Error: Failed to process image - {e}", file=sys.stderr)
        raise


def parse_arguments() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Test stability score detection using Vision Transformers with ONNX Runtime",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Use default environment variables
  python scripts/test_stability_score_detection.py --image /absolute/path/to/image.jpg

  # Specify model path and model name directly
  python scripts/test_stability_score_detection.py --image /absolute/path/to/image.jpg \\
      --model-path stability_onnx_model/model.onnx \\
      --model-name your-model-name-here

  # Use GPU if available
  python scripts/test_stability_score_detection.py --image /absolute/path/to/image.jpg --use-gpu
        """
    )
    parser.add_argument(
        "--image",
        type=str,
        required=True,
        help="Absolute path to the image file to classify"
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default=None,
        help="Path to ONNX model file (default: STABILITY_ONNX_MODEL_PATH env var)"
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default=None,
        help="Hugging Face model ID for processor (default: STABILITY_MODEL_NAME env var)"
    )
    parser.add_argument(
        "--use-gpu",
        action="store_true",
        help="Use GPU (CUDA) if available"
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
        # Check if it exists relatively just in case, but follow the absolute path rule of the original script
        if not image_path.exists():
            return 1

    # Determine providers based on --use-gpu flag
    providers = None
    if args.use_gpu:
        available_providers = ort.get_available_providers()
        providers = []
        if 'CoreMLExecutionProvider' in available_providers:
            providers.append('CoreMLExecutionProvider')
        if 'CUDAExecutionProvider' in available_providers:
            providers.append('CUDAExecutionProvider')
        if not providers:
            providers.append('CPUExecutionProvider')
    # If not --use-gpu, providers stays None, which allows auto-detection in load_model

    # Load model
    session, processor, model_desc = load_model(
        model_path=args.model_path,
        model_name=args.model_name,
        providers=providers
    )

    # Evaluate image
    try:
        stability_score, stable_score, details = evaluate_image(session, processor, image_path)

        # Display results
        print("\n" + "="*60)
        print("Stability Detection Results")
        print("="*60)
        print(f"Model: {model_desc}")
        print(f"Image: {image_path}")
        print(f"\nStability Score: {stability_score:.4f}")
        print(f"Stable Score:    {stable_score:.4f}")
        print(f"\nPredicted class: {'Unstable' if stability_score > 0.5 else 'Stable'}")
        print(f"Probabilities: {details['probabilities']}")
        print("="*60)

        return 0

    except FileNotFoundError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1
    except Exception as e:
        print(f"Error: Failed to process image - {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
