"""Flet Image Creator - AI Image Generation with Qwen Model.

This package provides a GUI application for generating images using the
Qwen diffusion model from HuggingFace.

Modules:
    qwenimage_fletui: Main application module containing the UI and generation logic.

Example:
    >>> import flet as ft
    >>> from src.qwenimage_fletui import main
    >>> ft.app(target=main)
"""

from src.qwenimage_fletui import (
    ASPECT_RATIOS,
    DEFAULT_ENHANCEMENT,
    DEFAULT_SEED,
    MODEL_NAME,
    GenerationParams,
    ImageGenerationOrchestrator,
    ImageSaver,
    ModelConfig,
    ModelLoader,
    QwenImageGenerator,
    UIUpdater,
    convert_image_to_base64,
    enhance_prompt,
    generate_filename,
    get_optimal_device,
    main,
    validate_generation_params,
)

__all__ = [
    # Constants
    "MODEL_NAME",
    "ASPECT_RATIOS",
    "DEFAULT_ENHANCEMENT",
    "DEFAULT_SEED",
    # Data classes
    "GenerationParams",
    "ModelConfig",
    # Classes
    "ModelLoader",
    "ImageSaver",
    "UIUpdater",
    "ImageGenerationOrchestrator",
    "QwenImageGenerator",
    # Functions
    "get_optimal_device",
    "validate_generation_params",
    "enhance_prompt",
    "convert_image_to_base64",
    "generate_filename",
    "main",
]

__version__ = "0.1.0"
