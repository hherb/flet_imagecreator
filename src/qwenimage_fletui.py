"""Flet-based GUI application for AI image generation using the Qwen diffusion model.

This module provides a complete user interface for text-to-image generation with
customizable parameters including aspect ratio, inference steps, CFG scale, and seed.
The application features real-time progress tracking, responsive image display,
and automatic device detection for optimal performance.

Typical usage:
    python src/qwenimage_fletui.py

Or programmatically:
    import flet as ft
    from qwenimage_fletui import main
    ft.app(target=main)
"""

import base64
import datetime
import io
import threading
from dataclasses import dataclass
from typing import Any, Callable, Dict, Optional, Tuple, Type, Union

import flet as ft
import torch
from diffusers import DiffusionPipeline
from PIL import Image

try:
    from diffusers.pipelines.qwen_image import QwenImagePipeline
except ImportError:
    # Fallback if the import path is different or not yet available
    QwenImagePipeline: Optional[Type[DiffusionPipeline]] = None

MODEL_NAME: str = "Qwen/Qwen-Image"
"""Default HuggingFace model identifier for the Qwen image generation model."""

# Centralized aspect ratio definitions to avoid duplication
ASPECT_RATIOS: Dict[str, Tuple[int, int]] = {
    "1:1 (Square)": (1328, 1328),
    "16:9 (Widescreen)": (1664, 928),
    "9:16 (Portrait)": (928, 1664),
    "4:3 (Standard)": (1472, 1140),
    "3:4 (Portrait Standard)": (1140, 1472),
}
"""Available aspect ratios with their corresponding (width, height) in pixels."""

DEFAULT_ENHANCEMENT: str = "Ultra HD, 4K, cinematic composition."
"""Default quality enhancement text appended to user prompts."""

DEFAULT_SEED: int = 42
"""Default seed value for reproducible image generation."""


# =============================================================================
# PURE FUNCTIONS - Business Logic Layer
# =============================================================================


@dataclass(frozen=True)
class GenerationParams:
    """Immutable container for validated image generation parameters.

    Attributes:
        prompt: The main text description for image generation.
        negative_prompt: Text describing what to avoid in the generated image.
        enhancement: Quality modifiers appended to the prompt.
        width: Output image width in pixels.
        height: Output image height in pixels.
        num_inference_steps: Number of denoising steps (higher = better quality, slower).
        cfg_scale: Classifier-free guidance scale (higher = more prompt adherence).
        seed: Random seed for reproducible generation.
    """

    prompt: str
    negative_prompt: str
    enhancement: str
    width: int
    height: int
    num_inference_steps: int
    cfg_scale: float
    seed: int


@dataclass
class ModelConfig:
    """Configuration for model loading and device settings.

    Attributes:
        device: Target compute device ('mps', 'cuda', or 'cpu').
        torch_dtype: PyTorch data type for model weights (e.g., torch.bfloat16).
    """

    device: str
    torch_dtype: torch.dtype

def get_optimal_device() -> Tuple[str, torch.dtype]:
    """Detect the best available compute device and appropriate data type.

    Checks for hardware acceleration in order of preference:
    1. MPS (Apple Silicon) - Best for Mac with M-series chips
    2. CUDA (NVIDIA GPU) - Best for systems with NVIDIA graphics cards
    3. CPU - Fallback for systems without GPU acceleration

    Returns:
        A tuple of (device_name, torch_dtype) where device_name is one of
        'mps', 'cuda', or 'cpu', and torch_dtype is the recommended data type
        for that device (bfloat16 for accelerated devices, float32 for CPU).

    Example:
        >>> device, dtype = get_optimal_device()
        >>> print(f"Using {device} with {dtype}")
        Using cuda with torch.bfloat16
    """
    if torch.backends.mps.is_available():
        return "mps", torch.bfloat16
    elif torch.cuda.is_available():
        return "cuda", torch.bfloat16
    else:
        return "cpu", torch.float32


def _parse_seed(seed_value: Any) -> int:
    """Parse and validate a seed value from user input.

    Args:
        seed_value: The seed value to parse (can be string, int, or None).

    Returns:
        A valid integer seed, defaulting to DEFAULT_SEED if parsing fails.
    """
    if seed_value is None:
        return DEFAULT_SEED

    try:
        seed_int = int(seed_value)
        # Ensure seed is non-negative (PyTorch requirement)
        return seed_int if seed_int >= 0 else DEFAULT_SEED
    except (ValueError, TypeError):
        return DEFAULT_SEED


def validate_generation_params(params: Dict[str, Any]) -> GenerationParams:
    """Validate UI input and convert to structured generation parameters.

    Performs validation and type conversion on raw UI parameters, providing
    sensible defaults for missing or invalid values.

    Args:
        params: Dictionary containing raw UI parameters with keys:
            - prompt: Text description for image generation
            - negative_prompt: Text describing what to avoid
            - enhancement: Quality enhancement text
            - aspect_ratio: Key from ASPECT_RATIOS dictionary
            - seed: Random seed (string or int)
            - steps: Number of inference steps
            - cfg_scale: Classifier-free guidance scale

    Returns:
        A validated GenerationParams instance with all parameters
        properly typed and defaulted.

    Example:
        >>> params = {"prompt": "A cat", "aspect_ratio": "1:1 (Square)"}
        >>> gen_params = validate_generation_params(params)
        >>> gen_params.width, gen_params.height
        (1328, 1328)
    """
    # Use centralized aspect ratio definitions
    aspect_ratio_key = params.get("aspect_ratio", "16:9 (Widescreen)")
    width, height = ASPECT_RATIOS.get(aspect_ratio_key, (1664, 928))

    return GenerationParams(
        prompt=str(params.get("prompt", "")),
        negative_prompt=str(params.get("negative_prompt", "")),
        enhancement=str(params.get("enhancement", DEFAULT_ENHANCEMENT)),
        width=width,
        height=height,
        num_inference_steps=int(params.get("steps", 50)),
        cfg_scale=float(params.get("cfg_scale", 4.0)),
        seed=_parse_seed(params.get("seed")),
    )

def enhance_prompt(prompt: str, enhancement: str = DEFAULT_ENHANCEMENT) -> str:
    """Append quality enhancement text to the user's prompt.

    Quality modifiers like 'Ultra HD, 4K' help guide the model toward
    higher quality outputs with better detail and composition.

    Args:
        prompt: The user's original text prompt.
        enhancement: Quality enhancement text to append. Defaults to
            DEFAULT_ENHANCEMENT. Empty or whitespace-only values are ignored.

    Returns:
        The original prompt with enhancement appended (space-separated),
        or just the original prompt if enhancement is empty.

    Example:
        >>> enhance_prompt("A cat sleeping")
        'A cat sleeping Ultra HD, 4K, cinematic composition.'
        >>> enhance_prompt("A cat", "")
        'A cat'
    """
    enhancement_stripped = enhancement.strip()
    if enhancement_stripped:
        return f"{prompt} {enhancement_stripped}"
    return prompt


def calculate_progress(current_step: int, total_steps: int) -> float:
    """Calculate completion percentage for progress bars.

    Args:
        current_step: The current step number (1-indexed typically).
        total_steps: The total number of steps in the process.

    Returns:
        A float between 0.0 and 1.0 representing completion percentage.
        Returns 0.0 if total_steps is zero or negative to avoid division errors.

    Example:
        >>> calculate_progress(25, 50)
        0.5
        >>> calculate_progress(0, 0)
        0.0
    """
    if total_steps <= 0:
        return 0.0
    return current_step / total_steps


def create_progress_message(current_step: int, total_steps: int) -> str:
    """Format progress text for display during generation.

    Args:
        current_step: The current step number.
        total_steps: The total number of steps.

    Returns:
        A human-readable progress string like "Step 25 / 50".
    """
    return f"Step {current_step} / {total_steps}"


def generate_filename(prefix: str = "qwen_generated", extension: str = "png") -> str:
    """Create a unique timestamped filename for saved images.

    Generates filenames in the format: {prefix}_{YYYYMMDD}_{HHMMSS}.{extension}
    This prevents filename conflicts when saving multiple images.

    Args:
        prefix: Filename prefix. Defaults to "qwen_generated".
        extension: File extension without dot. Defaults to "png".

    Returns:
        A unique filename string with current timestamp.

    Example:
        >>> generate_filename()
        'qwen_generated_20250115_143022.png'
    """
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"{prefix}_{timestamp}.{extension}"


def convert_image_to_base64(image: Image.Image) -> str:
    """Encode PIL Image as base64 string for Flet display.

    Flet UI requires base64-encoded images for the src_base64 parameter.
    This function always uses PNG format to preserve image quality.

    Args:
        image: A PIL Image object to encode.

    Returns:
        A base64-encoded string representation of the image.

    Raises:
        ValueError: If the image cannot be saved to PNG format.
    """
    img_buffer = io.BytesIO()
    image.save(img_buffer, format="PNG")
    img_buffer.seek(0)
    return base64.b64encode(img_buffer.getvalue()).decode("utf-8")


def create_progress_display_data(current_step: int, total_steps: int) -> Dict[str, Any]:
    """Build progress display data structure for UI updates.

    Creates a dictionary containing all necessary data for updating
    the progress UI components during image generation.

    Args:
        current_step: The current generation step (1-indexed).
        total_steps: The total number of inference steps.

    Returns:
        A dictionary containing:
            - progress_value: Float between 0.0 and 1.0
            - progress_text: Human-readable step counter
            - icon: Flet icon identifier for display
            - message: Primary status message
            - sub_message: Secondary informational message

    Note:
        The Qwen model doesn't support intermediate image previews during
        generation, so the sub_message reflects this limitation.
    """
    return {
        "progress_value": calculate_progress(current_step, total_steps),
        "progress_text": create_progress_message(current_step, total_steps),
        "icon": ft.Icons.HOURGLASS_EMPTY,
        "message": f"Generating step {current_step}/{total_steps}",
        "sub_message": "Preview not available for Qwen model",
    }

# =============================================================================
# SIDE EFFECTS - I/O Operations Layer
# =============================================================================


class ModelLoader:
    """Handle diffusion model loading and device configuration.

    This class provides static methods for loading the Qwen diffusion pipeline
    and creating seeded random number generators. It abstracts away the
    complexity of model initialization and device management.

    The class attempts to load the specialized QwenImagePipeline first for
    full feature support, falling back to the generic DiffusionPipeline
    if the specialized version is unavailable.
    """

    @staticmethod
    def load_diffusion_pipeline(
        model_name: str, config: ModelConfig
    ) -> DiffusionPipeline:
        """Load the Qwen diffusion pipeline with fallback handling.

        Attempts to load QwenImagePipeline first for future image editing support,
        then falls back to generic DiffusionPipeline if the specific pipeline
        is unavailable. Automatically moves the model to the specified device.

        Args:
            model_name: HuggingFace model identifier (e.g., "Qwen/Qwen-Image").
            config: ModelConfig containing device and dtype settings.

        Returns:
            A loaded DiffusionPipeline instance on the specified device.

        Raises:
            Exception: If model loading fails completely (no fallback succeeds).

        Note:
            Model download can take significant time on first run as the
            model weights are several gigabytes in size.
        """
        # Try to load as QwenImagePipeline first for image editing support
        if QwenImagePipeline is not None:
            try:
                pipe = QwenImagePipeline.from_pretrained(
                    model_name, torch_dtype=config.torch_dtype
                )
                print("Loaded as QwenImagePipeline - image editing supported")
                return pipe.to(config.device)
            except Exception as e:
                print(f"Failed to load as QwenImagePipeline: {e}")

        # Fallback to generic DiffusionPipeline
        pipe = DiffusionPipeline.from_pretrained(
            model_name, torch_dtype=config.torch_dtype
        )
        print(f"Loaded as {type(pipe).__name__}")
        return pipe.to(config.device)

    @staticmethod
    def create_generator(device: str, seed: int) -> torch.Generator:
        """Create a seeded random number generator for reproducible generation.

        The generator must match the model's device (MPS/CUDA/CPU) to avoid
        tensor device mismatch errors during generation.

        Args:
            device: Target device name ('mps', 'cuda', or 'cpu').
            seed: Random seed for reproducibility. Same seed produces
                identical results across runs.

        Returns:
            A PyTorch Generator initialized on the specified device with
            the given seed.

        Example:
            >>> gen = ModelLoader.create_generator("cuda", 42)
            >>> # Use gen in pipeline for reproducible results
        """
        return torch.Generator(device=device).manual_seed(seed)

class ImageSaver:
    """Handle saving generated images to disk.

    Provides a simple interface for persisting PIL Image objects to the
    filesystem. Supports automatic format detection based on file extension.
    """

    @staticmethod
    def save_image(image: Image.Image, filename: str) -> str:
        """Save PIL Image to filesystem with automatic format detection.

        The output format is determined by the file extension. PNG is
        recommended for quality preservation without compression artifacts.

        Args:
            image: The PIL Image object to save.
            filename: Target filename including extension (e.g., "output.png").

        Returns:
            The filename that was saved to (same as input).

        Raises:
            OSError: If the file cannot be written (permissions, disk space, etc.).
            ValueError: If the format cannot be determined from the extension.

        Example:
            >>> ImageSaver.save_image(my_image, "generated_art.png")
            'generated_art.png'
        """
        image.save(filename)
        return filename

class UIUpdater:
    """Manage UI updates and page refreshes in a centralized way.

    This class provides a consistent interface for updating various UI components
    (status text, progress bars, image displays) while ensuring proper
    page.update() calls for Flet to refresh the display.

    All methods automatically call page.update() after making changes,
    so callers don't need to manually trigger refreshes.

    Attributes:
        page: The Flet Page instance to update.
    """

    def __init__(self, page: ft.Page) -> None:
        """Initialize the UIUpdater with a Flet page reference.

        Args:
            page: The Flet Page instance that will be updated.
        """
        self.page: ft.Page = page

    def update_status(self, message: str, status_component: ft.Text) -> None:
        """Update status text component and refresh the UI.

        Use this method for user feedback during loading, errors,
        and completion states.

        Args:
            message: The status message to display.
            status_component: The Flet Text component to update.
        """
        status_component.value = message
        self.page.update()

    def update_progress(
        self,
        progress_data: Dict[str, Any],
        progress_components: Dict[str, ft.Control],
    ) -> None:
        """Update progress bar, text, and preview components during generation.

        Args:
            progress_data: Dictionary containing:
                - progress_value: Float 0.0-1.0 for progress bar
                - progress_text: String for step counter display
                - icon: Flet icon identifier
                - message: Primary status message
                - sub_message: Secondary informational message
            progress_components: Dictionary containing:
                - 'bar': ProgressBar control
                - 'text': Text control for step display
                - 'preview': Container for preview content
        """
        progress_components["bar"].value = progress_data["progress_value"]
        progress_components["text"].value = progress_data["progress_text"]

        progress_components["preview"].content = ft.Column(
            [
                ft.Icon(progress_data["icon"], size=40, color=ft.Colors.BLUE_400),
                ft.Text(
                    progress_data["message"], text_align=ft.TextAlign.CENTER, size=12
                ),
                ft.Text(
                    progress_data["sub_message"],
                    text_align=ft.TextAlign.CENTER,
                    size=10,
                    color=ft.Colors.GREY_600,
                ),
            ],
            horizontal_alignment=ft.CrossAxisAlignment.CENTER,
        )

        self.page.update()

    def show_progress_components(self, components: Dict[str, ft.Control]) -> None:
        """Make progress components visible during generation.

        Args:
            components: Dictionary of Flet controls to show.
        """
        for comp in components.values():
            comp.visible = True
        self.page.update()

    def hide_progress_components(self, components: Dict[str, ft.Control]) -> None:
        """Hide progress components after generation completes.

        Args:
            components: Dictionary of Flet controls to hide.
        """
        for comp in components.values():
            comp.visible = False
        self.page.update()

    def update_image_display(
        self, image_base64: str, image_container: ft.Container
    ) -> None:
        """Display generated image with responsive scaling and aspect ratio preservation.

        Creates a new container with the image, replacing any previous content.
        Uses CONTAIN fit to prevent distortion while filling available space.

        Args:
            image_base64: Base64-encoded PNG image string.
            image_container: The Flet Container to place the image in.
        """
        image_display = ft.Container(
            content=ft.Image(
                src_base64=image_base64,
                fit=ft.ImageFit.CONTAIN,  # Maintains aspect ratio, scales to fit
                expand=True,
            ),
            expand=True,
            alignment=ft.alignment.center,
            bgcolor=ft.Colors.BLACK12,
            border_radius=8,
            clip_behavior=ft.ClipBehavior.HARD_EDGE,
        )
        image_container.content = image_display
        self.page.update()

# =============================================================================
# ORCHESTRATION LAYER - Coordinates Logic and Side Effects
# =============================================================================


class ImageGenerationOrchestrator:
    """Coordinate image generation workflow with UI updates.

    This class manages the storage and saving of generated images. It acts as
    a bridge between the generation logic and the UI, maintaining state about
    the most recently generated image for save operations.

    Attributes:
        ui_updater: UIUpdater instance for refreshing the display.
        current_image: The most recently generated image, or None if no
            image has been generated yet.
    """

    def __init__(self, ui_updater: UIUpdater) -> None:
        """Initialize the orchestrator with a UI updater.

        Args:
            ui_updater: UIUpdater instance for managing display updates.
        """
        self.ui_updater: UIUpdater = ui_updater
        self.current_image: Optional[Image.Image] = None

    def save_current_image(self) -> Optional[str]:
        """Save the most recently generated image with timestamp filename.

        Creates a unique filename using the current timestamp to prevent
        overwrites when saving multiple images.

        Returns:
            The filename if successful, None if no image is available to save.

        Example:
            >>> filename = orchestrator.save_current_image()
            >>> if filename:
            ...     print(f"Saved to {filename}")
            Saved to qwen_generated_20250115_143022.png
        """
        if self.current_image is None:
            return None
        filename = generate_filename()
        ImageSaver.save_image(self.current_image, filename)
        return filename

class QwenImageGenerator:
    """Main interface for Qwen image generation model.

    This class handles model loading, device management, and image generation.
    It provides a high-level API for generating images from text prompts using
    the Qwen diffusion model from HuggingFace.

    Currently supports text-to-image generation only. Image editing capabilities
    will be added in future updates when the Qwen model supports them.

    Attributes:
        pipe: The loaded diffusion pipeline, or None if not yet loaded.
        device: The compute device being used ('mps', 'cuda', or 'cpu').
        torch_dtype: The PyTorch data type for model weights.
        model_loaded: Whether the model has been successfully loaded.

    Example:
        >>> generator = QwenImageGenerator()
        >>> generator.load_model(progress_callback=print)
        >>> params = GenerationParams(
        ...     prompt="A cat", negative_prompt="", enhancement="",
        ...     width=1024, height=1024, num_inference_steps=50,
        ...     cfg_scale=4.0, seed=42
        ... )
        >>> image = generator.generate_image(params)
    """

    def __init__(self) -> None:
        """Initialize the generator with default (unloaded) state."""
        self.pipe: Optional[DiffusionPipeline] = None
        self.device: Optional[str] = None
        self.torch_dtype: Optional[torch.dtype] = None
        self.model_loaded: bool = False

    def load_model(
        self, progress_callback: Optional[Callable[[str], None]] = None
    ) -> None:
        """Download and initialize the Qwen diffusion model.

        Automatically detects the best available device (MPS/CUDA/CPU) and
        configures appropriate data types for optimal performance. The model
        is only loaded once; subsequent calls are no-ops.

        Args:
            progress_callback: Optional function to receive status updates
                during loading. Called with string messages like
                "Loading model..." and "Model loaded successfully!".

        Note:
            First-time loading can take several minutes as the model
            weights (several GB) are downloaded from HuggingFace.
        """
        if self.model_loaded:
            return

        # Use pure function to determine device config
        self.device, self.torch_dtype = get_optimal_device()

        if progress_callback:
            progress_callback(f"Using device: {self.device} with dtype: {self.torch_dtype}")
            progress_callback("Loading model...")

        # Use side effect class to load pipeline
        config = ModelConfig(self.device, self.torch_dtype)
        self.pipe = ModelLoader.load_diffusion_pipeline(MODEL_NAME, config)

        self.model_loaded = True
        if progress_callback:
            progress_callback("Model loaded successfully!")

    def generate_image(
        self,
        params: GenerationParams,
        progress_callback: Optional[Callable[[Dict[str, Any]], None]] = None,
    ) -> Image.Image:
        """Generate image from text prompt using the loaded Qwen model.

        Args:
            params: Validated generation parameters including prompt,
                dimensions, inference steps, CFG scale, and seed.
            progress_callback: Optional function to receive progress updates
                during generation. Called with a dictionary containing
                progress_value, progress_text, icon, message, and sub_message.

        Returns:
            The generated PIL Image.

        Raises:
            RuntimeError: If the model has not been loaded yet.
            Exception: If image generation fails for any reason.

        Note:
            Generation time depends on the number of inference steps and
            the compute device. GPU acceleration significantly reduces time.
        """
        if not self.model_loaded or self.pipe is None or self.device is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")

        # Create progress callback for pipeline
        def step_callback(
            pipe: Any,
            step_index: int,
            timestep: Any,
            callback_kwargs: Dict[str, Any],
        ) -> Dict[str, Any]:
            if progress_callback:
                progress_data = create_progress_display_data(
                    step_index + 1, params.num_inference_steps
                )
                progress_callback(progress_data)
            return callback_kwargs

        # Generate image with enhanced prompt
        enhanced_prompt = enhance_prompt(params.prompt, params.enhancement)
        torch_generator = ModelLoader.create_generator(self.device, params.seed)

        try:
            # Qwen-Image currently only supports text-to-image generation
            image = self.pipe(
                prompt=enhanced_prompt,
                negative_prompt=params.negative_prompt,
                width=params.width,
                height=params.height,
                num_inference_steps=params.num_inference_steps,
                true_cfg_scale=params.cfg_scale,
                generator=torch_generator,
                callback_on_step_end=step_callback if progress_callback else None,
                callback_on_step_end_tensor_inputs=["latents"]
                if progress_callback
                else None,
            ).images[0]
        except TypeError as e:
            print(f"Callback parameters not supported: {e}")
            # Fallback without callback if not supported
            image = self.pipe(
                prompt=enhanced_prompt,
                negative_prompt=params.negative_prompt,
                width=params.width,
                height=params.height,
                num_inference_steps=params.num_inference_steps,
                true_cfg_scale=params.cfg_scale,
                generator=torch_generator,
            ).images[0]

        return image

def main(page: ft.Page) -> None:
    """Main Flet application entry point.

    Sets up the complete UI including model controls, parameter inputs,
    progress tracking, and image display. Handles all user interactions
    and coordinates between UI and generation logic.

    The UI is organized into:
    - A collapsible configuration section with prompt inputs and parameters
    - A progress section that appears during generation
    - An image display area that expands to fill available space

    Args:
        page: Flet page object for UI rendering.
    """
    page.title = f"Image Generator (using model {MODEL_NAME})"
    page.theme_mode = ft.ThemeMode.LIGHT
    page.window.width = 1200
    page.window.height = 800

    # Initialize components using architecture layers
    generator = QwenImageGenerator()
    ui_updater = UIUpdater(page)
    orchestrator = ImageGenerationOrchestrator(ui_updater)

    # State for collapsible configuration section
    config_collapsed: bool = False

    # UI Components - Prompt inputs
    prompt_field = ft.TextField(
        label="Prompt",
        multiline=True,
        min_lines=2,
        max_lines=3,
        value=(
            "A beautiful lake in a forest grove in the Alps. "
            "Snow capped mountains in the background. "
            "Mossy ground with mushrooms. Little dwarfs frolicking in the moss"
        ),
        expand=True,
    )

    negative_prompt_field = ft.TextField(
        label="Negative Prompt",
        multiline=True,
        min_lines=2,
        max_lines=3,
        value="",
        expand=True,
    )

    enhancement_field = ft.TextField(
        label="Enhancement",
        value=DEFAULT_ENHANCEMENT,
        expand=True,
    )

    # Aspect ratio selection - uses centralized ASPECT_RATIOS constant
    aspect_dropdown = ft.Dropdown(
        label="Aspect Ratio",
        value="16:9 (Widescreen)",
        options=[ft.dropdown.Option(key) for key in ASPECT_RATIOS.keys()],
        width=200,
    )

    steps_slider = ft.Slider(
        min=10,
        max=100,
        divisions=18,
        value=50,
        label="Steps: {value}",
        width=200,
    )

    cfg_slider = ft.Slider(
        min=1.0,
        max=10.0,
        divisions=18,
        value=4.0,
        label="CFG Scale: {value}",
        width=200,
    )

    seed_field = ft.TextField(
        label="Seed",
        value=str(DEFAULT_SEED),
        width=100,
    )

    def on_random_seed_click(_: ft.ControlEvent) -> None:
        """Generate a random seed and update the seed field."""
        random_seed = torch.randint(0, 2**32 - 1, (1,)).item()
        seed_field.value = str(random_seed)
        page.update()

    random_seed_btn = ft.ElevatedButton(
        "Random",
        on_click=on_random_seed_click,
        width=80,
    )
    
    
    # Test button to check if click events work
    def test_button_click(_: ft.ControlEvent) -> None:
        """Test function to verify UI interactions are working."""
        ui_updater.update_status("Button click detected!", status_text)
        print("Button clicked successfully")
    
    # Add a test button to verify click functionality
    test_btn = ft.ElevatedButton(
        "Test Click",
        icon=ft.Icons.BUG_REPORT,
        on_click=test_button_click,
        width=100
    )
    
    # Status and progress
    status_text = ft.Text("Ready to generate images", size=14)
    
    # Progress components
    progress_bar = ft.ProgressBar(
        width=400,
        color=ft.Colors.BLUE_400,
        bgcolor=ft.Colors.GREY_300,
        value=0,
        visible=False
    )
    
    progress_text = ft.Text(
        "Step 0 / 0", 
        size=12,
        visible=False
    )
    
    # Preview image container for intermediate steps
    preview_container = ft.Container(
        content=ft.Text("Generation preview will appear here", text_align=ft.TextAlign.CENTER),
        border=ft.border.all(1, ft.Colors.GREY_300),
        alignment=ft.alignment.center,
        border_radius=8,
        height=200,
        visible=False
    )
    
    # Image display - properly sized container that maintains aspect ratio
    image_container = ft.Container(
        content=ft.Text("Generated image will appear here", text_align=ft.TextAlign.CENTER, size=16, color=ft.Colors.GREY_600),
        border=ft.border.all(1, ft.Colors.GREY_400),
        alignment=ft.alignment.center,
        border_radius=8,
        expand=True,
        bgcolor=ft.Colors.GREY_50
    )
    
    
    # Buttons
    load_model_btn = ft.ElevatedButton(
        "Load Model",
        icon=ft.Icons.DOWNLOAD,
        width=120
    )
    
    generate_btn = ft.ElevatedButton(
        "Generate Image",
        icon=ft.Icons.BRUSH,
        disabled=True,
        width=140
    )
    
    save_btn = ft.ElevatedButton(
        "Save Image",
        icon=ft.Icons.SAVE,
        disabled=True,
        width=120
    )
    
    
    # Configuration section that can be collapsed
    config_container = ft.Container(
        content=ft.Column([
            # First row: Prompt and Negative Prompt side by side
            ft.Row([
                ft.Container(content=prompt_field, expand=True),
                ft.Container(content=negative_prompt_field, expand=True)
            ], spacing=10),
            
            # Enhancement field (full width)
            enhancement_field,
            
            # Information about upcoming features
            ft.Container(
                content=ft.Column([
                    ft.Text("ℹ️ Image Editing Coming Soon", size=14, weight=ft.FontWeight.BOLD, color=ft.Colors.BLUE_700),
                    ft.Text(
                        "Qwen-Image currently supports text-to-image generation only. " +
                        "Image editing capabilities will be added in a future release.",
                        size=12,
                        color=ft.Colors.GREY_700
                    ),
                    ft.Row([test_btn], spacing=10)
                ], spacing=10),
                bgcolor=ft.Colors.BLUE_50,
                padding=15,
                border_radius=8,
                border=ft.border.all(1, ft.Colors.BLUE_200)
            ),
            
            # Second row: Generation parameters
            ft.Row([
                aspect_dropdown,
                ft.Container(
                    content=ft.Column([
                        ft.Text("Inference Steps", size=12),
                        steps_slider
                    ]),
                    width=200
                ),
                ft.Container(
                    content=ft.Column([
                        ft.Text("CFG Scale", size=12),
                        cfg_slider
                    ]),
                    width=200
                ),
                ft.Row([
                    seed_field,
                    random_seed_btn
                ], spacing=5)
            ], spacing=20, alignment=ft.MainAxisAlignment.START),
            
            # Third row: Control buttons
            ft.Row([
                load_model_btn,
                generate_btn,
                save_btn
            ], spacing=10),
            
            # Status
            status_text
        ], spacing=15),
        padding=20,
        bgcolor=ft.Colors.BLUE_GREY_50,
        border_radius=8,
        border=ft.border.all(1, ft.Colors.BLUE_GREY_200),
        visible=True
    )
    
    # Toggle button for collapsing/expanding the configuration section
    def toggle_config_section(_: ft.ControlEvent) -> None:
        """Toggles visibility of configuration panel to save screen space.
        
        Updates button icon and tooltip text to reflect current state.
        """
        nonlocal config_collapsed
        config_collapsed = not config_collapsed
        config_container.visible = not config_collapsed
        
        # Update button icon
        if config_collapsed:
            toggle_btn.icon = ft.Icons.EXPAND_MORE
            toggle_btn.tooltip = "Show configuration"
        else:
            toggle_btn.icon = ft.Icons.EXPAND_LESS
            toggle_btn.tooltip = "Hide configuration"
        
        page.update()
    
    # Function to auto-collapse after generation
    def auto_collapse_after_generation() -> None:
        """Automatically hides configuration panel after successful generation.
        
        Provides more space for viewing the generated image.
        """
        nonlocal config_collapsed
        if not config_collapsed:
            config_collapsed = True
            config_container.visible = False
            toggle_btn.icon = ft.Icons.EXPAND_MORE
            toggle_btn.tooltip = "Show configuration"
            page.update()
    
    toggle_btn = ft.IconButton(
        icon=ft.Icons.EXPAND_LESS,
        tooltip="Hide configuration",
        on_click=toggle_config_section,
        icon_size=24
    )
    
    
    def load_model_thread() -> None:
        """Background thread for model loading to prevent UI blocking.
        
        Disables load button during process, updates UI with progress,
        enables generate button on success, shows error on failure.
        """
        try:
            load_model_btn.disabled = True
            page.update()
            
            def progress_callback(message: str) -> None:
                ui_updater.update_status(message, status_text)
            
            generator.load_model(progress_callback=progress_callback)
            
            load_model_btn.text = "Model Loaded"
            load_model_btn.icon = ft.Icons.CHECK
            generate_btn.disabled = False
            ui_updater.update_status("Model loaded successfully! Ready to generate images.", status_text)
            
        except Exception as e:
            ui_updater.update_status(f"Error loading model: {str(e)}", status_text)
            load_model_btn.disabled = False
        
        page.update()
    
    def generate_image_thread() -> None:
        """Background thread for image generation to prevent UI blocking.
        
        Manages complete generation workflow including progress display,
        error handling, and UI state management. Auto-collapses config on success.
        """
        try:
            print("Generate button clicked - starting thread")
            generate_btn.disabled = True
            save_btn.disabled = True
            
            # Show progress components using UI updater
            progress_components = {
                "bar": progress_bar,
                "text": progress_text, 
                "preview": preview_container
            }
            ui_updater.show_progress_components(progress_components)
            ui_updater.update_status("Generating image...", status_text)
            
            # Extract UI parameters using pure function
            ui_params = {
                "prompt": prompt_field.value,
                "negative_prompt": negative_prompt_field.value,
                "enhancement": enhancement_field.value,
                "aspect_ratio": aspect_dropdown.value,
                "seed": seed_field.value,
                "steps": steps_slider.value,
                "cfg_scale": cfg_slider.value
            }
            params = validate_generation_params(ui_params)
            print(f"Parameters: {params.width}x{params.height}, steps={params.num_inference_steps}, seed={params.seed}")
            
            # Progress callback function for UI updates during generation
            def on_progress(progress_data: Dict[str, Any]) -> None:
                ui_updater.update_progress(progress_data, progress_components)
            
            # Generate image using orchestrator
            image = generator.generate_image(params, progress_callback=on_progress)
            orchestrator.current_image = image
            
            # Convert and display image using pure functions and UI updater
            image_base64 = convert_image_to_base64(image)
            ui_updater.update_image_display(image_base64, image_container)
            
            # Hide progress and re-enable controls
            ui_updater.hide_progress_components(progress_components)
            generate_btn.disabled = False
            save_btn.disabled = False
            ui_updater.update_status("Image generated successfully!", status_text)
            
            # Auto-collapse configuration section after successful generation
            auto_collapse_after_generation()
            
        except Exception as e:
            ui_updater.update_status(f"Error generating image: {str(e)}", status_text)
            generate_btn.disabled = False
            # Re-define progress_components for error handling
            progress_components = {
                "bar": progress_bar,
                "text": progress_text, 
                "preview": preview_container
            }
            ui_updater.hide_progress_components(progress_components)
            page.update()
    
    def save_image(_: ft.ControlEvent) -> None:
        """Saves the currently displayed image with timestamp filename."""
        filename = orchestrator.save_current_image()
        if filename:
            ui_updater.update_status(f"Image saved as {filename}", status_text)
        else:
            ui_updater.update_status("No image to save", status_text)
    
    def on_generate_click(_: ft.ControlEvent) -> None:
        """Starts image generation in background thread."""
        print("Generate button clicked!")
        threading.Thread(target=generate_image_thread, daemon=True).start()
    
    load_model_btn.on_click = lambda _: threading.Thread(target=load_model_thread, daemon=True).start()
    generate_btn.on_click = on_generate_click
    save_btn.on_click = save_image
    
    # New vertical layout
    page.add(
        ft.Container(
            content=ft.Column([
                # Header with toggle button and title
                ft.Row([
                    toggle_btn,
                    ft.Text("Qwen Image Generator", size=24, weight=ft.FontWeight.BOLD),
                ], alignment=ft.MainAxisAlignment.START, vertical_alignment=ft.CrossAxisAlignment.CENTER),
                
                # Collapsible configuration section
                config_container,
                
                # Progress section (always visible when active)
                ft.Container(
                    content=ft.Column([
                        ft.Text("Generation Progress", size=14, weight=ft.FontWeight.BOLD, visible=False),
                        progress_text,
                        progress_bar,
                        preview_container
                    ], spacing=5),
                    padding=ft.padding.symmetric(horizontal=20, vertical=10)
                ),
                
                # Image display section (takes remaining space)
                ft.Container(
                    content=ft.Column([
                        ft.Text("Generated Image", size=18, weight=ft.FontWeight.BOLD),
                        image_container
                    ], spacing=10),
                    expand=True,
                    padding=20
                )
            ], spacing=10),
            padding=20,
            expand=True
        )
    )

if __name__ == "__main__":
    ft.app(target=main)