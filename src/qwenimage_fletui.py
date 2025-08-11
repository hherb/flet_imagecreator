import flet as ft
from diffusers import DiffusionPipeline
try:
    from diffusers.pipelines.qwen_image import QwenImagePipeline
except ImportError:
    # Fallback if the import path is different
    QwenImagePipeline = None
import torch
import io
import base64
from PIL import Image, ImageFilter
import threading
import os
from typing import Dict, Tuple, Optional, Any
from dataclasses import dataclass
import datetime

MODEL_NAME = "Qwen/Qwen-Image"

# =============================================================================
# PURE FUNCTIONS - Business Logic Layer
# =============================================================================

@dataclass
class GenerationParams:
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
    device: str
    torch_dtype: torch.dtype
    model_loaded: bool

def get_optimal_device() -> Tuple[str, torch.dtype]:
    """Detects the best available compute device and appropriate data type.
    
    Returns MPS for Apple Silicon, CUDA for NVIDIA GPUs, or CPU as fallback.
    Uses bfloat16 for accelerated devices to save memory and improve performance.
    """
    if torch.backends.mps.is_available():
        return "mps", torch.bfloat16
    elif torch.cuda.is_available():
        return "cuda", torch.bfloat16
    else:
        return "cpu", torch.float32

def validate_generation_params(params: Dict[str, Any]) -> GenerationParams:
    """Validates UI input and converts to structured generation parameters.
    
    Handles aspect ratio lookup, seed validation, and parameter type conversion.
    Invalid seeds default to 42 to ensure reproducible generation.
    """
    aspect_ratios = {
        "1:1 (Square)": (1328, 1328),
        "16:9 (Widescreen)": (1664, 928),
        "9:16 (Portrait)": (928, 1664),
        "4:3 (Standard)": (1472, 1140),
        "3:4 (Portrait Standard)": (1140, 1472)
    }
    
    width, height = aspect_ratios.get(params.get("aspect_ratio"), (1664, 928))
    seed = int(params.get("seed", 42)) if str(params.get("seed", "")).isdigit() else 42
    
    return GenerationParams(
        prompt=params.get("prompt", ""),
        negative_prompt=params.get("negative_prompt", ""),
        enhancement=params.get("enhancement", "Ultra HD, 4K, cinematic composition."),
        width=width,
        height=height,
        num_inference_steps=int(params.get("steps", 50)),
        cfg_scale=float(params.get("cfg_scale", 4.0)),
        seed=seed
    )

def enhance_prompt(prompt: str, enhancement: str = "Ultra HD, 4K, cinematic composition.") -> str:
    """Appends quality enhancement text to the user's prompt.
    
    Quality modifiers like 'Ultra HD, 4K' help guide the model toward
    higher quality outputs. Empty enhancements are ignored.
    """
    if enhancement.strip():
        return f"{prompt} {enhancement}"
    return prompt

def calculate_progress(current_step: int, total_steps: int) -> float:
    """Calculates completion percentage for progress bars (0.0 to 1.0)."""
    return current_step / total_steps

def create_progress_message(current_step: int, total_steps: int) -> str:
    """Formats progress text for display during generation."""
    return f"Step {current_step} / {total_steps}"

def generate_filename() -> str:
    """Creates unique timestamped filename for saved images.
    
    Format: qwen_generated_YYYYMMDD_HHMMSS.png
    Prevents filename conflicts when saving multiple images.
    """
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"qwen_generated_{timestamp}.png"

def convert_image_to_base64(image: Image.Image) -> str:
    """Encodes PIL Image as base64 string for Flet display.
    
    Flet UI requires base64-encoded images for src_base64 parameter.
    Always uses PNG format to preserve quality.
    """
    img_buffer = io.BytesIO()
    image.save(img_buffer, format='PNG')
    return base64.b64encode(img_buffer.getvalue()).decode()

def create_progress_display_data(current_step: int, total_steps: int) -> Dict[str, Any]:
    """Builds progress display data structure for UI updates.
    
    Returns dictionary with progress value, text, and UI elements.
    Qwen model doesn't support intermediate previews during generation.
    """
    return {
        "progress_value": calculate_progress(current_step, total_steps),
        "progress_text": create_progress_message(current_step, total_steps),
        "icon": ft.Icons.HOURGLASS_EMPTY,
        "message": f"Generating step {current_step}/{total_steps}",
        "sub_message": "Preview not available for Qwen model"
    }

# =============================================================================
# SIDE EFFECTS - I/O Operations Layer
# =============================================================================

class ModelLoader:
    """Handles diffusion model loading and device configuration."""
    
    @staticmethod
    def load_diffusion_pipeline(model_name: str, config: ModelConfig) -> DiffusionPipeline:
        """Loads the Qwen diffusion pipeline with fallback handling.
        
        Attempts to load QwenImagePipeline first for future image editing support,
        falls back to generic DiffusionPipeline if specific pipeline unavailable.
        Automatically moves model to specified device (MPS/CUDA/CPU).
        """
        # Try to load as QwenImagePipeline first for image editing support
        if QwenImagePipeline is not None:
            try:
                pipe = QwenImagePipeline.from_pretrained(model_name, torch_dtype=config.torch_dtype)
                print("Loaded as QwenImagePipeline - image editing supported")
                return pipe.to(config.device)
            except Exception as e:
                print(f"Failed to load as QwenImagePipeline: {e}")
        
        # Fallback to generic DiffusionPipeline
        pipe = DiffusionPipeline.from_pretrained(model_name, torch_dtype=config.torch_dtype)
        print(f"Loaded as {type(pipe).__name__}")
        return pipe.to(config.device)
    
    @staticmethod
    def create_generator(device: str, seed: int) -> torch.Generator:
        """Creates a seeded random number generator for reproducible generation.
        
        Generator must match the model's device (MPS/CUDA/CPU) to avoid errors.
        Same seed produces identical results for debugging and consistency.
        """
        return torch.Generator(device=device).manual_seed(seed)

class ImageSaver:
    """Handles saving generated images to disk."""
    
    @staticmethod
    def save_image(image: Image.Image, filename: str) -> None:
        """Saves PIL Image to filesystem with automatic format detection.
        
        Format determined by file extension. PNG recommended for quality preservation.
        """
        image.save(filename)

class UIUpdater:
    """Manages UI updates and page refreshes in a centralized way.
    
    Provides consistent interface for updating status, progress, and images
    while ensuring proper page.update() calls for Flet refresh.
    """
    
    def __init__(self, page: ft.Page) -> None:
        self.page = page
    
    def update_status(self, message: str, status_component: ft.Text) -> None:
        """Updates status text component and refreshes the UI.
        
        Use for user feedback during loading, errors, and completion states.
        """
        status_component.value = message
        self.page.update()
    
    def update_progress(self, progress_data: Dict[str, Any], progress_components: Dict[str, ft.Control]) -> None:
        """Updates progress bar, text, and preview components during generation.
        
        progress_data should contain progress_value, progress_text, icon, message, sub_message.
        progress_components should contain 'bar', 'text', and 'preview' controls.
        """
        progress_components["bar"].value = progress_data["progress_value"]
        progress_components["text"].value = progress_data["progress_text"]
        
        progress_components["preview"].content = ft.Column([
            ft.Icon(progress_data["icon"], size=40, color=ft.Colors.BLUE_400),
            ft.Text(progress_data["message"], text_align=ft.TextAlign.CENTER, size=12),
            ft.Text(progress_data["sub_message"], text_align=ft.TextAlign.CENTER, size=10, color=ft.Colors.GREY_600)
        ], horizontal_alignment=ft.CrossAxisAlignment.CENTER)
        
        self.page.update()
    
    def show_progress_components(self, components: Dict[str, ft.Control]) -> None:
        """Makes progress components visible during generation."""
        for comp in components.values():
            comp.visible = True
        self.page.update()
    
    def hide_progress_components(self, components: Dict[str, ft.Control]) -> None:
        """Hides progress components after generation completes."""
        for comp in components.values():
            comp.visible = False
        self.page.update()
    
    def update_image_display(self, image_base64: str, image_container: ft.Container) -> None:
        """Displays generated image with responsive scaling and aspect ratio preservation.
        
        Creates a new container with the image, replacing any previous content.
        Uses CONTAIN fit to prevent distortion while filling available space.
        """
        image_display = ft.Container(
            content=ft.Image(
                src_base64=image_base64,
                fit=ft.ImageFit.CONTAIN,  # Maintains aspect ratio, scales to fit
                expand=True
            ),
            expand=True,
            alignment=ft.alignment.center,
            bgcolor=ft.Colors.BLACK12,
            border_radius=8,
            clip_behavior=ft.ClipBehavior.HARD_EDGE
        )
        image_container.content = image_display
        self.page.update()

# =============================================================================
# ORCHESTRATION LAYER - Coordinates Logic and Side Effects
# =============================================================================

class ImageGenerationOrchestrator:
    """Coordinates image generation workflow with UI updates.
    
    Manages the complete generation process from parameter validation
    through progress tracking to final image storage.
    """
    
    def __init__(self, ui_updater: UIUpdater) -> None:
        self.ui_updater = ui_updater
        self.current_image: Optional[Image.Image] = None
    
    def generate_image_workflow(self, generator: 'QwenImageGenerator', params: GenerationParams, 
                               progress_callback: Optional[callable] = None) -> Image.Image:
        """Executes complete image generation workflow with progress tracking.
        
        Handles prompt enhancement, generator setup, pipeline execution,
        and progress reporting through callback system.
        """
        
        # Create progress callback for pipeline
        def step_callback(pipe: Any, step_index: int, timestep: Any, callback_kwargs: Dict[str, Any]) -> Dict[str, Any]:
            if progress_callback:
                progress_data = create_progress_display_data(step_index + 1, params.num_inference_steps)
                progress_callback(progress_data)
            return callback_kwargs
        
        # Generate image with enhanced prompt
        enhanced_prompt = enhance_prompt(params.prompt, params.enhancement)
        torch_generator = ModelLoader.create_generator(generator.device, params.seed)
        
        image = generator.pipe(
            prompt=enhanced_prompt,
            negative_prompt=params.negative_prompt,
            width=params.width,
            height=params.height,
            num_inference_steps=params.num_inference_steps,
            true_cfg_scale=params.cfg_scale,
            generator=torch_generator,
            callback_on_step_end=step_callback if progress_callback else None,
            callback_on_step_end_tensor_inputs=["latents"] if progress_callback else None
        ).images[0]
        
        self.current_image = image
        return image
    
    def save_current_image(self) -> Optional[str]:
        """Saves the most recently generated image with timestamp filename.
        
        Returns the filename if successful, None if no image to save.
        Creates unique filenames to prevent overwrites.
        """
        if self.current_image:
            filename = generate_filename()
            ImageSaver.save_image(self.current_image, filename)
            return filename
        return None

class QwenImageGenerator:
    """Main interface for Qwen image generation model.
    
    Handles model loading, device management, and image generation.
    Currently supports text-to-image only; image editing coming in future updates.
    """
    
    def __init__(self) -> None:
        self.pipe: Optional[DiffusionPipeline] = None
        self.device: Optional[str] = None
        self.torch_dtype: Optional[torch.dtype] = None
        self.model_loaded: bool = False
        
    def load_model(self, progress_callback: Optional[callable] = None) -> None:
        """Downloads and initializes the Qwen diffusion model.
        
        Automatically detects best available device (MPS/CUDA/CPU) and configures
        appropriate data types for optimal performance. Only loads once.
        
        Args:
            progress_callback: Optional function to receive status updates during loading
        """
        if self.model_loaded:
            return
            
        # Use pure function to determine device config
        self.device, self.torch_dtype = get_optimal_device()
        
        if progress_callback:
            progress_callback(f"Using device: {self.device} with dtype: {self.torch_dtype}")
            progress_callback("Loading model...")
        
        # Use side effect class to load pipeline
        config = ModelConfig(self.device, self.torch_dtype, False)
        self.pipe = ModelLoader.load_diffusion_pipeline(MODEL_NAME, config)
        
        self.model_loaded = True
        if progress_callback:
            progress_callback("Model loaded successfully!")
    
    def generate_image(self, params: GenerationParams, progress_callback: Optional[callable] = None) -> Image.Image:
        """Generates image from text prompt using loaded Qwen model.
        
        Args:
            params: Validated generation parameters including prompt, dimensions, steps, etc.
            progress_callback: Optional function to receive progress updates during generation
            
        Returns:
            Generated PIL Image
            
        Raises:
            Exception: If model not loaded or generation fails
        """
        if not self.model_loaded:
            raise Exception("Model not loaded. Please load the model first.")
        
        # Create progress callback for pipeline
        def step_callback(pipe: Any, step_index: int, timestep: Any, callback_kwargs: Dict[str, Any]) -> Dict[str, Any]:
            if progress_callback:
                progress_data = create_progress_display_data(step_index + 1, params.num_inference_steps)
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
                callback_on_step_end_tensor_inputs=["latents"] if progress_callback else None
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
                generator=torch_generator
            ).images[0]
        
        return image

def main(page: ft.Page) -> None:
    """Main Flet application entry point.
    
    Sets up the complete UI including model controls, parameter inputs,
    progress tracking, and image display. Handles all user interactions
    and coordinates between UI and generation logic.
    
    Args:
        page: Flet page object for UI rendering
    """
    page.title = f"Image Generator (using model {MODEL_NAME})"
    page.theme_mode = ft.ThemeMode.LIGHT
    page.window_width = 1200
    page.window_height = 800
    
    # Initialize components using new architecture
    generator = QwenImageGenerator()
    ui_updater = UIUpdater(page)
    orchestrator = ImageGenerationOrchestrator(ui_updater)
    
    # State for collapsible configuration section
    config_collapsed = False
    
    # UI Components
    prompt_field = ft.TextField(
        label="Prompt",
        multiline=True,
        min_lines=2,
        max_lines=3,
        value="A beautiful lake in a forest grove in the Alps. Snow capped mountains in the backround. Mossy ground with mushrooms. Little dwarfs frolicking in the moss",
        expand=True
    )
    
    negative_prompt_field = ft.TextField(
        label="Negative Prompt",
        multiline=True,
        min_lines=2,
        max_lines=3,
        value="",
        expand=True
    )
    
    enhancement_field = ft.TextField(
        label="Enhancement",
        value="Ultra HD, 4K, cinematic composition.",
        expand=True
    )
    
    # Aspect ratio selection
    aspect_ratios = {
        "1:1 (Square)": (1328, 1328),
        "16:9 (Widescreen)": (1664, 928),
        "9:16 (Portrait)": (928, 1664),
        "4:3 (Standard)": (1472, 1140),
        "3:4 (Portrait Standard)": (1140, 1472)
    }
    
    aspect_dropdown = ft.Dropdown(
        label="Aspect Ratio",
        value="16:9 (Widescreen)",
        options=[ft.dropdown.Option(key) for key in aspect_ratios.keys()],
        width=200
    )
    
    steps_slider = ft.Slider(
        min=10,
        max=100,
        divisions=18,
        value=50,
        label="Steps: {value}",
        width=200
    )
    
    cfg_slider = ft.Slider(
        min=1.0,
        max=10.0,
        divisions=18,
        value=4.0,
        label="CFG Scale: {value}",
        width=200
    )
    
    seed_field = ft.TextField(
        label="Seed",
        value="42",
        width=100
    )
    
    random_seed_btn = ft.ElevatedButton(
        "Random",
        on_click=lambda _: setattr(seed_field, 'value', str(torch.randint(0, 2**32-1, (1,)).item())),
        width=80
    )
    
    # Message container for image editing feedback
    image_edit_message = ft.Container(
        content=ft.Text(
            "Image editing mode: Your input image will guide the generation", 
            size=12, 
            color=ft.Colors.GREEN_700
        ),
        visible=False,
        padding=10,
        bgcolor=ft.Colors.GREEN_50,
        border_radius=5
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