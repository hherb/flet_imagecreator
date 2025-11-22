# Architecture Documentation

This document provides a detailed overview of the Flet Image Creator architecture, design decisions, and code organization.

## Design Philosophy

The application follows several key design principles:

1. **Separation of Concerns** - Business logic is separated from UI and I/O operations
2. **Pure Functions First** - Core logic is implemented as pure functions for testability
3. **Single Responsibility** - Each class has a single, well-defined purpose
4. **Type Safety** - Comprehensive type hints throughout the codebase
5. **Defensive Programming** - Validation at boundaries, sensible defaults

## Layer Diagram

```
┌─────────────────────────────────────────────────────────────────┐
│                         UI LAYER (Flet)                         │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────────┐  │
│  │   Prompts   │  │  Controls   │  │    Image Display        │  │
│  │   Fields    │  │  (sliders)  │  │    Container            │  │
│  └─────────────┘  └─────────────┘  └─────────────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                    ORCHESTRATION LAYER                          │
│  ┌──────────────────────────┐  ┌─────────────────────────────┐  │
│  │ ImageGenerationOrchestra │  │     QwenImageGenerator      │  │
│  │ (state management)       │  │     (model interface)       │  │
│  └──────────────────────────┘  └─────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                   SIDE EFFECTS LAYER (I/O)                      │
│  ┌──────────────┐  ┌──────────────┐  ┌─────────────────────┐    │
│  │ ModelLoader  │  │ ImageSaver   │  │    UIUpdater        │    │
│  │ (HuggingFace)│  │ (filesystem) │  │    (page refresh)   │    │
│  └──────────────┘  └──────────────┘  └─────────────────────┘    │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                    PURE FUNCTIONS LAYER                         │
│  ┌────────────────────┐  ┌─────────────────────────────────┐    │
│  │ validate_params()  │  │ enhance_prompt()                │    │
│  │ calculate_progress │  │ convert_image_to_base64()       │    │
│  │ get_optimal_device │  │ generate_filename()             │    │
│  └────────────────────┘  └─────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                       DATA LAYER                                │
│  ┌──────────────────────┐  ┌────────────────────────────────┐   │
│  │  GenerationParams    │  │      ModelConfig               │   │
│  │  (frozen dataclass)  │  │      (dataclass)               │   │
│  └──────────────────────┘  └────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────┘
```

## Component Details

### Data Classes

#### GenerationParams

```python
@dataclass(frozen=True)
class GenerationParams:
    prompt: str               # Main text description
    negative_prompt: str      # What to avoid
    enhancement: str          # Quality modifiers
    width: int                # Output width in pixels
    height: int               # Output height in pixels
    num_inference_steps: int  # Denoising iterations
    cfg_scale: float          # Classifier-free guidance scale
    seed: int                 # Random seed for reproducibility
```

This is a **frozen dataclass** (immutable) to ensure parameters cannot be accidentally modified after validation.

#### ModelConfig

```python
@dataclass
class ModelConfig:
    device: str          # 'mps', 'cuda', or 'cpu'
    torch_dtype: torch.dtype  # torch.bfloat16 or torch.float32
```

Configuration passed to the model loader, encapsulating device-specific settings.

### Pure Functions

These functions have no side effects and are deterministic:

| Function | Input | Output | Purpose |
|----------|-------|--------|---------|
| `get_optimal_device()` | None | `(str, dtype)` | Detect best compute device |
| `validate_generation_params()` | `Dict` | `GenerationParams` | Validate and convert UI params |
| `enhance_prompt()` | `str, str` | `str` | Append quality modifiers |
| `calculate_progress()` | `int, int` | `float` | Compute progress percentage |
| `create_progress_message()` | `int, int` | `str` | Format progress text |
| `generate_filename()` | `str, str` | `str` | Create timestamped filename |
| `convert_image_to_base64()` | `Image` | `str` | Encode image for display |

### Side Effects Classes

#### ModelLoader

Handles model loading from HuggingFace:

```python
class ModelLoader:
    @staticmethod
    def load_diffusion_pipeline(model_name: str, config: ModelConfig) -> DiffusionPipeline:
        """Load model with fallback from QwenImagePipeline to DiffusionPipeline"""

    @staticmethod
    def create_generator(device: str, seed: int) -> torch.Generator:
        """Create seeded RNG on correct device"""
```

#### ImageSaver

Simple file I/O for saving images:

```python
class ImageSaver:
    @staticmethod
    def save_image(image: Image.Image, filename: str) -> str:
        """Save PIL Image to disk"""
```

#### UIUpdater

Centralizes all Flet UI updates:

```python
class UIUpdater:
    def update_status(self, message: str, status_component: ft.Text) -> None
    def update_progress(self, progress_data: Dict, progress_components: Dict) -> None
    def show_progress_components(self, components: Dict) -> None
    def hide_progress_components(self, components: Dict) -> None
    def update_image_display(self, image_base64: str, container: ft.Container) -> None
```

All methods call `page.update()` automatically to refresh the Flet display.

### Orchestration Classes

#### QwenImageGenerator

Main interface to the Qwen diffusion model:

```python
class QwenImageGenerator:
    pipe: Optional[DiffusionPipeline]  # The loaded pipeline
    device: Optional[str]               # Current device
    torch_dtype: Optional[torch.dtype]  # Current dtype
    model_loaded: bool                   # Loading state

    def load_model(self, progress_callback: Optional[Callable[[str], None]] = None) -> None
    def generate_image(self, params: GenerationParams,
                       progress_callback: Optional[Callable[[Dict], None]] = None) -> Image.Image
```

#### ImageGenerationOrchestrator

Manages generated image state for save operations:

```python
class ImageGenerationOrchestrator:
    ui_updater: UIUpdater
    current_image: Optional[Image.Image]

    def save_current_image(self) -> Optional[str]
```

## Data Flow

### Image Generation Flow

```
User Input (UI)
     │
     ▼
┌────────────────────────────┐
│ validate_generation_params │  ← Converts dict to GenerationParams
└────────────────────────────┘
     │
     ▼
┌────────────────────────────┐
│     enhance_prompt()       │  ← Appends quality modifiers
└────────────────────────────┘
     │
     ▼
┌────────────────────────────┐
│ ModelLoader.create_generator│  ← Creates seeded RNG
└────────────────────────────┘
     │
     ▼
┌────────────────────────────┐
│     pipeline(...)          │  ← Diffusion process with callbacks
└────────────────────────────┘
     │
     ├──────────────────────────┐
     │                          │
     ▼                          ▼
┌──────────────┐        ┌───────────────────┐
│ PIL.Image    │        │ Progress Callbacks│
└──────────────┘        └───────────────────┘
     │                          │
     ▼                          ▼
┌──────────────────┐    ┌─────────────────┐
│convert_to_base64 │    │ UIUpdater       │
└──────────────────┘    │ .update_progress│
     │                  └─────────────────┘
     ▼
┌──────────────────┐
│ UIUpdater        │
│ .update_image    │
└──────────────────┘
```

### Model Loading Flow

```
User clicks "Load Model"
     │
     ▼
┌────────────────────────────┐
│   get_optimal_device()     │  ← Detects MPS/CUDA/CPU
└────────────────────────────┘
     │
     ▼
┌────────────────────────────┐
│     ModelConfig created    │  ← Encapsulates device settings
└────────────────────────────┘
     │
     ▼
┌────────────────────────────┐
│ ModelLoader.load_pipeline  │  ← Downloads/loads from HuggingFace
└────────────────────────────┘
     │
     ├─── Success ────────────────────────────────┐
     │                                            │
     ▼                                            ▼
┌──────────────────┐                    ┌──────────────────┐
│ Enable generate  │                    │ Store pipeline   │
│ button           │                    │ in generator     │
└──────────────────┘                    └──────────────────┘
```

## Threading Model

The application uses background threads to prevent UI blocking:

```
┌─────────────────────────────────────────────────────────────┐
│                      MAIN THREAD                            │
│  ┌────────────────────────────────────────────────────────┐ │
│  │                  Flet Event Loop                       │ │
│  │  - Handles UI events                                   │ │
│  │  - Processes page.update() calls                       │ │
│  │  - Renders components                                  │ │
│  └────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────┘
         │                              │
         │ spawns (daemon=True)         │ spawns (daemon=True)
         ▼                              ▼
┌─────────────────────┐       ┌─────────────────────┐
│ load_model_thread   │       │ generate_image_thread│
│ - Downloads model   │       │ - Runs diffusion    │
│ - Updates status    │       │ - Updates progress  │
└─────────────────────┘       └─────────────────────┘
```

### Thread Safety

- UI updates are performed through `page.update()` which is thread-safe
- Model loading happens once (guarded by `model_loaded` flag)
- Button states are managed to prevent concurrent operations

## Configuration Constants

```python
MODEL_NAME: str = "Qwen/Qwen-Image"
DEFAULT_ENHANCEMENT: str = "Ultra HD, 4K, cinematic composition."
DEFAULT_SEED: int = 42

ASPECT_RATIOS: Dict[str, Tuple[int, int]] = {
    "1:1 (Square)": (1328, 1328),
    "16:9 (Widescreen)": (1664, 928),
    "9:16 (Portrait)": (928, 1664),
    "4:3 (Standard)": (1472, 1140),
    "3:4 (Portrait Standard)": (1140, 1472),
}
```

These constants are defined at module level for easy modification and to avoid magic values.

## Error Handling Strategy

1. **Validation Errors** - Handled by `validate_generation_params()` with sensible defaults
2. **Model Loading Errors** - Caught and displayed in status text, button re-enabled
3. **Generation Errors** - Caught in thread, progress hidden, status updated
4. **Callback Errors** - Fallback generation without callbacks if TypeError occurs

## Extension Points

### Adding New Models

1. Modify `MODEL_NAME` constant
2. Update `ModelLoader.load_diffusion_pipeline()` if needed
3. Adjust `ASPECT_RATIOS` for model-specific dimensions

### Adding New UI Controls

1. Define control in `main()` function
2. Add to appropriate container/row
3. Extract value in `generate_image_thread()`
4. Update `validate_generation_params()` if new parameter needed

### Adding Post-Processing

1. Create new pure function for processing
2. Call after `generate_image()` returns
3. Update before `convert_image_to_base64()`
