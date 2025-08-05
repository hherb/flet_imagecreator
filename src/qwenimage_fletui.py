import flet as ft
from diffusers import DiffusionPipeline
import torch
import io
import base64
from PIL import Image
import threading
import os

MODEL_NAME = "Qwen/Qwen-Image"

class QwenImageGenerator:
    def __init__(self):
        self.pipe = None
        self.device = None
        self.torch_dtype = None
        self.model_loaded = False
        
    def load_model(self, progress_callback=None):
        if self.model_loaded:
            return
            
        model_name = MODEL_NAME
        
        # Device detection for Apple Silicon
        if torch.backends.mps.is_available():
            self.torch_dtype = torch.bfloat16
            self.device = "mps"
        elif torch.cuda.is_available():
            self.torch_dtype = torch.bfloat16
            self.device = "cuda"
        else:
            self.torch_dtype = torch.float32
            self.device = "cpu"
        
        if progress_callback:
            progress_callback(f"Using device: {self.device} with dtype: {self.torch_dtype}")
            progress_callback("Loading model...")
        
        # Load the pipeline
        self.pipe = DiffusionPipeline.from_pretrained(model_name, torch_dtype=self.torch_dtype)
        self.pipe = self.pipe.to(self.device)
        
        self.model_loaded = True
        if progress_callback:
            progress_callback("Model loaded successfully!")
    
    def generate_image(self, prompt, negative_prompt="", width=1664, height=928, 
                      num_inference_steps=50, cfg_scale=4.0, seed=42):
        if not self.model_loaded:
            raise Exception("Model not loaded. Please load the model first.")
        
        positive_magic = {
            "en": "Ultra HD, 4K, cinematic composition.",
            "zh": "超清，4K，电影级构图"
        }
        
        # Create generator with correct device
        generator = torch.Generator(device=self.device).manual_seed(seed)
        
        image = self.pipe(
            prompt=prompt + " " + positive_magic["en"],
            negative_prompt=negative_prompt,
            width=width,
            height=height,
            num_inference_steps=num_inference_steps,
            true_cfg_scale=cfg_scale,
            generator=generator
        ).images[0]
        
        return image

def main(page: ft.Page):
    page.title = f"Image Generator (using model {MODEL_NAME})"
    page.theme_mode = ft.ThemeMode.LIGHT
    page.window_width = 1200
    page.window_height = 800
    
    generator = QwenImageGenerator()
    
    # UI Components
    prompt_field = ft.TextField(
        label="Prompt",
        multiline=True,
        min_lines=3,
        max_lines=5,
        value="A beautiful lake in a forest grove in the Alps. Snow capped mountains in the backround. Mossy ground with mushrooms. Little dwarfs frolicking in the moss"
    )
    
    negative_prompt_field = ft.TextField(
        label="Negative Prompt",
        multiline=True,
        min_lines=2,
        max_lines=3,
        value="",
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
    
    # Status and progress
    status_text = ft.Text("Ready to generate images", size=14)
    progress_bar = ft.ProgressBar(visible=False)
    
    # Image display
    image_container = ft.Container(
        content=ft.Text("Generated image will appear here", text_align=ft.TextAlign.CENTER),
        width=512,
        height=512,
        border=ft.border.all(1, ft.Colors.GREY_400),
        alignment=ft.alignment.center,
        border_radius=8
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
    
    current_image = None
    
    def update_status(message):
        status_text.value = message
        page.update()
    
    def load_model_thread():
        try:
            progress_bar.visible = True
            load_model_btn.disabled = True
            page.update()
            
            generator.load_model(progress_callback=update_status)
            
            load_model_btn.text = "Model Loaded"
            load_model_btn.icon = ft.Icons.CHECK
            generate_btn.disabled = False
            progress_bar.visible = False
            update_status("Model loaded successfully! Ready to generate images.")
            
        except Exception as e:
            update_status(f"Error loading model: {str(e)}")
            load_model_btn.disabled = False
            progress_bar.visible = False
        
        page.update()
    
    def generate_image_thread():
        nonlocal current_image
        try:
            progress_bar.visible = True
            generate_btn.disabled = True
            save_btn.disabled = True
            page.update()
            
            update_status("Generating image...")
            
            width, height = aspect_ratios[aspect_dropdown.value]
            seed = int(seed_field.value) if seed_field.value.isdigit() else 42
            
            image = generator.generate_image(
                prompt=prompt_field.value,
                negative_prompt=negative_prompt_field.value,
                width=width,
                height=height,
                num_inference_steps=int(steps_slider.value),
                cfg_scale=cfg_slider.value,
                seed=seed
            )
            
            current_image = image
            
            # Convert image to base64 for display
            img_buffer = io.BytesIO()
            # Resize for display if too large
            display_image = image.copy()
            display_image.thumbnail((512, 512), Image.Resampling.LANCZOS)
            display_image.save(img_buffer, format='PNG')
            img_base64 = base64.b64encode(img_buffer.getvalue()).decode()
            
            image_container.content = ft.Image(
                src_base64=img_base64,
                width=512,
                height=512,
                fit=ft.ImageFit.CONTAIN
            )
            
            generate_btn.disabled = False
            save_btn.disabled = False
            progress_bar.visible = False
            update_status("Image generated successfully!")
            
        except Exception as e:
            update_status(f"Error generating image: {str(e)}")
            generate_btn.disabled = False
            progress_bar.visible = False
        
        page.update()
    
    def save_image(e):
        if current_image:
            # Create filename with timestamp
            import datetime
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"qwen_generated_{timestamp}.png"
            
            current_image.save(filename)
            update_status(f"Image saved as {filename}")
    
    load_model_btn.on_click = lambda _: threading.Thread(target=load_model_thread, daemon=True).start()
    generate_btn.on_click = lambda _: threading.Thread(target=generate_image_thread, daemon=True).start()
    save_btn.on_click = save_image
    
    # Layout
    page.add(
        ft.Container(
            content=ft.Column([
                ft.Text("Qwen Image Generator", size=24, weight=ft.FontWeight.BOLD),
                
                ft.Row([
                    ft.Container(
                        content=ft.Column([
                            ft.Text("Generation Parameters", size=18, weight=ft.FontWeight.BOLD),
                            prompt_field,
                            negative_prompt_field,
                            
                            ft.Row([
                                aspect_dropdown,
                                ft.Column([
                                    ft.Text("Inference Steps"),
                                    steps_slider
                                ]),
                                ft.Column([
                                    ft.Text("CFG Scale"),
                                    cfg_slider
                                ])
                            ]),
                            
                            ft.Row([
                                seed_field,
                                random_seed_btn,
                                load_model_btn,
                                generate_btn,
                                save_btn
                            ]),
                            
                            ft.Column([
                                status_text,
                                progress_bar
                            ])
                        ]),
                        width=650,
                        padding=20
                    ),
                    
                    ft.Container(
                        content=ft.Column([
                            ft.Text("Generated Image", size=18, weight=ft.FontWeight.BOLD),
                            image_container
                        ]),
                        width=550,
                        padding=20
                    )
                ])
            ]),
            padding=20
        )
    )

if __name__ == "__main__":
    ft.app(target=main)