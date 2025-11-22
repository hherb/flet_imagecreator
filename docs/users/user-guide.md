# User Guide

This comprehensive guide explains all features of Flet Image Creator.

## Interface Overview

The application interface is divided into several sections:

```
┌────────────────────────────────────────────────────────┐
│ [▼] Qwen Image Generator                               │
├────────────────────────────────────────────────────────┤
│ ┌─────────────────────┐ ┌─────────────────────┐        │
│ │ Prompt              │ │ Negative Prompt     │        │
│ │                     │ │                     │        │
│ └─────────────────────┘ └─────────────────────┘        │
│                                                        │
│ Enhancement: [Ultra HD, 4K, cinematic composition.]    │
│                                                        │
│ Aspect Ratio: [16:9 ▼]  Steps: [====50====]            │
│ CFG Scale: [====4.0====]  Seed: [42] [Random]          │
│                                                        │
│ [Load Model] [Generate Image] [Save Image]             │
│                                                        │
│ Status: Ready to generate images                       │
├────────────────────────────────────────────────────────┤
│                                                        │
│                  Generated Image                       │
│                                                        │
│              [Image appears here]                      │
│                                                        │
└────────────────────────────────────────────────────────┘
```

## Input Fields

### Prompt

The main text description of the image you want to create.

**Best Practices:**
- Be specific and descriptive
- Include details about style, lighting, and composition
- Mention colors, mood, and atmosphere
- Describe the main subject and background

**Examples:**

| Good Prompt | Why It Works |
|-------------|--------------|
| "A majestic mountain peak at golden hour, snow-capped summit glowing orange, dramatic clouds, pine forest in foreground" | Specific subject, lighting, colors, and composition |
| "Portrait of an elegant woman in Renaissance style, oil painting technique, soft lighting, detailed fabric textures" | Clear subject, artistic style, technique details |
| "Futuristic cityscape at night, neon lights reflecting on wet streets, flying cars, towering skyscrapers, cyberpunk atmosphere" | Setting, lighting, specific elements, mood |

### Negative Prompt

Describe what you want to avoid in the generated image.

**Common Negative Prompts:**
- `blurry, low quality, distorted` - For cleaner images
- `watermark, text, signature` - To avoid text overlays
- `cropped, out of frame` - For complete compositions
- `duplicate, multiple` - To avoid repeated elements

### Enhancement

Quality modifiers automatically appended to your prompt.

**Default:** `Ultra HD, 4K, cinematic composition.`

**Alternatives:**
- `highly detailed, sharp focus, professional photography`
- `masterpiece, best quality, intricate details`
- `photorealistic, 8K resolution, studio lighting`

Leave empty if you don't want any enhancement.

## Generation Parameters

### Aspect Ratio

Choose the dimensions for your generated image:

| Option | Dimensions | Best For |
|--------|------------|----------|
| 1:1 (Square) | 1328 x 1328 | Portraits, social media posts |
| 16:9 (Widescreen) | 1664 x 928 | Landscapes, desktop wallpapers |
| 9:16 (Portrait) | 928 x 1664 | Phone wallpapers, vertical art |
| 4:3 (Standard) | 1472 x 1140 | Traditional prints, presentations |
| 3:4 (Portrait Standard) | 1140 x 1472 | Book covers, vertical prints |

### Inference Steps

Controls the number of denoising iterations during generation.

| Value | Quality | Speed |
|-------|---------|-------|
| 10-20 | Draft quality | Very fast |
| 30-50 | Good quality | Moderate |
| 50-70 | High quality | Slower |
| 70-100 | Maximum quality | Slowest |

**Recommendation:** Start with 50 steps for a good balance of quality and speed.

### CFG Scale (Classifier-Free Guidance)

Controls how closely the image follows your prompt.

| Value | Effect |
|-------|--------|
| 1-2 | Very creative, may ignore prompt details |
| 3-5 | Balanced creativity and prompt adherence |
| 6-8 | Strong prompt adherence |
| 9-10 | Very strict, may reduce quality |

**Recommendation:** Start with 4.0 for most prompts.

### Seed

A number that determines the random starting point for generation.

- **Same seed + same prompt = same image** (for reproducibility)
- Click **"Random"** to generate a new random seed
- Note down seeds you like to recreate similar images

## Buttons

### Load Model

Downloads and initializes the AI model.

- Only needed once per session
- Model is cached after first download
- Button shows "Model Loaded" with a checkmark when ready

### Generate Image

Starts the image generation process.

- Only active after model is loaded
- Disabled during generation to prevent conflicts
- Progress bar shows generation progress

### Save Image

Saves the current image to your computer.

- Only active after an image is generated
- Files are saved with timestamp names (e.g., `qwen_generated_20250115_143022.png`)
- Saved in PNG format for best quality

## Configuration Panel

### Collapse/Expand

Click the arrow button (▼/▲) next to the title to show/hide the configuration panel.

**Auto-collapse:** After successfully generating an image, the configuration panel automatically hides to give more space to view the result. Click the expand button to show it again.

## Progress Tracking

During image generation:

1. **Progress Bar** - Shows percentage complete
2. **Step Counter** - Shows current step (e.g., "Step 25 / 50")
3. **Status Text** - Displays current operation

**Note:** The Qwen model doesn't support intermediate previews, so you'll only see the final result.

## Tips for Best Results

### 1. Write Detailed Prompts

Instead of: "a cat"

Try: "A fluffy orange tabby cat lounging on a velvet cushion, soft afternoon sunlight streaming through a window, warm cozy atmosphere, photorealistic"

### 2. Use Reference Styles

Add artistic style references:
- "in the style of Studio Ghibli"
- "oil painting technique"
- "digital art, trending on artstation"
- "photograph, DSLR, 85mm lens"

### 3. Experiment with Seeds

When you find an image you like:
1. Note the seed number
2. Make small prompt changes
3. Use the same seed for similar compositions

### 4. Iterate and Refine

1. Start with a basic prompt at 30 steps
2. Refine the prompt based on results
3. Increase steps for final version
4. Adjust CFG scale if needed

### 5. Use Negative Prompts Wisely

Only add negative prompts if you're seeing unwanted elements:
- Getting blurry images? Add "blurry, out of focus"
- Seeing text/watermarks? Add "watermark, text, signature"
- Wrong proportions? Add "distorted, deformed"

## Keyboard Shortcuts

Currently, the application uses mouse-based interaction. Future versions may include keyboard shortcuts.

## Performance Notes

- **First generation** after loading is typically slower (warmup)
- **GPU users** will see significantly faster generation
- **Higher steps** = longer generation time
- **Larger images** require more memory

## Saving Your Work

Generated images are saved to the current working directory (where you ran the application from).

To organize your images:
1. Create a folder for your projects
2. Run the application from that folder
3. All saved images will appear there
