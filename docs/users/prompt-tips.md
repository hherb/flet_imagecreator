# Prompt Writing Tips

Learn how to write effective prompts for better AI-generated images.

## Prompt Structure

A well-structured prompt typically includes:

```
[Subject] + [Details] + [Style] + [Lighting/Mood] + [Technical Quality]
```

**Example:**
> A majestic lion (subject) with a flowing golden mane, sitting on a rocky outcrop (details), digital painting style (style), dramatic sunset lighting with warm orange tones (lighting/mood), highly detailed, 4K resolution (quality)

## Subject Description

### Be Specific

| Vague | Better | Best |
|-------|--------|------|
| A dog | A golden retriever | A golden retriever puppy with fluffy fur |
| A house | A Victorian mansion | A three-story Victorian mansion with ornate gables |
| A person | A young woman | An elegant young woman with flowing auburn hair |

### Include Actions

Instead of static descriptions, add movement or actions:
- "A hawk soaring through clouds"
- "Children playing in autumn leaves"
- "A chef expertly tossing pizza dough"

## Detail Elements

### Environment/Background

Specify the setting:
- "in a mystical enchanted forest"
- "on a busy Tokyo street at night"
- "inside a cozy cottage with a fireplace"

### Composition

Guide the framing:
- "close-up portrait"
- "wide-angle landscape view"
- "bird's eye view of the city"
- "centered composition with rule of thirds"

### Colors

Be explicit about colors:
- "vibrant sunset colors of orange, pink, and purple"
- "muted earth tones with hints of gold"
- "high contrast black and white"

## Style References

### Art Styles

| Style | Example Prompt Addition |
|-------|------------------------|
| Photorealistic | "photorealistic, DSLR photography, 85mm lens" |
| Digital Art | "digital art, trending on artstation, detailed illustration" |
| Oil Painting | "oil painting, brushstroke texture, classical technique" |
| Watercolor | "watercolor painting, soft edges, paper texture" |
| Anime/Manga | "anime style, studio ghibli inspired, cel shaded" |
| Concept Art | "concept art, matte painting, cinematic composition" |
| Pop Art | "pop art style, bold colors, andy warhol inspired" |

### Artist References

You can reference famous artists (for style, not copying):
- "in the style of Van Gogh"
- "reminiscent of Monet's impressionism"
- "inspired by HR Giger's biomechanical aesthetic"

### Era/Period Styles

- "Art Nouveau decorative elements"
- "1980s synthwave aesthetic"
- "Renaissance painting techniques"
- "Minimalist modern design"

## Lighting and Mood

### Lighting Types

| Lighting | Effect |
|----------|--------|
| Golden hour | Warm, soft, romantic |
| Blue hour | Cool, mysterious, calm |
| Studio lighting | Professional, clean, controlled |
| Dramatic side lighting | Moody, artistic, depth |
| Soft diffused light | Gentle, flattering, even |
| Backlighting | Silhouettes, rim light, ethereal |
| Neon lighting | Cyberpunk, vibrant, modern |

### Mood Words

- **Peaceful:** serene, tranquil, calm, gentle
- **Dramatic:** intense, powerful, striking, bold
- **Mysterious:** enigmatic, foggy, shadowy, ethereal
- **Joyful:** bright, cheerful, vibrant, lively
- **Melancholic:** moody, somber, wistful, nostalgic

## Technical Quality Terms

### Resolution and Detail

- "highly detailed"
- "intricate details"
- "4K resolution" / "8K resolution"
- "ultra HD"
- "sharp focus"

### Photography Terms

- "depth of field"
- "bokeh background"
- "macro photography"
- "long exposure"
- "f/1.4 aperture"

### Rendering Quality

- "ray tracing"
- "global illumination"
- "subsurface scattering"
- "ambient occlusion"
- "volumetric lighting"

## Example Prompts by Category

### Landscapes

```
A breathtaking mountain vista at golden hour, snow-capped peaks
reflecting warm sunlight, pristine alpine lake in the foreground,
wildflowers dotting the meadow, dramatic clouds casting shadows,
photorealistic landscape photography, wide-angle lens,
highly detailed, 4K resolution
```

### Portraits

```
Portrait of an elderly craftsman with weathered hands, kind eyes
with deep wrinkles that tell stories, wearing a worn leather apron,
workshop background with soft natural light from a window,
documentary photography style, shallow depth of field,
85mm lens, cinematic color grading
```

### Fantasy

```
An ancient dragon perched atop a crystal spire, iridescent scales
shimmering with purple and blue, wings spread majestically against
a stormy sky, lightning illuminating the scene, floating islands
in the background, epic fantasy concept art, detailed illustration,
dramatic lighting, trending on artstation
```

### Architecture

```
Futuristic eco-friendly skyscraper covered in vertical gardens,
glass and steel structure with organic curves, solar panels
integrated into design, busy street level with pedestrians,
blue sky with wispy clouds, architectural visualization render,
clean modern aesthetic, photorealistic
```

### Food

```
Gourmet chocolate cake slice on a white ceramic plate, rich dark
ganache dripping down layers, fresh raspberries and mint garnish,
chocolate shavings scattered artfully, soft studio lighting,
shallow depth of field, food photography, mouth-watering,
delicious, highly detailed
```

## Negative Prompts

### When to Use Them

Use negative prompts to fix recurring issues:

| Problem | Negative Prompt |
|---------|-----------------|
| Blurry images | blurry, out of focus, soft |
| Watermarks | watermark, text, signature, logo |
| Bad anatomy | distorted, deformed, disfigured, bad anatomy |
| Extra limbs | extra fingers, extra limbs, duplicate |
| Low quality | low quality, low resolution, jpeg artifacts |
| Cropped images | cropped, cut off, out of frame |

### Effective Negative Prompt Template

```
blurry, low quality, distorted, watermark, text, signature,
cropped, out of frame, worst quality, low resolution
```

## Advanced Techniques

### Weighted Terms

Some diffusion models support weighted terms. While Qwen may handle this differently, you can emphasize important elements by:
- Repeating important words: "beautiful beautiful sunset"
- Front-loading key terms: Start with the most important subject
- Using specific adjectives: "extremely detailed" vs "detailed"

### Prompt Length

- **Short prompts** (< 20 words): Quick results, more AI interpretation
- **Medium prompts** (20-50 words): Good balance of control and creativity
- **Long prompts** (50+ words): More specific results, can sometimes confuse the model

### Iterative Refinement

1. Start with a basic prompt
2. Generate and review
3. Add details for elements you like
4. Add negative prompts for elements you don't like
5. Adjust style terms
6. Repeat until satisfied

## Common Mistakes to Avoid

1. **Being too vague:** "A nice picture" won't give good results
2. **Contradictory terms:** "dark bright sunny night" confuses the model
3. **Overloading prompts:** Too many concepts compete for attention
4. **Ignoring composition:** Always consider the overall arrangement
5. **Forgetting style:** Without style guidance, results are unpredictable

## Template for Quick Start

Copy and modify this template:

```
[Main subject with 2-3 descriptive details],
[setting/environment],
[style reference],
[lighting type],
[quality terms: highly detailed, 4K]
```

**Example using template:**
```
A wise old wizard with a long silver beard,
in an ancient library filled with floating books,
fantasy digital art style,
candlelit with warm golden glow,
highly detailed, 4K resolution
```
