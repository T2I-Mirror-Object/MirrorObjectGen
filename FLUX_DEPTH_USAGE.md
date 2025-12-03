# FLUX.1-Depth-dev Usage Guide

## Overview
The `flux_depth.py` script uses the FLUX.1-Depth-dev model from Black Forest Labs to generate images conditioned on depth maps.

## Prerequisites

### 1. Install Required Packages
```bash
pip install -U diffusers torch pillow
```

### 2. HuggingFace Authentication
You need to accept the model license and authenticate:

1. Visit: https://huggingface.co/black-forest-labs/FLUX.1-Depth-dev
2. Click "Agree and access repository"
3. Login via CLI:
   ```bash
   huggingface-cli login
   ```

## Usage

### Basic Usage
Generate an image using the default depth map:
```bash
python flux_depth.py --prompt "A beautiful teddy bear standing in front of an ornate mirror, photorealistic, high quality"
```

### Full Example with All Parameters
```bash
python flux_depth.py \
  --prompt "A cute teddy bear in front of a golden mirror" \
  --depth-map results/depth/scene_depth.png \
  --output results/flux_depth/my_image.png \
  --height 1024 \
  --width 1024 \
  --num-inference-steps 30 \
  --guidance-scale 10.0 \
  --seed 42 \
  --device cuda
```

### Complete Workflow

#### Step 1: Generate Depth Map
First, create a depth map from your prompt:
```bash
python depth_map_extraction.py --prompt "a teddy bear in front of a mirror"
```
This creates: `results/depth/scene_depth.png`

#### Step 2: Generate Image with FLUX
Then use the depth map to generate the final image:
```bash
python flux_depth.py --prompt "A photorealistic teddy bear standing in front of an ornate antique mirror, studio lighting, high quality, detailed fur texture"
```

## Command-Line Arguments

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--prompt` | str | **Required** | Text prompt for image generation |
| `--depth-map` | str | `results/depth/scene_depth.png` | Path to depth map image |
| `--output` | str | `results/flux_depth/output.png` | Output image path |
| `--height` | int | `1024` | Output image height |
| `--width` | int | `1024` | Output image width |
| `--num-inference-steps` | int | `30` | Number of denoising steps (higher = better quality, slower) |
| `--guidance-scale` | float | `10.0` | How closely to follow the prompt (higher = stricter) |
| `--seed` | int | `42` | Random seed for reproducibility |
| `--device` | str | `cuda` | Device to use (`cuda` or `cpu`) |

## Tips for Better Results

### Prompt Engineering
- Be specific about style: "photorealistic", "artistic", "digital art"
- Include quality modifiers: "high quality", "detailed", "8k"
- Describe lighting: "studio lighting", "natural light", "dramatic shadows"
- Add artistic style if desired: "in the style of [artist]"

### Parameter Tuning
- **num-inference-steps**: 
  - 20-30: Fast, good quality
  - 30-50: Better quality, slower
  - 50+: Diminishing returns

- **guidance-scale**:
  - 5-7: More creative, less strict to prompt
  - 10-15: Balanced (recommended)
  - 15+: Very strict to prompt, may reduce quality

### Example Prompts

**Photorealistic style:**
```bash
--prompt "A photorealistic teddy bear with soft brown fur standing in front of an ornate baroque mirror with golden frame, studio lighting, highly detailed, 8k quality"
```

**Artistic style:**
```bash
--prompt "A whimsical watercolor teddy bear in front of a vintage mirror, soft pastel colors, artistic style, dreamy atmosphere"
```

**Fantasy style:**
```bash
--prompt "A magical glowing teddy bear standing before an enchanted mirror portal, mystical energy, fantasy art, vibrant colors"
```

## Troubleshooting

### Error: "You need to accept the license"
- Visit https://huggingface.co/black-forest-labs/FLUX.1-Depth-dev
- Click "Agree and access repository"
- Run `huggingface-cli login`

### Error: "CUDA out of memory"
- Use `--device cpu` (slower but uses less memory)
- Reduce `--height` and `--width` to 512 or 768
- Close other applications using GPU

### Slow generation
- This is normal for first run (model download)
- Reduce `--num-inference-steps` to 20
- Ensure you're using `--device cuda` if available

## Cache Location
The model will be cached in `../hf_cache/` to avoid re-downloading.
First download is ~20GB and may take a while.

## Performance Notes
- **GPU (CUDA)**: ~1-2 minutes for 30 steps at 1024x1024
- **CPU**: ~30-60 minutes for same settings
- **First run**: Add extra time for model download (~20GB)
