import os
import json
import random
import torch
from diffusers import FluxPipeline, FluxControlPipeline
from PIL import Image

# Import your refactored function
from depth_map_extraction import generate_depth_for_prompt

# Configuration
DATASET_DIR = "dpo_dataset"
PROMPTS_FILE = "dpo_dataset/prompts.txt"  # A text file with 1 prompt per line
DEVICE = "cuda"

# Camera parameter ranges for randomization
CAMERA_DISTANCE_RANGE = (4.0, 7.0)      # Distance from origin
CAMERA_ELEVATION_RANGE = (15.0, 35.0)   # Elevation angle in degrees
# Azimuth: randomly choose from left or right view (avoid center)
CAMERA_AZIMUTH_LEFT_RANGE = (-30.0, -15.0)   # Left side view
CAMERA_AZIMUTH_RIGHT_RANGE = (15.0, 30.0)    # Right side view


def setup_pipelines():
    print("Loading models...")
    # 1. The "Loser" Generator (Vanilla FLUX)
    # This represents the un-aligned model we want to improve
    vanilla_pipe = FluxPipeline.from_pretrained(
        "black-forest-labs/FLUX.1-dev", 
        torch_dtype=torch.bfloat16
    ).to(DEVICE)

    # 2. The "Winner" Generator (Your Depth Pipeline)
    # This acts as the "Teacher" or "Oracle"
    depth_pipe = FluxControlPipeline.from_pretrained(
        "black-forest-labs/FLUX.1-Depth-dev", 
        torch_dtype=torch.bfloat16
    ).to(DEVICE)
    
    return vanilla_pipe, depth_pipe


def randomize_camera_params():
    """
    Randomize camera parameters within predefined ranges.
    Azimuth is randomly chosen from either left or right side view.
    
    Returns:
        tuple: (camera_distance, camera_elevation, camera_azimuth)
    """
    camera_distance = random.uniform(*CAMERA_DISTANCE_RANGE)
    camera_elevation = random.uniform(*CAMERA_ELEVATION_RANGE)
    
    # Randomly choose between left or right side view
    azimuth_range = random.choice([CAMERA_AZIMUTH_LEFT_RANGE, CAMERA_AZIMUTH_RIGHT_RANGE])
    camera_azimuth = random.uniform(*azimuth_range)
    
    return camera_distance, camera_elevation, camera_azimuth


def main():
    os.makedirs(f"{DATASET_DIR}/images", exist_ok=True)
    vanilla_pipe, depth_pipe = setup_pipelines()
    
    # List to store DPO metadata
    dpo_entries = []

    with open(PROMPTS_FILE, 'r') as f:
        prompts = [line.strip() for line in f if line.strip()]

    print(f"Found {len(prompts)} prompts to process.")

    for idx, prompt in enumerate(prompts):
        print(f"\n{'='*60}")
        print(f"Processing [{idx+1}/{len(prompts)}]: {prompt}")
        
        try:
            # Randomize camera parameters for each sample
            camera_distance, camera_elevation, camera_azimuth = randomize_camera_params()
            print(f"Camera params: distance={camera_distance:.2f}, "
                  f"elevation={camera_elevation:.2f}°, azimuth={camera_azimuth:.2f}°")
            
            # --- STEP A: Generate the 3D Ground Truth (Depth) ---
            # Call your PyTorch3D logic with randomized camera parameters
            depth_map_path = generate_depth_for_prompt(
                prompt=prompt,
                output_dir=f"{DATASET_DIR}/temp_depth",
                camera_distance=camera_distance,
                camera_elevation=camera_elevation,
                camera_azimuth=camera_azimuth
            )
            depth_image = Image.open(depth_map_path).convert("RGB")

            # --- STEP B: Generate the "Winner" (Chosen) ---
            # Conditioned on the correct physics
            print("Generating winner image (depth-conditioned)...")
            winner_image = depth_pipe(
                prompt=prompt, 
                control_image=depth_image,
                height=1024, width=1024,
                num_inference_steps=30,
                guidance_scale=10.0
            ).images[0]
            
            winner_filename = f"{idx:05d}_winner.png"
            winner_path = os.path.join(DATASET_DIR, "images", winner_filename)
            winner_image.save(winner_path)
            print(f"✓ Winner saved: {winner_filename}")

            # --- STEP C: Generate the "Loser" (Rejected) ---
            # Unconditioned, likely physically incorrect
            print("Generating loser image (vanilla)...")
            loser_image = vanilla_pipe(
                prompt=prompt,
                height=1024, width=1024,
                num_inference_steps=30,
                guidance_scale=3.5 
            ).images[0]
            
            loser_filename = f"{idx:05d}_loser.png"
            loser_path = os.path.join(DATASET_DIR, "images", loser_filename)
            loser_image.save(loser_path)
            print(f"✓ Loser saved: {loser_filename}")

            # --- STEP D: Log Entry ---
            entry = {
                "prompt": prompt,
                "chosen": f"images/{winner_filename}",
                "rejected": f"images/{loser_filename}",
                "depth_guide": depth_map_path,
                "camera_distance": camera_distance,
                "camera_elevation": camera_elevation,
                "camera_azimuth": camera_azimuth
            }
            dpo_entries.append(entry)
            
            # Save JSONL incrementally (safety against crashes)
            with open(os.path.join(DATASET_DIR, "dpo_data.jsonl"), 'w') as jf:
                for item in dpo_entries:
                    jf.write(json.dumps(item) + "\n")
            
            print(f"✓ Entry saved to dpo_data.jsonl")

        except Exception as e:
            print(f"✗ Failed on prompt '{prompt}': {e}")
            import traceback
            traceback.print_exc()

    print(f"\n{'='*60}")
    print("Data generation complete!")
    print(f"Total entries: {len(dpo_entries)}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()