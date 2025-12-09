import os
import json
import cv2
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from diffusers import FluxControlNetPipeline, FluxControlNetModel, FluxTransformer2DModel, AutoencoderKL, FlowMatchEulerDiscreteScheduler
from diffusers.optimization import get_scheduler
from peft import LoraConfig, get_peft_model, TaskType
from transformers import CLIPTokenizer, CLIPTextModel, T5Tokenizer, T5EncoderModel
from accelerate import Accelerator
from accelerate.utils import ProjectConfiguration
from tqdm.auto import tqdm
from PIL import Image
import numpy as np
import random

# ==========================================
# CONFIGURATION
# ==========================================
MODEL_NAME = "black-forest-labs/FLUX.1-dev"
CONTROLNET_MODEL_NAME = "Shakker-Labs/FLUX.1-dev-ControlNet-Depth"
OUTPUT_DIR = "flux_mirror_controlnet_lora"
DATASET_DIR = "./SynMirror"  # Point to your dataset root

# Weights for the Loss Function
# Mirror reflection is critical, so we give it 5.0x importance
MIRROR_LOSS_WEIGHT = 5.0 
# The object itself is also important, 2.0x importance
OBJECT_LOSS_WEIGHT = 2.0 

TRAIN_BATCH_SIZE = 1
GRADIENT_ACCUMULATION_STEPS = 4
LEARNING_RATE = 1e-4
MAX_TRAIN_STEPS = 5000
IMAGE_SIZE = 1024

# ==========================================
# 1. CUSTOM DATASET CLASS
# ==========================================
class SynMirrorDataset(Dataset):
    def __init__(self, root_dir, size=1024):
        self.root_dir = root_dir
        self.size = size
        self.metadata = []
        
        # Load metadata
        with open(os.path.join(root_dir, "metadata.jsonl"), "r") as f:
            for line in f:
                self.metadata.append(json.loads(line))

        # Transformations
        self.image_transforms = transforms.Compose([
            transforms.Resize((size, size), interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]) # Normalize to [-1, 1] for FLUX
        ])
        
        self.condition_transforms = transforms.Compose([
            transforms.Resize((size, size), interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.ToTensor(), # [0, 1]
        ])
        
        self.mask_transforms = transforms.Compose([
            transforms.Resize((size, size), interpolation=transforms.InterpolationMode.NEAREST),
            transforms.ToTensor(),
        ])

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx):
        entry = self.metadata[idx]
        
        # 1. Load RGB Target (Ground Truth)
        img_path = os.path.join(self.root_dir, entry["file_name"])
        image = Image.open(img_path).convert("RGB")
        
        # 2. Load Depth Condition
        depth_path = os.path.join(self.root_dir, entry["depth_file"])
        depth = Image.open(depth_path).convert("RGB") # ControlNet usually expects 3 channels
        
        # 3. Load Masks
        mirror_mask_path = os.path.join(self.root_dir, entry["mirror_masks"]) # Ensure jsonl key matches
        object_mask_path = os.path.join(self.root_dir, entry["object_masks"])
        
        mirror_mask = Image.open(mirror_mask_path).convert("L")
        object_mask = Image.open(object_mask_path).convert("L")

        # 4. Prompt
        prompt = entry["text"]

        return {
            "pixel_values": self.image_transforms(image),
            "conditioning_pixel_values": self.condition_transforms(depth),
            "mirror_mask": self.mask_transforms(mirror_mask),
            "object_mask": self.mask_transforms(object_mask),
            "prompt": prompt
        }

def collate_fn(examples):
    pixel_values = torch.stack([example["pixel_values"] for example in examples])
    conditioning_pixel_values = torch.stack([example["conditioning_pixel_values"] for example in examples])
    mirror_masks = torch.stack([example["mirror_mask"] for example in examples])
    object_masks = torch.stack([example["object_mask"] for example in examples])
    prompts = [example["prompt"] for example in examples]
    
    return {
        "pixel_values": pixel_values,
        "conditioning_pixel_values": conditioning_pixel_values,
        "mirror_masks": mirror_masks,
        "object_masks": object_masks,
        "prompts": prompts
    }

def compute_embeddings(batch, proportion_empty_prompts, flux_controlnet_pipeline, weight_dtype, is_train=True):
    prompt_batch = batch[args.caption_column]
    captions = []
    for caption in prompt_batch:
        if random.random() < proportion_empty_prompts:
            captions.append("")
        elif isinstance(caption, str):
            captions.append(caption)
        elif isinstance(caption, (list, np.ndarray)):
            # take a random caption if there are multiple
            captions.append(random.choice(caption) if is_train else caption[0])
    prompt_batch = captions
    prompt_embeds, pooled_prompt_embeds, text_ids = flux_controlnet_pipeline.encode_prompt(
        prompt_batch, prompt_2=prompt_batch
    )
    prompt_embeds = prompt_embeds.to(dtype=weight_dtype)
    pooled_prompt_embeds = pooled_prompt_embeds.to(dtype=weight_dtype)
    text_ids = text_ids.to(dtype=weight_dtype)

    # text_ids [512,3] to [bs,512,3]
    text_ids = text_ids.unsqueeze(0).expand(prompt_embeds.shape[0], -1, -1)
    return {"prompt_embeds": prompt_embeds, "pooled_prompt_embeds": pooled_prompt_embeds, "text_ids": text_ids}

# ==========================================
# 2. TRAINING SETUP
# ==========================================
def main():
    accelerator = Accelerator(
        gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
        mixed_precision="bf16",
        project_dir=OUTPUT_DIR
    )

    # Load Scheduler and Models
    noise_scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(MODEL_NAME, subfolder="scheduler")
    vae = AutoencoderKL.from_pretrained(MODEL_NAME, subfolder="vae", torch_dtype=torch.bfloat16)
    text_encoder = CLIPTextModel.from_pretrained(MODEL_NAME, subfolder="text_encoder", torch_dtype=torch.bfloat16)
    text_encoder_2 = T5EncoderModel.from_pretrained(MODEL_NAME, subfolder="text_encoder_2", torch_dtype=torch.bfloat16)
    tokenizer = CLIPTokenizer.from_pretrained(MODEL_NAME, subfolder="tokenizer")
    tokenizer_2 = T5Tokenizer.from_pretrained(MODEL_NAME, subfolder="tokenizer_2")
    transformer = FluxTransformer2DModel.from_pretrained(MODEL_NAME, subfolder="transformer", torch_dtype=torch.bfloat16)
    
    # LOAD CONTROLNET
    controlnet = FluxControlNetModel.from_pretrained(CONTROLNET_MODEL_NAME, torch_dtype=torch.bfloat16)

    pipeline = FluxControlNetPipeline(
        vae=vae,
        text_encoder=text_encoder,
        text_encoder_2=text_encoder_2,
        tokenizer=tokenizer,
        tokenizer_2=tokenizer_2,
        transformer=transformer,
        controlnet=controlnet,
        scheduler=noise_scheduler,
    )

    # FREEZE BASE MODELS
    vae.requires_grad_(False)
    text_encoder.requires_grad_(False)
    text_encoder_2.requires_grad_(False)
    transformer.requires_grad_(False)
    controlnet.requires_grad_(False) # We will only train LoRA adapters

    # ADD LORA TO CONTROLNET
    # We target the transformer blocks inside ControlNet
    lora_config = LoraConfig(
        r=16,
        lora_alpha=16,
        target_modules=["to_k", "to_q", "to_v", "to_out.0"], 
        bias="none",
    )
    controlnet.add_adapter(lora_config)
    
    # Move models to GPU
    vae.to(accelerator.device)
    text_encoder.to(accelerator.device)
    text_encoder_2.to(accelerator.device)
    transformer.to(accelerator.device)
    controlnet.to(accelerator.device)
    
    # Enable Gradient Checkpointing for memory saving
    controlnet.enable_gradient_checkpointing()

    # Optimizer (Only optimize ControlNet LoRA parameters)
    params_to_optimize = list(filter(lambda p: p.requires_grad, controlnet.parameters()))
    optimizer = torch.optim.AdamW(params_to_optimize, lr=LEARNING_RATE)

    # Dataset
    dataset = SynMirrorDataset(DATASET_DIR, size=IMAGE_SIZE)
    dataloader = DataLoader(dataset, batch_size=TRAIN_BATCH_SIZE, shuffle=True, collate_fn=collate_fn)

    # Prepare with Accelerator
    controlnet, optimizer, dataloader = accelerator.prepare(controlnet, optimizer, dataloader)

    # ==========================================
    # 3. TRAINING LOOP
    # ==========================================
    global_step = 0
    progress_bar = tqdm(range(MAX_TRAIN_STEPS), disable=not accelerator.is_local_main_process)
    
    controlnet.train()
    
    for epoch in range(100): # Arbitrary high number, we stop by global_step
        for batch in dataloader:
            with accelerator.accumulate(controlnet):
                # A. Encode Images to Latents (VAE)
                latents = vae.encode(batch["pixel_values"].to(dtype=torch.bfloat16)).latent_dist.sample()
                latents = latents * vae.config.scaling_factor
                
                # B. Add Noise (Flow Matching)
                noise = torch.randn_like(latents)
                bsz = latents.shape[0]
                timesteps = torch.rand((bsz,), device=latents.device) # Random t for Flow Matching
                
                # Interpolate between Noise and Image (x_t)
                noisy_latents = (1 - timesteps.view(bsz, 1, 1, 1)) * latents + timesteps.view(bsz, 1, 1, 1) * noise
                
                # C. Encode Prompts
                encoded_text = compute_embeddings(
                    batch=batch, # Our collate_fn returns "prompts" in the batch dict
                    proportion_empty_prompts=0.1, # 10% dropout for classifier-free guidance training
                    flux_controlnet_pipeline=pipeline,
                    weight_dtype=torch.bfloat16,
                    is_train=True
                )
                
                prompt_embeds = encoded_text["prompt_embeds"].to(accelerator.device)
                pooled_prompt_embeds = encoded_text["pooled_prompt_embeds"].to(accelerator.device)
                text_ids = encoded_text["text_ids"].to(accelerator.device)
                
                # D. ControlNet Forward Pass
                controlnet_image = batch["conditioning_pixel_values"].to(dtype=torch.bfloat16)
                
                down_block_res_samples, mid_block_sample = controlnet(
                    hidden_states=noisy_latents,
                    timestep=timesteps,
                    controlnet_cond=controlnet_image,
                    encoder_hidden_states=prompt_embeds, # Placeholder
                    pooled_projections=pooled_prompt_embeds, # Placeholder
                    return_dict=False,
                )

                # E. Predict Noise/Velocity (Teacher Model)
                # We feed the ControlNet residuals into the Base Transformer
                model_pred = transformer(
                    hidden_states=noisy_latents,
                    timestep=timesteps,
                    encoder_hidden_states=prompt_embeds,
                    pooled_projections=pooled_prompt_embeds,
                    block_controlnet_hidden_states=[sample for sample in down_block_res_samples],
                    mid_block_controlnet_hidden_state=mid_block_sample,
                    return_dict=False,
                )[0]

                # F. CALCULATE WEIGHTED LOSS
                # Target for Flow Matching is (noise - latents) usually, or velocity
                target = noise - latents 
                
                # 1. Base MSE Loss (per pixel)
                loss = F.mse_loss(model_pred.float(), target.float(), reduction="none")
                
                # 2. Resize Masks to Latent Space (Original is 1024x1024, Latent is 128x128)
                # VAE downsamples by factor of 8
                latent_h, latent_w = latents.shape[-2:]
                
                mirror_mask_resized = F.interpolate(batch["mirror_masks"], size=(latent_h, latent_w), mode="nearest")
                object_mask_resized = F.interpolate(batch["object_masks"], size=(latent_h, latent_w), mode="nearest")
                
                # 3. Create Weight Map
                # Base weight = 1.0
                # Mirror pixels = 1.0 + 5.0 = 6.0
                # Object pixels = 1.0 + 2.0 = 3.0
                weight_map = 1.0 + (mirror_mask_resized * MIRROR_LOSS_WEIGHT) + (object_mask_resized * OBJECT_LOSS_WEIGHT)
                
                # 4. Apply Weighted Loss
                weighted_loss = (loss * weight_map).mean()

                # Backprop
                accelerator.backward(weighted_loss)
                optimizer.step()
                optimizer.zero_grad()

            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1
                
            if global_step >= MAX_TRAIN_STEPS:
                break
    
    # Save
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        controlnet.save_pretrained(OUTPUT_DIR)

if __name__ == "__main__":
    main()