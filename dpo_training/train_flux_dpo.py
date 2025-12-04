import argparse
import torch
import torch.nn.functional as F
from accelerate import Accelerator
from diffusers import FluxPipeline, FlowMatchEulerDiscreteScheduler
from peft import LoraConfig, get_peft_model, PeftModel
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import os

# --- 1. Dataset Class ---
class SynMirrorDPODataset(Dataset):
    def __init__(self, jsonl_path, tokenizer_2, tokenizer_clip, size=1024):
        self.data = []
        import json
        with open(jsonl_path, 'r') as f:
            for line in f:
                self.data.append(json.loads(line))
        
        self.size = size
        self.tokenizer_2 = tokenizer_2      # T5 Tokenizer
        self.tokenizer_clip = tokenizer_clip # CLIP Tokenizer
        
        self.transforms = transforms.Compose([
            transforms.Resize(size, interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.CenterCrop(size),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        
        # Load Images
        img_w = Image.open(item['winner_path']).convert("RGB")
        img_l = Image.open(item['loser_path']).convert("RGB")
        
        # Process Images
        pixel_w = self.transforms(img_w)
        pixel_l = self.transforms(img_l)
        
        # Process Text (Simplified for brevity - in prod use standard Flux processing)
        prompt = item['prompt']
        
        return {
            "pixel_w": pixel_w,
            "pixel_l": pixel_l,
            "prompt": prompt
        }

# --- 2. The DPO Loss for Flow Matching ---
def compute_dpo_loss(model_w_loss, model_l_loss, ref_w_loss, ref_l_loss, beta=0.1):
    """
    DPO Loss = -log sigmoid( beta * (Reward_Winner - Reward_Loser) )
    Reward = Reference_Loss - Model_Loss (Lower loss is better reward)
    """
    reward_w = ref_w_loss - model_w_loss
    reward_l = ref_l_loss - model_l_loss
    
    logits = beta * (reward_w - reward_l)
    loss = -torch.nn.functional.logsigmoid(logits)
    
    return loss.mean()

# --- 3. Main Training Loop ---
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_path", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="flux-dpo-lora")
    parser.add_argument("--steps", type=int, default=1000)
    args = parser.parse_args()

    accelerator = Accelerator(gradient_accumulation_steps=1)
    
    # A. Load Models
    print("Loading FLUX.1-dev...")
    model_id = "black-forest-labs/FLUX.1-dev"
    
    # We only train the Transformer (U-Net equivalent)
    pipeline = FluxPipeline.from_pretrained(model_id, torch_dtype=torch.bfloat16)
    
    transformer = pipeline.transformer
    vae = pipeline.vae
    text_encoder = pipeline.text_encoder
    text_encoder_2 = pipeline.text_encoder_2
    tokenizer = pipeline.tokenizer
    tokenizer_2 = pipeline.tokenizer_2
    
    # Freeze standard components
    vae.requires_grad_(False)
    text_encoder.requires_grad_(False)
    text_encoder_2.requires_grad_(False)
    transformer.requires_grad_(False)
    
    # B. Add LoRA Adapter
    print("Adding LoRA adapters...")
    lora_config = LoraConfig(
        r=16,
        lora_alpha=16,
        target_modules=["to_k", "to_q", "to_v", "to_out.0"], # Standard Attention targets
        init_lora_weights="gaussian"
    )
    transformer = get_peft_model(transformer, lora_config)
    transformer.print_trainable_parameters()
    
    # C. Optimizer
    optimizer = torch.optim.AdamW(transformer.parameters(), lr=1e-5)
    
    # D. Prepare Dataset
    dataset = SynMirrorDPODataset(args.dataset_path, tokenizer_2, tokenizer)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True)
    
    # E. Move to device
    transformer, optimizer, dataloader = accelerator.prepare(transformer, optimizer, dataloader)
    vae.to(accelerator.device)
    text_encoder.to(accelerator.device)
    text_encoder_2.to(accelerator.device)
    
    # --- Training Loop ---
    print("Starting DPO Training...")
    global_step = 0
    
    for batch in dataloader:
        if global_step >= args.steps: break
        
        with accelerator.accumulate(transformer):
            # 1. Encode Images to Latents (Winner & Loser)
            # (In production, pre-compute these to save VRAM)
            with torch.no_grad():
                latents_w = vae.encode(batch["pixel_w"].to(torch.bfloat16)).latent_dist.sample()
                latents_l = vae.encode(batch["pixel_l"].to(torch.bfloat16)).latent_dist.sample()
                latents_w = (latents_w - vae.config.shift_factor) * vae.config.scaling_factor
                latents_l = (latents_l - vae.config.shift_factor) * vae.config.scaling_factor
                
                # Encode Prompts
                # (Simplification: relying on pipeline logic for full encoding is complex, 
                # usually you pre-compute embeddings. Here we assume we have a function `encode_prompt`)
                # For this snippet, assume we have pre-computed `prompt_embeds` and `pooled_prompt_embeds`
                # prompt_embeds, pooled_prompt_embeds = pipeline.encode_prompt(batch['prompt'], ...)
                pass # Placeholder for text encoding logic

            # 2. Sample Timestep & Noise (Flow Matching)
            # FLUX uses Rectified Flow: t in [0, 1]
            bsz = latents_w.shape[0]
            t = torch.rand((bsz,), device=accelerator.device)
            noise = torch.randn_like(latents_w)
            
            # Interpolate (Forward Process)
            # x_t = (1-t)x_0 + t * noise
            # Target Velocity (u) = noise - x_0
            noisy_latents_w = (1 - t.view(-1,1,1,1)) * latents_w + t.view(-1,1,1,1) * noise
            noisy_latents_l = (1 - t.view(-1,1,1,1)) * latents_l + t.view(-1,1,1,1) * noise
            
            target_v_w = noise - latents_w
            target_v_l = noise - latents_l
            
            # 3. Forward Pass - Policy Model (LoRA Enabled)
            # We predict velocity 'v'
            # TODO: Add encoder_hidden_states and pooled_projections
            pred_v_w = transformer(hidden_states=noisy_latents_w, timestep=t).sample
            pred_v_l = transformer(hidden_states=noisy_latents_l, timestep=t).sample
            
            model_loss_w = F.mse_loss(pred_v_w, target_v_w, reduction="none").mean()
            model_loss_l = F.mse_loss(pred_v_l, target_v_l, reduction="none").mean()
            
            # 4. Forward Pass - Reference Model (LoRA Disabled)
            with torch.no_grad():
                with transformer.disable_adapter():
                    ref_v_w = transformer(hidden_states=noisy_latents_w, timestep=t).sample
                    ref_v_l = transformer(hidden_states=noisy_latents_l, timestep=t).sample
                    
                    ref_loss_w = F.mse_loss(ref_v_w, target_v_w, reduction="none").mean()
                    ref_loss_l = F.mse_loss(ref_v_l, target_v_l, reduction="none").mean()
            
            # 5. DPO Step
            loss = compute_dpo_loss(model_loss_w, model_loss_l, ref_loss_w, ref_loss_l, beta=0.5)
            
            accelerator.backward(loss)
            optimizer.step()
            optimizer.zero_grad()
            
            print(f"Step {global_step}: Loss={loss.item()}")
            global_step += 1

    # Save LoRA
    transformer.save_pretrained(args.output_dir)

if __name__ == "__main__":
    main()