# save as run_langsam_all_masks.py
from PIL import Image
import numpy as np
import torch
import os
from datetime import datetime

from lang_sam import LangSAM  # https://github.com/luca-medeiros/lang-segment-anything


def _to_numpy_mask(m):
    """Convert mask to uint8 [0,255] HxW array."""
    if isinstance(m, torch.Tensor):
        m = m.detach().cpu().numpy()
    else:
        m = np.array(m)
    m = np.squeeze(m)

    if m.dtype == bool:
        m = m.astype(np.uint8) * 255
    elif m.dtype != np.uint8:
        m = (m >= 0.5).astype(np.uint8) * 255
    return m


def main():
    # 1) load inputs from current directory
    src_path = "val"
    image_list = []
    for file in os.listdir(src_path):
        if file.endswith(".png"):
            image_list.append(Image.open(os.path.join(src_path, file)).convert("RGB"))

    prompt_list = [
        "copper spirit kettle.mirror.",
        "chair.mirror.",
        "ginger jar.mirror.",
        "wooden barrel.mirror."
    ]
    # 2) init model
    model = LangSAM()
    # 3) predict
    results = model.predict(image_list, prompt_list)

    if not results or "masks" not in results[0] or len(results[0]["masks"]) == 0:
        raise RuntimeError("No mask returned by LangSAM for the given prompt/image.")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_path = "segmentation_result"

    for idx, result in enumerate(results):
        # mkdir if not exists the folder result1
        os.makedirs(f"{base_path}/result{idx}", exist_ok=True)
        # save the mask
        for j, m in enumerate(result["masks"]):
            mask_np = _to_numpy_mask(m)
            Image.fromarray(mask_np, mode="L").save(f"{base_path}/result{idx}/mask_{j}_{timestamp}.png")
            print(f"Saved {base_path}/result{idx}/mask_{j}_{timestamp}.png")

if __name__ == "__main__":
    main()
