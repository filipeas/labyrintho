import torch
import numpy as np
from minerva.transforms.transform import (
    Transpose,
    PadCrop,
    CastTo
)

class Compose:
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, x):
        for t in self.transforms:
            x = t(x)
        return x

class Segmenter:
    def __init__(self, multimask_output: bool, height: float, width: float, model, device):
        self.multimask_output = multimask_output
        self.height = height
        self.width = width
        self.model = model
        self.device = device
        self.clear_prev_low_res_logits()
    
    def clear_prev_low_res_logits(self):
        self.prev_low_res_logits = None # torch.zeros(1, 1, H, W, device=self.device) # used to send last mask to current prediction

    def update_prev_low_res_logits(self, prev_low_res_logits):
        self.prev_low_res_logits = prev_low_res_logits
    
    def segment(self, image: np.ndarray, points: list):
        # apply transforms in image
        transform_pipeline = Compose([
            Transpose([2, 0, 1]),
            PadCrop(
                target_h_size=self.height,
                target_w_size=self.width,
                padding_mode="reflect",
                constant_values=0,
                seed=42,
            ),
            CastTo("float32")
        ])
        input_image = torch.from_numpy(transform_pipeline(image)).to(self.device)

        # convert positive and negative points for SAM pattern
        input_points = torch.tensor([[x, y] for x, y, _ in points], dtype=torch.long).unsqueeze(0).to(self.device)  # shape: (N, 2)
        input_labels = torch.tensor([label for _, _, label in points], dtype=torch.long).unsqueeze(0).to(self.device)  # shape: (N,)
        
        # send to model
        print(f"Segmentando com pontos={points} | image shape={input_image.shape}")
        batch_dict = {
            "image": input_image,
            "original_size": (input_image.shape[1], input_image.shape[2]),
            "point_coords": input_points,
            "point_labels": input_labels
        }
        if self.prev_low_res_logits is not None:
            batch_dict["mask_inputs"] = self.prev_low_res_logits
        
        self.model.eval()
        with torch.no_grad():
            outputs = self.model([batch_dict], multimask_output=self.multimask_output)
        low_res_logits = torch.stack([output["low_res_logits"].squeeze(0) for output in outputs]) # [B, 3, H, W] if multimask_output else [B, 1, H, W]
        masks_logits = torch.stack([output["masks_logits"].squeeze(0) for output in outputs]) # [B, 3, H, W] if multimask_output else [B, 1, H, W]
        
        del outputs, batch_dict
        torch.mps.empty_cache()

        return low_res_logits, masks_logits