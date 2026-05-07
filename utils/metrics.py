# -----------------------------------------------------------------------------
# If you use this code in your research, please cite our paper:
# Blur-Resistant Hyperspectral Image Super-Resolution via Dual-Degradation Fusion Model
# Thanks
# -----------------------------------------------------------------------------
import torch

def sam_calculator(
    pred: torch.Tensor, 
    target: torch.Tensor, 
    eps: float = 1e-8,
    degree: bool = False
) -> torch.Tensor:

    pred = pred.float()
    target = target.float()

    original_dim = pred.dim()
    if original_dim == 4: 
        pred = pred.permute(0, 2, 3, 1).reshape(-1, pred.shape[1])  
        target = target.permute(0, 2, 3, 1).reshape(-1, target.shape[1])  
    elif original_dim == 2:  
        pred = pred.reshape(-1, pred.shape[1])  
        target = target.reshape(-1, target.shape[1]) 
    elif original_dim == 1: 
        pred = pred.unsqueeze(0)  
        target = target.unsqueeze(0) 
    else:
        raise ValueError(f"input shape {original_dim} wrong")

    dot_product = (pred * target).sum(dim=1)  # [N]
    pred_norm = torch.norm(pred, p=2, dim=1)  # [N]
    target_norm = torch.norm(target, p=2, dim=1)  # [N]
    cos_sim = dot_product / (pred_norm * target_norm + eps) 
    cos_sim = torch.clamp(cos_sim, -1.0, 1.0)
    sam = torch.acos(cos_sim)
    if degree:
        sam = sam * (180.0 / torch.pi)
    return sam.mean()


def uqi_calculator(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:

    pred = pred.squeeze(0)
    target = target.squeeze(0)

    C, H, W = pred.shape

    mu_pred = torch.mean(pred, dim=(1, 2), keepdim=True)  
    mu_target = torch.mean(target, dim=(1, 2), keepdim=True)
    sigma_pred = torch.mean((pred - mu_pred) ** 2, dim=(1, 2), keepdim=True) 
    sigma_target = torch.mean((target - mu_target) **2, dim=(1, 2), keepdim=True)  
    sigma_pred_target = torch.mean((pred - mu_pred) * (target - mu_target), dim=(1, 2), keepdim=True)
    numerator = 4 * sigma_pred_target * mu_pred * mu_target 
    denominator = (sigma_pred + sigma_target) * (mu_pred** 2 + mu_target **2) 
    denominator = torch.clamp(denominator, min=1e-10)
    uqi_per_channel = numerator / denominator 
    uqi = torch.mean(uqi_per_channel)  
    
    return uqi

def ergas_calculator(pred: torch.Tensor, target: torch.Tensor, scale_factor) -> torch.Tensor:

    pred = pred.squeeze(0)
    target = target.squeeze(0)

    C, H, W = pred.shape
    mse_per_channel = torch.mean((pred - target) **2, dim=(1, 2))  # [C]
    mean_target_per_channel = torch.mean(target, dim=(1, 2))  # [C]
    mean_target_per_channel = torch.clamp(mean_target_per_channel, min=1e-10)
    r = scale_factor
    ergas = (100.0 / r) * torch.sqrt( (1.0 / C) * torch.sum(mse_per_channel / (mean_target_per_channel** 2)) )
    
    return ergas