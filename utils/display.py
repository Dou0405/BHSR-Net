# -----------------------------------------------------------------------------
# If you use this code in your research, please cite our paper:
# Blur-Resistant Hyperspectral Image Super-Resolution via Dual-Degradation Fusion Model
# Thanks
# -----------------------------------------------------------------------------
import os
import torch
import numpy as np
from scipy.io import savemat
from utils.dataloaders import get_dataloaders_dataparallel
from utils.metrics import *
import matplotlib.pyplot as plt
from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure
from models import BHSR


def to_grayscale(tensor):
    gray_tensor = torch.mean(tensor, dim=1, keepdim=True)
    return gray_tensor

def visualize_hsi(mat_path, rgb_channels, dataset='CAVE', save_dir=None):

    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec
    import numpy as np
    from scipy.io import loadmat
    import os

    data = loadmat(mat_path)
    pred = data['pred']
    HrHSI_true = data['HrHSI_true']
    
    num_channels = pred.shape[0]
    if len(rgb_channels) != 3:
        raise ValueError("rgb_channels must contain exactly 3 channel indices")
    for ch in rgb_channels:
        if ch < 0 or ch >= num_channels:
            raise ValueError(f"Channel {ch} out of range (0-{num_channels-1})")
    
    pred_rgb = np.stack([pred[ch] for ch in rgb_channels], axis=-1)
    true_rgb = np.stack([HrHSI_true[ch] for ch in rgb_channels], axis=-1)

    diff_rgb = true_rgb - pred_rgb
    diff = np.mean(np.abs(diff_rgb), axis=-1)
    
    fig = plt.figure(figsize=(18, 6))
    gs = gridspec.GridSpec(1, 3, width_ratios=[1, 1, 1])
    
    ax1 = plt.subplot(gs[0])
    im1 = ax1.imshow(pred_rgb)
    ax1.set_title(f'Predicted (Ch{",".join(map(str, rgb_channels))})', fontsize=12)
    ax1.axis('off')
    
    ax2 = plt.subplot(gs[1])
    im2 = ax2.imshow(true_rgb)
    ax2.set_title(f'Ground Truth (Ch{",".join(map(str, rgb_channels))})', fontsize=12)
    ax2.axis('off')
    
    ax3 = plt.subplot(gs[2])
    im3 = ax3.imshow(diff, cmap='viridis')
    ax3.set_title('RGB Difference (GT - Pred)', fontsize=12)
    ax3.axis('off')
    
    pos3 = ax3.get_position()
    cax = fig.add_axes([pos3.x0 + pos3.width + 0.01,
                        pos3.y0 - pos3.height * 0.063,
                        0.01,
                        pos3.height * 1.12])
    cbar = plt.colorbar(im3, cax=cax)
    plt.subplots_adjust(left=0.05, right=0.9, wspace=0.15)
    
    if save_dir is None:
        save_dir = os.path.dirname(mat_path)
    os.makedirs(save_dir, exist_ok=True)
    base_name = os.path.splitext(os.path.basename(mat_path))[0]
    ch_str = f"ch{rgb_channels[0]}_{rgb_channels[1]}_{rgb_channels[2]}"
    save_path = os.path.join(save_dir, f'{base_name}_rgb_{ch_str}.png')
    
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Visualization saved to: {save_path}")
    print(f"Image shape: C={pred.shape[0]}, H={pred.shape[1]}, W={pred.shape[2]}")
    print(f"Selected channels: {rgb_channels}")

def visualize_inputs(X, Y, HrHSI, save_path):

    X_np = X[0, 0].cpu().numpy()
    Y_np = Y[0, 0].cpu().numpy()
    HrHSI_np = HrHSI[0, 0].cpu().numpy()
    
    plt.figure(figsize=(15, 5))
    plt.subplot(131)
    plt.imshow(X_np, cmap='viridis')
    plt.colorbar()
    plt.title("Input X (First Channel)")
    plt.subplot(132)
    plt.imshow(Y_np, cmap='viridis')
    plt.colorbar()
    plt.title("Input Y (First Channel)")
    plt.subplot(133)
    plt.imshow(HrHSI_np, cmap='viridis')
    plt.colorbar()
    plt.title("Ground Truth HrHSI (First Channel)")
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

def run_display(model, stagenum, split, blur, log_dir, dataset, data_path, datarange, kernel_path = None, model_name='zsl', batch_size=1, scale=32, device=None, normalize=True,test_loader=None,sgm1=1,sgm2=1):
    save_dir = os.path.join(log_dir, 'results')
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    psf = torch.load(os.path.join(log_dir, 'psf.pth'))
    srf = torch.load(os.path.join(log_dir, 'srf.pth'))

    if torch.is_tensor(psf):
        psf = psf.cpu().numpy()
    if torch.is_tensor(srf):
        srf = srf.cpu().numpy()

    plt.figure(figsize=(8, 6))
    plt.imshow(psf, cmap='gray')
    plt.title("PSF Visualization")
    plt.colorbar()
    plt.savefig(os.path.join(save_dir, 'psf_visualization.png'))
    plt.close()

    plt.figure(figsize=(8, 6))
    plt.imshow(srf, cmap='gray')
    plt.title("SRF Visualization")
    plt.colorbar()
    plt.savefig(os.path.join(save_dir, 'srf_visualization.png'))
    plt.close()

    model_path = os.path.join(log_dir, 'best_model.pth')
    model.load_state_dict(torch.load(model_path))
    model.eval()

    all_psnrs = []
    all_ssims = []
    all_sams = []
    all_uqis = []
    all_ergas = []

    psnr_calculator = PeakSignalNoiseRatio(data_range=1).to(device)
    ssim_calculator = StructuralSimilarityIndexMeasure(data_range=1, gaussian_kernel=True, kernel_size=11, sigma=1.5).to(device)
    
    with torch.no_grad():
        for i, batch in enumerate(test_loader):
            LrHSI, HrMSI, HrHSI, realHrMSI = [tensor.float() for tensor in batch]
            LrHSI, HrMSI, HrHSI, realHrMSI = LrHSI.to(device), HrMSI.to(device), HrHSI.to(device), realHrMSI.to(device)
            
            if dataset == 'chikusei':
                rgb_channels = [55, 35, 11]
            elif dataset == 'CAVE':
                rgb_channels = [29, 15, 5]
            elif dataset == 'Harvard':
                rgb_channels = [29, 15, 7]                
            elif dataset == 'houston':
                rgb_channels = [42, 34, 10]
            elif dataset == 'liao':
                rgb_channels = [28, 18, 9]

            input_visualization_path = os.path.join(save_dir, f'inputs_{i}.png')
            visualize_inputs(LrHSI, HrMSI, HrHSI, input_visualization_path)

            pred, y_ = model(LrHSI, HrMSI)
            pred = pred[-1].to(device)
            HrHSI = HrHSI.to(device)
            
            pred = torch.clamp(pred, 0, 1)

            current_psnr = psnr_calculator(pred, HrHSI).item()
            current_ssim = ssim_calculator(pred, HrHSI).item()
            current_sam = sam_calculator(pred, HrHSI).item()
            current_uqi = uqi_calculator(pred, HrHSI).item()
            current_ergas = ergas_calculator(pred, HrHSI, scale_factor=scale).item()

            print(f"Image {i}: PSNR: {current_psnr:.2f}, SSIM: {current_ssim:.4f}, SAM: {current_sam:.4f}, UQI: {current_uqi:.4f}, ERGAS: {current_ergas:.4f}")
            if not np.isnan(current_psnr) and not np.isinf(current_psnr):
                all_psnrs.append(current_psnr)
            else:
                print(f"Warning: PSNR for image {i} is NaN or inf.")

            if not np.isnan(current_ssim) and not np.isinf(current_ssim):
                all_ssims.append(current_ssim)
            else:
                print(f"Warning: SSIM for image {i} is NaN or inf.")
            
            if not np.isnan(current_sam) and not np.isinf(current_sam):
                all_sams.append(current_sam)
            else:
                print(f"Warning: SAM for image {i} is NaN or inf.")

            if not np.isnan(current_uqi) and not np.isinf(current_uqi):
                all_uqis.append(current_uqi)
            else:
                print(f"Warning: UQI for image {i} is NaN or inf.")

            if not np.isnan(current_ergas) and not np.isinf(current_ergas):
                all_ergas.append(current_ergas)
            else:
                print(f"Warning: ERGAS for image {i} is NaN or inf.")
            pred_np = pred.cpu().numpy().squeeze()
            HrHSI_np = HrHSI.cpu().numpy().squeeze()

            mat_path = os.path.join(save_dir, f'predicted_hsi_{i}.mat')
            savemat(mat_path, {'pred': pred_np, 'HrHSI_true': HrHSI_np})

            visualize_hsi(mat_path, rgb_channels=rgb_channels, save_dir=save_dir, dataset=dataset)

    if all_psnrs and all_ssims:
        mean_psnr = np.mean(all_psnrs)
        mean_ssim = np.mean(all_ssims)
        mean_sam = np.mean(all_sams)
        mean_uqi = np.mean(all_uqis)
        mean_ergas = np.mean(all_ergas)
        std_psnr = np.std(all_psnrs)
        std_ssim = np.std(all_ssims)
        std_sam = np.std(all_sams)
        std_uqi = np.std(all_uqis)
        std_ergas = np.std(all_ergas)

        print("\nOverall Results:")
        print(f"Mean PSNR: {mean_psnr:.2f} ± {std_psnr:.2f}")
        print(f"Mean SSIM: {mean_ssim:.4f} ± {std_ssim:.4f}")
        print(f"Mean SAM:  {mean_sam:.3f} ± {std_sam:.3f}")
        print(f"Mean UQI:  {mean_uqi:.4f} ± {std_uqi:.4f}")
        print(f"Mean ERGAS: {mean_ergas:.4f} ± {std_ergas:.4f}")
        
        with open(os.path.join(save_dir, 'metrics.txt'), 'w') as f:
            f.write(f"Mean PSNR: {mean_psnr:.2f} ± {std_psnr:.2f}\n")
            f.write(f"Mean SSIM: {mean_ssim:.4f} ± {std_ssim:.4f}\n")
            f.write(f"Mean SAM:  {mean_sam:.3f}  ± {std_sam:.3f}\n")
            f.write(f"Mean UQI:  {mean_uqi:.4f} ± {std_uqi:.4f}\n")
            f.write(f"Mean ERGAS: {mean_ergas:.4f} ± {std_ergas:.4f}\n")
            f.write("\nPer-image metrics:\n")
            for i, (psnr, ssim) in enumerate(zip(all_psnrs, all_ssims)):
                f.write(f"Image {i}: PSNR: {psnr:.2f}, SSIM: {ssim:.4f}, SAM: {all_sams[i]:.4f}, UQI: {all_uqis[i]:.4f}, ERGAS: {all_ergas[i]:.4f}\n")
    else:
        print("No valid PSNR or SSIM values were calculated.")
