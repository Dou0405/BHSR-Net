# -----------------------------------------------------------------------------
# If you use this code in your research, please cite our paper:
# Blur-Resistant Hyperspectral Image Super-Resolution via Dual-Degradation Fusion Model
# Thanks
# -----------------------------------------------------------------------------
import os
import sys
import socket
import subprocess
import torch
import numpy as np
import time
import random
import matplotlib.pyplot as plt
import argparse
from models import *
from loss import *
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure
from utils.degradation import getPSF, getSRF
from utils.display import run_display

from utils.metrics import *
from utils.dataloaders import get_dataloaders_dataparallel


# ===================== 1. Random Seed Setup =====================
def set_seed(seed: int = 100):
    """
    Fix all random seeds to ensure full experiment reproducibility
    :param seed: Global random seed value
    """
    os.environ['PYTHONHASHSEED'] = str(seed)  # Fix Python hash seed
    random.seed(seed)                          # Fix Python built-in random seed
    np.random.seed(seed)                       # Fix NumPy random seed
    torch.manual_seed(seed)                    # Fix PyTorch CPU random seed
    torch.cuda.manual_seed(seed)               # Fix PyTorch single-GPU random seed
    torch.cuda.manual_seed_all(seed)           # Fix PyTorch multi-GPU random seed
    torch.backends.cudnn.deterministic = True  # Enforce deterministic CuDNN algorithms
    torch.backends.cudnn.benchmark = False     # Disable CuDNN auto-optimization (critical for reproducibility)
    print(f"set random seed as: {seed}")


set_seed(40)  # Set global random seed to 40


# ===================== 2. Dataset Split Definitions =====================
# Standard train/test splits for common hyperspectral datasets (from literature)
CAVE_MHF_split = {  'test': ['beads.mat', 'cloth.mat', 'face.mat', 'fake_and_real_food.mat', 'fake_and_real_lemons.mat', 
                           'fake_and_real_peppers.mat', 'fake_and_real_strawberries.mat', 'fake_and_real_sushi.mat', 'glass_tiles.mat', 'oil_painting.mat', 
                           'paints.mat', 'photo_and_face.mat', 'pompoms.mat', 'real_and_fake_apples.mat', 'real_and_fake_peppers.mat', 'sponges.mat', 
                           'stuffed_toys.mat', 'superballs.mat', 'thread_spools.mat', 'watercolors.mat'], 
                    'train': ['balloons.mat', 'cd.mat', 'chart_and_stuffed_toy.mat', 'clay.mat', 'egyptian_statue.mat', 'fake_and_real_beers.mat', 
                        'fake_and_real_lemon_slices.mat', 'fake_and_real_tomatoes.mat', 'feathers.mat', 'flowers.mat', 'hairs.mat', 'jelly_beans.mat']}

CAVE_UTAL_split = {'test': [ 'real_and_fake_apples.mat', 'superballs.mat', 'chart_and_stuffed_toy.mat', 'hairs.mat',  'fake_and_real_lemons.mat',
                            'fake_and_real_lemon_slices.mat', 'fake_and_real_sushi.mat', 'egyptian_statue.mat', 'glass_tiles.mat', 'jelly_beans.mat',
                            'fake_and_real_peppers.mat', 'clay.mat', 'pompoms.mat', 'watercolors.mat', 'fake_and_real_tomatoes.mat', 'flowers.mat', 
                            'paints.mat', 'photo_and_face.mat', 'cloth.mat', 'beads.mat'],
                  'train': ['balloons.mat', 'cd.mat', 'face.mat', 'fake_and_real_food.mat', 'fake_and_real_strawberries.mat',
                            'fake_and_real_beers.mat', 'stuffed_toys.mat', 'oil_painting.mat', 'thread_spools.mat', 'sponges.mat',
                            'real_and_fake_peppers.mat', 'feathers.mat']}

Harvard_split = {'test': ['imgd9','imgd7','imgc1','imgb2','imgh3','imgf7','imge7','imga1','imga7','img1','imgd2',
                          'imge4','imgb0','imgc5','imgf6','imgb6','imga5','imgd8','imgf1','imgh0','imgb5','imgb4',
                          'imga2','imgc9','imgb9','imgh1','imgc2','imge3','imgb8','imgf5','imgc7','imgf4','imgf2',
                          'imge1', 'imgb1', 'imgd3', 'img2',  'imge5', 'imgc8', 'imge2'],  
                'train': ['imge6', 'imgc4', 'imgf8', 'imgb7', 'imgd4',  'imga6', 'imgh2', 'imgb3', 'imgf3',  'imge0']}

chikusei_split = {'test':['chikusei01','chikusei02','chikusei03','chikusei04','chikusei05','chikusei06','chikusei07','chikusei08'],
                  'train':['chikusei09','chikusei10','chikusei11','chikusei12','chikusei13','chikusei14','chikusei15','chikusei16']}

houston_split ={'test':['img2','img3','img4'],'train':['img1']}

liao_split ={'test':['img0'],'train':['img1','img2','img3','img4','img5']}  # 5 training samples have overlapping areas but no overlap with test sample; region matches our paper


# ===================== 3. Command Line Argument Parsing =====================
parser = argparse.ArgumentParser(description="Train a hyperspectral image super-resolution (HSI-SR) model.")

# Model and data configuration
parser.add_argument("--model", type=str, default='UMMHF', help="Model architecture name.")
parser.add_argument("--dataset", type=str, default='CAVE', help="Dataset name (CAVE/Harvard/chikusei/houston/liao).")
parser.add_argument("--stagenum", type=int, default=10, help="Number of iterative stages in the model.")
parser.add_argument("--batch_size", type=int, default=16, help="Training batch size per GPU.")
parser.add_argument("--split", type=str, default='cave_utal', help="Dataset split protocol (mhf/utal).")

# Training hyperparameters
parser.add_argument("--epochs", type=int, default=200, help="Maximum number of training epochs.")
parser.add_argument("--lr", type=float, default=1e-2, help="Initial learning rate.")
parser.add_argument("--patience_limit", type=int, default=100, help="Early stopping patience (epochs without improvement).")

# Degradation parameters (spatial/spectral)
parser.add_argument("--scale", type=int, default=32, help="Spatial downsampling scale factor.")
parser.add_argument("--kernel_size", type=int, default=7, help="Kernel size for spatial blur (PSF).")
parser.add_argument("--sigma", type=float, default=4, help="Standard deviation for Gaussian PSF.")
parser.add_argument("--blur", type=float, default=0.0, help="Additive Gaussian blur level (dB).")

# Additional functionality flags
parser.add_argument("--resume", type=str, help="Path to pretrained checkpoint for resuming training.")
parser.add_argument("--anisotropy", action="store_true", help="Use anisotropic Gaussian PSF instead of isotropic.")
parser.add_argument("--Gaussian", action="store_true", help="Enable Gaussian blur degradation.")
parser.add_argument("--pre_psf", type=str, default=None, help="Path to predefined PSF kernel file.")
parser.add_argument("--angle", type=float, default=0, help="Rotation angle for anisotropic Gaussian PSF (degrees).")
parser.add_argument("--display", action="store_true", help="Visualize reconstruction results after training.")
parser.add_argument("--normalize", action="store_true", help="Normalize input data to [0,1] range.")
parser.add_argument("--sgm1", type=float, default=1, help="Sigma1 parameter for BHSR model.")
parser.add_argument("--sgm2", type=float, default=2, help="Sigma2 parameter for BHSR model.")


# Parse command line arguments
args = parser.parse_args()

# ===================== 4. Parameter Initialization & Dataset Configuration =====================
# Extract parsed arguments
model_name = args.model
dataset = args.dataset
datarange = None
sgm1 = args.sgm1
sgm2 = args.sgm2
srf = ""
C = 31  # Number of spectral bands in high-resolution HSI
c = 3   # Number of spectral bands in high-resolution MSI

# Configure dataset-specific parameters (data range, SRF path, band counts, split)
if dataset == 'CAVE':
    datarange = 65535
    srf = "xxxx/Nikon_srf.mat"
    C = 31
    c = 3
    split = CAVE_UTAL_split
elif dataset == 'Harvard':
    datarange = 0.0616312
    srf = "xxxx/Nikon_srf.mat"
    C = 31
    c = 3
    split = Harvard_split
elif dataset == 'chikusei':
    datarange = 15133
    srf = "xxxx/landsat_srf.mat"
    C = 128
    c = 3
    split = chikusei_split
elif dataset == 'houston':
    datarange = 25924
    srf = "xxxx/srf_houston18_worldview2.mat"
    C = 46
    c = 8
    split = houston_split
elif dataset == 'liao':
    srf = None
    C = 149
    c = 8
    split = liao_split

# Extract remaining training parameters
stagenum = args.stagenum
batch_size = args.batch_size
epochs = args.epochs
lr = args.lr
patience_limit = args.patience_limit
scale = args.scale
sigma = args.sigma
anisotropy = args.anisotropy
Gaussian = args.Gaussian
angle = args.angle
kernel_size = args.kernel_size
blur = args.blur
split_name = args.split
resume_model = args.resume
psf_path = args.pre_psf
normalize = args.normalize

# Validate kernel size (must be odd for symmetric padding)
if kernel_size % 2 == 0:
    raise ValueError("Kernel size must be an odd integer.")


# ===================== 5. Logging & Environment Setup =====================
# Generate timestamp for unique log directory naming
now = datetime.now()
time_str = now.strftime("%Y-%m-%d_%H-%M-%S")

# Create experiment log directory
log_dir = os.path.join("logs", f"{dataset}{time_str}stg{stagenum}scale{scale}blur{blur}")
if not os.path.exists(log_dir):
    os.makedirs(log_dir)

# Create log file path
log_file = os.path.join(log_dir, 'output.txt')

# Logger class: Simultaneously output to console and log file
class Logger:
    def __init__(self, log_file):
        self.terminal = sys.stdout  # Save original stdout
        self.log = open(log_file, 'w')  # Open log file in write mode

    def write(self, message):
        self.terminal.write(message)  # Write to console
        self.log.write(message)       # Write to log file
        self.log.flush()              # Force immediate write to disk

    def flush(self):
        self.terminal.flush()
        self.log.flush()

# Redirect standard output to our custom Logger
sys.stdout = Logger(log_file)


# Set computation device (GPU preferred)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Enforce GPU usage (CPU training is not supported for this model)
if device.type == 'cpu':
    raise RuntimeError("CUDA is not available. Please check your GPU installation.")


# ===================== 6. Degradation Kernel Generation (PSF/SRF) =====================
# Generate/load Point Spread Function (PSF: spatial blur) and Spectral Response Function (SRF: spectral downsampling)
psf, psf_name = getPSF(
    kernel_size=kernel_size, 
    scale=scale, 
    sigma=sigma, 
    anisotropy=anisotropy, 
    Gaussian=Gaussian, 
    angle=angle, 
    predefined_kernel=psf_path
)
srf, srf_name = getSRF(srf)  

# Print core experiment configuration
print(f"Model: {model_name}, StageNum: {stagenum}, Batch size: {batch_size}, Split: {split_name}, "
      f"Learning rate: {lr}, Patience limit: {patience_limit}, Scale: {scale}, "
      f"PSF: {psf_name}, SRF: {srf_name}, Blur: {blur}, Normalize: {normalize}")


# ===================== 7. Data Loading =====================
print("Loading data...")
# Get multi-GPU compatible data loaders
train_loader, test_loader = get_dataloaders_dataparallel(
    batch_size=batch_size, 
    split=split, 
    num_workers=8 * torch.cuda.device_count(),  # 8 workers per GPU
    data_path='xxx', 
    psf=psf, 
    srf=srf, 
    scale_factor=scale, 
    dataset=dataset, 
    datarange=datarange, 
    blur=blur, 
    normalize=normalize, 
    kernel_path=psf_name
)


# ===================== 8. Model Initialization =====================
print("Loading model...")

# Initialize BHSR model and wrap with DataParallel for multi-GPU training
model = BHSR(stage_num=stagenum, C=C, c=c, sigma1=sgm1, sigma2=sgm2)
model = torch.nn.DataParallel(model).to(device)  

# Load pretrained checkpoint if specified
if resume_model:
    if os.path.isfile(resume_model):
        print(f"Loading pretrained weights from: {resume_model}")
        model.load_state_dict(torch.load(resume_model, map_location=device))
    else:
        raise FileNotFoundError(f"Checkpoint not found at: {resume_model}")
    
print(f"{model_name} training on device: {device}")


# ===================== 9. Optimizer, Loss & Learning Rate Scheduler =====================
optimizer = torch.optim.Adam(model.parameters(), lr=lr)  # Adam optimizer
# Reduce learning rate when validation loss plateaus
scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.7, patience=20, verbose=True)
# Custom composite loss function for HSI-SR
criterion = Losses(scale=scale, model_name=model_name, blur=blur).to(device)


# ===================== 10. Degradation Kernel Saving & Visualization =====================
# Save PSF and SRF tensors for reproducibility
torch.save(psf, os.path.join(log_dir, 'psf.pth'))
torch.save(srf, os.path.join(log_dir, 'srf.pth'))

# Visualize and save PSF as image
def save_psf_visualization(psf, save_path):
    plt.figure(figsize=(10, 10))
    psf_vis = psf.squeeze()
    if len(psf_vis.shape) > 2:
        psf_vis = np.mean(psf_vis, axis=0)  # Average over spectral bands for visualization
    plt.imshow(psf_vis, cmap='viridis')
    plt.colorbar(label='Intensity')
    plt.title('Point Spread Function (PSF)')
    plt.axis('off')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

# Visualize and save SRF as image
def save_srf_visualization(srf, save_path):
    plt.figure(figsize=(10, 5))
    srf_vis = srf.squeeze()
    plt.imshow(srf_vis, cmap='viridis', aspect='auto')
    plt.colorbar(label='Response')
    plt.title('Spectral Response Function (SRF)')
    plt.xlabel('Input Spectral Band')
    plt.ylabel('Output Spectral Band')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

# Generate and save visualizations
save_psf_visualization(psf, os.path.join(log_dir, 'psf.png'))
save_srf_visualization(srf, os.path.join(log_dir, 'srf.png'))


# ===================== 11. Pre-Training Setup & TensorBoard =====================
# Print experiment summary
print(f"Training started at: {time_str}")
print(f"PSF configuration: {psf_name}")
print(f"SRF configuration: {srf_name}")
print(f"Total trainable parameters: {sum(p.numel() for p in model.parameters()):,}")
print("-" * 80)

# Find a free network port for TensorBoard
def find_free_port():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(('', 0))  # Bind to a random available port
        return s.getsockname()[1]  # Return the port number

free_port = find_free_port()

# Initialize TensorBoard writer
writer = SummaryWriter(log_dir)

# Initialize tracking variables for best model and early stopping
best_val_loss = float("inf")
best_val_psnr = -float("inf")
patience = 0

# Start TensorBoard process in the background
subprocess.Popen([
    os.path.join(os.path.dirname(sys.executable), 'tensorboard'),
    '--logdir', log_dir,
    '--host', 'localhost',
    '--port', str(free_port),
    '--load_fast=false'
])


# ===================== 12. Training & Validation Functions =====================
def train_one_epoch(model, dataloader, optimizer, criterion, device, epoch):
    """
    Train the model for one complete epoch
    :param model: PyTorch model instance
    :param dataloader: Training data loader
    :param optimizer: Optimizer instance
    :param criterion: Loss function
    :param device: Computation device (cuda/cpu)
    :param epoch: Current epoch number
    :return: Average training loss over the epoch
    """
    model.train()  # Set model to training mode (enables dropout/batch norm)
    train_loss = 0.0
    
    for batch in dataloader: 
        # Unpack batch data: LrHSI (Low-res HSI), HrMSI (High-res MSI), HrHSI (Ground truth High-res HSI)
        LrHSI, HrMSI, HrHSI = [tensor.float() for tensor in batch]  
        LrHSI, HrMSI, HrHSI = LrHSI.to(device), HrMSI.to(device), HrHSI.to(device) 

        # Forward pass: Get model predictions
        pred, y_ = model(LrHSI, HrMSI) 

        # Calculate composite loss
        loss = criterion(pred, HrHSI, y_, HrMSI, epoch)  

        # Backward pass and optimization
        optimizer.zero_grad()           # Clear accumulated gradients
        loss.backward(retain_graph=True) # Compute gradients
        # Gradient clipping to prevent exploding gradients
        if dataset != 'liao':
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
        else:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
        optimizer.step()                # Update model weights
        
        # Accumulate loss (adjust for batch size and number of GPUs)
        train_loss += loss.item() * HrHSI.size(0) * torch.cuda.device_count()

    return train_loss / len(dataloader.dataset)  # Return average loss per sample


def test_one_epoch(model, stage_num, dataloader, criterion, device, writer=None, epoch=None):
    """
    Evaluate the model on the validation/test set
    :param model: PyTorch model instance
    :param stage_num: Number of model stages
    :param dataloader: Validation/test data loader
    :param criterion: Loss function
    :param device: Computation device (cuda/cpu)
    :param writer: TensorBoard writer instance (optional)
    :param epoch: Current epoch number (optional)
    :return: Tuple of (avg_test_loss, avg_psnr, avg_ssim, avg_sam, avg_uqi, avg_ergas)
    """
    model.eval()  # Set model to evaluation mode (disables dropout/batch norm)
    test_loss = 0.0

    # Initialize metric calculators
    psnr_calculator = PeakSignalNoiseRatio(data_range=1).to(device)
    ssim_calculator = StructuralSimilarityIndexMeasure(
        data_range=1, gaussian_kernel=True, kernel_size=11, sigma=1.5
    ).to(device)

    # Initialize metric accumulators
    total_psnr = 0.0
    total_ssim = 0.0
    total_sam = 0.0
    total_uqi = 0.0
    total_ergas = 0.0
    num_batches = 0

    with torch.no_grad():  # Disable gradient computation (saves memory and speed)
        for batch in dataloader:
            # Unpack batch data
            LrHSI, HrMSI, HrHSI, _ = [tensor.float() for tensor in batch] 
            LrHSI, HrMSI, HrHSI = LrHSI.to(device), HrMSI.to(device), HrHSI.to(device) 

            # Forward pass
            pred, y_ = model(LrHSI, HrMSI) 
            
            # Calculate loss
            test_loss += criterion(pred, HrHSI, y_, HrMSI, epoch)
            
            # Calculate evaluation metrics (use final stage output pred[-1])
            batch_psnr = psnr_calculator(pred[-1], HrHSI).item()
            batch_ssim = ssim_calculator(pred[-1], HrHSI).item()
            batch_sam = sam_calculator(pred[-1], HrHSI).item()
            batch_uqi = uqi_calculator(pred[-1], HrHSI).item()
            batch_ergas = ergas_calculator(pred[-1], HrHSI, scale_factor=scale).item()

            # Accumulate metrics
            total_psnr += batch_psnr
            total_ssim += batch_ssim
            total_sam += batch_sam
            total_uqi += batch_uqi
            total_ergas += batch_ergas

            num_batches += 1

    # Compute average metrics
    avg_psnr = total_psnr / num_batches
    avg_ssim = total_ssim / num_batches
    avg_sam = total_sam / num_batches
    avg_uqi = total_uqi / num_batches
    avg_ergas = total_ergas / num_batches

    # Log metrics to TensorBoard
    if writer is not None and epoch is not None:
        writer.add_scalar("Metrics/PSNR", avg_psnr, epoch)
        writer.add_scalar("Metrics/SSIM", avg_ssim, epoch)
        writer.add_scalar("Metrics/SAM", avg_sam, epoch)
        writer.add_scalar("Metrics/UQI", avg_uqi, epoch)
        writer.add_scalar("Metrics/ERGAS", avg_ergas, epoch)
        writer.add_scalar("Test/Loss", test_loss / len(dataloader), epoch)

    return (test_loss / len(dataloader.dataset), 
            avg_psnr, avg_ssim, avg_sam, avg_uqi, avg_ergas)


# ===================== 13. Main Training Loop =====================
for epoch in range(epochs):

    start_time = time.time()  # Record epoch start time

    # 1. Train for one epoch
    train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device, epoch)

    # 2. Evaluate on validation set
    test_loss, avg_psnr, avg_ssim, avg_sam, avg_uqi, avg_ergas = test_one_epoch(
        model, stagenum, test_loader, criterion, device, writer, epoch
    )

    # 3. Update learning rate based on validation loss
    scheduler.step(test_loss)

    # 4. Log training progress to TensorBoard
    writer.add_scalar("Loss/Train", train_loss, epoch)
    writer.add_scalar("Loss/Val", test_loss, epoch)
    writer.add_scalar("Learning Rate", optimizer.param_groups[0]['lr'], epoch)

    # 5. Save best model checkpoint (based on validation loss)
    if test_loss < best_val_loss:
        if epoch > 40:  # Skip first 40 epochs to avoid saving unstable initial models
            best_val_loss = test_loss
        torch.save(model.state_dict(), os.path.join(log_dir, "best_model.pth"))
        patience = 0  # Reset early stopping counter
    else:
        patience += 1
        # Trigger early stopping if no improvement for patience_limit epochs
        if patience > patience_limit:
            print(f"Early stopping triggered at epoch {epoch+1} (no improvement for {patience_limit} epochs)")
            break

    # 6. Print epoch summary
    end_time = time.time()
    epoch_time = end_time - start_time
    print(f"Epoch {epoch+1}/{epochs}: "
          f"Train Loss {train_loss:.4f}, Val Loss {test_loss:.4f}, "
          f"PSNR {avg_psnr:.4f}, SSIM {avg_ssim:.4f}, SAM {avg_sam:.4f}, "
          f"UQI {avg_uqi:.4f}, ERGAS {avg_ergas:.4f}, Time: {epoch_time:.2f}s")


# ===================== 14. Training Completion & Cleanup =====================
print(f"\nTraining completed at: {datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}")
print(f"Experiment results saved to: {log_dir}")
print(f"Best validation loss achieved: {best_val_loss:.4f}")

writer.close()  # Close TensorBoard writer

# Run result visualization if enabled
if args.display:
    run_display(
        model=model, 
        stagenum=stagenum, 
        model_name=model_name, 
        split=split, 
        blur=blur, 
        log_dir=log_dir, 
        data_path='xx', 
        dataset=dataset, 
        datarange=datarange, 
        batch_size=1, 
        scale=args.scale, 
        device=device,
        normalize=normalize, 
        kernel_path=psf_name, 
        test_loader=test_loader,
        sgm1=sgm1,
        sgm2=sgm2
    )