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
from utils.bicubic import run_bicubic
from utils.metrics import *
from utils.dataloaders import get_dataloaders_dataparallel
from thop import profile


def set_seed(seed: int = 100):
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print(f"set random seed as: {seed}")



set_seed(40)  

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

Harvard_hard = {'test': ['imgd9','imgd7','imgc1','imgb2','imgh3','imgf7','imge7','imga1','imga7','img1','imgd2',
                          'imge4','imgb0','imgc5','imgf6','imgb6','imga5','imgd8','imgf1','imgh0','imgb5','imgb4',
                          'imga2','imgc9','imgb9','imgh1','imgc2','imge3','imgb8','imgf5','imgc7','imgf4','imgf2',
                          'imge1', 'imgb1', 'imgd3', 'img2',  'imge5', 'imgc8', 'imge2'],  
                'train': ['imge6', 'imgc4', 'imgf8', 'imgb7', 'imgd4',  'imga6', 'imgh2', 'imgb3', 'imgf3',  'imge0']}

Harvard_UTAL_split = {'test': [ 'imgb9', 'imgh3','imgc5', 'imga7', 'imgb4', 'imgh0', 'imgd7', 'imge7', 'imgb6', 'imga5', 
                                'imgf7', 'imgc2', 'imgf5','imgb2', 'imge3', 'imgc1', 'imga1', 'imgc9', 'imgb5', 'img1', 
                                'imgb0', 'imgd8', 'imgb8'],  
                    'train': ['imge6', 'imgc4', 'imgf8', 'imgb7', 'imgd4', 'imgb1', 'imge1', 'imga6', 'imgh2', 'imgb3', 
                              'imgf3', 'imgf4', 'imge0', 'imgd3', 'img2', 'imgf2', 'imge5', 'imgc8', 'imge2', 'imgc7']}

chikusei_split = {'test':['chikusei01','chikusei02','chikusei03','chikusei04','chikusei05','chikusei06','chikusei07','chikusei08'],
                  'train':['chikusei09','chikusei10','chikusei11','chikusei12','chikusei13','chikusei14','chikusei15','chikusei16']}

houston_split ={'test':['img2','img3','img4'],'train':['img1']}
liao_split ={'test':['img0'],'train':['img1','img2','img3','img4','img5']}


parser = argparse.ArgumentParser(description="Train a hyperspectral image reconstruction model.")
parser.add_argument("--model", type=str, default='UMMHF', help="Model name.")
parser.add_argument("--dataset", type=str, default='CAVE', help="Dataset name.")
parser.add_argument("--stagenum", type=int, default=10, help="Model's stage.")
parser.add_argument("--batch_size", type=int, default=16, help="Batch size for training.")
parser.add_argument("--split", type=str, default='cave_utal', help="Split for training and testing, mhf or utal.")
parser.add_argument("--epochs", type=int, default=200, help="Number of training epochs.")
parser.add_argument("--lr", type=float, default=1e-2, help="Learning rate.")
parser.add_argument("--patience_limit", type=int, default=100, help="Early stopping patience limit.")
parser.add_argument("--scale", type=int, default=32, help="Scale factor for spatial downsampling.")
parser.add_argument("--kernel_size", type=int, default=11, help="Kernel size for convolution.")
parser.add_argument("--sigma", type=float, default=4, help="Sigma for Gaussian kernel.")
parser.add_argument("--noise", type=float, default=0.0, help="Noise level for the input data (dB).")
parser.add_argument("--resume", type=str, help="Path to the pretrained model to resume training from.")
parser.add_argument("--anisotropy", action="store_true", help="Use anisotropic Gaussian kernel.")
parser.add_argument("--Gaussian", action="store_true", help="Use Gaussian kernel.")
parser.add_argument("--pre_psf", type=str, default=None, help="Predefined PSF kernel path.")
parser.add_argument("--angle", type=float, default=0, help="Angle for anisotropic Gaussian kernel.")
parser.add_argument("--display", action="store_true", help="Display the results after training.")
parser.add_argument("--normalize", action="store_true", help="Normalize the input data.")
parser.add_argument("--sgm1", type=float, default=1)
parser.add_argument("--sgm2", type=float, default=2)

args = parser.parse_args()

model_name = args.model
dataset = args.dataset
datarange = None
sgm1 = args.sgm1
sgm2 = args.sgm2
srf = ""
C=31
c=3
if dataset == 'CAVE':
    datarange = 65535
    srf = "/media/data/dyx/UMMHF6/data/predefined_SRF/Nikon/Nikon_srf.mat"
    C=31
    c=3
    split = CAVE_UTAL_split
elif dataset == 'Harvard':
    datarange = 0.0616312
    srf = "/media/data/dyx/UMMHF6/data/predefined_SRF/Nikon/Nikon_srf.mat"
    C=31
    c=3
    split = Harvard_hard
elif dataset == 'chikusei':
    datarange = 15133
    srf = "/media/data/dyx/UMMHF6/data/predefined_SRF/landsat/landsat_srf.mat"
    C=128
    c=3
    split = chikusei_split
elif dataset == 'houston':
    datarange = 25924
    srf = "/media/data/dyx/UMMHF6/data/predefined_SRF/worldview2/srf_houston18_worldview2.mat"
    C=46
    c=8
    split = houston_split
elif dataset == 'liao':
    srf = "/media/data/dyx/UMMHF6/data/predefined_SRF/Nikon/Nikon_srf.mat"
    C=149
    c=8
    split = liao_split

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
noise= args.noise
split_name = args.split
resume_model = args.resume
psf_path = args.pre_psf
normalize = args.normalize


if kernel_size % 2 == 0:
    raise ValueError("Kernel size must be an odd number.")


now = datetime.now()

time_str = now.strftime("%Y-%m-%d_%H-%M-%S")

log_dir = os.path.join("logs", dataset+time_str+"stg"+str(stagenum)+"scale"+str(scale)+"noise"+str(noise))
if not os.path.exists(log_dir):
    os.makedirs(log_dir)


log_file = os.path.join(log_dir, 'output.txt')


class Logger:
    def __init__(self, log_file):
        self.terminal = sys.stdout
        self.log = open(log_file, 'w')

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
        self.log.flush()

    def flush(self):
        self.terminal.flush()
        self.log.flush()


sys.stdout = Logger(log_file)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if device.type == 'cpu':
    raise RuntimeError("CUDA is not available. Please check your GPU setup.")


psf, psf_name = getPSF(kernel_size=kernel_size, scale=scale, sigma=sigma, anisotropy=anisotropy, Gaussian=Gaussian, angle=angle, predefined_kernel=psf_path)  # 示例参数
srf, srf_name = getSRF(srf)


print(f"Model: {model_name}, StageNum: {stagenum}, Batch size: {batch_size}, Split: {split_name}, Learning rate: {lr}, Patience limit: {patience_limit}, "
      f"Scale: {scale}, PSF: {psf_name}, SRF: {srf_name}, Noise: {noise}, Normalize: {normalize}")

print("Loaing data...")
train_loader, test_loader = get_dataloaders_dataparallel(batch_size=batch_size, split=split, num_workers=8 * torch.cuda.device_count(), data_path='/media/data/dyx/UMMHF6/data', psf=psf, srf=srf, scale_factor=scale, dataset=dataset, datarange=datarange, noise=noise, normalize=normalize, kernel_path=psf_name)
print("Loaing model...")


model = UM(stage_num=stagenum, C=C, c=c, sigma1=sgm1, sigma2=sgm2)
model = torch.nn.DataParallel(model).to(device) 

if resume_model:
    if os.path.isfile(resume_model):
        print(f"Loading model from {resume_model}")
        model.load_state_dict(torch.load(resume_model, map_location=device))
    else:
        raise FileNotFoundError(f"No checkpoint found at '{resume_model}'")
    
print(model_name, "training on", device)

optimizer = torch.optim.Adam(model.parameters(), lr=lr)
scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.7, patience=20)
criterion = Losses(scale=scale, model_name = model_name, blur = noise).to(device)


torch.save(psf, os.path.join(log_dir, 'psf.pth'))
torch.save(srf, os.path.join(log_dir, 'srf.pth'))


def save_psf_visualization(psf, save_path):
    plt.figure(figsize=(10, 10))
    psf_vis = psf.squeeze()
    if len(psf_vis.shape) > 2:
        psf_vis = np.mean(psf_vis, axis=0)
    plt.imshow(psf_vis, cmap='viridis')
    plt.colorbar()
    plt.title('PSF Visualization')
    plt.savefig(save_path)
    plt.close()

def save_srf_visualization(srf, save_path):
    plt.figure(figsize=(10, 5))
    srf_vis = srf.squeeze()
    plt.imshow(srf_vis, cmap='viridis', aspect='auto')
    plt.colorbar()
    plt.title('SRF Visualization')
    plt.xlabel('Source Channel')
    plt.ylabel('Target Channel')
    plt.savefig(save_path)
    plt.close()

save_psf_visualization(psf, os.path.join(log_dir, 'psf.png'))
save_srf_visualization(srf, os.path.join(log_dir, 'srf.png'))


print(f"Training started at {time_str}")
print(f"PSF name: {psf_name}")
print(f"SRF name: {srf_name}")
print(f"Model parameters: {sum(p.numel() for p in model.parameters())}")
print("-" * 80)

def find_free_port():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(('', 0))  
        return s.getsockname()[1]  

free_port = find_free_port()

writer = SummaryWriter(log_dir)

best_val_loss = float("inf")
best_val_psnr = -100000000.0
patience = 0

# 启动 TensorBoard
subprocess.Popen([
    os.path.join(os.path.dirname(sys.executable), 'tensorboard'),
    '--logdir', log_dir,
    '--host', 'localhost',
    '--port', str(free_port),
    '--load_fast=false'
])

def train_one_epoch(model, dataloader, optimizer, criterion, device, epoch):
    model.train()
    train_loss = 0.0
    for batch in dataloader: 
        LrHSI, HrMSI, HrHSI = [tensor.float() for tensor in batch] 
        LrHSI, HrMSI, HrHSI = LrHSI.to(device), HrMSI.to(device), HrHSI.to(device)

        pred, y_  = model(LrHSI, HrMSI) 

        loss = criterion(pred, HrHSI, y_, HrMSI, epoch)  


        optimizer.zero_grad()
        loss.backward(retain_graph=True)
        if dataset !='liao':
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
        else: 
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
        optimizer.step() 
        train_loss += loss.item()* HrHSI.size(0)* torch.cuda.device_count()

    return train_loss / len(dataloader.dataset)


def test_one_epoch(model, stage_num, dataloader, criterion, device, writer=None, epoch=None, fine_tune_criterion = None, optimizer=None):
    model.eval()
    test_loss = 0.0

    psnr_calculator = PeakSignalNoiseRatio(data_range=1).to(device)
    ssim_calculator = StructuralSimilarityIndexMeasure(data_range=1, gaussian_kernel=True, kernel_size=11, sigma=1.5).to(device)

    total_psnr = 0.0
    total_ssim = 0.0
    total_sam = 0.0
    total_uqi = 0.0
    total_ergas = 0.0
    num_batches = 0

    with torch.no_grad():
        for batch in dataloader:
            LrHSI, HrMSI, HrHSI, _ = [tensor.float() for tensor in batch] 
            LrHSI, HrMSI, HrHSI    = LrHSI.to(device), HrMSI.to(device), HrHSI.to(device)


            pred, y_ = model(LrHSI, HrMSI) 
            test_loss += criterion(pred, HrHSI, y_, HrMSI, epoch)
            batch_psnr = psnr_calculator(pred[-1], HrHSI).item()
            batch_ssim = ssim_calculator(pred[-1], HrHSI).item()
            batch_sam  = sam_calculator(pred[-1], HrHSI).item()
            batch_uqi  = uqi_calculator(pred[-1], HrHSI).item()
            batch_ergas = ergas_calculator(pred[-1], HrHSI, scale_factor=scale).item()

            total_psnr += batch_psnr
            total_ssim += batch_ssim
            total_sam  += batch_sam
            total_uqi  += batch_uqi
            total_ergas += batch_ergas

            num_batches += 1


    avg_psnr = total_psnr / num_batches
    avg_ssim = total_ssim / num_batches
    avg_sam  = total_sam  / num_batches
    avg_uqi  = total_uqi  / num_batches
    avg_ergas = total_ergas / num_batches


    if writer is not None and epoch is not None:
        writer.add_scalar("Metrics/PSNR", avg_psnr, epoch)
        writer.add_scalar("Metrics/SSIM", avg_ssim, epoch)
        writer.add_scalar("Metrics/SAM",  avg_sam,  epoch)
        writer.add_scalar("Metrics/UQI",  avg_uqi,  epoch)
        writer.add_scalar("Metrics/ERGAS", avg_ergas, epoch)
        writer.add_scalar("Test/Loss", test_loss / len(dataloader), epoch)

    return test_loss / len(dataloader.dataset), avg_psnr, avg_ssim, avg_sam, avg_uqi, avg_ergas



for epoch in range(epochs):

    start_time = time.time()

    train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device, epoch)

    test_loss, avg_psnr, avg_ssim, avg_sam, avg_uqi, avg_ergas = test_one_epoch(model, stagenum, test_loader, criterion, device, writer, epoch)

    scheduler.step(test_loss)

    writer.add_scalar("Loss/Train", train_loss, epoch)
    writer.add_scalar("Loss/Val", test_loss, epoch)
    writer.add_scalar("Learning Rate", optimizer.param_groups[0]['lr'], epoch)

    if test_loss < best_val_loss:
        if epoch>40: 
            best_val_loss = test_loss
        torch.save(model.state_dict(), os.path.join(log_dir, "best_model.pth"))
        patience = 0
    else:
        patience += 1
        if patience > patience_limit:
            print("Early stopping triggered!")
            break
    end_time = time.time()
    epoch_time = end_time - start_time

    print(f"Epoch {epoch+1}/{epochs}: Train Loss {train_loss:.4f}, Val Loss {test_loss:.4f}, PSNR {avg_psnr:.4f}, SSIM {avg_ssim:.4f}, SAM {avg_sam:.4f}, UQI {avg_uqi:.4f}, ERGAS {avg_ergas:.4f}, Time: {epoch_time:.2f}s")

print(f"\nTraining finished at {datetime.now().strftime('%Y-%m-%d_%H-%M-%S')},saved in {log_dir}")
print(f"Best validation loss: {best_val_loss:.4f}")

writer.close()

if args.display:

    # run_bicubic(stagenum=stagenum, model_name=model_name, split=split, noise=noise, log_dir=log_dir, data_path='/media/data/dyx/UMMHF6/data', dataset=dataset, datarange=datarange, batch_size=1, scale=args.scale, device=device,normalize=normalize, kernel_path=psf_name, test_loader=test_loader)
    run_display(model=model, stagenum=stagenum, model_name=model_name, split=split, noise=noise, log_dir=log_dir, data_path='/media/data/dyx/UMMHF6/data', dataset=dataset, datarange=datarange, batch_size=1, scale=args.scale, device=device,normalize=normalize, kernel_path=psf_name, test_loader=test_loader,sgm1=sgm1,sgm2=sgm2)


