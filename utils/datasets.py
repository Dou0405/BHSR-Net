# -----------------------------------------------------------------------------
# If you use this code in your research, please cite our paper:
# Blur-Resistant Hyperspectral Image Super-Resolution via Dual-Degradation Fusion Model
# Thanks
# -----------------------------------------------------------------------------
import os
import numpy as np
import torch.utils.data as data
import scipy.io as sio
from utils.degradation import getPSF, spatial_blur

np.random.seed(42)

def extract_patches_small(img):
    C, H, W = img.shape
    window_H = H // 4  
    window_W = W // 4  
    stride_H = int(window_H // 2)     
    stride_W = int(window_W // 2)     
    nh = (H - window_H) // stride_H + 1  
    nw = (W - window_W) // stride_W + 1  
    
    patches = np.empty((nh * nw, C, window_H, window_W), dtype=img.dtype)
    
    patch_idx = 0
    for i in range(nh):
        for j in range(nw):
            h_start = i * stride_H
            h_end = h_start + window_H
            w_start = j * stride_W
            w_end = w_start + window_W
            
            patches[patch_idx] = img[:, h_start:h_end, w_start:w_end]
            patch_idx += 1
    
    return patches

class HSI_MSI_Data(data.Dataset): 
    def __init__(self, mode, psf, srf, split, dataset, datarange, kernel_path=None, aug=False, scale_factor=8, data_path='data', blur=0, normalize=True):
        if mode not in ['train','test']:
            raise ValueError("mode must be 'train' or 'test'")
        self.mode = mode
        self.data_path = data_path
        self.aug = aug
        self.dataset = dataset
        if self.dataset == 'chikusei':
            self.srfname = "landsat"
        elif self.dataset == 'houston':
            self.srfname = "worldview2"
        else:
            self.srfname = "Nikon"
        self.datarange = datarange
        self.psf = psf
        self.srf = srf
        self.scale = scale_factor
        self.blur = blur
        self.split = split
        self.normalize = normalize
        self.kernel_path = kernel_path
        if self.mode == 'train':
            self.HrHSI_patches, self.HrMSI_patches, self.LrHSI_patches = self.load_data()
        else:
            self.HrHSIs, self.HrMSIs, self.LrHSIs, self.realHrMSI= self.load_data()

    def load_data(self):
        key = 'hsi'  

        gt_dir = os.path.join(self.data_path, self.dataset, 'gt')
        LrHSI_dir = os.path.join(self.data_path, self.dataset, "scale"+str(self.scale)+"+"+self.kernel_path+"+"+self.srfname, 'LrHSI')
        HrMSI_dir = os.path.join(self.data_path, self.dataset, "scale"+str(self.scale)+"+"+self.kernel_path+"+"+self.srfname, 'HrMSI')
        img_names= self.split[self.mode]

        all_gt = []
        all_HrMSI = []  
        all_LrHSI = []


        for img_name in img_names:
            gt_path = os.path.join(gt_dir, img_name)
            LrHSI_path = os.path.join(LrHSI_dir, img_name)
            HrMSI_path = os.path.join(HrMSI_dir, img_name)

            gt = sio.loadmat(gt_path)[key]
            LrHSI = sio.loadmat(LrHSI_path)['data']
            HrMSI = sio.loadmat(HrMSI_path)['data']

            gt = np.transpose(gt, (2, 0, 1))
            LrHSI = np.transpose(LrHSI, (2, 0, 1))
            HrMSI = np.transpose(HrMSI, (2, 0, 1))
            
            all_gt.append(gt)
            all_HrMSI.append(HrMSI)
            all_LrHSI.append(LrHSI)

        GTs = np.array(all_gt, dtype=np.float32)   
        HrMSIs = np.array(all_HrMSI, dtype=np.float32)
        LrHSIs = np.array(all_LrHSI, dtype=np.float32)

        if self.normalize:

            GTs = GTs / self.datarange
            HrMSIs = HrMSIs / self.datarange
            LrHSIs = LrHSIs / self.datarange
        
        HrMSIs_real = HrMSIs.copy()

        if self.blur > 0:
            sigma = self.blur 
            blur_kernel, _ = getPSF(kernel_size=5, sigma=sigma, anisotropy=False, Gaussian=True, angle=0, predefined_kernel=None)
            HrMSIs = spatial_blur(blur_kernel, HrMSIs, scale=1)


        if self.mode == 'test':
            return GTs, HrMSIs, LrHSIs, HrMSIs_real

        elif self.mode == 'train':

            GT_patches = []
            HrMSI_patches = []
            LrHSI_patches = []
            for i in range(GTs.shape[0]):

                GT = GTs[i]
                HrMSI = HrMSIs[i]
                LrHSI = LrHSIs[i]

                GT_patch = extract_patches_small(GT)
                HrMSI_patch = extract_patches_small(HrMSI)
                LrHSI_patch = extract_patches_small(LrHSI)

                GT_patches.append(GT_patch)
                HrMSI_patches.append(HrMSI_patch)
                LrHSI_patches.append(LrHSI_patch)

            GT_patches = np.concatenate(GT_patches, axis=0)
            HrMSI_patches = np.concatenate(HrMSI_patches, axis=0)
            LrHSI_patches = np.concatenate(LrHSI_patches, axis=0)

            return GT_patches, HrMSI_patches, LrHSI_patches

            
    def __len__(self):
        if self.mode == 'train':
            return len(self.HrHSI_patches)
        else:
            return len(self.HrHSIs)
    
    def __getitem__(self, index):
        if self.mode == 'train':
            HrHSI_patch = self.HrHSI_patches[index]
            HrMSI_patch = self.HrMSI_patches[index]
            LrHSI_patch = self.LrHSI_patches[index]

            return  LrHSI_patch, HrMSI_patch, HrHSI_patch
        else:
            HrHSI = self.HrHSIs[index]
            HrMSI = self.HrMSIs[index]
            LrHSI = self.LrHSIs[index]

            return  LrHSI, HrMSI, HrHSI, self.realHrMSI[index]


class real_Data(data.Dataset): 
    def __init__(self, mode, split, dataset, aug=False, data_path='data', normalize=True):
        if mode not in ['train','test']:
            raise ValueError("mode must be 'train' or 'test'")
        self.mode = mode
        self.data_path = data_path
        self.aug = aug
        self.dataset = dataset
        self.split = split
        self.normalize = normalize
        self.HrHSIs, self.HrMSIs, self.LrHSIs, self.realHrMSI= self.load_data()

    def load_data(self):

        gt_dir = os.path.join(self.data_path, self.dataset, 'gt')
        LrHSI_dir = os.path.join(self.data_path, self.dataset, 'lrhsi')
        HrMSI_dir = os.path.join(self.data_path, self.dataset, 'hrmsi')
        img_names= self.split[self.mode]

        all_gt = []
        all_HrMSI = []  
        all_LrHSI = []

        for img_name in img_names:
            gt_path = os.path.join(gt_dir, img_name)
            LrHSI_path = os.path.join(LrHSI_dir, img_name)
            HrMSI_path = os.path.join(HrMSI_dir, img_name)

            gt = sio.loadmat(gt_path)['hsi']
            LrHSI = sio.loadmat(LrHSI_path)['hsi']
            HrMSI = sio.loadmat(HrMSI_path)['msi']

            gt = np.transpose(gt, (2, 0, 1))
            LrHSI = np.transpose(LrHSI, (2, 0, 1))
            HrMSI = np.transpose(HrMSI, (2, 0, 1))
            
            all_gt.append(gt)
            all_HrMSI.append(HrMSI)
            all_LrHSI.append(LrHSI)

        GTs = np.array(all_gt, dtype=np.float32)   
        HrMSIs = np.array(all_HrMSI, dtype=np.float32)
        LrHSIs = np.array(all_LrHSI, dtype=np.float32)

        if self.normalize:

            GTs = (GTs + 0.012797) / 1.64148
            HrMSIs = (HrMSIs + 0.012797) / 1.64148
            LrHSIs = (LrHSIs + 0.012797) / 1.64148
        
        return GTs, HrMSIs, LrHSIs, HrMSIs

    def __len__(self):
        return len(self.HrHSIs)
    
    def __getitem__(self, index):

        HrHSI = self.HrHSIs[index]
        HrMSI = self.HrMSIs[index]
        LrHSI = self.LrHSIs[index]
        realHrMSI = self.realHrMSI[index]
        if self.mode == 'train':
            return  LrHSI, HrMSI, HrHSI
        else:
            return  LrHSI, HrMSI, HrHSI, realHrMSI

