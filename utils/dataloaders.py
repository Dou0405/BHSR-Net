# -----------------------------------------------------------------------------
# If you use this code in your research, please cite our paper:
# Blur-Resistant Hyperspectral Image Super-Resolution via Dual-Degradation Fusion Model
# Thanks
# -----------------------------------------------------------------------------
from torch.utils.data import DataLoader
from utils.datasets import HSI_MSI_Data, real_Data

def get_dataloaders_dataparallel(split, kernel_path=None, batch_size=32, num_workers=4, data_path='./data', datarange=None, psf=None, srf=None, scale_factor=8, dataset='CAVE', blur=0, normalize=False):
    if dataset != 'liao':
            
        train_dataset = HSI_MSI_Data(
            mode='train', 
            split=split,
            blur=blur,
            data_path=data_path, 
            psf=psf, 
            srf=srf, 
            dataset=dataset, 
            datarange=datarange,
            aug=False, 
            scale_factor=scale_factor,
            normalize=normalize,
            kernel_path=kernel_path
        )

        test_dataset = HSI_MSI_Data(
            mode='test', 
            split=split,
            blur=blur,
            data_path=data_path, 
            psf=psf, 
            srf=srf, 
            dataset=dataset, 
            datarange=datarange,
            aug=False,
            scale_factor=scale_factor,
            normalize=normalize,
            kernel_path=kernel_path
        )
    else:

        train_dataset = real_Data(
            mode='train', 
            split=split,
            data_path=data_path, 
            dataset=dataset, 
            aug=False, 
            normalize=normalize
        )

        test_dataset = real_Data(
            mode='test', 
            split=split,
            data_path=data_path, 
            dataset=dataset, 
            aug=False,  
            normalize=normalize
        )

    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=num_workers, 
        pin_memory=True  
    )

    test_loader = DataLoader(
        test_dataset, 
        batch_size=1, 
        shuffle=False,  
        num_workers=num_workers, 
        pin_memory=True
    )

    return train_loader, test_loader
