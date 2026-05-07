# BHSR-Net
The official code for BHSR-Net.

## 1. Introduction
The deep unfolding network represents a promising research avenue in fusion-based hyperspectral image super-resolution (HSI-SR). However, most current deep unfolding methodologies are anchored in idealized observation models, which overlook the degradation of the multispectral image (MSI), hindering their SR performance and practical applicability.

To address this problem, this paper establishes a novel Dual-Degradation Fusion (D$^{2}$-Fusion) model, which incorporates both HSI degradation and MSI blurring into the HSI-SR modelling process. Subsequently, we apply the second-order semismooth Newton algorithm to solve the optimization problem in the D$^{2}$-Fusion model. The solution steps are then mapped into an end-to-end trainable network, termed Blur-resistant Hyperspectral image Super-Resolution Network (BHSR-Net).

To the best of our knowledge, the proposed network is the first successful attempt to consider MSI blurring artifacts in HSI-SR tasks. It offers several distinct advantages:
* **Mathematical Correspondence**: The network structure maintains a strict mathematical correspondence with the optimization algorithm, ensuring each module retains strong physical interpretability.
* **High Performance**: The network exhibits superior SR performance and strong generalization ability on both standard and real-world scenarios across five datasets.
* **Efficiency**: The network demonstrates excellent learning efficiency with a compact architecture.

## 2. File Structure
The project structure is organized as follows based on the provided repository:

* **Core Files**:
    * `models.py`: Defines the architecture of the BHSR-Net.
    * `modules.py`: Contains the specific neural network components and iteration blocks.
    * `loss.py`: Implements the composite loss functions used for training.
* **Utilities**:
    * `utils/dataloaders.py`: Reference implementation for data loading.
    * `utils/datasets.py`: Logic for dataset handling and patch extraction.
    * `utils/metrics.py`: Evaluation metrics including PSNR, SSIM, SAM, UQI, and ERGAS.
    * `utils/display.py`: Tools for result visualization and saving `.mat` files.
* **Scripts**:
    * `data/SRF/`: Contains Spectral Response Functions (SRF) for various datasets.
    * `train.py`: A reference training script provided for convenience.

## 3. How to Reproduce
To reproduce the results presented in our paper, please follow these guidelines:

* **Core Components**: `models.py`, `modules.py`, and `loss.py` are the essential files of the project. While we provide a `train.py` script, it is intended for reference as our training strategy is straightforward and can be customized.
* **Data Preparation**: We strongly suggest performing preprocessing (such as generating LrHSI and HrMSI) before starting the training process. Loading pre-prepared data directly will significantly improve training efficiency.
* **Customization**: The provided `utils/datasets.py` and `utils/dataloaders.py` are reference templates. We recommend implementing your own data handling logic tailored to your specific environment.
* **Spectral Response Functions**: Different datasets utilize different SRFs. Ensure you use the correct `.mat` file from `data/SRF/` corresponding to your dataset.
* **Support**: If you encounter any issues, please feel free to open an issue or contact **yongxuan@buaa.edu.cn** directly for assistance.

## 4. Citation
If you find our work helpful for your research, please cite our paper: Blur-Resistant Hyperspectral Image Super-Resolution via Dual-Degradation Fusion Model
