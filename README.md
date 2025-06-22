# Brain Tumor Segmentation with U-Net

This project implements a U-Net-based model for segmenting brain tumor tissues from MRI images. The project includes dataset preprocessing, model training, data augmentation, and modifications to the U-Net architecture to incorporate residual connections.

## 📌 Project Overview
- **Model:** U-Net (Base and Modified with Residual Connections)
- **Dataset:** Brain tumor MRI dataset (available on Kaggle and Google Drive)
- **Framework:** PyTorch
- **Evaluation Metrics:** Jaccard Index & Dice Coefficient
- **Loss Function:** Reported for training, validation, and test datasets

## 📂 Dataset
The dataset consists of MRI scans from **110 patients**, where each patient’s images are stored in separate folders. Each image has a corresponding mask (ending with `_mask`) indicating the tumor region.

🔗 **MRI Brain Tumor Dataset**: Download from Kaggle (mateuszbuda/lgg-mri-segmentation) or via the Google Drive link specified in `config.yaml`.

### 🔹 Data Splitting
- **Training:** 80%
- **Validation:** 10%
- **Test:** 10%
- **Note:** The split is done at the **patient level**, not at the individual image level.

## 🔧 Installation & Setup
1. Clone the repository:
   ```bash
   git clone https://github.com/Alireza-Ghafouri/Brain-Tumor-Segmentation.git
   cd Brain-Tumor-Segmentation
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Configure dataset path, hyperparameters, and augmentation settings in `config.yaml`.
4. Run the main script:
   ```bash
   python main.py
   ```

## 🚀 Features
- **Configurable parameters:** All hyperparameters, dataset paths, and settings are easily adjustable via `config.yaml`.
- **Augmentations:** Can be toggled on/off and modified in the config file.
- **Loss & Evaluation Metrics:** Jaccard and Dice scores are logged and plotted after training.
- **Model Selection:** Users can choose between the **standard U-Net** and the **modified U-Net with residual connections**.
- **Results & Outputs:** Trained model checkpoints and evaluation plots are saved in the specified directories.

## 🔄 Model Variants
### 1️⃣ **Base U-Net**
- Standard U-Net model used for segmentation.
- Pretrained model available from [PyTorch Hub](https://pytorch.org/hub/mateuszbuda_brain-segmentation-pytorch_unet/).

### 2️⃣ **Residual U-Net**
- Enhances performance using **skip connections with 1×1 convolutions** to match input-output feature maps.
- Can be enabled in the config file.

## 📊 Evaluation
After training, evaluation metrics (loss, Jaccard, and Dice scores) are plotted and saved. These results can be found in the **output directory** (specified in `config.yaml`).

## 📜 License
This project was developed as an **academic assignment** and is **free to use**.

---

Feel free to modify the config file and experiment with different settings! 🚀

