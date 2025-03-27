import os
import random
import pickle
import kagglehub
import gdown
import zipfile
from tqdm import tqdm


def download_from_gdrive(gdrive_url, save_path):
    """
    Downloads a file from Google Drive and saves it locally.

    Args:
        gdrive_url (str): The shareable Google Drive link.
        save_path (str): The destination path to save the file.

    Returns:
        str: Path to the downloaded file.
    """
    # Extract file ID from the URL
    file_id = gdrive_url.split("/d/")[1].split("/")[0]
    download_url = f"https://drive.google.com/uc?id={file_id}"

    # Ensure the save directory exists
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    print("📥 Downloading dataset from Google Drive...")
    gdown.download(download_url, save_path, quiet=False)

    if os.path.exists(save_path):
        print(f"✅ Download complete: {save_path}")
    else:
        print("❌ Download failed!")

    # return save_path


def download_dataset_kaggle(dataset_name, save_path=None):
    """
    Download dataset from Kaggle using kagglehub.

    Args:
        dataset_name (str): The Kaggle dataset name (e.g., "mateuszbuda/lgg-mri-segmentation").
        save_path (str, optional): Custom save directory. If None, the default kagglehub path is used.

    Returns:
        str: Path to the downloaded dataset.
    """
    print("📥 Downloading dataset from Kaggle...")
    dataset_path = kagglehub.dataset_download(dataset_name)

    if save_path:
        os.makedirs(save_path, exist_ok=True)
        os.system(f"cp -r {dataset_path}/* {save_path}")

    print(f"✅ Dataset downloaded at: {save_path or dataset_path}")
    # return save_path or dataset_path


def unzip_dataset(zip_path, extract_to):
    """
    Extracts a dataset only if it hasn't been extracted already, with a progress bar.

    Args:
        zip_path (str): Path to the zip file.
        extract_to (str): Directory where files should be extracted.
    """
    if os.path.exists(extract_to) and os.listdir(extract_to):
        print(f"✅ Dataset already extracted: {extract_to}")
        return

    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        file_list = zip_ref.namelist()
        with tqdm(
            total=len(file_list), desc="📂 Extracting dataset", unit="files"
        ) as pbar:
            for file in file_list:
                zip_ref.extract(file, extract_to)
                pbar.update(1)  # Update tqdm progress bar
    print("✅ Extraction complete.")


def get_patient_list(dataset_path, save_path, filename="patient_list.pkl"):
    """
    Retrieves the list of patient directories. If the list is already saved, it loads it from a pickle file.

    Args:
        dataset_path (str): Path where the dataset is stored.
        save_path (str): Path to save/load the patient list.
        filename (str): The name of the saved file.

    Returns:
        list: A list of patient directory names.
    """
    save_path = os.path.join(save_path, filename)
    if os.path.exists(save_path):
        print("📂 Loading patient list from cache...")
        with open(save_path, "rb") as f:
            return pickle.load(f)

    print("🔍 Extracting patient list from dataset directory...")
    new_dataset_path = os.path.join(dataset_path, "lgg-mri-segmentation", "kaggle_3m")
    patient_list = [
        d
        for d in os.listdir(new_dataset_path)
        if os.path.isdir(os.path.join(new_dataset_path, d))
    ]

    with open(save_path, "wb") as f:
        pickle.dump(patient_list, f)

    print(f"✅ Patient list saved to {save_path}")
    return patient_list


def split_dataset(patient_list, train_ratio=0.8, valid_ratio=0.1, seed=4):
    """
    Splits the patient list into train, validation, and test sets.

    Args:
        patient_list (list): List of patient directory names.
        train_ratio (float): Ratio of the training set.
        valid_ratio (float): Ratio of the validation set.
        seed (int): Random seed for reproducibility.

    Returns:
        tuple: (train_list, valid_list, test_list)
    """
    random.seed(seed)
    random.shuffle(patient_list)

    total = len(patient_list)
    train_split = int(total * train_ratio)
    valid_split = int(total * (train_ratio + valid_ratio))

    train_data = patient_list[:train_split]
    valid_data = patient_list[train_split:valid_split]
    test_data = patient_list[valid_split:]

    print(
        f"📊 Train: {len(train_data)} ({train_ratio*100}%) | Validation: {len(valid_data)} ({valid_ratio*100}%) | Test: {len(test_data)} ({(1-train_ratio-valid_ratio)*100}%)"
    )

    return train_data, valid_data, test_data


def get_image_paths(dataset_path, patient_list):
    """
    Extracts image and mask file paths.

    Args:
        dataset_path (str): Path where patient directories are stored.
        patient_list (list): List of patient directories.

    Returns:
        tuple: (X, Y) where X contains image paths and Y contains mask paths.
    """
    X, Y = [], []
    for patient in patient_list:
        patient_dir = os.path.join(
            dataset_path, "lgg-mri-segmentation", "kaggle_3m", patient
        )
        images = os.listdir(patient_dir)

        for img in images:
            if (
                img.endswith(".tif") or img.endswith(".png") or img.endswith(".jpg")
            ):  # Adjust based on dataset format
                img_path = os.path.join(patient_dir, img)
                if "mask" in img:
                    Y.append(img_path)
                    X.append(img_path.replace("_mask", ""))

    return X, Y
