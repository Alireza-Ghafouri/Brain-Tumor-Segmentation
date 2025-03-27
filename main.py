from omegaconf import OmegaConf
import os
import torch
from dataloader import prepare_dataloaders
from model import UNet_with_Residual
from train import train_and_evaluate


def main():
    config = OmegaConf.load(os.path.join("config", "config.yaml"))

    DATASET_DIR = config.paths.data.dataset_dir
    weights_path = config.paths.weights_root
    plots_path = config.paths.plots_root
    saves_path = config.paths.saves_root

    os.makedirs(DATASET_DIR, exist_ok=True)
    os.makedirs(weights_path, exist_ok=True)
    os.makedirs(plots_path, exist_ok=True)
    os.makedirs(saves_path, exist_ok=True)

    device = torch.device(config.training.device)

    dataloaders, dataset_sizes = prepare_dataloaders(
        gdrive_url=config.data.gdrive_url,
        dataset_zip_path=config.paths.data.zip_path,
        dataset_path=config.paths.data.dataset_dir,
        save_path=config.paths.saves_root,
        batch_size=config.training.batch_size,
        augmentation=config.training.augmentation,
        train_ratio=config.data.train_ratio,
        valid_ratio=config.data.valid_ratio,
        seed=config.seed,
    )

    if config.training.model == "Unet_Modified":
        model = UNet_with_Residual()
    else:
        model = torch.hub.load(
            "mateuszbuda/brain-segmentation-pytorch",
            "unet",
            in_channels=3,
            out_channels=1,
            init_features=32,
            pretrained=False,
        )

    train_and_evaluate(
        model,
        dataloaders,
        dataset_sizes,
        config.training.model,
        config.paths.saves_root,
        config.paths.weights_root,
        config.paths.plots_root,
        config.training.num_epochs,
        config.optimizer.lr,
        config.optimizer.weight_decay,
        config.lr_scheduler.step_size,
        config.lr_scheduler.gamma,
        device,
    )


if __name__ == "__main__":
    main()
