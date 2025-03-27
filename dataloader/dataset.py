from torch.utils.data import Dataset
from torchvision import transforms
import cv2
import albumentations as A


class MRI_Dataset(Dataset):

    def __init__(self, input_dir, label_dir, augmentation=None, transform=None):

        self.input_dir = input_dir
        self.label_dir = label_dir
        self.transform = transform
        self.augmentation = augmentation

    def __len__(self):
        return len(self.input_dir)

    def __getitem__(self, idx):

        input_img = cv2.imread(self.input_dir[idx])
        input_img = cv2.cvtColor(input_img, cv2.COLOR_BGR2RGB)

        label_img = cv2.imread(self.label_dir[idx], 0)

        # input_m , input_s = np.mean(input_img, axis=(0, 1)), np.std(input_img, axis=(0, 1))

        input_transform = transforms.Compose(
            [
                transforms.ToPILImage(),
                # transforms.Resize((256,256)),
                transforms.ToTensor(),
                # transforms.Normalize(mean=input_m, std=input_s)
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ]
        )

        label_transform = transforms.Compose(
            [
                transforms.ToPILImage(),
                # transforms.Resize((256,256)),
                transforms.ToTensor(),
            ]
        )

        augment_transform = A.Compose(
            [
                A.HorizontalFlip(p=0.2),
                A.Rotate(limit=(-10, 10), p=0.2, border_mode=0),
                A.Affine(
                    scale=(0.9, 1.1), shear=(-15, 15), translate_percent=(0, 0.1), p=0.2
                ),
            ]
        )

        if self.augmentation:
            transformed = augment_transform(image=input_img, mask=label_img)
            input_img = transformed["image"]
            label_img = transformed["mask"]

        if self.transform:
            input_img = input_transform(input_img)
            label_img = label_transform(label_img)

        return input_img, label_img
