from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from pathlib import Path
import torch, sys

def get_dataloaders(data_root, img_size=480, batch_size=32, num_workers=None):
    data_root = Path(data_root)
    is_mps = torch.backends.mps.is_available()
    if num_workers is None:
        num_workers = 0 if is_mps else 2

    train_tfms = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
    ])
    test_tfms = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
    ])

    train_ds = datasets.ImageFolder(data_root / "train", transform=train_tfms)
    test_ds  = datasets.ImageFolder(data_root / "test",  transform=test_tfms)

    # assert identical class folders (ImageFolder maps by alphabetical order)
    if train_ds.classes != test_ds.classes:
        print("[warn] Train/Test class folders differ. Aligning by creating empty missing dirs is recommended.",
              file=sys.stderr)
        print("train:", train_ds.classes, "\n test:", test_ds.classes, file=sys.stderr)

    train_dl = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=False, persistent_workers=False
    )
    test_dl = DataLoader(
        test_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=False, persistent_workers=False
    )
    return train_dl, test_dl
