from pathlib import Path
import csv
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

MANIFEST_PATH = (
    PROJECT_ROOT
    / "data"
    / "manifests"
    / "patch_manifest.csv"
)


# ============================================================
# OIL SPILL DATASET
# ============================================================

class OilSpillDataset(Dataset):
    """
    PyTorch Dataset for preprocessed Sentinel-1 oil-spill patches.

    Each sample contains:

        image      -> [2, 512, 512]
        mask       -> [1, 512, 512]
        valid_mask -> [1, 512, 512]

    Metadata:
        dataset
        patch_name
        source_image
    """

    def __init__(
        self,
        manifest_path=MANIFEST_PATH,
        split="train"
    ):

        self.manifest_path = Path(
            manifest_path
        )

        self.split = split

        if not self.manifest_path.exists():

            raise FileNotFoundError(
                f"Manifest not found:\n"
                f"{self.manifest_path}"
            )

        # ----------------------------------------------------
        # Load manifest
        # ----------------------------------------------------

        with open(
            self.manifest_path,
            "r",
            encoding="utf-8"
        ) as file:

            reader = csv.DictReader(file)

            self.records = [
                row
                for row in reader
                if row["split"] == split
            ]

        if not self.records:

            raise ValueError(
                f"No records found for split: "
                f"{split}"
            )

        print(
            f"{split.capitalize()} samples: "
            f"{len(self.records):,}"
        )

    # ========================================================
    # LENGTH
    # ========================================================

    def __len__(self):

        return len(
            self.records
        )

    # ========================================================
    # GET ITEM
    # ========================================================

    def __getitem__(
        self,
        index
    ):

        record = self.records[
            index
        ]

        # ----------------------------------------------------
        # Paths
        # ----------------------------------------------------

        image_path = Path(
            record["image_path"]
        )

        mask_path = Path(
            record["mask_path"]
        )

        invalid_path = Path(
            record["invalid_path"]
        )

        # ----------------------------------------------------
        # Load NumPy arrays
        # ----------------------------------------------------

        image = np.load(
            image_path,
            allow_pickle=False
        )

        mask = np.load(
            mask_path,
            allow_pickle=False
        )

        invalid_mask = np.load(
            invalid_path,
            allow_pickle=False
        )

        # ----------------------------------------------------
        # Convert to tensors
        # ----------------------------------------------------

        image = torch.from_numpy(image).float()

        mask = torch.from_numpy(mask).float()

        invalid_mask = torch.from_numpy(invalid_mask).float()

        # ----------------------------------------------------
        # Add channel dimension to masks
        # ----------------------------------------------------

        mask = mask.unsqueeze(
            0
        )

        invalid_mask = (
            invalid_mask.unsqueeze(
                0
            )
        )

        # ----------------------------------------------------
        # Valid mask
        #
        # invalid = 1
        # valid   = 0
        #
        # We want:
        # valid   = 1
        # invalid = 0
        # ----------------------------------------------------

        valid_mask = 1.0 - (
            invalid_mask
        )

        # ----------------------------------------------------
        # Return sample
        # ----------------------------------------------------

        return {
            "image": image,
            "mask": mask,
            "valid_mask": valid_mask,

            "dataset": record[
                "dataset"
            ],

            "patch_type": record[
                "patch_type"
            ],

            "patch_name": record[
                "patch_name"
            ],

            "source_image": record[
                "source_image"
            ]
        }


# ============================================================
# CREATE DATALOADER
# ============================================================

def create_dataloader(
    manifest_path=MANIFEST_PATH,
    split="train",
    batch_size=4,
    shuffle=None,
    num_workers=4
):
    """
    Create a PyTorch DataLoader.

    num_workers=0 is recommended initially on Windows.
    """

    dataset = OilSpillDataset(
        manifest_path=manifest_path,
        split=split
    )

    # --------------------------------------------------------
    # Default shuffle behavior
    # --------------------------------------------------------

    if shuffle is None:

        shuffle = (
            split == "train"
        )

    # --------------------------------------------------------
    # DataLoader
    # --------------------------------------------------------

    loader_kwargs = {
        "batch_size": batch_size,
        "shuffle": shuffle,
        "num_workers": num_workers,
        "pin_memory": torch.cuda.is_available(),
    }

    if num_workers > 0:
        loader_kwargs["persistent_workers"] = True
        loader_kwargs["prefetch_factor"] = 2

    loader = DataLoader(
        dataset,
        **loader_kwargs
    )

    return (
        dataset,
        loader
    )


# ============================================================
# TEST DATASET
# ============================================================

def test_dataset():

    print("\n" + "=" * 75)
    print(
        "PYTORCH DATASET TEST"
    )
    print("=" * 75)

    # --------------------------------------------------------
    # TRAIN
    # --------------------------------------------------------

    train_dataset, train_loader = (
        create_dataloader(
            split="train",
            batch_size=4,
            shuffle=True,
            num_workers=0
        )
    )

    # --------------------------------------------------------
    # VALIDATION
    # --------------------------------------------------------

    validation_dataset, validation_loader = (
        create_dataloader(
            split="validation",
            batch_size=4,
            shuffle=False,
            num_workers=0
        )
    )

    # --------------------------------------------------------
    # Dataset lengths
    # --------------------------------------------------------

    print("\nDataset sizes:")

    print(
        f"Train: "
        f"{len(train_dataset):,}"
    )

    print(
        f"Validation: "
        f"{len(validation_dataset):,}"
    )

    # --------------------------------------------------------
    # Inspect single sample
    # --------------------------------------------------------

    print(
        "\nInspecting one training sample..."
    )

    sample = train_dataset[0]

    print(
        f"Image shape: "
        f"{sample['image'].shape}"
    )

    print(
        f"Image dtype: "
        f"{sample['image'].dtype}"
    )

    print(
        f"Image range: "
        f"{sample['image'].min().item():.6f} "
        f"to "
        f"{sample['image'].max().item():.6f}"
    )

    print(
        f"Mask shape: "
        f"{sample['mask'].shape}"
    )

    print(
        f"Mask dtype: "
        f"{sample['mask'].dtype}"
    )

    print(
        f"Mask unique values: "
        f"{torch.unique(sample['mask']).tolist()}"
    )

    print(
        f"Valid mask shape: "
        f"{sample['valid_mask'].shape}"
    )

    print(
        f"Valid pixels: "
        f"{sample['valid_mask'].mean().item() * 100:.2f}%"
    )

    print(
        f"Dataset category: "
        f"{sample['dataset']}"
    )

    print(
        f"Patch type: "
        f"{sample['patch_type']}"
    )

    print(
        f"Patch name: "
        f"{sample['patch_name']}"
    )

    # --------------------------------------------------------
    # Inspect batch
    # --------------------------------------------------------

    print(
        "\nInspecting training batch..."
    )

    batch = next(
        iter(train_loader)
    )

    print(
        f"Image batch shape: "
        f"{batch['image'].shape}"
    )

    print(
        f"Mask batch shape: "
        f"{batch['mask'].shape}"
    )

    print(
        f"Valid mask batch shape: "
        f"{batch['valid_mask'].shape}"
    )

    print(
        f"Image batch dtype: "
        f"{batch['image'].dtype}"
    )

    print(
        f"Mask batch dtype: "
        f"{batch['mask'].dtype}"
    )

    # --------------------------------------------------------
    # Validation batch
    # --------------------------------------------------------

    print(
        "\nInspecting validation batch..."
    )

    validation_batch = next(
        iter(validation_loader)
    )

    print(
        f"Validation image batch: "
        f"{validation_batch['image'].shape}"
    )

    print(
        f"Validation mask batch: "
        f"{validation_batch['mask'].shape}"
    )

    # --------------------------------------------------------
    # Final
    # --------------------------------------------------------

    expected_image_shape = (
        4,
        2,
        512,
        512
    )

    expected_mask_shape = (
        4,
        1,
        512,
        512
    )

    if (
        tuple(
            batch["image"].shape
        )
        != expected_image_shape
    ):

        raise AssertionError(
            "Unexpected image batch shape: "
            f"{batch['image'].shape}"
        )

    if (
        tuple(
            batch["mask"].shape
        )
        != expected_mask_shape
    ):

        raise AssertionError(
            "Unexpected mask batch shape: "
            f"{batch['mask'].shape}"
        )

    print("\n" + "=" * 75)

    print(
        "🎉 PYTORCH DATASET TEST PASSED"
    )

    print("=" * 75)


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    test_dataset()