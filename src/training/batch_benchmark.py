from pathlib import Path
import sys
import time

import torch
from torch.amp import autocast, GradScaler
from torch.optim import Adam


# ============================================================
# PROJECT PATH
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

SRC_ROOT = PROJECT_ROOT / "src"

if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))


# ============================================================
# PROJECT IMPORTS
# ============================================================

from data.dataset import create_dataloader
from models.unet import UNet
from training.losses import BCEDiceLoss


# ============================================================
# CONFIG
# ============================================================

MANIFEST_PATH = (
    PROJECT_ROOT
    / "data"
    / "manifests"
    / "patch_manifest.csv"
)

NUM_WORKERS = 6

BATCH_SIZES = [4]

INPUT_CHANNELS = 2

OUTPUT_CHANNELS = 1

BASE_FEATURES = 64

LEARNING_RATE = 1e-4

WARMUP_BATCHES = 2

TEST_BATCHES = 10


# ============================================================
# DEVICE
# ============================================================

device = torch.device("cuda")

print("=" * 75)
print("U-NET BATCH SIZE STRESS TEST")
print("=" * 75)

print(
    f"\nGPU: {torch.cuda.get_device_name(0)}"
)

print(
    f"VRAM: "
    f"{torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB"
)

torch.backends.cudnn.benchmark = True


# ============================================================
# TEST ONE BATCH SIZE
# ============================================================

def test_batch_size(batch_size):

    print("\n" + "=" * 75)

    print(
        f"TESTING BATCH SIZE: {batch_size}"
    )

    print("=" * 75)

    # --------------------------------------------------------
    # Clear GPU memory
    # --------------------------------------------------------

    torch.cuda.empty_cache()

    torch.cuda.reset_peak_memory_stats()

    # --------------------------------------------------------
    # Dataset
    # --------------------------------------------------------

    _, loader = create_dataloader(
        manifest_path=MANIFEST_PATH,
        split="train",
        batch_size=batch_size,
        shuffle=True,
        num_workers=NUM_WORKERS
    )

    # --------------------------------------------------------
    # Model
    # --------------------------------------------------------

    model = UNet(
        input_channels=INPUT_CHANNELS,
        output_channels=OUTPUT_CHANNELS,
        base_features=BASE_FEATURES
    ).to(device)

    criterion = BCEDiceLoss()

    optimizer = Adam(
        model.parameters(),
        lr=LEARNING_RATE
    )

    scaler = GradScaler("cuda")

    model.train()

    # --------------------------------------------------------
    # Warm-up
    # --------------------------------------------------------

    try:

        iterator = iter(loader)

        for _ in range(WARMUP_BATCHES):

            batch = next(iterator)

            images = batch["image"].to(
                device,
                non_blocking=True
            )

            masks = batch["mask"].to(
                device,
                non_blocking=True
            )

            valid_masks = batch[
                "valid_mask"
            ].to(
                device,
                non_blocking=True
            )

            optimizer.zero_grad(
                set_to_none=True
            )

            with autocast(
                device_type="cuda",
                dtype=torch.float16
            ):

                logits = model(images)

                loss = criterion(
                    logits,
                    masks,
                    valid_masks
                )

            scaler.scale(
                loss
            ).backward()

            scaler.step(
                optimizer
            )

            scaler.update()

        torch.cuda.synchronize()

        print(
            "Warm-up: PASSED"
        )

        # ----------------------------------------------------
        # Benchmark
        # ----------------------------------------------------

        start = time.perf_counter()

        batches_done = 0

        while batches_done < TEST_BATCHES:

            try:

                batch = next(iterator)

            except StopIteration:

                iterator = iter(loader)

                batch = next(iterator)

            images = batch["image"].to(
                device,
                non_blocking=True
            )

            masks = batch["mask"].to(
                device,
                non_blocking=True
            )

            valid_masks = batch[
                "valid_mask"
            ].to(
                device,
                non_blocking=True
            )

            optimizer.zero_grad(
                set_to_none=True
            )

            with autocast(
                device_type="cuda",
                dtype=torch.float16
            ):

                logits = model(images)

                loss = criterion(
                    logits,
                    masks,
                    valid_masks
                )

            scaler.scale(
                loss
            ).backward()

            scaler.step(
                optimizer
            )

            scaler.update()

            batches_done += 1

        torch.cuda.synchronize()

        elapsed = (
            time.perf_counter() - start
        )

        average = (
            elapsed / TEST_BATCHES
        )

        # ----------------------------------------------------
        # Memory
        # ----------------------------------------------------

        peak_memory = (
            torch.cuda.max_memory_allocated()
            / 1024**3
        )

        reserved = (
            torch.cuda.max_memory_reserved()
            / 1024**3
        )

        # ----------------------------------------------------
        # Results
        # ----------------------------------------------------

        print(
            f"\nTime for {TEST_BATCHES} batches: "
            f"{elapsed:.2f} sec"
        )

        print(
            f"Average batch time: "
            f"{average:.3f} sec"
        )

        print(
            f"Batches/hour: "
            f"{3600 / average:.0f}"
        )

        print(
            f"Peak GPU allocated: "
            f"{peak_memory:.2f} GB"
        )

        print(
            f"Peak GPU reserved: "
            f"{reserved:.2f} GB"
        )

        print(
            "\nSTATUS: ✅ STABLE"
        )

        return {
            "batch_size": batch_size,
            "average": average,
            "peak_memory": peak_memory,
            "status": "STABLE"
        }

    except torch.cuda.OutOfMemoryError:

        print(
            "\nSTATUS: ❌ CUDA OUT OF MEMORY"
        )

        torch.cuda.empty_cache()

        return {
            "batch_size": batch_size,
            "average": None,
            "peak_memory": None,
            "status": "OOM"
        }

    except Exception as e:

        print(
            f"\nSTATUS: ❌ ERROR"
        )

        print(
            f"Error: {e}"
        )

        torch.cuda.empty_cache()

        return {
            "batch_size": batch_size,
            "average": None,
            "peak_memory": None,
            "status": "ERROR"
        }

    finally:

        del model
        del optimizer
        del scaler

        torch.cuda.empty_cache()


# ============================================================
# MAIN
# ============================================================

def main():

    results = []

    for batch_size in BATCH_SIZES:

        result = test_batch_size(
            batch_size
        )

        results.append(result)

    # ========================================================
    # SUMMARY
    # ========================================================

    print("\n\n")

    print("=" * 75)

    print("FINAL RESULTS")

    print("=" * 75)

    print(
        f"{'Batch':<10}"
        f"{'Time/Batch':<15}"
        f"{'Batches/Hour':<18}"
        f"{'VRAM':<12}"
        f"{'Status'}"
    )

    print("-" * 75)

    for result in results:

        batch_size = result[
            "batch_size"
        ]

        status = result[
            "status"
        ]

        if status == "STABLE":

            average = result[
                "average"
            ]

            memory = result[
                "peak_memory"
            ]

            print(
                f"{batch_size:<10}"
                f"{average:.3f} sec{'':<7}"
                f"{3600 / average:<18.0f}"
                f"{memory:.2f} GB{'':<5}"
                f"✅ STABLE"
            )

        else:

            print(
                f"{batch_size:<10}"
                f"{'-':<15}"
                f"{'-':<18}"
                f"{'-':<12}"
                f"❌ {status}"
            )

    # ========================================================
    # BEST BATCH SIZE
    # ========================================================

    stable_results = [
        r for r in results
        if r["status"] == "STABLE"
    ]

    if stable_results:

        best = min(
            stable_results,
            key=lambda x: x["average"]
        )

        print(
            "\n" + "=" * 75
        )

        print(
            f"BEST BATCH SIZE: "
            f"{best['batch_size']}"
        )

        print(
            f"Average time: "
            f"{best['average']:.3f} sec/batch"
        )

        print(
            f"Peak VRAM: "
            f"{best['peak_memory']:.2f} GB"
        )

        print("=" * 75)


# ============================================================
# WINDOWS SAFE ENTRY POINT
# ============================================================

if __name__ == "__main__":

    main()