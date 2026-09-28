"""Train the RISE multi-atlas Siamese network."""

from pathlib import Path
from typing import Dict, Mapping, Optional, Tuple

import matplotlib

matplotlib.use("Agg")
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader

from .config import TrainingConfig, parse_training_config
from .models import DEFAULT_ATLAS_CLASS_COUNTS, MultiAtlasSiameseNetwork
from .utils import (
    PairedHemisphereDataset,
    build_model,
    configure_logger,
    load_paired_dataset,
    make_loader,
    move_batch,
    resolve_device,
    subset_dataset,
)


def soft_cross_entropy(logits: torch.Tensor, soft_targets: torch.Tensor) -> torch.Tensor:
    """Average cross entropy against parcel-wise probability targets."""

    return -(soft_targets * F.log_softmax(logits, dim=1)).sum(dim=1).mean()


def similarity_distance_loss(
    embeddings: torch.Tensor,
    distances: torch.Tensor,
    epsilon: float = 1e-8,
) -> torch.Tensor:
    """Correlate embedding similarity negatively with geodesic distance."""
    similarity = embeddings @ embeddings.T
    mask = ~torch.eye(distances.size(0), dtype=torch.bool, device=distances.device)
    similarity = similarity[mask]
    distances = distances[mask]
    similarity = (similarity - similarity.mean()) / (similarity.std() + epsilon)
    distances = (distances - distances.mean()) / (distances.std() + epsilon)
    return (similarity * distances).mean()


def multi_atlas_loss(
    logits: Mapping[str, torch.Tensor],
    embedding: torch.Tensor,
    soft_targets: torch.Tensor,
    distances: torch.Tensor,
    config: TrainingConfig,
) -> torch.Tensor:
    """Combine soft CE, geodesic correlation, and embedding L2 losses.

    The eight atlas losses receive equal weight, matching the original code.
    """

    offset = 0
    classification_losses = []
    for atlas, width in DEFAULT_ATLAS_CLASS_COUNTS.items():
        classification_losses.append(
            soft_cross_entropy(logits[atlas], soft_targets[:, offset : offset + width])
        )
        offset += width
    classification = torch.stack(classification_losses).mean()
    geodesic = similarity_distance_loss(embedding, distances)
    embedding_l2 = embedding.norm(p=2, dim=1).square().mean()
    return (
        classification
        + config.geodesic_loss_weight * geodesic
        + config.embedding_l2_weight * embedding_l2
    )


def run_epoch(
    model: MultiAtlasSiameseNetwork,
    loader: DataLoader,
    geodesic: torch.Tensor,
    device: torch.device,
    config: TrainingConfig,
    optimizer=None,
) -> Tuple[float, Dict[str, float]]:
    """Run one training or validation epoch.

    An optimizer enables gradient updates; without one this is validation.
    """

    training = optimizer is not None
    model.train(training)
    loss_total = 0.0
    parcel_total = 0
    correct = {name: 0 for name in model.atlas_names}
    context = torch.enable_grad() if training else torch.no_grad()

    with context:
        for batch in loader:
            (
                left_x,
                left_y,
                left_soft,
                left_parcel,
                right_x,
                right_y,
                right_soft,
                right_parcel,
            ) = move_batch(batch, device)
            if left_x.shape[0] < 2:
                continue

            left_logits, left_embedding, right_logits, right_embedding = model(
                left_x, right_x
            )
            left_distances = geodesic[left_parcel][:, left_parcel]
            right_distances = geodesic[right_parcel][:, right_parcel]
            left_loss = multi_atlas_loss(
                left_logits, left_embedding, left_soft, left_distances, config
            )
            right_loss = multi_atlas_loss(
                right_logits, right_embedding, right_soft, right_distances, config
            )
            loss = (left_loss + right_loss) / 2.0

            if training:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()

            batch_size = left_x.shape[0]
            loss_total += loss.item() * batch_size
            parcel_total += batch_size
            for index, atlas in enumerate(model.atlas_names):
                correct[atlas] += int(
                    (left_logits[atlas].argmax(1) == left_y[:, index]).sum()
                )
                correct[atlas] += int(
                    (right_logits[atlas].argmax(1) == right_y[:, index]).sum()
                )

    if parcel_total == 0:
        raise RuntimeError("No batches with at least two parcels were available")
    accuracies = {
        name: correct[name] / (2.0 * parcel_total) for name in model.atlas_names
    }
    return loss_total / parcel_total, accuracies


def plot_training_history(history, output_dir: Path) -> None:
    """Save loss and per-atlas validation-accuracy curves."""

    frame = pd.DataFrame(history)

    figure, axis = plt.subplots(dpi=300)
    axis.plot(frame["epoch"], frame["train_loss"], color="red")
    axis.plot(frame["epoch"], frame["validation_loss"], color="green")
    axis.plot(frame["epoch"], frame["mean_validation_accuracy"], color="blue")
    axis.set_xlabel("Epoch")
    axis.legend(
        ["Train loss", "Validation loss", "Validation accuracy"],
        loc="upper right",
    )
    figure.tight_layout()
    figure.savefig(output_dir / "loss.png")
    plt.close(figure)

    figure, axis = plt.subplots(dpi=300)
    for atlas in DEFAULT_ATLAS_CLASS_COUNTS:
        axis.plot(
            frame["epoch"],
            frame["{}_accuracy".format(atlas.lower())],
            label=atlas,
        )
    axis.set_xlabel("Epoch")
    axis.set_ylabel("Validation accuracy")
    axis.legend(loc="upper right")
    figure.tight_layout()
    figure.savefig(output_dir / "accuracy.png")
    plt.close(figure)


def train_repeat(
    config: TrainingConfig,
    dataset: PairedHemisphereDataset,
    geodesic: torch.Tensor,
    device: torch.device,
    repeat_index: int,
    logger,
    initial_checkpoint: Optional[Path] = None,
) -> Path:
    """Train one random subject split and save best and final checkpoints."""

    subjects = np.unique(dataset.left.subject_ids)
    train_subjects, validation_subjects = train_test_split(
        subjects,
        test_size=config.validation_ratio,
    )
    train_data = subset_dataset(dataset, train_subjects)
    validation_data = subset_dataset(dataset, validation_subjects)
    train_loader = make_loader(train_data, config, shuffle=True, drop_last=True)
    validation_loader = make_loader(
        validation_data, config, shuffle=True, drop_last=True
    )

    model = build_model(config, device)
    if initial_checkpoint is not None:
        try:
            state = torch.load(initial_checkpoint, map_location=device, weights_only=True)
        except TypeError:
            state = torch.load(initial_checkpoint, map_location=device)
        model.load_state_dict(state)
        logger.info("repeat=%d initialized from %s", repeat_index, initial_checkpoint)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer,
        step_size=config.scheduler_step_size,
        gamma=config.scheduler_gamma,
    )

    repeat_dir = config.model_dir / "repeat_{}".format(repeat_index)
    repeat_dir.mkdir(parents=True, exist_ok=True)

    history = []
    best_loss = float("inf")
    stale_epochs = 0
    for epoch in range(config.epochs):
        train_loss, _ = run_epoch(
            model, train_loader, geodesic, device, config, optimizer
        )
        validation_loss, validation_accuracy = run_epoch(
            model, validation_loader, geodesic, device, config
        )
        row = {
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "validation_loss": validation_loss,
            "mean_validation_accuracy": float(
                np.mean(list(validation_accuracy.values()))
            ),
            "learning_rate": optimizer.param_groups[0]["lr"],
        }
        row.update(
            {
                "{}_accuracy".format(name.lower()): value
                for name, value in validation_accuracy.items()
            }
        )
        history.append(row)
        pd.DataFrame(history).to_csv(repeat_dir / "history.csv", index=False)
        torch.save(model.state_dict(), repeat_dir / "last_epoch.pth")

        if validation_loss <= best_loss:
            best_loss = validation_loss
            stale_epochs = 0
            torch.save(model.state_dict(), repeat_dir / "best_validloss_epoch.pth")
        else:
            stale_epochs += 1

        logger.info(
            "repeat=%d epoch=%d train_loss=%.4f validation_loss=%.4f validation_accuracy=%.4f",
            repeat_index,
            epoch + 1,
            train_loss,
            validation_loss,
            row["mean_validation_accuracy"],
        )
        scheduler.step()
        if (
            config.early_stopping_patience
            and stale_epochs >= config.early_stopping_patience
        ):
            logger.info(
                "repeat=%d early stopping after %d stale epochs",
                repeat_index,
                stale_epochs,
            )
            break
    plot_training_history(history, repeat_dir)
    return repeat_dir


def train_network(config: TrainingConfig) -> None:
    """Load training resources and train every configured repeat."""

    config.model_dir.mkdir(parents=True, exist_ok=True)
    logger = configure_logger(config.model_dir, "train.log")
    device = resolve_device(config.device)
    logger.info("device=%s train_data=%s", device, config.train_data)

    dataset, _ = load_paired_dataset(config.train_data, config)
    if not config.geodesic_distances.is_file():
        raise FileNotFoundError(str(config.geodesic_distances))

    # The first CSV column contains parcel names and must not be cast to float.
    geodesic_columns = pd.read_csv(config.geodesic_distances, nrows=0).columns[1:]
    geodesic_values = pd.read_csv(
        config.geodesic_distances,
        usecols=geodesic_columns,
        dtype=np.float32,
    ).to_numpy(dtype=np.float32, copy=False)
    geodesic = torch.from_numpy(geodesic_values).to(device)

    for repeat_index in range(config.repeats):
        print("training repeat {}/{}".format(repeat_index + 1, config.repeats))
        train_repeat(config, dataset, geodesic, device, repeat_index, logger)


def main(argv=None) -> None:
    config = parse_training_config(argv, description="Train the RISE neural network.")
    train_network(config)


if __name__ == "__main__":
    main()
