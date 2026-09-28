"""Fine-tune repeated RISE HCP checkpoints on HCP-D data."""
from pathlib import Path
import numpy as np
import pandas as pd
import torch

from .finetune_config import FineTuneConfig, parse_finetune_config
from .train import train_repeat
from .utils import configure_logger, load_paired_dataset, resolve_device

def pretrained_checkpoint_path(config: FineTuneConfig, repeat_index: int) -> Path:
    path = config.model_dir.parent / config.pretrained_model_name / "repeat_{}".format(repeat_index) / "best_validloss_epoch.pth"
    if not path.is_file():
        raise FileNotFoundError("Missing pretrained checkpoint for repeat {}: {}".format(repeat_index, path))
    return path

def finetune_network(config: FineTuneConfig) -> None:
    config.model_dir.mkdir(parents=True, exist_ok=True)
    logger = configure_logger(config.model_dir, "finetune.log")
    device = resolve_device(config.device)
    logger.info("device=%s train_data=%s pretrained_model=%s", device, config.train_data, config.pretrained_model_name)
    dataset, _ = load_paired_dataset(config.train_data, config)
    if not config.geodesic_distances.is_file():
        raise FileNotFoundError(str(config.geodesic_distances))
    columns = pd.read_csv(config.geodesic_distances, nrows=0).columns[1:]
    values = pd.read_csv(config.geodesic_distances, usecols=columns, dtype=np.float32).to_numpy(dtype=np.float32, copy=False)
    geodesic = torch.from_numpy(values).to(device)
    for repeat_index in range(config.repeats):
        checkpoint = pretrained_checkpoint_path(config, repeat_index)
        print("fine-tuning repeat {}/{}".format(repeat_index + 1, config.repeats))
        train_repeat(config, dataset, geodesic, device, repeat_index, logger, initial_checkpoint=checkpoint)

def main(argv=None) -> None:
    finetune_network(parse_finetune_config(argv, description="Fine-tune RISE HCP models on HCP-D."))
if __name__ == "__main__":
    main()
