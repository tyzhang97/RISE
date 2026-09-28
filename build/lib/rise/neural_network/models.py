"""Neural-network architectures used by RISE."""

from collections import OrderedDict
from typing import Mapping, Sequence, Tuple

import torch
from torch import nn


DEFAULT_ATLAS_CLASS_COUNTS = OrderedDict([
    ("DK", 34),
    ("AAL", 38),
    ("HarvardOxford", 48),
    ("NM", 49),
    ("BN", 105),
    ("Glasser", 176),
    ("Schaefer200", 100),
    ("Schaefer400", 200),
])


class MultiAtlasSiameseNetwork(nn.Module):
    """Shared MLP encoder with hemisphere-specific multi-atlas heads.

    The default 378 -> 512 -> 512 -> 256 encoder and eight output heads match
    the effective architecture in the historical ``MultiAtlasNet_Siamese``.
    Both hemispheres share every encoder parameter but use separate classifiers.
    """

    def __init__(
        self,
        input_dim: int = 378,
        hidden_dims: Sequence[int] = (512, 512),
        embedding_dim: int = 256,
        atlas_class_counts: Mapping[str, int] = DEFAULT_ATLAS_CLASS_COUNTS,
    ) -> None:
        super().__init__()
        if len(hidden_dims) != 2:
            raise ValueError("hidden_dims must contain exactly two layer widths")

        self.hidden1 = nn.Linear(input_dim, hidden_dims[0])
        self.ln1 = nn.LayerNorm(hidden_dims[0])
        self.dropout = nn.Dropout(0.2)
        self.hidden2 = nn.Linear(hidden_dims[0], hidden_dims[1])
        self.ln2 = nn.LayerNorm(hidden_dims[1])
        self.hidden3 = nn.Linear(hidden_dims[1], embedding_dim)
        self.ln3 = nn.LayerNorm(embedding_dim)

        self.atlas_names = tuple(atlas_class_counts.keys())
        self.output_DK_lh = nn.Linear(embedding_dim, atlas_class_counts["DK"])
        self.output_AAL_lh = nn.Linear(embedding_dim, atlas_class_counts["AAL"])
        self.output_Harvard_lh = nn.Linear(embedding_dim, atlas_class_counts["HarvardOxford"])
        self.output_MICCAI_lh = nn.Linear(embedding_dim, atlas_class_counts["NM"])
        self.output_BN_lh = nn.Linear(embedding_dim, atlas_class_counts["BN"])
        self.output_Glasser_lh = nn.Linear(embedding_dim, atlas_class_counts["Glasser"])
        self.output_Sch200_lh = nn.Linear(embedding_dim, atlas_class_counts["Schaefer200"])
        self.output_Sch400_lh = nn.Linear(embedding_dim, atlas_class_counts["Schaefer400"])

        self.output_DK_rh = nn.Linear(embedding_dim, atlas_class_counts["DK"])
        self.output_AAL_rh = nn.Linear(embedding_dim, atlas_class_counts["AAL"])
        self.output_Harvard_rh = nn.Linear(embedding_dim, atlas_class_counts["HarvardOxford"])
        self.output_MICCAI_rh = nn.Linear(embedding_dim, atlas_class_counts["NM"])
        self.output_BN_rh = nn.Linear(embedding_dim, atlas_class_counts["BN"])
        self.output_Glasser_rh = nn.Linear(embedding_dim, atlas_class_counts["Glasser"])
        self.output_Sch200_rh = nn.Linear(embedding_dim, atlas_class_counts["Schaefer200"])
        self.output_Sch400_rh = nn.Linear(embedding_dim, atlas_class_counts["Schaefer400"])

    def encode(self, features: torch.Tensor) -> torch.Tensor:
        features = torch.relu(self.ln1(self.hidden1(features)))
        features = torch.relu(self.ln2(self.hidden2(features)))
        return torch.relu(self.ln3(self.hidden3(features)))

    def forward(
        self, left_features: torch.Tensor, right_features: torch.Tensor
    ) -> Tuple[Mapping[str, torch.Tensor], torch.Tensor, Mapping[str, torch.Tensor], torch.Tensor]:
        left_embedding = self.encode(left_features)
        right_embedding = self.encode(right_features)
        left_logits = OrderedDict([
            ("DK", self.output_DK_lh(left_embedding)),
            ("AAL", self.output_AAL_lh(left_embedding)),
            ("HarvardOxford", self.output_Harvard_lh(left_embedding)),
            ("NM", self.output_MICCAI_lh(left_embedding)),
            ("BN", self.output_BN_lh(left_embedding)),
            ("Glasser", self.output_Glasser_lh(left_embedding)),
            ("Schaefer200", self.output_Sch200_lh(left_embedding)),
            ("Schaefer400", self.output_Sch400_lh(left_embedding)),
        ])
        right_logits = OrderedDict([
            ("DK", self.output_DK_rh(right_embedding)),
            ("AAL", self.output_AAL_rh(right_embedding)),
            ("HarvardOxford", self.output_Harvard_rh(right_embedding)),
            ("NM", self.output_MICCAI_rh(right_embedding)),
            ("BN", self.output_BN_rh(right_embedding)),
            ("Glasser", self.output_Glasser_rh(right_embedding)),
            ("Schaefer200", self.output_Sch200_rh(right_embedding)),
            ("Schaefer400", self.output_Sch400_rh(right_embedding)),
        ])
        return left_logits, left_embedding, right_logits, right_embedding
