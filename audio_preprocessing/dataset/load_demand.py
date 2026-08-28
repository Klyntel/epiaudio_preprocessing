"""DEMAND environmental-noise loader used by EpiAudio."""

from pathlib import Path

from audio_preprocessing.dataset._common import build_audio_dataset
from audio_preprocessing.dataset.base_loader import ZenodoLoader
from audio_preprocessing.datasets import AudioDataset, LabelSource

RECORD_ID = "1227121"
CATEGORY_LABELS = {
    "D": "Domestic",
    "N": "Nature",
    "O": "Office",
    "P": "Public",
    "S": "Street",
    "T": "Transportation",
}


def collect(root: Path):
    for archive in root.iterdir():
        if not archive.is_dir():
            continue
        for environment in archive.iterdir():
            if not environment.is_dir() or environment.name[:1] not in CATEGORY_LABELS:
                continue
            for path in environment.glob("*.wav"):
                yield path, CATEGORY_LABELS[environment.name[0]]


class DEMANDLoader(ZenodoLoader):
    record_id = RECORD_ID
    label_source = LabelSource.GOLD

    def __init__(
        self,
        root=None,
        *,
        split_ratios=(0.8, 0.1),
        seed=42,
        only=None,
        do_not_download=None,
        prepare=True,
    ) -> None:
        super().__init__(
            root, only=only, do_not_download=do_not_download, prepare=prepare
        )
        self.split_ratios = split_ratios
        self.seed = seed

    def build_dataset(self) -> AudioDataset:
        if self.root is None:
            raise ValueError("DEMANDLoader requires a root path")
        return build_audio_dataset(
            collect(self.root),
            self.split_ratios,
            self.seed,
            source_dataset="DEMAND",
            channel_format="mono",
            label_source=self.label_source,
        )

