"""Load the DEMAND noise dataset into an AudioDataset.

DEMAND extracts to ``<archive>/<ENV>/chNN.wav`` (e.g. ``DKITCHEN_16k/DKITCHEN/ch01.wav``).
The label is the broad category derived from the environment's first letter.

Run ``python -m audio_preprocessing.dataset.load_demand`` from the repo root to download a small subset
and build the dataset.
"""

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


def collect(input_path):
    """Yield ``(wav_path, category_label)`` for every channel WAV under ``input_path``.

    Walks ``<archive>/<ENV>/chNN.wav`` and labels each file by the environment's category.
    """
    for archive_dir in input_path.iterdir():
        if not archive_dir.is_dir():
            continue
        for env_dir in archive_dir.iterdir():
            if not env_dir.is_dir():
                continue
            label = CATEGORY_LABELS[env_dir.name[0]]
            for file in env_dir.iterdir():
                if file.is_file() and file.suffix == ".wav":
                    yield file, label


class DEMANDLoader(ZenodoLoader):
    """Prepare and build the DEMAND environmental-noise dataset."""

    record_id = RECORD_ID
    label_source = LabelSource.GOLD

    def __init__(
        self,
        root: str | Path | None = None,
        *,
        split_ratios: list[float] | tuple[float, float] = (0.8, 0.1),
        seed: int = 42,
        only: list[str] | None = None,
        do_not_download: list[str] | None = None,
        prepare: bool = True,
    ) -> None:
        super().__init__(
            root=root,
            only=only,
            do_not_download=do_not_download,
            prepare=prepare,
        )
        self.split_ratios = split_ratios
        self.seed = seed

    def build_dataset(self) -> AudioDataset:
        if self.root is None:
            raise ValueError("DEMANDLoader needs a root path. Use prepare=False with an existing local root.")

        return build_audio_dataset(
            collect(self.root),
            self.split_ratios,
            self.seed,
            source_dataset="DEMAND",
            channel_format="mono",
            label_source=self.label_source,
        )


def main():
    # Full DEMAND record (~7.4 GB: all 15 environments, 16k + 48k):
    #   DEMANDLoader(do_not_download=["scripts"])()
    # Small subset for testing transforms: two 16 kHz environments (~250 MB).
    return DEMANDLoader(only=["DKITCHEN_16k", "NPARK_16k"])()


if __name__ == "__main__":
    demand = main()
    print(demand.info())
