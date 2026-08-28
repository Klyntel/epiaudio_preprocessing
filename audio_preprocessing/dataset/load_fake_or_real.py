"""Load the Kaggle Fake or Real Dataset."""

import zipfile
import warnings
from pathlib import Path
from typing import Any
import requests

from tqdm import tqdm

from audio_preprocessing.dataset.base_loader import BaseLoader
from audio_preprocessing.datasets import AudioDataset, LabelSource, DATA_FEATURES
from audio_preprocessing.dataset._common import build_audio_record, splits_to_audio_dataset


SOURCE_DATASET = "Fake-or-Real"
KAGGLE_DATASET = "mohammedabdeldayem/the-fake-or-real-dataset"
KAGGLE_VERSION = 2
VERSIONS = ["for-2sec", "for-norm", "for-original", "for-rerec"]
SPLIT_MAP = {
    "training": "train",
    "validation": "valid",
    "testing": "eval"
}
DEFAULT_ROOT = "data/fake_or_real"


def _download_kaggle_dataset(destination: Path) -> None:
    """Download and extract an entire public Kaggle dataset and extracts."""
    # It doesn't seem possible to download specific directories, and
    # using the API to find specific file names resulted in 404 errors
    # due to the number of requests required.
    destination.mkdir(parents=True, exist_ok=True)
    archive_path = destination / "dataset.zip.part"
    url = f"https://www.kaggle.com/api/v1/datasets/download/{KAGGLE_DATASET}"
    with requests.get(
        url,
        params={"datasetVersionNumber": KAGGLE_VERSION},
        stream=True,
        timeout=(15, 300),
    ) as response:
        response.raise_for_status()
        total = int(response.headers.get("content-length", 0))
        with (
            archive_path.open("wb") as handle,
            tqdm(total=total, unit="B", unit_scale=True, desc=archive_path.name) as bar,
        ):
            for chunk in response.iter_content(chunk_size=1 << 20):
                if chunk:
                    handle.write(chunk)
                    bar.update(len(chunk))

    with zipfile.ZipFile(archive_path) as archive:
        archive.extractall(destination)
    archive_path.unlink()


def _find_audio_files(input_dir: Path) -> list[Path]:
    """Recursively find all .wav and .mp3 files under input_dir."""
    return list(
        path for path in input_dir.rglob("*")
        if path.is_file() and path.suffix.lower() in {".wav", ".mp3"}
    )


class FakeOrRealLoader(BaseLoader):
    """ Download a specific version of the Fake or Real dataset
    and convert it to an AudioDataset.
    """
    label_source = LabelSource.GOLD

    def __init__(
        self,
        root: str | Path | None = None,
        *,
        version: str = "for-2sec",
        prepare: bool = True
    ) -> None:
        super().__init__(root=root, prepare=prepare)

        if version not in VERSIONS:
            raise ValueError(f"version must be in {VERSIONS}, got {version}.")

        self.version = version
        root = DEFAULT_ROOT if root is None else root
        self.root = Path(root)

    def _create_row(self, path: Path) -> dict[str, Any]:
        parent_path = path.parent
        label = parent_path.stem
        split = SPLIT_MAP[parent_path.parent.stem]
        record = build_audio_record(
            path,
            label,
            split=split,
            source_dataset=f"{SOURCE_DATASET}_{self.version}"
        )

        return record

    def prepare_raw(self) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        _download_kaggle_dataset(self.root)

    def build_dataset(self) -> AudioDataset:
        path = self.root / self.version
        if not path.is_dir():
            raise ValueError(f"{path} is not a directory.")

        audio_file_paths = _find_audio_files(path)
        num_files = len(audio_file_paths)
        if num_files == 0:
            raise ValueError(f"No audio files found in {path}.")

        splits = {split: [] for split in SPLIT_MAP.values()}
        pbar = tqdm(audio_file_paths)
        for i, path in enumerate(pbar):
            pbar.set_description(f"Processing audio files: {i + 1}/{num_files}")
            try:
                record = self._create_row(path)
                splits[record["split"]].append(record)
            except Exception as e:
                warnings.warn(f"Error in processing {path}: {e}, skipping.", stacklevel=2)

        dataset = splits_to_audio_dataset(
            splits,
            features=DATA_FEATURES,
            label_source=self.label_source,
        )

        return dataset


def main() -> AudioDataset:
    return FakeOrRealLoader()()


if __name__ == "__main__":
    fake_or_real = main()
    print(fake_or_real.info())