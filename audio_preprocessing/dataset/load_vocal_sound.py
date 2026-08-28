import requests
import warnings
import zipfile
from pathlib import Path
from typing import Any

from tqdm import tqdm
import pandas as pd
from datasets.features.features import Features, Value

from audio_preprocessing.dataset._common import build_audio_record, splits_to_audio_dataset
from audio_preprocessing.dataset.base_loader import BaseLoader
from audio_preprocessing.datasets import DATA_FEATURES, AudioDataset, LabelSource


SOURCE_DATASET = "VocalSound"
DEFAULT_ROOT = f"data/{SOURCE_DATASET}_16k"
DOWNLOAD_URLS = {
    "16k" : "https://www.dropbox.com/s/c5ace70qh1vbyzb/vs_release_16k.zip?dl=1",
    "44k" : "https://www.dropbox.com/s/ybgaprezl8ubcce/vs_release_44k.zip?dl=1"
}
AUDIO_DIR_NAMES = {
    "16k" : "audio_16k",
    "44k" : "data_44k"
}
SPLIT_MAP = {
    "tr": "train",
    "val": "valid",
    "te": "eval"
}
METADATA_COLUMNS = [
    "spk_id",
    "gender",
    "age",
    "country",
    "native language",
    "health condition (no=no problem)"
]
METADATA_SCHEMA = {
    "spk_id": Value("string"),
    "gender": Value("string"),
    "age": Value("int16"),
    "country": Value("string"),
    "native language": Value("string"),
    "health condition (no=no problem)": Value("string")
}
VOCAL_SOUND_FEATURES = Features({
    **DATA_FEATURES,
    **METADATA_SCHEMA
})


class VocalSoundLoader(BaseLoader):
    """Load the VocalSound  into an AudioDataset."""
    label_source = LabelSource.GOLD

    def __init__(
        self,
        root: str | Path | None = None,
        *,
        prepare: bool = True,
        sample_rate: str = "16k",
        **config: Any
    ) -> None:
        super().__init__(root=root, prepare=prepare, **config)

        if sample_rate not in DOWNLOAD_URLS:
            raise ValueError(f"sample_rate must be in {sorted(list(DOWNLOAD_URLS.keys()))}, got {sample_rate}")
        self.sample_rate = sample_rate

        self._create_root()

    def _create_root(self) -> None:
        if self.root is None:
            self.root = Path(DEFAULT_ROOT)

        self.root.mkdir(parents=True, exist_ok=True)

    def _download(self, destination: Path) -> None:
        archive_path = destination / "dataset.zip.part"
        url = DOWNLOAD_URLS[self.sample_rate]
        with requests.get(
            url,
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

    def _get_audio_file_paths(self, spk_ids: set[str]) -> list[Path]:
        path = self.root / AUDIO_DIR_NAMES[self.sample_rate]
        all_file_paths = sorted(path.rglob("*.wav"))

        audio_file_paths = []
        for file_path in all_file_paths:
            spk_id = str(file_path.stem).split("_")[0]
            if spk_id in spk_ids:
                audio_file_paths.append(file_path)

        return audio_file_paths

    def _load_metadata(self, split: str) -> tuple[Path, dict[str, dict[str, Any]]]:
        metadata_path = self.root / "meta" / f"{split}_meta.csv"
        metadata_df = pd.read_csv(metadata_path)
        metadata_df.columns = METADATA_COLUMNS.copy()

        records = metadata_df.to_dict("records")
        metadata: dict[str, dict[str, Any]] = dict()

        for record in records:
            spk_id = record["spk_id"]
            metadata[spk_id] = record

        return metadata_path, metadata

    def _build_split(self, split: str) -> list[dict[str, Any]]:
        metadata_path, metadata = self._load_metadata(split)

        records = []
        audio_file_paths = self._get_audio_file_paths(set(metadata.keys()))
        num_files = len(audio_file_paths)
        pbar = tqdm(audio_file_paths)
        for i, file_path in enumerate(pbar):
            pbar.set_description(f"Processing file {i + 1} / {num_files}")
            spk_id, _, label = str(file_path.stem).split("_")
            try:
                record = build_audio_record(
                    str(file_path),
                    label,
                    split=SPLIT_MAP[split],
                    source_dataset=f"{SOURCE_DATASET}_{self.sample_rate}",
                    metadata_path=str(metadata_path)
                )
            except Exception as e:
                warnings.warn(f"Error in processing {file_path}: {e}, skipping.", stacklevel=2)
                continue

            record |= metadata[spk_id]
            records.append(record)

        return records

    def prepare_raw(self) -> None:
        self._download(self.root)

    def build_dataset(self) -> AudioDataset:
        splits = dict()
        for source_split, split in SPLIT_MAP.items():
            splits[split] = self._build_split(source_split)

        audio_dataset = splits_to_audio_dataset(
            splits,
            features=VOCAL_SOUND_FEATURES,
            label_source=self.label_source
        )

        return audio_dataset


def main():
    return VocalSoundLoader()()


if __name__ == "__main__":
    print(main().info())