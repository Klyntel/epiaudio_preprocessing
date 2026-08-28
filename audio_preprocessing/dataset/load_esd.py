import zipfile
import warnings
from pathlib import Path
from typing import Any
import requests
from tqdm import tqdm
from datasets.features.features import Features, Value
from audio_preprocessing.dataset.base_loader import BaseLoader
from audio_preprocessing.datasets import AudioDataset, LabelSource, DATA_FEATURES
from audio_preprocessing.dataset._common import (
    read_tsv_rows,
    split_records_by_key,
    build_audio_record,
    splits_to_audio_dataset
)


SOURCE_DATASET = "Emotional Speech Dataset"
KAGGLE_DATASET = "nguyenthanhlim/emotional-speech-dataset-esd"
KAGGLE_VERSION = 1
FOLDER_NAME = "Emotion Speech Dataset"
DEFAULT_ROOT = "data/esd"
LANGUAGE = {i: "Chinese" if i <= 10 else "English" for i in range(1, 21)}
SPEAKER_SEX = {i: "Female" if i in [1,2,3,7,9,15,16,17,18] else "Male" for i in range(1, 21)}
ESD_FEATURES = Features({
    **DATA_FEATURES,
    "speaker_id": Value("int16"),
    "speaker_sex": Value("string"),
    "language": Value("string"),
    "transcript": Value("string")
})
DEFAULT_SPLIT_RATIOS = (.8, .1)


def _download_kaggle_dataset(destination: Path) -> None:
    """Download and extract an entire public Kaggle dataset and extracts."""
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


class ESDLoader(BaseLoader):
    """ Download a specific version of the Emotional Speech dataset
    and convert it to an AudioDataset.
    """
    label_source = LabelSource.GOLD

    def __init__(
        self,
        root: str | Path | None = None,
        *,
        prepare: bool = True,
        split_ratios: tuple[float, float] = DEFAULT_SPLIT_RATIOS
    ) -> None:
        super().__init__(root=root, prepare=prepare)

        self.root: Path = Path(DEFAULT_ROOT) if root is None else self.root
        self.split_ratios = split_ratios

    def _records(self, input_dir: Path) -> list[dict[str, Any]]:
        """Build one record per .wav file under input_dir,
        merging in transcript/language/sex metadata.
        """
        num_files = len(list(input_dir.rglob("*.wav")))

        pbar = tqdm()
        num_records_built = 0
        records = []
        for folder in input_dir.iterdir():
            if not folder.is_dir():
                continue

            folder_name = folder.name
            text_file_path = folder / f"{folder_name}.txt"
            if not text_file_path.is_file():
                raise ValueError(f"No file {text_file_path} found.")

            transcripts = {row[0]: row[1] for row in read_tsv_rows(text_file_path)}
            idx = int(folder_name)

            for emotion_subfolder in folder.iterdir():
                if not emotion_subfolder.is_dir():
                    continue

                label = emotion_subfolder.name
                for path in emotion_subfolder.glob("*.wav"):
                    audio_record = build_audio_record(
                        str(path),
                        label,
                        source_dataset=SOURCE_DATASET,
                        metadata_path=str(text_file_path),
                    )

                    file_name = path.stem
                    if file_name not in transcripts:
                        warnings.warn(f"Transcript could not be found for {path}")
                    metadata_values = {
                        "speaker_id": idx,
                        "speaker_sex": SPEAKER_SEX[idx],
                        "language": LANGUAGE[idx],
                        "transcript": transcripts.get(file_name, None)
                    }

                    records.append({**audio_record, **metadata_values})
                    num_records_built += 1
                    pbar.set_description(f"Built {num_records_built} / {num_files} records")

        pbar.close()

        return records

    def prepare_raw(self) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        _download_kaggle_dataset(self.root)

    def build_dataset(self) -> AudioDataset:
        path = self.root / FOLDER_NAME
        if not path.is_dir():
            raise ValueError(f"{path} is not a directory.")

        records = self._records(path)

        split_records = split_records_by_key(
            records,
            key_fn=lambda record: (record["class_list"][0], record["speaker_id"]),
            split_ratios=self.split_ratios
        )
        for split, rows in split_records.items():
            for record in rows:
                record["split"] = split

        dataset = splits_to_audio_dataset(
            split_records,
            features=ESD_FEATURES,
            label_source=self.label_source,
        )

        return dataset


def main() -> AudioDataset:
    return ESDLoader()()


if __name__ == "__main__":
    esd = main()
    print(esd.info())