import subprocess
import warnings
from pathlib import Path
from typing import Any
from tqdm import tqdm
import pandas as pd
from datasets.features.features import Features, Value
from audio_preprocessing.dataset._common import build_audio_record, splits_to_audio_dataset
from audio_preprocessing.dataset.base_loader import BaseLoader
from audio_preprocessing.datasets import DATA_FEATURES, AudioDataset, LabelSource


DEFAULT_ROOT = "data/meld"
S3_PATH = "s3://team-audio-data/raw/content/meld"

SOURCE_DATASET = "MELD"
SPLITS = ("train", "valid", "eval")
COLUMN_MAP = {
    "Sr No.": "sr no.",
    "Utterance": "utterance",
    "Speaker": "speaker",
    "Sentiment": "sentiment",
    "Dialogue_ID": "dialogue_id",
    "Utterance_ID": "utterance_id",
    "Season": "season",
    "Episode": "episode",
    "StartTime": "start_time",
    "EndTime": "end_time"
}
MELD_FEATURES = Features({
    **DATA_FEATURES,
    "sr no.": Value("int16"),
    "utterance": Value("string"),
    "speaker": Value("string"),
    "sentiment": Value("string"),
    "dialogue_id": Value("int16"),
    "utterance_id": Value("int16"),
    "season": Value("int16"),
    "episode": Value("int16"),
    "start_time": Value("string"),
    "end_time": Value("string")
})


class MELDLoader(BaseLoader):
    """Load the MELD dataset into an AudioDataset.

    WARNING: This dataset extends the EmotionLines Dataset.

    Documentation: https://affective-meld.github.io/
    GitHub: https://github.com/declare-lab/MELD/
    License: GPL-3.0
    """
    label_source = LabelSource.GOLD

    def __init__(
        self,
        root: str | Path | None = None,
        *,
        prepare: bool = True,
        **config: Any
    ) -> None:
        super().__init__(root=root, prepare=prepare, **config)
        self.root = self.root or Path(DEFAULT_ROOT)

    def _get_metadata(self, split: str) -> tuple[str, dict[str, dict[str, Any]]]:
        path = str(self.root / f"{split}.csv")
        df = pd.read_csv(path)
        records = df.to_dict("records")
        records = {
            f"dia{record["Dialogue_ID"]}_utt{record["Utterance_ID"]}": record
            for record in records
        }

        return path, records

    def _build_records(self, split: str) -> list[dict[str, Any]]:
        metadata_path, metadata = self._get_metadata(split)

        audio_file_paths = list((self.root / split).rglob("*.wav"))
        num_audio_files = len(audio_file_paths)

        records = []
        records_built = 0
        pbar = tqdm(audio_file_paths)

        for audio_file_path in pbar:
            file_name = audio_file_path.stem
            if file_name not in metadata:
                warnings.warn(f"Metadata not found for {file_name}; skipping.")
                continue
            file_metadata = metadata[file_name]
            record: dict[str, Any] = build_audio_record(
                    audio_file_path,
                    file_metadata["Emotion"],
                    split=split,
                    source_dataset=SOURCE_DATASET,
                    metadata_path=metadata_path
            ) | {COLUMN_MAP[col]: file_metadata[col] for col in COLUMN_MAP}

            records.append(record)
            records_built += 1
            pbar.set_description(f"Built {records_built} / {num_audio_files} records.")

        pbar.close()

        return records

    def prepare_raw(self) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        subprocess.run(["aws", "s3", "sync", S3_PATH, str(self.root)], check=True)

    def build_dataset(self) -> AudioDataset:
        splits = dict()
        for split in SPLITS:
            splits[split] = self._build_records(split)

        audio_dataset = splits_to_audio_dataset(
            splits,
            features=MELD_FEATURES,
            label_source=self.label_source
        )

        return audio_dataset


def main():
    return MELDLoader()()


if __name__ == "__main__":
    print(main().info())
