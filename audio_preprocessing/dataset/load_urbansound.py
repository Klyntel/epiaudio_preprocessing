"""UrbanSound8K loader used by EpiAudio."""

from pathlib import Path

from datasets import load_dataset

from audio_preprocessing.dataset._common import build_audio_record, splits_to_audio_dataset
from audio_preprocessing.dataset.base_loader import ZenodoLoader
from audio_preprocessing.datasets import AudioDataset, Event, LabelSource

RECORD_ID = "1203745"


class UrbanSoundLoader(ZenodoLoader):
    record_id = RECORD_ID
    label_source = LabelSource.GOLD

    def build_dataset(self) -> AudioDataset:
        if self.root is None:
            raise ValueError("UrbanSoundLoader requires a root path")
        root = Path(self.root) / "UrbanSound8K" / "UrbanSound8K"
        metadata_path = Path("metadata/UrbanSound8K.csv")
        raw = load_dataset("csv", data_files=str(root / metadata_path))["train"]
        records = {}
        for row in raw:
            fold = str(row["fold"])
            label = str(row["class"])
            path = root / "audio" / f"fold{fold}" / str(row["slice_file_name"])
            if path not in records:
                records[path] = build_audio_record(
                    path,
                    label,
                    split="train",
                    source_dataset="ZenodoUrbanSound8k",
                    metadata_path=metadata_path,
                )
            elif label not in records[path]["class_list"]:
                records[path]["class_list"].append(label)
            records[path]["events"].append(
                Event(label, float(row["start"]), float(row["end"]) - float(row["start"])).to_dict()
            )
        return splits_to_audio_dataset(
            {"train": records.values()}, label_source=self.label_source
        )

