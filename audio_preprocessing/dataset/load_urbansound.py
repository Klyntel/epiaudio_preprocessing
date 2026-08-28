from pathlib import Path
from typing import Any, cast

from datasets import load_dataset
from tqdm import tqdm

from audio_preprocessing.dataset.base_loader import ZenodoLoader
from audio_preprocessing.dataset._common import build_audio_record, splits_to_audio_dataset
from audio_preprocessing.datasets import DATA_FEATURES, Event, AudioDataset, LabelSource


SOURCE_DATASET = "ZenodoUrbanSound8k"
RECORD_ID = "1203745"


class UrbanSoundLoader(ZenodoLoader):
    record_id = RECORD_ID
    label_source = LabelSource.GOLD

    def build_dataset(self) -> AudioDataset:
        parent_path = Path(self.root) / "UrbanSound8K/UrbanSound8K"
        csv_path = Path("metadata/UrbanSound8K.csv")

        raw_ds = load_dataset("csv", data_files=str(parent_path / csv_path))

        record_collect = {}
        for raw_row in tqdm(raw_ds["train"], total=raw_ds["train"].num_rows):
            row = cast(dict[str, Any], raw_row)
            fold = str(row["fold"])
            class_name = str(row["class"])
            start = float(row["start"])
            end = float(row["end"])
            file_path = parent_path / "audio" / f"fold{fold}" / str(row["slice_file_name"])
            if file_path not in record_collect:
                record_collect[file_path] = build_audio_record(
                    file_path,
                    class_name,
                    split="train",
                    source_dataset=SOURCE_DATASET,
                    metadata_path=str(csv_path),
                    channel_format=None,
                    environment=None,
                )
            else:
                record_collect[file_path]["class_list"].append(class_name)

            record_collect[file_path]["events"].append(
                Event(
                    class_name,
                    start,
                    end - start,
                ).to_dict()
            )

        return splits_to_audio_dataset(
            {"train": record_collect.values()},
            features=DATA_FEATURES,
            label_source=self.label_source,
        )


if __name__ == "__main__":
    ub8k = UrbanSoundLoader()()
    print(ub8k.info())
