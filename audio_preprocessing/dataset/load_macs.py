"""Load the MACS multi-annotator acoustic-scene dataset."""

from __future__ import annotations

import warnings
import json
from pathlib import Path
from typing import Any
import yaml
from tqdm import tqdm
import pandas as pd
from datasets.features.features import Features, Value
from audio_preprocessing.dataset._common import (
    find_unique,
    build_audio_record,
    splits_to_audio_dataset,
    split_records_by_key,
)
from audio_preprocessing.dataset.base_loader import ZenodoLoader
from audio_preprocessing.datasets import DATA_FEATURES, AudioDataset, LabelSource


METADATA_RECORD_ID = "5114771"
AUDIO_RECORD_ID = "2589280"
RECORD_IDs = (METADATA_RECORD_ID, AUDIO_RECORD_ID)
DO_NOT_DOWNLOAD = ["LICENSE.txt", "TAU-urban-acoustic-scenes-2019-development.doc"]

SOURCE_DATASET = "MACS_v1"
ANNOTATIONS_FILE_NAME = "MACS.yaml"
COMPETENCE_FILE_NAME = "MACS_competence.csv"
METADATA_FILE_NAME = "meta.csv"
DEFAULT_ROOT = Path("data/macs")

DEFAULT_SPLIT_RATIOS = (0.8, 0.1)
MACS_FEATURES = Features(
    {
        **DATA_FEATURES,
        "annotations": Value("string"),
        "identifier": Value("string"),
        "source_label": Value("string"),
    }
)


class MACSLoader(ZenodoLoader):
    """Download selected archive parts and index the official location-disjoint fold."""

    record_ids = RECORD_IDs
    label_source = LabelSource.GOLD

    def __init__(
        self,
        root: str | Path | None = None,
        *,
        split_ratios: tuple[float, float] = DEFAULT_SPLIT_RATIOS,
        prepare: bool = True,
    ) -> None:
        super().__init__(root=root, do_not_download=DO_NOT_DOWNLOAD, prepare=prepare)

        self.root = self.root or DEFAULT_ROOT
        self.split_ratios = split_ratios

    def _get_annotations(self) -> dict[str, list[dict[str, Any]]]:
        annotations_path = find_unique(self.root, ANNOTATIONS_FILE_NAME, SOURCE_DATASET)
        annotations_list = yaml.load(
            annotations_path.read_text(), Loader=yaml.SafeLoader
        )["files"]
        annotations = {f["filename"]: f["annotations"] for f in annotations_list}

        competence_path = find_unique(self.root, COMPETENCE_FILE_NAME, SOURCE_DATASET)
        competence_df = pd.read_csv(competence_path, sep="\t")
        competences = competence_df.set_index("annotator_id")["competence"].to_dict()

        for file_annotations in annotations.values():
            for annotation in file_annotations:
                annotator_id = annotation["annotator_id"]
                annotation["annotator_competence"] = competences[annotator_id]

        return annotations

    def _get_metadata(self) -> tuple[Path, dict[str, dict[str, str]]]:
        metadata_path = find_unique(self.root, METADATA_FILE_NAME, SOURCE_DATASET)
        metadata_records = pd.read_csv(metadata_path, sep="\t").to_dict("records")
        metadata = {
            record["filename"].replace("audio/", ""): {
                k: v for k, v in record.items() if k != "filename"
            }
            for record in metadata_records
        }

        return metadata_path, metadata

    def _build_records(self) -> list[dict[str, Any]]:
        metadata_path, metadata = self._get_metadata()
        annotations = self._get_annotations()

        audio_file_paths = list(self.root.rglob("*.wav"))
        num_files = len(audio_file_paths)
        num_records_built = 0
        pbar = tqdm()

        records = []
        for audio_file_path in audio_file_paths:
            audio_file_name = audio_file_path.name
            if audio_file_name not in metadata:
                warnings.warn(f"No metadata found for {audio_file_name}, skipping")
                continue

            metadata_record = metadata[audio_file_path.name]
            audio_record = build_audio_record(
                str(audio_file_path),
                metadata_record["scene_label"],
                source_dataset=SOURCE_DATASET,
                metadata_path=str(metadata_path),
            )
            audio_record = {
                **audio_record,
                "annotations": json.dumps(annotations.get(audio_file_path.name, [])),
                "identifier": metadata_record["identifier"],
                "source_label": metadata_record["source_label"],
            }
            records.append(audio_record)

            num_records_built += 1
            pbar.set_description(f"Built {num_records_built} / {num_files} records")

        pbar.close()

        return records

    def build_dataset(self) -> AudioDataset:
        records = self._build_records()

        split_records = split_records_by_key(
            records,
            key_fn=lambda record: record["class_list"][0],
            split_ratios=self.split_ratios,
        )
        for split, rows in split_records.items():
            for record in rows:
                record["split"] = split

        dataset = splits_to_audio_dataset(
            split_records,
            features=MACS_FEATURES,
            label_source=self.label_source,
        )

        return dataset


def main():
    return MACSLoader()()


if __name__ == "__main__":
    print(main().info())
