import warnings
from typing import Any
import json
from collections.abc import Iterable
from pathlib import Path
import requests
from tqdm import tqdm
import pandas as pd
from datasets.features.features import Features, Value
from audio_preprocessing.dataset._common import (
    find_unique,
    validate_choices,
    build_audio_record,
    splits_to_audio_dataset,
    split_records_by_key
)
from audio_preprocessing.dataset.base_loader import ZenodoLoader
from audio_preprocessing.datasets import DATA_FEATURES, AudioDataset, LabelSource


RECORD_ID = "3351307"
DO_NOT_DOWNLOAD = ["LICENSE.pdf"]
ANOMALY_CONDITIONS_BASE_URL = (
    "https://raw.githubusercontent.com/YumaKoizumi/ToyADMOS-dataset/master/anomaly_conditions"
)
ANOMALY_CONDITION_FILE_NAMES = {
    "ToyCar": "ToyCar_anomay_condition.xlsx",
    "ToyConveyor": "ToyConveyor_anomay_condition.xlsx",
    "ToyTrain": "ToyTrain_anomay_condition.xlsx",
}

SOURCE_DATASET = "ToyADMOS_v1"
DEFAULT_ROOT = Path("data/toyadmos")
SUBSETS = ("ToyCar", "ToyConveyor", "ToyTrain")
LABELS = {
    "EnvironmentalNoise": "noise",
    "NormalSound": "normal",
    "AnomalousSound": "anomalous"
}
TOYADMOS_FEATURES = Features({
    **DATA_FEATURES,
    "subset": Value("string"),
    "case": Value("string"),
    "metadata": Value("string")
})
DEFAULT_SPLIT_RATIOS = (.8, .1)


def _download_file(url: str, destination: Path) -> None:
    if destination.is_file():
        return
    partial = destination.with_suffix(destination.suffix + ".part")
    try:
        with requests.get(url, stream=True, timeout=60) as response:
            response.raise_for_status()
            with partial.open("wb") as handle:
                for chunk in response.iter_content(chunk_size=1 << 20):
                    if chunk:
                        handle.write(chunk)
        partial.replace(destination)
    except Exception:
        partial.unlink(missing_ok=True)
        raise


def _df_to_dict(df: pd.DataFrame, key: str) -> dict[str, dict[str, Any]]:
    records = dict()
    for record in df.to_dict("records"):
        records[record[key]] = record

    return records


class ToyADMOSLoader(ZenodoLoader):
    """Download and load the ToyADMOS dataset."""
    record_id = RECORD_ID
    label_source = LabelSource.GOLD

    def __init__(
        self,
        root: str | Path | None = None,
        *,
        subsets: Iterable[str] = SUBSETS,
        split_ratios: tuple[float, float] = DEFAULT_SPLIT_RATIOS,
        prepare: bool = True
    ) -> None:
        self.subsets = list(validate_choices(
            subsets,
            name="subset",
            context=SOURCE_DATASET,
            allowed=SUBSETS
        ))
        self.split_ratios = split_ratios
        super().__init__(
            root=root,
            do_not_download=DO_NOT_DOWNLOAD,
            only=self.subsets,
            prepare=prepare
        )
        self.root = self.root or DEFAULT_ROOT

    def _download_anomaly_conditions(self) -> Path:
        """Download the per-subset anomaly-condition workbooks published alongside the dataset.

        These ``.xlsx`` files (https://github.com/YumaKoizumi/ToyADMOS-dataset/tree/master/anomaly_conditions)
        document which recorded sounds count as anomalous per toy/machine and are not part of
        the Zenodo archive, so they're fetched separately from the maintainer's GitHub repo.
        """
        anomaly_conditions_root = self.root / "anomaly_conditions"
        anomaly_conditions_root.mkdir(parents=True, exist_ok=True)
        for subset in self.subsets:
            filename = ANOMALY_CONDITION_FILE_NAMES[subset]
            _download_file(f"{ANOMALY_CONDITIONS_BASE_URL}/{filename}", anomaly_conditions_root / filename)
        return anomaly_conditions_root

    def _get_metadata(self) -> dict[str, Any]:
        metadata = dict()
        for subset in self.subsets:
            metadata_file_name = f"{subset}_anomay_condition.xlsx"
            metadata_path = find_unique(self.root, metadata_file_name, SOURCE_DATASET)
            metadata_df = pd.read_excel(metadata_path)
            metadata[subset] = _df_to_dict(metadata_df, "Name")
            metadata[subset] |= {"path": str(metadata_path)}

        return metadata

    def _build_records(self) -> list[dict[str, Any]]:
        metadata = self._get_metadata()

        audio_file_paths = []
        for subset in self.subsets:
            paths = (self.root / subset).rglob("*.wav")
            audio_file_paths.extend(paths)

        num_audio_files = len(audio_file_paths)
        records_built = 0
        records = []
        pbar = tqdm(audio_file_paths)

        for audio_file_path in audio_file_paths:
            strings = audio_file_path.stem.split("_")
            subset = strings[1] if strings[1] != "train" else "ToyTrain"
            case = strings[2][4:]
            name = strings[3]
            label = LABELS[audio_file_path.parent.name.split("_")[0]]
            if label == "anomalous":
                metadata_path = metadata[subset]["path"]
                file_metadata = metadata[subset][name]
            else:
                metadata_path = ""
                file_metadata = dict()
            try:
                record = build_audio_record(
                    audio_file_path,
                    label,
                    source_dataset=SOURCE_DATASET,
                    metadata_path=metadata_path
                )
            except Exception as e:
                msg = f"Error in processing {audio_file_path}: {e}, skipping."
                warnings.warn(msg, stacklevel=2)
                continue
            record |= {
                "subset": subset,
                "case": case,
                "metadata": json.dumps(file_metadata)
            }

            records.append(record)
            records_built += 1
            pbar.set_description(f"Built {records_built} / {num_audio_files} records.")

        pbar.close()

        return records

    def prepare_raw(self) -> None:
        super().prepare_raw()
        self._download_anomaly_conditions()

    def build_dataset(self) -> AudioDataset:
        records = self._build_records()

        split_records = split_records_by_key(
            records,
            key_fn=lambda record: (record["class_list"][0], record["subset"]),
            split_ratios=self.split_ratios
        )
        for split, rows in split_records.items():
            for record in rows:
                record["split"] = split

        dataset = splits_to_audio_dataset(
            split_records,
            features=TOYADMOS_FEATURES,
            label_source=self.label_source,
        )

        return dataset


def main():
    return ToyADMOSLoader()()


if __name__ == "__main__":
    print(main().info())
