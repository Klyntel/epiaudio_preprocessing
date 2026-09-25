"""Load AnomalyMachine-50K from the Hugging Face Hub into a path-backed AudioDataset.

AnomalyMachine-50K is a fully synthetic industrial-machine-sound corpus (CC-BY-4.0), generated
by deterministic signal-processing rules rather than recorded or human-labeled -- treat its
labels as ``SYNTHETIC``, not ``GOLD``. It ships an official, stratified train/val/test split.

The dataset is hosted under an individual Hugging Face account rather than an institutional or
lab source, and its own citation lists the author as "Anonymous". The CC-BY-4.0 license and
non-gated access are independently confirmed, but treat its provenance and quality as less
established than a dataset like FSD50K or TAU Urban 2022.

See ``_materialize_row`` for why this loader ignores the dataset's own ``file_path`` column when
naming local files.

Run ``python -m audio_preprocessing.dataset.load_anomalymachine50k`` for a smoke test against
just the ``test`` split (~3.4 GB). The full dataset (~21.9 GB, all splits) is what
``AnomalyMachine50KLoader()`` downloads by default; pass ``splits=[...]`` to scope it down.
"""

import hashlib
import os
from collections.abc import Iterable
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, cast

from datasets import DatasetDict, load_dataset
from datasets.features.features import Features, Value
from torchcodec.decoders import AudioDecoder
from torchcodec.encoders import AudioEncoder

from audio_preprocessing.dataset._common import splits_to_audio_dataset
from audio_preprocessing.dataset.base_loader import HFLoader
from audio_preprocessing.datasets import DATA_FEATURES, AudioDataset, LabelSource

# Cap materialization threads on high-core machines to avoid overwhelming storage.
_DEFAULT_MATERIALIZE_WORKERS = min(64, (os.cpu_count() or 8))
_SPLITS = {"train": "train", "val": "valid", "test": "eval"}

ANOMALYMACHINE50K_FEATURES = Features({
    **DATA_FEATURES,
    "operating_condition": Value("string"),
    "anomaly_subtype": Value("string"),
    "snr_level": Value("string"),
})


class AnomalyMachine50KLoader(HFLoader):
    dataset_name = "mandipgoswami/AnomalyMachine-50K"
    label_source = LabelSource.SYNTHETIC

    def __init__(
        self,
        *args,
        materialize_workers: int | None = None,
        splits: Iterable[str] | None = None,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.materialize_workers = (
            _DEFAULT_MATERIALIZE_WORKERS
            if materialize_workers is None
            else materialize_workers
        )
        if self.materialize_workers < 1:
            raise ValueError("materialize_workers must be a positive integer.")
        self.splits = None if splits is None else tuple(splits)
        if self.splits is not None:
            unknown = [split for split in self.splits if split not in _SPLITS]
            if unknown:
                raise ValueError(
                    f"Unknown AnomalyMachine-50K split(s) {unknown}; valid: {list(_SPLITS)}"
                )

    def prepare_raw(self) -> None:
        if self.splits is None:
            # Full dataset (~21.9 GB); every official split.
            super().prepare_raw()
            return

        dataset_name = self.dataset_name
        if dataset_name is None:
            raise ValueError("HFLoader requires a dataset_name.")

        # HFLoader.prepare_raw() always fetches every split; downloading only self.splits
        # (e.g. main()'s "test", ~3.4 GB) needs a per-split load_dataset(split=...) call
        # instead, wrapped back into a DatasetDict so build_dataset() sees the same shape.
        kwargs: dict[str, Any] = {"streaming": self.streaming}
        if self.revision is not None:
            kwargs["revision"] = self.revision
        if self.cache_dir is not None:
            kwargs["cache_dir"] = str(self.cache_dir)
        kwargs.update(self.load_kwargs)
        self.raw = DatasetDict({
            split: load_dataset(dataset_name, self.config_name, split=split, **kwargs)
            for split in self.splits
        })

    def _materialize_row(self, row, split, split_dir):
        """Write one HF clip to a local FLAC (idempotent) and return its canonical record.

        ``file_path`` (e.g. ``audio/fan_normal_load_anomalous_bearing_fault_1234.wav``) is
        generation-time metadata, not a real path in this repo, and it is not on its own a
        documented unique key (nothing in it depends on ``snr_level``). The local filename is
        a hash of ``file_path`` plus ``snr_level`` instead -- a stable identifier derived from
        the row's own content rather than its position in the split, so a cached FLAC stays
        correctly paired with its row even if the upstream row order ever changes.
        """
        content_key = hashlib.sha256(f"{row['file_path']}|{row['snr_level']}".encode()).hexdigest()[:16]
        path = split_dir / f"{row['machine_type']}_{row['label']}_{row['anomaly_subtype']}_{content_key}.flac"
        if path.is_file():
            meta = AudioDecoder(str(path)).metadata
            sample_rate = cast(int, meta.sample_rate)
            duration = cast(float, meta.duration_seconds)
            num_channels = cast(int, meta.num_channels)
        else:
            samples = row["audio"].get_all_samples()
            partial = path.with_suffix(".part.flac")
            AudioEncoder(samples.data, sample_rate=samples.sample_rate).to_file(str(partial))
            partial.replace(path)
            sample_rate = samples.sample_rate
            num_channels = samples.data.shape[0]
            duration = samples.data.shape[-1] / sample_rate
        return {
            "audio_path": str(path),
            "sample_rate": sample_rate,
            "clip_offset": 0.0,
            "clip_duration": duration,
            "class_list": [row["label"]],
            "split": split,
            "source_dataset": "AnomalyMachine-50K",
            "metadata_path": "",
            "events": [],
            "num_channels": num_channels,
            "channel_format": {1: "mono", 2: "stereo"}.get(num_channels, f"{num_channels}-channel"),
            "environment": row["machine_type"],
            "operating_condition": row["operating_condition"],
            "anomaly_subtype": row["anomaly_subtype"],
            "snr_level": row["snr_level"],
        }

    def build_dataset(self) -> AudioDataset:
        raw = self.raw_dataset()
        audio_root = Path(self.root or "data/anomalymachine50k") / "audio"
        data = {}
        for source_split, rows in raw.items():
            split = _SPLITS[str(source_split)]
            split_dir = audio_root / str(source_split)
            split_dir.mkdir(parents=True, exist_ok=True)
            with ThreadPoolExecutor(max_workers=self.materialize_workers) as pool:
                records = list(
                    pool.map(
                        lambda row: self._materialize_row(row, split, split_dir),
                        rows,
                    )
                )
            data[split] = records
        return splits_to_audio_dataset(
            data, features=ANOMALYMACHINE50K_FEATURES, label_source=self.label_source
        )


def main():
    # Full dataset is ~21.9 GB; smoke-test against just the smallest split (~3.4 GB).
    return AnomalyMachine50KLoader(splits=["test"])()


if __name__ == "__main__":
    anomalymachine50k = main()
    print(anomalymachine50k.info())
