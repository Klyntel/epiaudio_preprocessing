"""Load the SONYC Urban Sound Tagging (SONYC-UST) dataset into an AudioDataset.

Each 10 s recording carries one multi-label annotation row per annotator in
``annotations.csv``. A clip that has an ``annotator_id == 0`` row uses that row's presence
values as the SONYC team's verified ground truth; otherwise every citizen-annotator row for
that clip is combined by logical OR per fine class (a tag counts as present if at least one
annotator marked it), matching the official DCASE baseline's training-label aggregation.

``download_zenodo`` extracts each ``audio-N.tar.gz`` into its own ``audio-N/`` directory (and
removes the archive), so those per-archive directories are flattened into one ``audio/``
directory here, mirroring the dataset's own ``unpack_audio.sh``.

Run ``python -m audio_preprocessing.dataset.load_sonyc_ust`` from the repo root to download a
small subset (one audio archive) and build the dataset.
"""

from __future__ import annotations

import csv
import re
import shutil
from pathlib import Path

from torchcodec.decoders import AudioDecoder  # pyright: ignore[reportPrivateImportUsage]

from audio_preprocessing.dataset._common import splits_to_audio_dataset
from audio_preprocessing.dataset.base_loader import ZenodoLoader
from audio_preprocessing.datasets import DATA_FEATURES, AudioDataset, LabelSource

RECORD_ID = "3966543"
ANNOTATIONS_FILENAME = "annotations.csv"
SPLIT_MAP = {"train": "train", "validate": "valid", "test": "eval"}
BOROUGH_NAMES = {"1": "Manhattan", "3": "Brooklyn", "4": "Queens"}

_FINE_COLUMN_RE = re.compile(r"^\d+-[0-9X]_(?P<name>.+)_presence$")


def _consolidate_audio_archives(root: Path) -> None:
    """Flatten ``download_zenodo``'s per-archive ``audio-N/`` directories into ``audio/``."""
    root = Path(root)
    audio_dir = root / "audio"
    for extract_dir in sorted(p for p in root.glob("audio-*") if p.is_dir()):
        audio_dir.mkdir(exist_ok=True)
        for wav in extract_dir.rglob("*.wav"):
            wav.replace(audio_dir / wav.name)
        shutil.rmtree(extract_dir)


def _fine_label_columns(fieldnames):
    """Return ``(column, class name)`` pairs for every fine-grained presence column."""
    matches = ((name, _FINE_COLUMN_RE.match(name)) for name in fieldnames)
    return [(name, match.group("name")) for name, match in matches if match]


def _aggregate_annotations(path: Path):
    """Yield ``(audio_filename, split, borough_code, fine_labels)`` per unique clip.

    Rows in ``annotations.csv`` are per-annotator votes; see the module docstring for how they
    are combined into one label set per clip.
    """
    with path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        fine_columns = _fine_label_columns(reader.fieldnames)
        groups: dict[str, dict] = {}
        for row in reader:
            group = groups.setdefault(
                row["audio_filename"],
                {"split": row["split"], "borough": row["borough"], "rows": []},
            )
            group["rows"].append(row)

    for filename, group in groups.items():
        rows = group["rows"]
        ground_truth = next((row for row in rows if row["annotator_id"] == "0"), None)
        source_rows = [ground_truth] if ground_truth is not None else rows
        labels = sorted({
            name
            for column, name in fine_columns
            if any(row[column] == "1" for row in source_rows)
        })
        yield filename, group["split"], group["borough"], labels


def collect(root: Path, *, allow_missing_audio: bool = False):
    """Yield canonical SONYC-UST records built from ``annotations.csv`` and local audio files.
    """
    root = Path(root)
    annotations_path = root / ANNOTATIONS_FILENAME
    if not annotations_path.exists():
        raise FileNotFoundError(f"{ANNOTATIONS_FILENAME} not found under {root}.")

    audio_dir = root / "audio"
    if not audio_dir.is_dir():
        raise FileNotFoundError(
            f"No 'audio' directory found under {root}. Use prepare=True, or prepare=False "
            "with a root where the audio-*.tar.gz archives have already been consolidated "
            "into audio/ (as prepare_raw or the dataset's own unpack_audio.sh would do)."
        )

    skipped = 0
    for filename, split_name, borough_code, labels in _aggregate_annotations(annotations_path):
        split = SPLIT_MAP.get(split_name)
        if split is None:
            raise ValueError(f"Unknown SONYC-UST split {split_name!r} for {filename!r}.")

        audio_path = audio_dir / filename
        if not audio_path.exists():
            if not allow_missing_audio:
                raise FileNotFoundError(
                    f"Annotated audio file {filename!r} not found under {audio_dir}"
                    " The full dataset should not be missing files."
                )
            skipped += 1
            continue

        borough = BOROUGH_NAMES.get(borough_code)
        if borough is None:
            raise ValueError(f"Unknown SONYC-UST borough code {borough_code!r} for {filename!r}.")

        audio = AudioDecoder(str(audio_path)).metadata
        num_channels = audio.num_channels
        if num_channels is None:
            raise ValueError(f"Could not determine the channel count for {audio_path}.")

        yield {
            "audio_path": str(audio_path),
            "sample_rate": audio.sample_rate,
            "clip_offset": 0.0,
            "clip_duration": audio.duration_seconds,
            "class_list": labels,
            "split": split,
            "source_dataset": "SONYC-UST",
            "metadata_path": str(annotations_path),
            "events": [],
            "num_channels": num_channels,
            "channel_format": {1: "mono", 2: "stereo"}.get(num_channels, f"{num_channels}-channel"),
            "environment": borough,
        }

    if skipped:
        print(f"SONYC-UST: skipped {skipped} annotation(s) with no local audio file under {audio_dir}.")


class SONYCUSTLoader(ZenodoLoader):
    """Download and index SONYC-UST using its official train/validate/test splits."""

    record_id = RECORD_ID
    label_source = LabelSource.GOLD

    def prepare_raw(self) -> None:
        super().prepare_raw()
        _consolidate_audio_archives(self.root)

    def build_dataset(self) -> AudioDataset:
        if self.root is None:
            raise ValueError(
                "SONYCUSTLoader needs a root path. Use prepare=False with an existing local "
                "extraction, or prepare=True to download the record."
            )

        records = {split: [] for split in ("train", "valid", "eval")}
        for record in collect(self.root, allow_missing_audio=self.only is not None):
            records[record["split"]].append(record)

        return splits_to_audio_dataset(
            records,
            features=DATA_FEATURES,
            label_source=self.label_source,
        )


def main():
    # Full dataset (~13.3 GB, 19 audio archives): SONYCUSTLoader()
    # Small subset for testing transforms: metadata plus the smallest audio archive (~366 MB).
    return SONYCUSTLoader(only=["audio-18", "annotations"])()


if __name__ == "__main__":
    sonyc_ust = main()
    print(sonyc_ust.info())
