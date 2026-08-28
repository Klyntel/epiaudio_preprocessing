"""Load TAU Urban Acoustic Scenes 2022 Mobile development data.

The release contains 1-second, single-channel acoustic-scene clips and one official
location-disjoint train/test fold. The public test ground truth is mapped to this project's
``valid`` split.

Run ``python -m audio_preprocessing.dataset.load_tau_urban_2022`` from the repository root to
download archive part 16 (about 435 MB), the metadata archive, and build a bounded subset.
"""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

from datasets.features.features import Features, Value  # pyright: ignore[reportMissingImports]
from torchcodec.decoders import AudioDecoder  # pyright: ignore[reportMissingImports, reportPrivateImportUsage]

from audio_preprocessing.dataset._common import (
    find_unique,
    read_tsv_dicts,
    splits_to_audio_dataset,
)
from audio_preprocessing.dataset.base_loader import ZenodoLoader
from audio_preprocessing.datasets import DATA_FEATURES, AudioDataset, LabelSource


RECORD_ID = "6337421"
SOURCE_DATASET = "TAU Urban Acoustic Scenes 2022 Mobile"
RELEASE = "TAU-urban-acoustic-scenes-2022-mobile-development"
META_ARCHIVE = f"{RELEASE}.meta.zip"
AUDIO_ARCHIVE_TEMPLATE = f"{RELEASE}.audio.{{part}}.zip"
VALID_AUDIO_PARTS = tuple(range(1, 17))
DEFAULT_ROOT = Path("data/tau-urban-2022")

TAU_URBAN_2022_FEATURES = Features(
    {
        **DATA_FEATURES,
        "city": Value("string"),
        "location_id": Value("string"),
        "segment_id": Value("string"),
        "subsegment_id": Value("string"),
        "device": Value("string"),
        "device_type": Value("string"),
    }
)


def _normalize_relative_path(value: str) -> str:
    return value.strip().replace("\\", "/").removeprefix("./")


def _filename_metadata(relative: str) -> dict[str, str]:
    path = Path(relative)
    if path.parent.as_posix() != "audio" or path.suffix.lower() != ".wav":
        raise ValueError(f"Unexpected TAU Urban 2022 audio path {relative!r}.")
    parts = path.stem.rsplit("-", 5)
    if len(parts) != 6:
        raise ValueError(f"Unexpected TAU Urban 2022 filename {path.name!r}.")
    scene, city, location, segment, subsegment, device = parts
    if not all(parts):
        raise ValueError(f"Unexpected TAU Urban 2022 filename {path.name!r}.")
    return {
        "scene_label": scene,
        "city": city,
        "location_id": f"{city}-{location}",
        "segment_id": segment,
        "subsegment_id": subsegment,
        "device": device,
        "device_type": "simulated" if device.startswith("s") else "real",
    }


def _split_rows(root: Path) -> tuple[dict[str, dict[str, str]], dict[str, Path]]:
    split_paths = {
        "train": find_unique(root, "fold1_train.csv", "TAU Urban 2022"),
        "valid": find_unique(root, "fold1_evaluate.csv", "TAU Urban 2022"),
    }
    test_path = find_unique(root, "fold1_test.csv", "TAU Urban 2022")

    split_rows: dict[str, dict[str, str]] = {}
    split_members: dict[str, set[str]] = {}
    for split, metadata_path in split_paths.items():
        rows = read_tsv_dicts(metadata_path, ("filename", "scene_label"))
        members = set()
        for row in rows:
            relative = _normalize_relative_path(row["filename"])
            if relative in members:
                raise ValueError(f"Duplicate TAU Urban 2022 {split} path {relative!r}.")
            if relative in split_rows:
                raise ValueError(
                    f"TAU Urban 2022 path {relative!r} appears in multiple splits."
                )
            members.add(relative)
            split_rows[relative] = {
                "split": split,
                "scene_label": row["scene_label"],
            }
        split_members[split] = members

    test_rows = read_tsv_dicts(test_path, ("filename",))
    test_members = {_normalize_relative_path(row["filename"]) for row in test_rows}
    if len(test_members) != len(test_rows):
        raise ValueError(f"Duplicate TAU Urban 2022 test paths in {test_path}.")
    if test_members != split_members["valid"]:
        raise ValueError(
            "TAU Urban 2022 fold1_test.csv and fold1_evaluate.csv contain different paths."
        )
    return split_rows, split_paths


def _attach_source_metadata(
    meta_path: Path,
    split_rows: dict[str, dict[str, str]],
) -> None:
    found = set()
    for row in read_tsv_dicts(
        meta_path, ("filename", "scene_label", "identifier", "source_label")
    ):
        relative = _normalize_relative_path(row["filename"])
        if relative not in split_rows:
            continue
        if relative in found:
            raise ValueError(f"Duplicate TAU Urban 2022 metadata path {relative!r}.")
        found.add(relative)

        parsed = _filename_metadata(relative)
        expected = split_rows[relative]
        if (
            row["scene_label"] != expected["scene_label"]
            or parsed["scene_label"] != row["scene_label"]
        ):
            raise ValueError(
                f"Conflicting TAU Urban 2022 scene labels for {relative!r}."
            )
        if row["identifier"] != parsed["location_id"]:
            raise ValueError(
                f"Conflicting TAU Urban 2022 location metadata for {relative!r}."
            )
        if row["source_label"] != parsed["device"]:
            raise ValueError(
                f"Conflicting TAU Urban 2022 device metadata for {relative!r}."
            )
        expected.update(parsed)

    missing = sorted(set(split_rows) - found)
    if missing:
        raise ValueError(
            f"TAU Urban 2022 meta.csv is missing {len(missing)} official split path(s), "
            f"including {missing[0]!r}."
        )

    locations = {
        split: {
            row["location_id"] for row in split_rows.values() if row["split"] == split
        }
        for split in ("train", "valid")
    }
    overlap = sorted(locations["train"] & locations["valid"])
    if overlap:
        raise ValueError(
            "TAU Urban 2022 official splits share location identifiers, including "
            f"{overlap[0]!r}."
        )


def _available_audio(root: Path) -> dict[str, Path]:
    available = {}
    for path in root.rglob("*.wav"):
        audio_positions = [
            index for index, part in enumerate(path.parts) if part == "audio"
        ]
        if not audio_positions:
            continue
        relative = Path(*path.parts[audio_positions[-1] :]).as_posix()
        if relative in available:
            raise ValueError(f"Multiple TAU Urban 2022 audio files match {relative!r}.")
        available[relative] = path
    if not available:
        raise FileNotFoundError(f"No TAU Urban 2022 WAV files found under {root}.")
    return available


def _build_record(audio_path: Path, metadata_path: Path, spec: dict[str, str]) -> dict:
    audio = AudioDecoder(str(audio_path)).metadata
    if audio.sample_rate is None:
        raise ValueError(f"Could not determine the sample rate for {audio_path}.")
    if audio.duration_seconds is None:
        raise ValueError(f"Could not determine the duration for {audio_path}.")
    if audio.num_channels is None:
        raise ValueError(f"Could not determine the channel count for {audio_path}.")

    return {
        "audio_path": str(audio_path),
        "sample_rate": audio.sample_rate,
        "clip_offset": 0.0,
        "clip_duration": audio.duration_seconds,
        "class_list": [spec["scene_label"]],
        "split": spec["split"],
        "source_dataset": SOURCE_DATASET,
        "metadata_path": str(metadata_path),
        "events": [],
        "num_channels": audio.num_channels,
        "channel_format": "mono"
        if audio.num_channels == 1
        else f"{audio.num_channels}-channel",
        "environment": spec["scene_label"],
        "city": spec["city"],
        "location_id": spec["location_id"],
        "segment_id": spec["segment_id"],
        "subsegment_id": spec["subsegment_id"],
        "device": spec["device"],
        "device_type": spec["device_type"],
    }


class TAUUrban2022Loader(ZenodoLoader):
    """Download selected archive parts and index the official location-disjoint fold."""

    record_id = RECORD_ID
    label_source = LabelSource.GOLD

    def __init__(
        self,
        root: str | Path | None = None,
        *,
        audio_parts: Iterable[int] | None = None,
        prepare: bool = True,
    ) -> None:
        parts = VALID_AUDIO_PARTS if audio_parts is None else tuple(audio_parts)
        if not parts:
            raise ValueError(
                "TAUUrban2022Loader requires at least one audio archive part."
            )
        if len(set(parts)) != len(parts):
            raise ValueError(
                "TAUUrban2022Loader audio_parts must not contain duplicates."
            )
        invalid = sorted(set(parts) - set(VALID_AUDIO_PARTS))
        if invalid:
            raise ValueError(
                f"Unknown TAU Urban 2022 audio part(s) {invalid}; valid parts are 1 through 16."
            )
        self.audio_parts = tuple(sorted(parts))
        only = [
            META_ARCHIVE,
            *(AUDIO_ARCHIVE_TEMPLATE.format(part=part) for part in self.audio_parts),
        ]
        loader_root = root if root is not None or not prepare else DEFAULT_ROOT
        super().__init__(root=loader_root, only=only, prepare=prepare)

    def build_dataset(self) -> AudioDataset:
        if self.root is None:
            raise ValueError(
                "TAUUrban2022Loader needs a root path. Use prepare=False with an existing "
                "extraction, or prepare=True to download metadata and audio archive parts."
            )

        split_rows, split_paths = _split_rows(self.root)
        _attach_source_metadata(
            find_unique(self.root, "meta.csv", "TAU Urban 2022"),
            split_rows,
        )
        records: dict[str, list[dict]] = {"train": [], "valid": []}
        for relative, audio_path in sorted(_available_audio(self.root).items()):
            spec = split_rows.get(relative)
            if spec is None:
                continue
            records[spec["split"]].append(
                _build_record(audio_path, split_paths[spec["split"]], spec)
            )
        if not any(records.values()):
            raise ValueError(
                "Downloaded TAU Urban 2022 audio does not intersect the official fold."
            )

        dataset = splits_to_audio_dataset(
            records,
            features=TAU_URBAN_2022_FEATURES,
            label_source=self.label_source,
        )
        dataset.split = next(
            split for split, rows in dataset.data.items() if rows.num_rows
        )
        return dataset


def main():
    return TAUUrban2022Loader(audio_parts=[16])()


if __name__ == "__main__":
    tau_urban_2022 = main()
    print(tau_urban_2022.info())
