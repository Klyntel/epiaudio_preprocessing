"""Load CochlScene acoustic-scene recordings with the official manual splits.

The Zenodo release is a 53.9 GB split ZIP. ``prepare=True`` downloads and extracts every
segment; for an existing extraction, point ``root`` at any ancestor of the ``CochlScene``
directory and use ``prepare=False``.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

from datasets.features.features import Features, Value  # pyright: ignore[reportMissingImports]

from audio_preprocessing.dataset._common import (
    build_audio_record,
    splits_to_audio_dataset,
)
from audio_preprocessing.dataset.base_loader import ZenodoLoader
from audio_preprocessing.datasets import DATA_FEATURES, AudioDataset, LabelSource


RECORD_ID = "7080122"
SOURCE_DATASET = "CochlScene"
DEFAULT_ROOT = Path("data/cochlscene")
SOURCE_SPLITS = {"Train": "train", "Val": "valid", "Test": "eval"}
SPLIT_DIRECTORIES = {split: source for source, split in SOURCE_SPLITS.items()}
VALID_SPLITS = tuple(SPLIT_DIRECTORIES)
SCENE_CLASSES = (
    "Bus",
    "Cafe",
    "Car",
    "CrowdedIndoor",
    "Elevator",
    "Kitchen",
    "Park",
    "ResidentialArea",
    "Restaurant",
    "Restroom",
    "Street",
    "Subway",
    "SubwayStation",
)

COCHLSCENE_FEATURES = Features(
    {
        **DATA_FEATURES,
        "user_id": Value("string"),
        "submission_id": Value("string"),
        "segment_id": Value("string"),
    }
)


def _release_root(root: Path) -> Path:
    """Find the unique CochlScene directory containing at least one official split."""
    if not root.exists():
        raise FileNotFoundError(f"CochlScene root {root} does not exist.")

    candidates = []
    for candidate in (root, *sorted(root.rglob("CochlScene"))):
        if candidate.is_dir() and any(
            (candidate / source_split).is_dir() for source_split in SOURCE_SPLITS
        ):
            candidates.append(candidate)

    unique = {candidate.resolve(): candidate for candidate in candidates}
    if not unique:
        expected = ", ".join(SOURCE_SPLITS)
        raise FileNotFoundError(
            f"No CochlScene release with an official split directory ({expected}) "
            f"found under {root}."
        )
    if len(unique) > 1:
        raise ValueError(f"Multiple CochlScene releases found under {root}.")
    return next(iter(unique.values()))


def _filename_metadata(path: Path, scene_class: str) -> dict[str, str]:
    parts = path.stem.rsplit("_", 3)
    if len(parts) != 4:
        raise ValueError(f"Unexpected CochlScene filename {path.name!r}.")
    filename_class, user_id, submission_id, segment_id = parts
    if filename_class != scene_class:
        raise ValueError(
            f"CochlScene filename class {filename_class!r} disagrees with directory "
            f"class {scene_class!r} for {path}."
        )
    if not all((user_id, submission_id, segment_id)):
        raise ValueError(f"Unexpected CochlScene filename {path.name!r}.")
    return {
        "user_id": user_id,
        "submission_id": submission_id,
        "segment_id": segment_id,
    }


def _collect_records(
    release_root: Path,
    splits: Sequence[str],
) -> dict[str, list[dict]]:
    records: dict[str, list[dict]] = {split: [] for split in splits}
    users: dict[str, set[str]] = {split: set() for split in splits}
    split_inputs: dict[str, tuple[Path, list[Path]]] = {}

    for split in splits:
        source_split = SPLIT_DIRECTORIES[split]
        split_root = release_root / source_split
        if not split_root.is_dir():
            raise FileNotFoundError(
                f"CochlScene requested split {split!r} directory {split_root} is missing."
            )
        audio_paths = sorted(
            path
            for path in split_root.rglob("*")
            if path.is_file() and path.suffix.lower() == ".wav"
        )
        if not audio_paths:
            raise FileNotFoundError(
                f"No CochlScene WAV files found for requested split {split!r} under "
                f"{split_root}."
            )
        split_inputs[split] = (split_root, audio_paths)

    for split, (split_root, audio_paths) in split_inputs.items():
        for audio_path in audio_paths:
            relative = audio_path.relative_to(split_root)
            if len(relative.parts) != 2:
                raise ValueError(
                    f"Unexpected CochlScene audio path {relative.as_posix()!r}."
                )
            scene_class = relative.parts[0]
            if scene_class not in SCENE_CLASSES:
                raise ValueError(f"Unknown CochlScene scene class {scene_class!r}.")
            metadata = _filename_metadata(audio_path, scene_class)
            record = build_audio_record(
                str(audio_path),
                scene_class,
                split=split,
                source_dataset=SOURCE_DATASET,
                environment=scene_class,
            )
            if record["sample_rate"] != 44100:
                raise ValueError(
                    f"CochlScene audio {audio_path} must have a 44.1 kHz sample rate."
                )
            if record["num_channels"] != 1:
                raise ValueError(
                    f"CochlScene audio {audio_path} must be single-channel."
                )
            record.update(metadata)
            records[split].append(record)
            users[split].add(metadata["user_id"])

    split_names = tuple(records)
    for index, left in enumerate(split_names):
        for right in split_names[index + 1 :]:
            overlap = sorted(users[left] & users[right])
            if overlap:
                raise ValueError(
                    f"CochlScene official splits {left!r} and {right!r} share user "
                    f"{overlap[0]!r}."
                )
    return records


class CochlSceneLoader(ZenodoLoader):
    """Download CochlScene and index selected participant-disjoint official splits.

    All official splits are required by default; ``splits`` explicitly selects a subset.
    """

    record_id = RECORD_ID
    label_source = LabelSource.GOLD

    def __init__(
        self,
        root: str | Path | None = None,
        *,
        splits: Sequence[str] = VALID_SPLITS,
        prepare: bool = True,
    ) -> None:
        self.splits = tuple(splits)
        if not self.splits:
            raise ValueError("CochlSceneLoader requires at least one split.")
        unknown = [split for split in self.splits if split not in SPLIT_DIRECTORIES]
        if unknown:
            raise ValueError(
                f"Unknown CochlScene split(s) {unknown}; valid splits: {list(VALID_SPLITS)}"
            )
        if len(set(self.splits)) != len(self.splits):
            raise ValueError("CochlSceneLoader splits must not contain duplicates.")
        super().__init__(root=root, prepare=prepare)

    def build_dataset(self) -> AudioDataset:
        if self.root is None:
            raise ValueError(
                "CochlSceneLoader needs a root path. Use prepare=False with an existing "
                "extraction, or prepare=True to download the full multipart release."
            )

        dataset = splits_to_audio_dataset(
            _collect_records(_release_root(self.root), self.splits),
            features=COCHLSCENE_FEATURES,
            label_source=self.label_source,
        )
        dataset.split = self.splits[0]
        return dataset


def main():
    return CochlSceneLoader(root=DEFAULT_ROOT, prepare=False)()


if __name__ == "__main__":
    cochlscene = main()
    print(cochlscene.info())
