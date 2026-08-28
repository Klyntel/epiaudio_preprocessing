"""Load VGGSound video clips (audio track only) from the Hugging Face Hub rehost.

~10 s clip as a raw `.mp4` (the original YouTube video, audio track included) 

Use `archives=(...)` to select specific shards (default `("00",)`, the smallest); the full
~338 GB corpus has no smaller official subset. Acquisition is opt-in (`prepare=False` by default):
pass `prepare=True` to download missing selected archives. 

Possible Refactor Note: Use huggingfacehub for download
"""

from __future__ import annotations

import csv
import shutil
import tarfile
import warnings
from collections.abc import Sequence
from pathlib import Path

from datasets.features.features import Features, Value

from audio_preprocessing.dataset._common import (
    build_audio_record,
    splits_to_audio_dataset,
    validate_choices,
)
from audio_preprocessing.dataset.base_loader import BaseLoader
from audio_preprocessing.datasets import DATA_FEATURES, AudioDataset, LabelSource


SOURCE_DATASET = "VGGSound"
HF_REPO_ID = "Loie/VGGSound"
DEFAULT_ROOT = Path("data/vggsound")
CSV_FILENAME = "vggsound.csv"

# The corpus ships as 20 independent ~17 GB archives
# each index in the form of ## from 00 to 19
ARCHIVE_IDS = tuple(f"{index:02d}" for index in range(20))

SOURCE_SPLITS = {"train": "train", "test": "eval"}
VALID_SPLITS = tuple(SOURCE_SPLITS.values())

VGGSOUND_FEATURES = Features(
    {
        **DATA_FEATURES,
        "sample_id": Value("string"),
        "youtube_id": Value("string"),
        "start_seconds": Value("int64"),
        "source_archive": Value("string"),
    }
)


def _index_csv(path: Path) -> dict[str, tuple[str, int, str, str]]:
    """Return ``{stem: (youtube_id, start_seconds, label, source_split)}`` from ``vggsound.csv``."""
    index: dict[str, tuple[str, int, str, str]] = {}
    with path.open(newline="", encoding="utf-8") as handle:
        for line_number, row in enumerate(csv.reader(handle), start=1):
            if len(row) != 4:
                raise ValueError(
                    f"Malformed VGGSound CSV row {line_number} in {path}: "
                    f"expected 4 columns, got {len(row)}."
                )
            youtube_id, start_seconds, label, source_split = (value.strip() for value in row)
            if not youtube_id or not label or source_split not in SOURCE_SPLITS:
                raise ValueError(f"Malformed VGGSound CSV row {line_number} in {path}: {row}.")
            try:
                start_seconds_int = int(start_seconds)
            except ValueError as error:
                raise ValueError(
                    f"Non-integer start_seconds in VGGSound CSV row {line_number} in {path}: "
                    f"{start_seconds!r}."
                ) from error

            stem = f"{youtube_id}_{start_seconds_int:06d}"
            if stem in index:
                raise ValueError(f"Duplicate VGGSound CSV entry for {stem!r} in {path}.")
            index[stem] = (youtube_id, start_seconds_int, label, source_split)

    if not index:
        raise ValueError(f"VGGSound CSV {path} contains no rows.")
    return index


def _source_archive_for(root: Path, path: Path) -> str:
    parts = path.relative_to(root).parts
    return parts[0] if len(parts) > 1 else ""


def _scan_videos(root: Path, archives: Sequence[str]) -> dict[str, tuple[Path, str]]:
    """Scan for ``.mp4`` files, scoped to ``archives`` when a ``vggsound_NN/`` layout is present.

    ``prepare=True`` extracts each archive into its own ``root/vggsound_{archive}/`` directory
    (see ``_prepare_archive``), so a selected archive's videos can be scoped and validated there.
    A hand-untarred root has no such wrapper (the official tarball is flat), so if no
    ``vggsound_NN/`` directory exists anywhere under root, fall back to scanning the whole root;
    in that layout ``archives`` cannot be scoped or validated since files carry no archive marker.
    """
    videos: dict[str, tuple[Path, str]] = {}
    structured = any(path.is_dir() for path in root.glob("vggsound_*"))

    if structured:
        scan_roots = []
        missing = []
        for archive in archives:
            archive_dir = root / f"vggsound_{archive}"
            if archive_dir.is_dir():
                scan_roots.append(archive_dir)
            else:
                missing.append(archive_dir.name)
        if missing:
            raise FileNotFoundError(
                f"VGGSound archive dir(s) not found under {root}: {missing}. "
                "Use prepare=True to download them."
            )
    else:
        warnings.warn(
            f"No vggsound_NN archive directories found under {root}; scanning the whole root as a "
            "flat, hand-extracted layout. The `archives` selection cannot be scoped or validated "
            "in this layout, so every .mp4 under the root is included regardless of `archives`.",
            stacklevel=2,
        )
        scan_roots = [root]

    for scan_root in scan_roots:
        for path in sorted(scan_root.rglob("*.mp4")):
            stem = path.stem
            if stem in videos:
                raise ValueError(f"Duplicate VGGSound video stem {stem!r}: {videos[stem][0]} and {path}.")
            videos[stem] = (path, _source_archive_for(root, path))
    return videos


def _extract_tar_gz(archive: Path, extract_dir: Path) -> None:
    extract_part = extract_dir.with_name(f"{extract_dir.name}.part")
    shutil.rmtree(extract_part, ignore_errors=True)
    try:
        with tarfile.open(archive) as handle:
            handle.extractall(extract_part, filter="data")
        extract_part.replace(extract_dir)
    finally:
        shutil.rmtree(extract_part, ignore_errors=True)


def _prepare_archive(root: Path, archive: str) -> None:
    from huggingface_hub import hf_hub_download

    extract_dir = root / f"vggsound_{archive}"
    marker = extract_dir / ".prepared"
    if marker.is_file():
        return

    local_path = Path(
        hf_hub_download(HF_REPO_ID, f"vggsound_{archive}.tar.gz", repo_type="dataset", local_dir=str(root))
    )
    _extract_tar_gz(local_path, extract_dir)
    marker.touch()
    local_path.unlink(missing_ok=True)


class VGGSoundLoader(BaseLoader):
    """Download and index selected VGGSound archives from the ``Loie/VGGSound`` HF Hub rehost.

    Each archive stores raw per-clip ``.mp4`` files directly
    
    Labels come from an automatic image-classifier + audio-verification pipeline with no per-clip 
    human review (arXiv:2004.14368).
    """

    label_source = LabelSource.SYNTHETIC

    def __init__(
        self,
        root: str | Path | None = None,
        *,
        archives: Sequence[str] = ("00",),
        prepare: bool = False,
    ) -> None:
        super().__init__(root=root, prepare=prepare)
        self.archives = validate_choices(
            archives, name="archive", context="VGGSound", allowed=ARCHIVE_IDS
        )

    def prepare_raw(self) -> None:
        from huggingface_hub import hf_hub_download

        root = self.root or DEFAULT_ROOT
        root.mkdir(parents=True, exist_ok=True)
        self.root = root

        csv_path = root / CSV_FILENAME
        if not csv_path.is_file():
            hf_hub_download(HF_REPO_ID, CSV_FILENAME, repo_type="dataset", local_dir=str(root))

        for archive in self.archives:
            _prepare_archive(root, archive)

    def build_dataset(self) -> AudioDataset:
        if self.root is None:
            raise ValueError(
                "VGGSoundLoader needs a root path. Use prepare=False with an existing "
                "extraction, or prepare=True to download selected archives."
            )

        csv_path = self.root / CSV_FILENAME
        if not csv_path.is_file():
            raise FileNotFoundError(
                f"VGGSound metadata not found at {csv_path}. Use prepare=True to download it."
            )
        index = _index_csv(csv_path)
        videos = _scan_videos(self.root, self.archives)

        missing_metadata = sorted(set(videos) - set(index))
        if missing_metadata:
            raise ValueError(
                f"VGGSound video(s) under {self.root} have no matching row in {csv_path}: "
                f"{missing_metadata[:3]}."
            )

        records: dict[str, list[dict]] = {split: [] for split in VALID_SPLITS}
        for stem, (path, source_archive) in videos.items():
            youtube_id, start_seconds, label, source_split = index[stem]
            split = SOURCE_SPLITS[source_split]
            record = build_audio_record(
                path,
                label,
                split=split,
                source_dataset=SOURCE_DATASET,
                metadata_path=str(csv_path),
            )
            record.update(
                {
                    "sample_id": stem,
                    "youtube_id": youtube_id,
                    "start_seconds": start_seconds,
                    "source_archive": source_archive,
                }
            )
            records[split].append(record)

        if not any(records.values()):
            raise ValueError(f"No VGGSound videos found under {self.root}.")

        return splits_to_audio_dataset(
            records,
            features=VGGSOUND_FEATURES,
            label_source=self.label_source,
        )


def main():
    # Archive "00" is one of the 20 ~17 GB shards which after download and unpack is 33GB on disk.
    return VGGSoundLoader(archives=("00",), prepare=True)()


if __name__ == "__main__":
    vggsound = main()
    print(vggsound.info())
