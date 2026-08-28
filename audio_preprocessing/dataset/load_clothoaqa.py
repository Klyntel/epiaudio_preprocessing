"""Load Clotho-AQA audio question-answer annotations from Zenodo into an AudioDataset."""

from __future__ import annotations

import csv
import io
from collections.abc import Sequence
from pathlib import Path

from datasets.features.features import Features, List, Value

from audio_preprocessing.dataset._common import (
    audio_by_name,
    build_audio_record,
    find_unique,
    splits_to_audio_dataset,
)
from audio_preprocessing.dataset.base_loader import ZenodoLoader
from audio_preprocessing.datasets import DATA_FEATURES, AudioDataset, LabelSource


RECORD_ID = "6473207"
SOURCE_DATASET = "ClothoAQA"
SOURCE_SPLITS = {
    "train": "train",
    "val": "valid",
    "test": "eval",
}
SPLIT_SOURCES = {split: source for source, split in SOURCE_SPLITS.items()}
VALID_SPLITS = tuple(SPLIT_SOURCES)
QA_COLUMNS = ("file_name", "QuestionText", "answer", "confidence")
# The source metadata CSV carries an unlabeled leading index column (a pandas artifact).
METADATA_COLUMNS = (
    "",
    "file_name",
    "keywords",
    "sound_link",
    "start_end_samples",
    "manufacturer",
    "license",
)
# Two clips are misspelled upstream, in opposite directions: clotho_aqa_metadata.csv writes
# "18_44" where audio_files.zip and the QA CSVs write "18.44", and audio_files.zip writes
# "souffle_me.tallique.wav" where both CSVs write "souffle_me_tallique.wav". Both are in train,
# which cannot load until each shipped spelling maps onto the QA spelling that keys the records.
UPSTREAM_FILENAME_TYPOS = {
    "Geese1 - (Apollonia&#39;s sPA) 18_44 05.10.wav": "Geese1 - (Apollonia&#39;s sPA) 18.44 05.10.wav",
    "souffle_me.tallique.wav": "souffle_me_tallique.wav",
}

CLOTHOAQA_FEATURES = Features(
    {
        **DATA_FEATURES,
        "sample_id": Value("string"),
        "question": Value("string"),
        "answers": List(Value("string")),
        "confidences": List(Value("string")),
        "keywords": Value("string"),
        "sound_link": Value("string"),
        "start_end_samples": Value("string"),
        "manufacturer": Value("string"),
        "license": Value("string"),
        "qa_path": Value("string"),
    }
)


def _read_csv(path: Path, expected_columns: tuple[str, ...]) -> list[dict[str, str]]:
    try:
        text = path.read_text(encoding="utf-8-sig")
    except UnicodeDecodeError:
        text = path.read_text(encoding="latin-1")

    reader = csv.DictReader(io.StringIO(text))
    if tuple(reader.fieldnames or ()) != expected_columns:
        raise ValueError(
            f"Unexpected Clotho-AQA columns in {path}: expected {list(expected_columns)}, "
            f"got {reader.fieldnames}."
        )

    rows = []
    for line_number, row in enumerate(reader, start=2):
        if None in row or any(row[column] is None for column in expected_columns):
            raise ValueError(f"Malformed Clotho-AQA CSV row {line_number} in {path}.")
        row = {
            column: row[column] if column in ("", "file_name") else row[column].strip()
            for column in expected_columns
        }
        rows.append(row)

    if not rows:
        raise ValueError(f"Clotho-AQA CSV {path} contains no rows.")
    return rows


def _canonical_filename(filename: str) -> str:
    """Return the QA CSV spelling of an upstream-misspelled clip name."""
    return UPSTREAM_FILENAME_TYPOS.get(filename, filename)


def _index_metadata(path: Path) -> dict[str, dict[str, str]]:
    indexed: dict[str, dict[str, str]] = {}
    for line_number, row in enumerate(_read_csv(path, METADATA_COLUMNS), start=2):
        filename = row["file_name"]
        if not filename:
            raise ValueError(f"Clotho-AQA metadata {path} line {line_number} has an empty file_name.")
        canonical = _canonical_filename(filename)
        if canonical in indexed:
            raise ValueError(f"Duplicate Clotho-AQA metadata entry for {filename!r} in {path}.")
        indexed[canonical] = row
    return indexed


def _index_audio(root: Path) -> dict[str, Path]:
    indexed: dict[str, Path] = {}
    for filename, path in audio_by_name(root, SOURCE_DATASET).items():
        canonical = _canonical_filename(filename)
        if canonical in indexed:
            raise ValueError(
                f"Clotho-AQA audio files {indexed[canonical].name!r} and {filename!r} under "
                f"{root} both index as {canonical!r}."
            )
        indexed[canonical] = path
    return indexed


def _group_qa_rows(path: Path) -> dict[tuple[str, str], dict[str, list[str]]]:
    """Group per-annotator QA rows by (file_name, question), preserving first-seen order.

    Every group holds three annotator answers except one in train, where a clip was asked the
    same question text twice and its six rows merge into a single group.
    """
    groups: dict[tuple[str, str], dict[str, list[str]]] = {}
    for row in _read_csv(path, QA_COLUMNS):
        key = (row["file_name"], row["QuestionText"])
        group = groups.setdefault(key, {"answers": [], "confidences": []})
        group["answers"].append(row["answer"])
        group["confidences"].append(row["confidence"])
    return groups


def _collect_split(
    root: Path,
    source_split: str,
    split: str,
    audio: dict[str, Path],
    metadata_by_name: dict[str, dict[str, str]],
    metadata_path: Path,
) -> list[dict]:
    qa_path = find_unique(root, f"clotho_aqa_{source_split}.csv", SOURCE_DATASET)
    groups = _group_qa_rows(qa_path)

    filenames = {filename for filename, _ in groups}
    missing_audio = sorted(filenames - audio.keys())
    if missing_audio:
        raise ValueError(f"Clotho-AQA {source_split} audio missing for files: {missing_audio[:3]}.")
    missing_metadata = sorted(filenames - metadata_by_name.keys())
    if missing_metadata:
        raise ValueError(
            f"Clotho-AQA {source_split} metadata missing for files: {missing_metadata[:3]}."
        )

    records = []
    question_index: dict[str, int] = {}
    for (filename, question), answer_group in groups.items():
        record = build_audio_record(
            audio[filename],
            None,
            split=split,
            source_dataset=SOURCE_DATASET,
            metadata_path=str(metadata_path),
            environment="",
        )
        if record["sample_rate"] != 44100 or record["num_channels"] != 1:
            raise ValueError(
                f"Clotho-AQA audio {audio[filename]} must be mono 44.1 kHz; found "
                f"{record['num_channels']} channel(s) at {record['sample_rate']} Hz."
            )

        index = question_index.get(filename, 0)
        question_index[filename] = index + 1
        metadata = metadata_by_name[filename]
        record.update(
            {
                "sample_id": f"{Path(filename).stem}:{index}",
                "question": question,
                "answers": answer_group["answers"],
                "confidences": answer_group["confidences"],
                "keywords": metadata["keywords"],
                "sound_link": metadata["sound_link"],
                "start_end_samples": metadata["start_end_samples"],
                "manufacturer": metadata["manufacturer"],
                "license": metadata["license"],
                "qa_path": str(qa_path),
            }
        )
        records.append(record)
    return records


class ClothoAQALoader(ZenodoLoader):
    """Download and index selected official splits of Clotho-AQA.

    Clotho-AQA reuses Clotho's audio clips but ships them as one combined
    ``audio_files.zip`` rather than per-split archives, so every split selection downloads
    the full ~3.1 GB archive; only the (small) question/answer CSVs are narrowed by ``splits``.
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
            raise ValueError("ClothoAQALoader requires at least one split.")
        unknown = [split for split in self.splits if split not in SPLIT_SOURCES]
        if unknown:
            raise ValueError(
                f"Unknown Clotho-AQA split(s) {unknown}; valid splits: {list(VALID_SPLITS)}."
            )
        if len(set(self.splits)) != len(self.splits):
            raise ValueError("ClothoAQALoader splits must not contain duplicates.")

        only = ["LICENSE", "audio_files", "clotho_aqa_metadata"]
        only.extend(f"clotho_aqa_{SPLIT_SOURCES[split]}" for split in self.splits)
        super().__init__(root=root, only=only, prepare=prepare)

    def build_dataset(self) -> AudioDataset:
        if self.root is None:
            raise ValueError(
                "ClothoAQALoader needs a root path. Use prepare=False with an existing local "
                "extraction, or prepare=True to download selected splits."
            )

        metadata_path = find_unique(self.root, "clotho_aqa_metadata.csv", SOURCE_DATASET)
        metadata_by_name = _index_metadata(metadata_path)
        audio = _index_audio(self.root)

        records = {
            split: _collect_split(
                self.root, SPLIT_SOURCES[split], split, audio, metadata_by_name, metadata_path
            )
            for split in self.splits
        }
        dataset = splits_to_audio_dataset(
            records,
            features=CLOTHOAQA_FEATURES,
            label_source=self.label_source,
        )
        dataset.split = self.splits[0]
        return dataset


def main():
    # The val CSV is the smallest split file, but every split still needs the one combined
    # ~3.1 GB audio_files.zip.
    return ClothoAQALoader(splits=("valid",), prepare=True)()


if __name__ == "__main__":
    clothoaqa = main()
    print(clothoaqa.info())
