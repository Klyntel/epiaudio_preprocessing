"""Load the official AISHELL-1 Mandarin speech recognition corpus into an AudioDataset.

AISHELL-1 (OpenSLR SLR33, Apache License 2.0, free for academic use) contains roughly
141,925 utterances read by 400 speakers from a range of Mandarin accent regions, recorded
at 16 kHz / 16-bit mono in a quiet studio; manual transcription accuracy is above 95% per
the corpus release notes, following professional annotation and quality inspection. It
ships as a single ~15 GB archive (``data_aishell.tgz``) whose extraction creates a wrapping
``data_aishell/`` directory (confirmed against the Kaldi and WeNet ``aishell_data_prep.sh``
recipes, both of which read ``data_aishell/wav`` and ``data_aishell/transcript`` after
extracting the archive) containing:

- ``wav/`` -- not ``.wav`` files directly, but a further layer of ``.tar.gz`` archives sitting
  flat inside it (one or more; Kaldi's ``download_and_untar.sh`` and lhotse's
  ``recipes/aishell.py`` both just glob ``wav/*.tar.gz`` rather than assuming a fixed count).
  Each one's internal members already carry the split name as a path prefix (e.g.
  ``train/S0002/<utterance_id>.wav``), so extracting all of them directly into ``wav/``
  reconstructs the expected ``wav/{train,dev,test}/<speaker_id>/<utterance_id>.wav`` tree.
  This second extraction pass is undocumented on the OpenSLR page itself and only discoverable
  from those recipes (or from actually extracting the archive), so ``prepare_raw`` performs
  it as part of preparation rather than leaving it to the caller.
- ``transcript/aishell_transcript_v0.8.txt`` -- one ``<utterance_id> <text>`` line per
  utterance. The delimiter between the two fields is not documented as strictly a single
  space, so parsing splits on the first run of whitespace rather than assuming one.

The transcript file and the audio tree are not perfectly 1:1 in the public release; this is
a known gap the official Kaldi/WeNet data-prep recipes work around by filtering to the
intersection of the two rather than treating it as an error. This loader does the same,
deriving each utterance's speaker from its parent directory (also matching those recipes,
rather than assuming a fixed-width speaker prefix in the filename), and warns with a count
of any skipped utterances per split instead of failing.

There is no per-speaker demographic metadata file for AISHELL-1, unlike its sibling
AISHELL-3: the corpus's separate ``resource_aishell.tgz`` only adds a pronunciation
lexicon, which this loader has no use for and therefore does not download.

Citation: Bu, Du, Na, Wu, Zheng, "AIShell-1: An Open-Source Mandarin Speech Corpus and A
Speech Recognition Baseline," O-COCOSDA 2017 (https://arxiv.org/abs/1709.05522).

Run standalone with an existing local extraction: ``python -m
audio_preprocessing.dataset.load_aishell1``.
"""

from __future__ import annotations

import tarfile
import warnings
from pathlib import Path

import requests
from datasets.features.features import Features, Value
from tqdm import tqdm

from audio_preprocessing.dataset._common import build_audio_record, splits_to_audio_dataset
from audio_preprocessing.dataset.base_loader import BaseLoader
from audio_preprocessing.datasets import AudioDataset, DATA_FEATURES, LabelSource

SOURCE_DATASET = "AISHELL-1"
ARCHIVE_URL = "https://www.openslr.org/resources/33/data_aishell.tgz"
DEFAULT_ROOT = Path("data/aishell1")

# Written by extraction; also the transcript index this loader parses, so it doubles as the
# sanity check that an extraction actually happened.
TRANSCRIPT_RELATIVE_PATH = Path("data_aishell/transcript/aishell_transcript_v0.8.txt")

# Official directory name per split; each maps to this project's split names.
SPLIT_DIRS = {"train": "train", "valid": "dev", "eval": "test"}

AISHELL1_FEATURES = Features({
    **DATA_FEATURES,
    "sample_id": Value("string"),
    "speaker_id": Value("string"),
    "text": Value("string"),
})


def _stream_to_partial(url: str, partial: Path) -> None:
    """Stream ``url`` into ``partial``, resuming an existing prefix when possible."""
    resume = partial.exists()
    while True:
        offset = partial.stat().st_size if resume else 0
        headers = {"Range": f"bytes={offset}-"} if offset else {}
        with requests.get(url, stream=True, headers=headers, timeout=60) as response:
            if offset and response.status_code == 416:
                # The .part is already at least as long as the server file; it can only
                # be stale (a prior run wrote a bad tail), so discard and restart clean.
                partial.unlink()
                resume = False
                continue
            response.raise_for_status()
            resumed = bool(offset and response.status_code == 206)
            start = offset if resumed else 0
            length = int(response.headers.get("content-length", 0))
            total = start + length if length else None
            with (
                partial.open("ab" if resumed else "wb") as handle,
                tqdm(total=total, initial=start, unit="B", unit_scale=True, desc=partial.name) as progress,
            ):
                for chunk in response.iter_content(chunk_size=1 << 20):
                    if chunk:
                        handle.write(chunk)
                        progress.update(len(chunk))
            return


def _prepare(root: Path) -> None:
    """Download and extract ``data_aishell.tgz`` into ``root`` unless already done."""
    marker = root / ".prepared"
    if marker.is_file():
        return

    archive = root / "data_aishell.tgz"
    partial = archive.with_suffix(archive.suffix + ".part")
    if not archive.is_file():
        _stream_to_partial(ARCHIVE_URL, partial)
        partial.replace(archive)

    with tarfile.open(archive) as handle:
        handle.extractall(root, filter="data")

    # data_aishell.tgz only unpacks wav/ into further nested archives (see the module
    # docstring); each extracts in place since its members already carry the split name as
    # a path prefix. Matches the Kaldi/WeNet/lhotse recipes' own cleanup of this corpus.
    wav_dir = root / "data_aishell" / "wav"
    for nested_archive in sorted(wav_dir.glob("*.tar.gz")):
        with tarfile.open(nested_archive) as handle:
            handle.extractall(wav_dir, filter="data")
        nested_archive.unlink()

    if not (root / TRANSCRIPT_RELATIVE_PATH).is_file():
        raise RuntimeError(
            f"{archive.name} did not extract the expected AISHELL-1 layout under {root}."
        )
    marker.write_text("", encoding="utf-8")
    archive.unlink()


def _read_transcripts(path: Path) -> dict[str, str]:
    """Parse ``aishell_transcript_v0.8.txt`` into ``{utterance_id: text}``."""
    transcripts: dict[str, str] = {}
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            line = line.strip()
            if not line:
                continue
            fields = line.split(maxsplit=1)
            if len(fields) != 2:
                raise ValueError(f"Malformed transcript line in {path}:{line_number}.")
            utterance_id, text = fields
            if utterance_id in transcripts:
                raise ValueError(f"Duplicate transcript entry for {utterance_id!r} in {path}.")
            transcripts[utterance_id] = " ".join(text.split())
    return transcripts


def _records(
    corpus_dir: Path,
    split_dir: str,
    split: str,
    transcripts: dict[str, str],
    transcript_path: Path,
):
    wav_root = corpus_dir / "wav" / split_dir
    paths = sorted(wav_root.rglob("*.wav"))
    if not paths:
        raise FileNotFoundError(f"No AISHELL-1 audio found under {wav_root}.")

    skipped = 0
    for audio_path in tqdm(paths, desc=f"AISHELL-1 {split_dir}"):
        utterance_id = audio_path.stem
        text = transcripts.get(utterance_id)
        if text is None:
            skipped += 1
            continue

        record = build_audio_record(
            audio_path,
            None,
            split=split,
            source_dataset=SOURCE_DATASET,
            metadata_path=str(transcript_path),
        )
        if record["sample_rate"] != 16000 or record["num_channels"] != 1:
            raise ValueError(
                f"Expected 16 kHz mono audio for {utterance_id}, got "
                f"{record['sample_rate']} Hz / {record['num_channels']} channel(s)."
            )
        record.update({
            "sample_id": utterance_id,
            "speaker_id": audio_path.parent.name,
            "text": text,
        })
        yield record

    if skipped:
        warnings.warn(
            f"AISHELL-1 {split_dir}: {skipped} audio file(s) under {wav_root} had no "
            f"matching transcript entry in {transcript_path} and were skipped.",
            stacklevel=2,
        )


class AISHELL1Loader(BaseLoader):
    """Index AISHELL-1, optionally downloading it from OpenSLR.

    Transcripts were produced through professional annotation with quality inspection (per
    the corpus release notes), so ``label_source`` is gold. The archive is ~15 GB, so
    preparation is opt-in.
    """

    label_source = LabelSource.GOLD

    def __init__(self, root: str | Path | None = None, *, prepare: bool = False) -> None:
        super().__init__(root=root, prepare=prepare)

    def prepare_raw(self) -> None:
        root = self.root or DEFAULT_ROOT
        root.mkdir(parents=True, exist_ok=True)
        self.root = root
        _prepare(root)

    def build_dataset(self) -> AudioDataset:
        if self.root is None:
            raise ValueError(
                "AISHELL1Loader needs a root path. Use prepare=False with an existing "
                "extraction or prepare=True to download it (~15 GB)."
            )

        transcript_path = self.root / TRANSCRIPT_RELATIVE_PATH
        if not transcript_path.is_file():
            raise FileNotFoundError(
                f"Expected an extracted AISHELL-1 layout ({TRANSCRIPT_RELATIVE_PATH}) under "
                f"{self.root}. Use prepare=True to download and extract it."
            )

        corpus_dir = self.root / "data_aishell"
        nested_archives = sorted((corpus_dir / "wav").glob("*.tar.gz"))
        if nested_archives:
            raise FileNotFoundError(
                f"Found {len(nested_archives)} unextracted archive(s) under "
                f"{corpus_dir / 'wav'} (e.g. {nested_archives[0].name}); data_aishell.tgz "
                "only unpacks wav/ into further nested archives (see the module docstring). "
                "Extract those too, or use prepare=True to have this loader do it."
            )

        transcripts = _read_transcripts(transcript_path)
        records = {
            split: list(_records(corpus_dir, split_dir, split, transcripts, transcript_path))
            for split, split_dir in SPLIT_DIRS.items()
        }
        return splits_to_audio_dataset(
            records,
            features=AISHELL1_FEATURES,
            label_source=self.label_source,
        )


def main():
    # The 15 GB archive has no smaller official subset (see the module docstring), so the
    # standalone smoke test reads an existing local extraction instead of downloading.
    return AISHELL1Loader(prepare=False)()


if __name__ == "__main__":
    dataset = main()
    print(dataset.info())
