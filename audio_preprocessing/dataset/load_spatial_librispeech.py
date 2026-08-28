"""Load Spatial LibriSpeech (Apple, CC BY 4.0) into an AudioDataset.

Spatial LibriSpeech places LibriSpeech utterances in simulated rooms. Each clip has one speaker at a
fixed position, recorded as 4-channel first-order ambisonics (ACN order, 16 kHz). A single
metadata.parquet holds every clip's split, transcription, source position, and room/noise details;
the ambisonics FLAC for each clip is downloaded by id from Apple's server.

The full set is about 323 GB, so a build downloads only the clips it needs. Use ``split``, ``lite``,
or ``limit`` to keep a build small.

Run ``python -m audio_preprocessing.dataset.load_spatial_librispeech`` from the repo root to build a
small example.
"""

from __future__ import annotations

import itertools
import math
import random
import time
import urllib.error
import urllib.request
from collections.abc import Callable, Iterable, Iterator
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import TypeVar

import pandas as pd
from datasets.features.features import Features, Value  # pyright: ignore[reportMissingImports]

from audio_preprocessing.dataset._common import build_audio_record, splits_to_audio_dataset
from audio_preprocessing.dataset.base_loader import BaseLoader
from audio_preprocessing.datasets import (
    DATA_FEATURES,
    AudioDataset,
    ContinuousData,
    ContinuousEvent,
    LabelSource,
)

SLS = "https://docs-assets.developer.apple.com/ml-research/datasets/spatial-librispeech/v1"
SOURCE_DATASET = "spatial_librispeech"
DEFAULT_ROOT = Path("data/spatial_librispeech")
METADATA = "metadata.parquet"
SPLITS = ("train", "test")
CHANNEL_FORMAT = "foa"
SAMPLE_RATE = 16000
NUM_CHANNELS = 4
_META = "speech/librispeech_metadata"
_TRANSCRIPTION = f"{_META}/transcription"

_T = TypeVar("_T")
_R = TypeVar("_R")

# Extra fields to keep, as ``{row_column: (parquet_column, converter)}``. These are not schema
# columns, so they land in the MDS metadata field; the source direction is stored separately, in
# events[].frame_array. The converter turns each parquet value into a plain Python type (which keeps
# the streaming publisher's JSON happy) and turns Apple's radians into the degrees this repo uses.
_PROVENANCE = {
    "speaking_azimuth_deg": ("speech/speaking_azimuth", math.degrees),
    "speaking_elevation_deg": ("speech/speaking_elevation", math.degrees),
    "room_id": ("room/room_id", int),
    "room_volume": ("room/volume", float),
    "noise_snr_db": ("noise/snr", float),
    "librispeech_subset": (f"{_META}/subset", str),
    "reader_sex": (f"{_META}/reader_sex", str),
}

SLS_FEATURES = Features({
    **DATA_FEATURES,
    "sample_id": Value("string"),
    "text": Value("string"),
    "speaking_azimuth_deg": Value("float64"),
    "speaking_elevation_deg": Value("float64"),
    "room_id": Value("int64"),
    "room_volume": Value("float64"),
    "noise_snr_db": Value("float64"),
    "librispeech_subset": Value("string"),
    "reader_sex": Value("string"),
})

_READ_COLUMNS = [
    "sample_id", "split", "lite_version",
    "speech/azimuth", "speech/elevation", "speech/distance", _TRANSCRIPTION,
    *(column for column, _ in _PROVENANCE.values()),
]


def _clip_url(sample_id: int) -> str:
    return f"{SLS}/ambisonics/{sample_id:06d}.flac"


def _download_clip(sample_id: int, retries: int = 8) -> bytes:
    """Download one ambisonics FLAC, retrying transient failures with exponential backoff.

    Apple's server returns intermittent 5xx under sustained load, so a full ~221k-clip crawl will hit
    some; we back off (0.5s, 1s, 2s, ... capped at 30s, with jitter) rather than retry in a tight loop
    that would just re-hit the same blip. A 4xx is permanent (a genuinely missing id) and is not
    retried.
    """
    url = _clip_url(sample_id)
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(url, timeout=90) as response:
                return response.read()
        except urllib.error.HTTPError as error:
            if error.code < 500 or attempt == retries - 1:
                raise
        except (urllib.error.URLError, TimeoutError):
            if attempt == retries - 1:
                raise
        time.sleep(min(30.0, 0.5 * 2**attempt) + random.uniform(0, 0.5))
    raise RuntimeError("unreachable")


def _fetch_clip(sample_id: int, dest: Path, retries: int = 3) -> None:
    """Download one ambisonics FLAC to ``dest``, skipping it if it is already there.

    The write is atomic (via a ``.part`` file) so an interrupted run leaves no half-written clip.
    """
    if dest.is_file():
        return
    tmp = dest.with_suffix(".part")
    tmp.write_bytes(_download_clip(sample_id, retries))
    tmp.replace(dest)


def _bounded_map(
    fn: Callable[[_T], _R], items: Iterable[_T], workers: int
) -> Iterator[tuple[_T, _R]]:
    """Run ``fn`` over ``items`` with at most ``workers`` calls in flight, yielding ``(item,
    result)`` pairs as each finishes.

    A full build can have hundreds of thousands of clips. Submitting them all at once would queue
    that many tasks (and, for the publisher, hold every downloaded clip in memory). Instead we prime
    a fixed window and refill it as tasks complete, so memory stays flat regardless of the input
    size. Pairs come back in completion order, not input order.
    """
    pending = iter(items)
    with ThreadPoolExecutor(max_workers=workers) as pool:
        in_flight = {pool.submit(fn, item): item for item in itertools.islice(pending, workers)}
        while in_flight:
            future = next(as_completed(in_flight))
            item = in_flight.pop(future)
            yield item, future.result()
            nxt = next(pending, None)
            if nxt is not None:
                in_flight[pool.submit(fn, nxt)] = nxt


def _select_clips(frame: pd.DataFrame, *, lite: bool, split: str | None, limit: int | None) -> pd.DataFrame:
    """Apply the ``lite``/``split``/``limit`` filters shared by the loader and streaming publisher.

    The lite subset spans both splits, so any ``split``/``lite`` combination still has rows; the only
    way to select nothing is ``limit <= 0``, which is rejected here.
    """
    if limit is not None and limit < 1:
        raise ValueError(f"limit must be a positive integer, got {limit}.")
    if lite:
        frame = frame.loc[frame["lite_version"]]
    if split is not None:
        frame = frame.loc[frame["split"] == split]
    if limit is not None:
        frame = frame.groupby("split", group_keys=False).head(limit)
    return frame.reset_index(drop=True)


def _direction_event(azimuth_rad: float, elevation_rad: float, distance_m: float, duration: float) -> dict:
    """The speaker's fixed direction for one clip, as a ``ContinuousEvent``.

    Spatial datasets store direction in events[].frame_array, so this matches them (see load_bat). A
    ``ContinuousEvent`` is a time series with a whole-number frame rate, and the speaker does not
    move, so we sample once per second and repeat the one position across the clip. Apple's angles
    use the same frame as this repo (counterclockwise, 0 is front, +90 is left) -- checked against
    the ambisonics intensity vector -- so we only turn radians into degrees.

    The position is constant, so a plain ``Event`` with the direction in additional_metadata would
    carry the same information. We use a ``ContinuousEvent`` because it is what the probe sweeps read:
    the probe source windows a clip and takes each window's label from the frames in events[].
    frame_array, so it needs a frame per window, and this keeps the schema identical to the datasets
    whose sources really do move (e.g. STARSS23), which those sweeps also read.
    """
    frames_per_sec = 1
    frame_count = max(1, math.ceil(duration * frames_per_sec - 1e-9))
    frame = ContinuousData(
        azimuth=math.degrees(azimuth_rad),
        elevation=math.degrees(elevation_rad),
        distance=distance_m,
    ).to_dict()
    return ContinuousEvent.build_start_end(
        start=0.0,
        end=duration,
        label="speech",
        frame_array=[frame] * frame_count,
        frames_per_sec=frames_per_sec,
    ).to_dict()


class SpatialLibriSpeechLoader(BaseLoader):
    """Prepare and build Spatial LibriSpeech from Apple's ambisonics FLAC and metadata parquet.

    The full dataset is large, so a build downloads only the clips it needs. ``split`` limits to one
    of ``train``/``test``, ``lite`` restricts to Apple's flagged lite subset, and ``limit`` caps the
    number of clips per split for small builds.
    """

    label_source = LabelSource.SYNTHETIC

    def __init__(
        self,
        root: str | Path | None = None,
        *,
        split: str | None = None,
        lite: bool = False,
        limit: int | None = None,
        download_workers: int = 16,
        prepare: bool = True,
    ) -> None:
        if split is not None and split not in SPLITS:
            raise ValueError(f"Unknown split {split!r}. Choose from: {', '.join(SPLITS)}")
        super().__init__(root=root, prepare=prepare)
        self.split = split
        self.lite = lite
        self.limit = limit
        self.download_workers = download_workers
        self._metadata: pd.DataFrame | None = None

    def _parquet_path(self) -> Path:
        return (self.root or DEFAULT_ROOT) / METADATA

    def _selected(self) -> pd.DataFrame:
        """Read the metadata parquet once and apply the ``lite``/``split``/``limit`` filters."""
        if self._metadata is None:
            frame = pd.read_parquet(self._parquet_path(), columns=_READ_COLUMNS)
            self._metadata = _select_clips(frame, lite=self.lite, split=self.split, limit=self.limit)
        return self._metadata

    def prepare_raw(self) -> None:
        root = Path(self.root) if self.root is not None else DEFAULT_ROOT
        (root / "ambisonics").mkdir(parents=True, exist_ok=True)
        self.root = root

        if not self._parquet_path().is_file():
            print(f"downloading {METADATA} ...")
            tmp = self._parquet_path().with_suffix(".part")
            urllib.request.urlretrieve(f"{SLS}/{METADATA}", tmp)
            tmp.replace(self._parquet_path())

        ids = [int(sample_id) for sample_id in self._selected()["sample_id"]]
        missing = [i for i in ids if not (root / "ambisonics" / f"{i:06d}.flac").is_file()]
        print(f"fetching {len(missing)} clips ({len(ids) - len(missing)} already present) ...")
        ambisonics = root / "ambisonics"
        done = 0
        for _sample_id, _ in _bounded_map(
            lambda i: _fetch_clip(i, ambisonics / f"{i:06d}.flac"), missing, self.download_workers
        ):
            done += 1
            if done % 500 == 0:
                print(f"  {done}/{len(missing)}")

    def build_dataset(self) -> AudioDataset:
        if self.root is None:
            raise ValueError(
                "SpatialLibriSpeechLoader needs a root path. Use prepare=False with an existing local root."
            )
        parquet_path = self._parquet_path()
        ambisonics_dir = Path(self.root) / "ambisonics"

        records: dict[str, list[dict]] = {}
        for fields in self._selected().to_dict("records"):
            record = self._build_record(fields, parquet_path, ambisonics_dir)
            records.setdefault(record["split"], []).append(record)

        dataset = splits_to_audio_dataset(records, features=SLS_FEATURES, label_source=self.label_source)
        # A build can hold just one split. Point the active split at whichever split is present, so
        # reading it does not fail when "train" is absent.
        if "train" not in records:
            dataset.split = next(iter(records))
        return dataset

    def _build_record(self, fields: dict, parquet_path: Path, ambisonics_dir: Path) -> dict:
        sample_id = int(fields["sample_id"])
        record = build_audio_record(
            str(ambisonics_dir / f"{sample_id:06d}.flac"),
            "speech",
            split=str(fields["split"]),
            source_dataset=SOURCE_DATASET,
            metadata_path=str(parquet_path),
            channel_format=CHANNEL_FORMAT,
            environment="",
        )
        # build_audio_record probes the file, so the channel count and rate here are the clip's real
        # shape. Check them before trusting the foa label: a corrupt or wrong local file would
        # otherwise be published as foa while being, say, mono.
        if record["num_channels"] != NUM_CHANNELS or record["sample_rate"] != SAMPLE_RATE:
            raise ValueError(
                f"Spatial LibriSpeech clip {sample_id:06d} is {record['num_channels']}-channel at "
                f"{record['sample_rate']} Hz, expected {NUM_CHANNELS}-channel FOA at {SAMPLE_RATE} Hz. "
                f"The local file may be truncated or wrong: {record['audio_path']}"
            )
        record["sample_id"] = str(sample_id)
        record["text"] = str(fields[_TRANSCRIPTION])
        record["events"] = [
            _direction_event(
                fields["speech/azimuth"],
                fields["speech/elevation"],
                fields["speech/distance"],
                record["clip_duration"],
            )
        ]
        for name, (column, convert) in _PROVENANCE.items():
            record[name] = convert(fields[column])
        return record


def main():
    # Example run: a few clips per split.
    return SpatialLibriSpeechLoader(limit=5)()


if __name__ == "__main__":
    dataset = main()
    print(dataset.info())
