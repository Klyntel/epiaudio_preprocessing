# Original work: https://github.com/Klyntel/EpiAudio/blob/656c2b35965a44bf25309aea2d29f86249ccc4c3/epiaudio/dataset/load_avspeech.py
from datasets import Dataset, DatasetDict, Audio, load_dataset

from audio_preprocessing.datasets import AudioDataset

DATASET_NAME = "ProgramComputer/avspeech-visual-audio"
DEFAULT_COLUMNS = ["clip_id", "avspeech_metadata", "audio"]
DEFAULT_SPLITS = ["train", "test"]
# AVSpeech ships "train"/"test" on the Hub; expose them under our train/valid/eval convention.
SPLIT_NAMES = {"train": "train", "test": "eval"}


def _split_limit(num_rows: dict[str, int] | int, split: str) -> int:
    limit = num_rows if isinstance(num_rows, int) else num_rows[split]
    if limit <= 0:
        raise ValueError(
            "AVSpeech is materialized into memory; pass a positive row limit "
            "for each requested split."
        )
    return limit


def _materialize_avspeech_split(split: str, *, columns: list[str] | None = None, num_rows: int) -> Dataset:
    """Stream a bounded number of AVSpeech rows and materialize them as a Dataset."""
    columns = columns or DEFAULT_COLUMNS
    dataset = load_dataset(DATASET_NAME, split=split, streaming=True).select_columns(columns)
    records = list(dataset.take(num_rows))
    ds = Dataset.from_list(records)
    if "audio" in ds.column_names:
        ds = ds.cast_column("audio", Audio())
    return ds


def load_avspeech(
    splits: list[str] | None = None,
    columns: list[str] | None = None,
    num_rows: dict[str, int] | int | None = None,
) -> AudioDataset:
    """Return a bounded, materialized AVSpeech sample as an AudioDataset.

    AVSpeech is very large, so callers must pass ``num_rows`` explicitly. Pass one
    integer to use the same limit for every split, or a dict keyed by raw Hub split
    names (for example ``{"train": 100, "test": 100}``).
    """
    if num_rows is None:
        raise ValueError(
            "AVSpeech is too large to materialize by default; pass num_rows as "
            "an int or as a dict keyed by split."
        )

    splits = splits or DEFAULT_SPLITS
    columns = columns or DEFAULT_COLUMNS

    data = DatasetDict({
        SPLIT_NAMES.get(split, split): _materialize_avspeech_split(
            split,
            columns=columns,
            num_rows=_split_limit(num_rows, split),
        )
        for split in splits
    })
    return AudioDataset(data=data)
