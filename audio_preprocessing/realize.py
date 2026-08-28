"""Audio reads needed by EpiAudio's tokenizer input path."""

from typing import Any, cast


def open_source(uri):
    """Return a local audio path or an opened S3 stream."""
    text = str(uri)
    if not text.startswith("s3://"):
        return text
    try:
        import s3fs
    except ImportError as error:
        raise ImportError("Reading s3:// audio requires the s3fs package.") from error
    return s3fs.S3FileSystem().open(text, "rb")


def read_window(uri, offset, duration):
    """Read float32 audio shaped ``[channels, frames]`` from one time window."""
    from torchcodec.decoders import AudioDecoder

    decoder = AudioDecoder(cast(Any, open_source(uri)))
    start = 0.0 if offset is None else float(offset)
    stop = None if duration is None else start + float(duration)
    samples = decoder.get_samples_played_in_range(start, stop)
    return samples.data.numpy(), decoder.metadata.sample_rate

