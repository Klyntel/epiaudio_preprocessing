import math
import re


SECONDS_RE = re.compile(r"^\s*([-+]?\d+(?:\.\d+)?)\s*(s|sec|secs|second|seconds)?\s*$", re.IGNORECASE)
SAMPLE_RATE_RE = re.compile(r"^\s*([-+]?\d+(?:\.\d+)?)\s*(k?hz)?\s*$", re.IGNORECASE)


def is_missing(value):
    return value is None or (isinstance(value, float) and math.isnan(value))


def to_non_negative_seconds(value):
    if is_missing(value):
        return value
    if isinstance(value, bool):
        raise TypeError("Expected a non-negative seconds value, got bool")
    if isinstance(value, int | float):
        seconds = float(value)
        if seconds < 0:
            raise ValueError(f"expected a non-negative seconds value, got `{value}`")
        return seconds
    if hasattr(value, "item"):
        return to_non_negative_seconds(value.item())
    if isinstance(value, str):
        match = SECONDS_RE.match(value)
        if not match:
            raise ValueError(f"Cannot parse `{value}` as seconds")

        seconds = float(match.group(1))
        if seconds < 0:
            raise ValueError(f"expected a non-negative seconds value, got `{value}`")
        return seconds

    raise TypeError(f"Cannot parse {type(value).__name__} as seconds")


def to_sample_rate(value):
    if is_missing(value):
        return value
    if hasattr(value, "item"):
        return to_sample_rate(value.item())
    if isinstance(value, bool):
        raise TypeError("Expected a positive integer sample rate, got bool")

    sample_rate = value
    multiplier = 1
    if isinstance(value, str):
        match = SAMPLE_RATE_RE.match(value)
        if not match:
            raise ValueError(f"Cannot parse `{value}` as a sample rate")
        sample_rate = match.group(1)
        unit = match.group(2)
        if unit and unit.lower() == "khz":
            multiplier = 1000

    number = float(sample_rate) * multiplier
    if not number.is_integer():
        raise ValueError(f"Expected an integer sample rate, got `{sample_rate}`")

    sample_rate = int(number)
    if sample_rate <= 0:
        raise ValueError(f"Expected a positive sample rate, got `{value}`")
    return sample_rate


def samples_to_seconds(samples, sample_rate):
    if is_missing(samples):
        return samples
    if hasattr(samples, "item"):
        samples = samples.item()

    sample_count = float(samples)
    if sample_count < 0:
        raise ValueError(f"Expected a non-negative sample count, got `{samples}`")

    rate = to_sample_rate(sample_rate)
    if is_missing(rate):
        raise ValueError(f"cannot convert samples to seconds without a sample rate, got `{sample_rate}`")
    return sample_count / rate


def seconds_to_samples(seconds, sample_rate):
    seconds = to_non_negative_seconds(seconds)
    if is_missing(seconds):
        return seconds

    rate = to_sample_rate(sample_rate)
    if is_missing(rate):
        raise ValueError(f"cannot convert seconds to samples without a sample rate, got `{sample_rate}`")
    return round(seconds * rate)


def to_list(value):
    if is_missing(value):
        return []
    if isinstance(value, list):
        return list(value)
    if isinstance(value, tuple | set):
        return list(value)
    return [value]
