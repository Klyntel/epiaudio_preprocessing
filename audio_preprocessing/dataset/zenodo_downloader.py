# Adapted from: https://github.com/Klyntel/EpiAudio/blob/656c2b35965a44bf25309aea2d29f86249ccc4c3/epiaudio/dataset/zenodo_downloader.py
"""Generic downloader for public Zenodo records.

Given a Zenodo record id this fetches the record's file listing, downloads the files
(optionally narrowed to an allowlist or widened-minus an excludelist), and extracts archives.
``.zip``, ``.7z``, and tar archives (``.tar``, ``.tar.gz``/``.tgz``, ``.tar.bz2``/``.tbz2``,
``.tar.xz``/``.txz``) are unpacked into a per-archive subdirectory and the archive itself is
removed; other files are kept as-is.

Split (spanned) zip archives — a ``stem.zip`` plus one or more ``stem.z01``, ``stem.z02``
segments — are downloaded as a group and combined with Info-ZIP's ``zip -s 0`` before
extraction, since Python's ``zipfile`` cannot read split archives directly.

Split 7z archives — ``stem.7z.001``, ``stem.7z.002``, ... volumes, with no separate
unnumbered ``.7z`` file — are downloaded as a group and combined by concatenating the
volumes in order, since 7z's split format (unlike zip's) is just the archive's bytes cut
into fixed-size chunks.
"""

import re
import shutil
import tarfile
import subprocess
import zipfile

from dataclasses import dataclass
from pathlib import Path

import py7zr
import requests
from tqdm import tqdm

_TAR_SUFFIXES = (".tar.gz", ".tar.bz2", ".tar.xz", ".tgz", ".tbz2", ".txz", ".tar")


def _archive_stem(filename: str) -> str:
    """Filename with its archive extension(s) removed, handling compound tar suffixes."""
    for suffix in _TAR_SUFFIXES:
        if filename.endswith(suffix):
            return filename[: -len(suffix)]
    return Path(filename).stem


@dataclass(frozen=True)
class _SplitZipGroup:
    zip_filename: str
    segment_filenames: tuple[str, ...]


@dataclass(frozen=True)
class _Split7zGroup:
    part_filenames: tuple[str, ...]  # ordered stem.7z.001, stem.7z.002, ...


_SPLIT_7Z_RE = re.compile(r"^(.*)\.7z\.(\d+)$", re.IGNORECASE)


def _matches(filename: str, names: list[str]) -> bool:
    """True if filename matches any entry in names.

    An entry matches when it equals the file's stem or is a substring of the filename, so a
    caller can pass either the full key (``DKITCHEN_16k``) or a friendly prefix (``DKITCHEN``).
    """
    stem = _archive_stem(filename)
    return any(name == stem or name in filename for name in names)


def _split_zip_groups(filenames: list[str]) -> dict[str, _SplitZipGroup]:
    """Return complete split-ZIP groups using their exact API filenames."""
    zip_filenames: dict[str, str] = {}
    segment_filenames: dict[str, list[str]] = {}
    for filename in filenames:
        path = Path(filename)
        suffix = path.suffix
        if suffix.lower() == ".zip":
            zip_filenames[path.stem] = filename
        elif re.fullmatch(r"\.z\d+", suffix, re.IGNORECASE):
            segment_filenames.setdefault(path.stem, []).append(filename)

    return {
        stem: _SplitZipGroup(
            zip_filename=zip_filename,
            segment_filenames=tuple(sorted(segment_filenames[stem])),
        )
        for stem, zip_filename in zip_filenames.items()
        if stem in segment_filenames
    }


def _split_7z_groups(filenames: list[str]) -> dict[str, _Split7zGroup]:
    """Return split-7z groups keyed by the archive's stem (``stem.7z.001`` -> ``stem``)."""
    parts: dict[str, list[tuple[int, str]]] = {}
    for filename in filenames:
        match = _SPLIT_7Z_RE.match(filename)
        if match is None:
            continue
        stem, volume = match.group(1), int(match.group(2))
        parts.setdefault(stem, []).append((volume, filename))

    return {
        stem: _Split7zGroup(part_filenames=tuple(name for _, name in sorted(volumes)))
        for stem, volumes in parts.items()
    }


def _extract_zip_cli(archive: Path, extract_dir: Path) -> None:
    """Extract a zip with the 7z CLI, for compression methods Python's zipfile cannot decode."""
    if not shutil.which("7z"):
        raise RuntimeError(
            f"{archive.name} uses a zip compression Python cannot decode (e.g. deflate64). "
            "Install the 7z CLI (p7zip) to extract it."
        )
    extract_dir.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        ["7z", "x", "-y", f"-o{extract_dir}", str(archive)], check=True, stdout=subprocess.DEVNULL
    )


def _extract(archive: Path, into: Path) -> bool:
    """Extract a .zip, .7z, or tar archive into ``into/<archive stem>/``. Returns True if handled.

    Extracts into a staging directory first and renames it into place only on success, so a
    mid-extraction failure (truncated archive, disk full) can't leave a partial directory that a
    later run's "already present" skip-check would mistake for a completed extraction.
    """
    suffix = archive.suffix.lower()
    is_tar = archive.name.lower().endswith(_TAR_SUFFIXES)
    if suffix not in (".zip", ".7z") and not is_tar:
        return False

    extract_dir = into / _archive_stem(archive.name)
    extract_part = extract_dir.with_name(f"{extract_dir.name}.part")
    if extract_dir.exists():
        return True

    shutil.rmtree(extract_part, ignore_errors=True)
    try:
        print(f"Extracting {archive.name} -> {extract_dir}/")
        if suffix == ".zip":
            try:
                with zipfile.ZipFile(archive) as zf:
                    zf.extractall(extract_part)
            except NotImplementedError:
                # Some large Zenodo zips use deflate64, which zipfile cannot decode; the 7z CLI can.
                shutil.rmtree(extract_part, ignore_errors=True)
                _extract_zip_cli(archive, extract_part)
        elif is_tar:
            with tarfile.open(archive) as tf:
                # "data" filter blocks path traversal/unsafe members from untrusted tar downloads
                tf.extractall(extract_part, filter="data")
        else:
            with py7zr.SevenZipFile(archive) as sz:
                sz.extractall(extract_part)
        extract_part.replace(extract_dir)
    finally:
        shutil.rmtree(extract_part, ignore_errors=True)
    return True


def _extract_split_zip(zip_path: Path, into: Path) -> None:
    """Combine and extract ``stem.zNN`` + ``stem.zip`` with Info-ZIP."""
    extract_dir = into / zip_path.stem
    extract_part = extract_dir.with_name(f"{extract_dir.name}.part")
    if extract_dir.exists():
        return
    if shutil.which("zip") is None:
        raise RuntimeError(
            f"{zip_path.name} is a split zip archive and needs the `zip` tool (Info-ZIP) "
            "to combine its segments. Install `zip` (for example, `apt-get install zip`), "
            "or extract the archive manually, then retry."
        )

    combined = zip_path.with_name(f"{zip_path.stem}_combined.zip")
    shutil.rmtree(extract_part, ignore_errors=True)
    combined.unlink(missing_ok=True)
    try:
        print(f"Combining split archive {zip_path.name} -> {combined.name}")
        subprocess.run(
            ["zip", "-q", "-s", "0", str(zip_path), "--out", str(combined)],
            check=True,
        )
        print(f"Extracting {zip_path.name} -> {extract_dir}/")
        with zipfile.ZipFile(combined) as zf:
            zf.extractall(extract_part)
        extract_part.replace(extract_dir)
    finally:
        combined.unlink(missing_ok=True)
        shutil.rmtree(extract_part, ignore_errors=True)


def _extract_split_7z(stem: str, part_filenames: tuple[str, ...], into: Path) -> None:
    """Concatenate ``stem.7z.001``, ``stem.7z.002``, ... volumes and extract into ``into/stem/``."""
    extract_dir = into / stem
    extract_part = extract_dir.with_name(f"{extract_dir.name}.part")
    if extract_dir.exists():
        return

    combined = into / f"{stem}_combined.7z"
    shutil.rmtree(extract_part, ignore_errors=True)
    combined.unlink(missing_ok=True)
    try:
        print(f"Combining split archive {stem}.7z.* -> {combined.name}")
        with combined.open("wb") as handle:
            for filename in part_filenames:
                with (into / filename).open("rb") as part:
                    shutil.copyfileobj(part, handle, length=1 << 20)
        print(f"Extracting {stem}.7z -> {extract_dir}/")
        with py7zr.SevenZipFile(combined) as sz:
            sz.extractall(extract_part)
        extract_part.replace(extract_dir)
    finally:
        combined.unlink(missing_ok=True)
        shutil.rmtree(extract_part, ignore_errors=True)


def _file_is_valid(file_info: dict, path: Path) -> bool:
    """Return whether an existing file matches Zenodo's reported size."""
    if not path.is_file():
        return False

    expected_size = file_info.get("size")
    return expected_size is None or path.stat().st_size == int(expected_size)


def _download_file(file_info: dict, dest: Path) -> None:
    """Download, size-check, and atomically promote one Zenodo file."""
    chunk_size = 1 << 20
    expected_size = file_info.get("size")
    expected_size = None if expected_size is None else int(expected_size)
    part = dest.with_name(f"{dest.name}.part")
    part.unlink(missing_ok=True)
    dest.parent.mkdir(parents=True, exist_ok=True)

    bytes_written = 0
    try:
        with requests.get(file_info["links"]["self"], stream=True) as response:
            response.raise_for_status()
            total = expected_size
            if total is None:
                total = int(response.headers.get("content-length", 0))
            with (
                part.open("wb") as handle,
                tqdm(total=total, unit="B", unit_scale=True, desc=dest.name) as bar,
            ):
                for chunk in response.iter_content(chunk_size=chunk_size):
                    if not chunk:
                        continue
                    handle.write(chunk)
                    bytes_written += len(chunk)
                    bar.update(len(chunk))

        if expected_size is not None and bytes_written != expected_size:
            raise RuntimeError(
                f"Size mismatch for {dest.name}: expected {expected_size} bytes, "
                f"downloaded {bytes_written}"
            )
        part.replace(dest)
    except BaseException:
        part.unlink(missing_ok=True)
        raise


def download_zenodo(
    record_id: str,
    output_dir: str = "",
    only: list[str] | None = None,
    do_not_download: list[str] | None = None,
) -> Path:
    """Download a (public) record from Zenodo.

    Args:
        record_id: The Zenodo record id (the number in the record URL).
        output_dir: Where to write the files. Defaults to ``data/zenodo_<record_id>`` under
            the current working directory.
        only: If given, download only files matching one of these names (by stem or
            substring). Use this to grab a cheap subset of a large record.
        do_not_download: Files matching one of these names are skipped. Applied after ``only``.

    Returns:
        Path: The directory the files were written to.
    """
    only = only or []
    do_not_download = do_not_download or []

    output_path = (
        Path(output_dir) if output_dir else Path.cwd() / "data" / f"zenodo_{record_id}"
    )
    output_path.mkdir(parents=True, exist_ok=True)

    response = requests.get(f"https://zenodo.org/api/records/{record_id}")
    response.raise_for_status()
    record = response.json()

    selected = []
    for file_info in record["files"]:
        filename = file_info["key"]
        if only and not _matches(filename, only):
            continue
        if do_not_download and _matches(filename, do_not_download):
            continue
        selected.append(file_info)

    # All pieces must be downloaded before a split archive can be combined.
    filenames = [file_info["key"] for file_info in selected]
    split_zip_groups = _split_zip_groups(filenames)
    split_7z_groups = _split_7z_groups(filenames)
    split_filenames = {
        filename
        for group in split_zip_groups.values()
        for filename in (group.zip_filename, *group.segment_filenames)
    } | {filename for group in split_7z_groups.values() for filename in group.part_filenames}

    for file_info in selected:
        filename = file_info["key"]
        # An extracted archive needs no further validation; its source archive is removed.
        extract_dir = output_path / _archive_stem(filename)
        if extract_dir.exists():
            print(f"Skipping {filename} (already present)")
            continue

        dest = output_path / filename
        if dest.exists() and _file_is_valid(file_info, dest):
            print(f"Skipping {filename} (already present and verified)")
        else:
            if dest.exists():
                print(f"Redownloading {filename} (existing file failed verification)")
            _download_file(file_info, dest)

        if filename in split_filenames:
            continue
        if _extract(dest, output_path):
            dest.unlink()  # archive unpacked; drop the archive itself

    for group in split_zip_groups.values():
        zip_path = output_path / group.zip_filename
        if not zip_path.exists():
            continue
        _extract_split_zip(zip_path, output_path)
        for filename in (group.zip_filename, *group.segment_filenames):
            (output_path / filename).unlink(missing_ok=True)

    for stem, group in split_7z_groups.items():
        if not (output_path / group.part_filenames[0]).exists():
            continue
        _extract_split_7z(stem, group.part_filenames, output_path)
        for filename in group.part_filenames:
            (output_path / filename).unlink(missing_ok=True)

    return output_path
