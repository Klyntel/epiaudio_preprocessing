"""Small Zenodo downloader supporting the archives used by retained loaders."""

import shutil
import tarfile
import zipfile
from pathlib import Path

import requests
from tqdm import tqdm


def _stem(name: str) -> str:
    for suffix in (".tar.gz", ".tgz", ".tar", ".zip"):
        if name.lower().endswith(suffix):
            return name[: -len(suffix)]
    return Path(name).stem


def _matches(name: str, choices: list[str]) -> bool:
    return any(choice == _stem(name) or choice in name for choice in choices)


def _safe_zip_members(archive: zipfile.ZipFile, destination: Path):
    root = destination.resolve()
    members = archive.infolist()
    for member in members:
        if not (root / member.filename).resolve().is_relative_to(root):
            raise ValueError(f"Unsafe ZIP member path: {member.filename!r}")
    return members


def _extract(path: Path, destination: Path) -> None:
    target = destination / _stem(path.name)
    if target.exists():
        return
    partial = target.with_name(target.name + ".part")
    shutil.rmtree(partial, ignore_errors=True)
    try:
        if path.suffix.lower() == ".zip":
            with zipfile.ZipFile(path) as archive:
                archive.extractall(partial, members=_safe_zip_members(archive, partial))
        elif tarfile.is_tarfile(path):
            with tarfile.open(path) as archive:
                archive.extractall(partial, filter="data")
        else:
            return
        partial.replace(target)
        path.unlink()
    finally:
        shutil.rmtree(partial, ignore_errors=True)


def download_zenodo(
    record_id: str,
    output_dir: str = "",
    only: list[str] | None = None,
    do_not_download: list[str] | None = None,
) -> Path:
    """Download and extract selected files from a public Zenodo record."""
    output = Path(output_dir) if output_dir else Path.cwd() / "data" / f"zenodo_{record_id}"
    output.mkdir(parents=True, exist_ok=True)
    response = requests.get(f"https://zenodo.org/api/records/{record_id}", timeout=60)
    response.raise_for_status()
    for info in response.json()["files"]:
        name = info["key"]
        if only and not _matches(name, only):
            continue
        if do_not_download and _matches(name, do_not_download):
            continue
        target = output / name
        extracted = output / _stem(name)
        if extracted.exists():
            continue
        if not target.exists() or target.stat().st_size != int(info["size"]):
            partial = target.with_name(target.name + ".part")
            with requests.get(info["links"]["self"], stream=True, timeout=60) as download:
                download.raise_for_status()
                with partial.open("wb") as handle, tqdm(
                    total=int(info["size"]), unit="B", unit_scale=True, desc=name
                ) as progress:
                    for chunk in download.iter_content(1 << 20):
                        handle.write(chunk)
                        progress.update(len(chunk))
            partial.replace(target)
        _extract(target, output)
    return output
