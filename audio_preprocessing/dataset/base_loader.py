"""Base classes for dataset loader implementations."""

from __future__ import annotations

import json
import subprocess
import urllib.request
from abc import ABC, abstractmethod
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from datasets import load_dataset

from audio_preprocessing.datasets import AudioDataset, LabelSource
from audio_preprocessing.dataset.zenodo_downloader import download_zenodo


_LABEL_SOURCE_OVERRIDE_ERROR = (
    "label_source is declared by the loader class and cannot be overridden at construction"
)


class BaseLoader(ABC):
    """Common lifecycle for dataset loaders.

    ``prepare_raw`` is intentionally file/source-level preparation only. Dataset
    semantics such as split mapping, labels, metadata, and event schema conversion
    belong in ``build_dataset``.
    """

    label_source: LabelSource | str = LabelSource.UNKNOWN

    def __init__(
        self,
        root: str | Path | None = None,
        *,
        prepare: bool = True,
        **config: Any,
    ) -> None:
        if "label_source" in config:
            raise TypeError(_LABEL_SOURCE_OVERRIDE_ERROR)
        self.root = Path(root) if root is not None else None
        self.should_prepare = prepare
        self.config = config

    def prepare_raw(self) -> None:
        """Make raw files available locally before building the dataset."""

    @abstractmethod
    def build_dataset(self) -> AudioDataset:
        """Build an ``AudioDataset`` from prepared raw data."""

    def __call__(self) -> AudioDataset:
        if self.should_prepare:
            self.prepare_raw()
        return self.build_dataset()


class ZenodoLoader(BaseLoader):
    """Loader base for datasets acquired from one or more public Zenodo records."""

    record_id: str | None = None
    record_ids: Iterable[str] | None = None

    def __init__(
        self,
        root: str | Path | None = None,
        *,
        record_id: str | None = None,
        record_ids: Iterable[str] | None = None,
        only: list[str] | None = None,
        do_not_download: list[str] | None = None,
        prepare: bool = True,
        **config: Any,
    ) -> None:
        super().__init__(root=root, prepare=prepare, **config)
        if record_id is not None and record_ids is not None:
            raise ValueError("Pass either record_id or record_ids, not both.")
        if record_id is not None:
            self.record_id = record_id
            self.record_ids = None
        elif record_ids is not None:
            self.record_id = None
            self.record_ids = (record_ids,) if isinstance(record_ids, str) else tuple(record_ids)
        self.only = only
        self.do_not_download = do_not_download

    def _record_ids(self) -> tuple[str, ...]:
        if self.record_id is not None and self.record_ids is not None:
            raise ValueError("ZenodoLoader cannot declare both record_id and record_ids.")

        if isinstance(self.record_ids, str):
            record_ids = (self.record_ids,)
        elif self.record_ids is not None:
            record_ids = tuple(self.record_ids)
        elif self.record_id is not None:
            record_ids = (self.record_id,)
        else:
            record_ids = ()

        if not record_ids:
            raise ValueError("ZenodoLoader requires record_id or record_ids.")
        if len(set(record_ids)) != len(record_ids):
            raise ValueError("ZenodoLoader record_ids must not contain duplicates.")
        return record_ids

    def prepare_raw(self) -> None:
        record_ids = self._record_ids()

        # A single record keeps the flat layout (``root`` is the extraction directory)
        if self.record_ids is None:
            self.root = download_zenodo(
                record_ids[0],
                output_dir=str(self.root) if self.root is not None else "",
                only=self.only,
                do_not_download=self.do_not_download,
            )
            return

        # Multiple records each get their own ``zenodo_<id>`` subdirectory. ``root``
        # becomes the shared parent that ``build_dataset`` scans.
        if self.root is not None:
            base_dir = Path(self.root)
        else:
            joined = "+".join(record_ids)
            base_dir = Path.cwd() / "data" / f"zenodo_{joined}"

        for record_id in record_ids:
            download_zenodo(
                record_id,
                output_dir=str(base_dir / f"zenodo_{record_id}"),
                only=self.only,
                do_not_download=self.do_not_download,
            )
        self.root = base_dir


class HFLoader(BaseLoader):
    """Loader base for datasets acquired through Hugging Face Datasets."""

    dataset_name: str | None = None
    config_name: str | None = None

    def __init__(
        self,
        root: str | Path | None = None,
        *,
        dataset_name: str | None = None,
        config_name: str | None = None,
        revision: str | None = None,
        cache_dir: str | Path | None = None,
        streaming: bool = False,
        prepare: bool = True,
        **load_kwargs: Any,
    ) -> None:
        if "label_source" in load_kwargs:
            raise TypeError(_LABEL_SOURCE_OVERRIDE_ERROR)
        super().__init__(root=root, prepare=prepare)
        if dataset_name is not None:
            self.dataset_name = dataset_name
        if config_name is not None:
            self.config_name = config_name
        self.revision = revision
        self.cache_dir = cache_dir
        self.streaming = streaming
        self.load_kwargs = load_kwargs
        self.raw: Any = None

    def prepare_raw(self) -> None:
        dataset_name = self.dataset_name
        if dataset_name is None:
            raise ValueError("HFLoader requires a dataset_name.")

        kwargs: dict[str, Any] = {"streaming": self.streaming}
        if self.revision is not None:
            kwargs["revision"] = self.revision
        if self.cache_dir is not None:
            kwargs["cache_dir"] = str(self.cache_dir)
        kwargs.update(self.load_kwargs)

        self.raw = load_dataset(dataset_name, self.config_name, **kwargs)

    def raw_dataset(self) -> Any:
        if self.raw is None:
            self.prepare_raw()
        return self.raw


def _run(cmd: list[str]) -> None:
    print(f"$ {' '.join(cmd)}")
    subprocess.run(cmd, check=True)


def _parse_lfs_pointer(path: Path) -> dict[str, Any] | None:
    """Return {"oid": "sha256:...", "size": int} if path is an LFS pointer, else None."""
    try:
        with open(path, "rb") as f:
            header = f.read(8)
        if header != b"version ":
            return None
        text = path.read_text()
        oid = size = None
        for line in text.splitlines():
            if line.startswith("oid "):
                oid = line.split(" ", 1)[1].strip()
            elif line.startswith("size "):
                size = int(line.split(" ", 1)[1].strip())
        if oid and size is not None:
            return {"oid": oid, "size": size}
    except (OSError, ValueError):
        pass
    return None


def is_lfs_pointer(path: str | Path) -> bool:
    """Return True if ``path`` is an unresolved Git LFS pointer file rather than real content.

    Dataset-specific ``collect()`` functions can use this to raise a clear error when audio
    hasn't been resolved yet (e.g. ``prepare=False`` against a bare clone).
    """
    return _parse_lfs_pointer(Path(path)) is not None


def _lfs_batch_urls(batch_url: str, objects: list[dict[str, Any]], chunk_size: int = 50) -> dict[str, str]:
    """Call the LFS batch API and return {oid: download_url} for each object.

    Chunks requests to avoid 413 errors from large batches.
    """
    result: dict[str, str] = {}
    for i in range(0, len(objects), chunk_size):
        chunk = objects[i:i + chunk_size]
        body = json.dumps({
            "operation": "download",
            "transfers": ["basic"],
            "objects": [{"oid": o["oid"].replace("sha256:", ""), "size": o["size"]} for o in chunk],
        }).encode()
        req = urllib.request.Request(
            batch_url,
            data=body,
            headers={
                "Accept": "application/vnd.git-lfs+json",
                "Content-Type": "application/vnd.git-lfs+json",
            },
        )
        with urllib.request.urlopen(req) as resp:
            data = json.loads(resp.read())
        for obj in data["objects"]:
            if "actions" in obj and "download" in obj["actions"]:
                result[f"sha256:{obj['oid']}"] = obj["actions"]["download"]["href"]
    return result


def _resolve_lfs_pointers(root: Path, subdirs: list[str] | None, batch_url: str) -> None:
    """Download and replace every LFS pointer file found under ``subdirs`` (or all of ``root``)."""
    search_roots = [root / d for d in subdirs] if subdirs else [root]

    pointers: dict[Path, dict[str, Any]] = {}
    for search_root in search_roots:
        for path in search_root.rglob("*"):
            if not path.is_file():
                continue
            ptr = _parse_lfs_pointer(path)
            if ptr:
                pointers[path] = ptr

    if not pointers:
        print("No LFS pointers found -- files may already be resolved.")
        return

    print(f"Found {len(pointers)} LFS pointer(s). Fetching download URLs...")
    oid_to_url = _lfs_batch_urls(batch_url, list(pointers.values()))

    for i, (path, ptr) in enumerate(pointers.items(), 1):
        url = oid_to_url.get(ptr["oid"])
        if not url:
            print(f"  [{i}/{len(pointers)}] WARNING: no URL for {path.name}, skipping")
            continue
        print(f"  [{i}/{len(pointers)}] {path.relative_to(root)}")
        tmp = path.with_suffix(path.suffix + ".tmp")
        try:
            urllib.request.urlretrieve(url, tmp)
            tmp.replace(path)
        except Exception as exc:
            tmp.unlink(missing_ok=True)
            print(f"    ERROR: {exc}")


class GitLFSLoader(BaseLoader):
    """Loader base for datasets hosted as a plain GitHub repo with audio stored via Git LFS.

    Clones the repo excluding LFS blobs, then resolves the requested LFS pointer files directly
    through the GitHub LFS batch API, so no ``git-lfs`` binary is required.
    """

    repo_url: str | None = None
    branch: str = "main"

    def __init__(
        self,
        root: str | Path | None = None,
        *,
        repo_url: str | None = None,
        branch: str | None = None,
        lfs_dirs: list[str] | None = None,
        prepare: bool = True,
        **config: Any,
    ) -> None:
        super().__init__(root=root, prepare=prepare, **config)
        if repo_url is not None:
            self.repo_url = repo_url
        if branch is not None:
            self.branch = branch
        self.lfs_dirs = lfs_dirs

    def prepare_raw(self) -> None:
        if self.repo_url is None:
            raise ValueError(f"{type(self).__name__} requires a repo_url.")

        if self.root is None:
            repo_name = self.repo_url.rstrip("/").rsplit("/", 1)[-1].removesuffix(".git")
            self.root = Path.cwd() / "data" / f"{repo_name}_raw"

        if not self.root.exists():
            print(f"Cloning {self.repo_url} into {self.root} (metadata only, no LFS)...")
            _run(["git", "clone", "--no-checkout", "-c", "lfs.fetchexclude=*", self.repo_url, str(self.root)])

        _run(["git", "-C", str(self.root), "-c", "lfs.fetchexclude=*", "checkout", self.branch])

        print(f"Resolving LFS objects under {self.lfs_dirs or 'the entire repo'}...")
        _resolve_lfs_pointers(self.root, self.lfs_dirs, self.lfs_batch_url())

    def lfs_batch_url(self) -> str:
        """GitHub's LFS batch API endpoint for ``repo_url`` (which must end in ``.git``)."""
        return f"{self.repo_url}/info/lfs/objects/batch"
