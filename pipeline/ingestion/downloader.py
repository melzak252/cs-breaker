"""Download, verify, and extract CS2 GOTV demos."""

from __future__ import annotations

import gzip
import hashlib
import os
import re
import shutil
import subprocess
import http.client
import tarfile
import urllib.error
import urllib.parse
import urllib.request
import zipfile
import zlib
from pathlib import Path, PurePosixPath
from typing import Callable, List, Optional

from pipeline.contracts import DemoFileMeta


class DemoDownloadError(RuntimeError):
    """A demo download could not be completed."""


class ChecksumVerificationError(DemoDownloadError):
    """A downloaded demo did not match its expected SHA-256 digest."""


class ArchiveExtractionError(RuntimeError):
    """An archive is invalid, unsafe, or cannot be extracted."""


class DemoDownloader:
    """Stream demo archives to disk and extract any contained map demos."""

    CHUNK_SIZE = 1024 * 1024

    def __init__(
        self,
        chunk_size: int = CHUNK_SIZE,
        progress_callback: Optional[Callable[[int, Optional[int]], None]] = None,
    ) -> None:
        if chunk_size <= 0:
            raise ValueError("chunk_size must be positive")
        self.chunk_size = chunk_size
        self.progress_callback = progress_callback
        self.downloaded_bytes = 0
        self.total_bytes: Optional[int] = None

    def download_file(
        self,
        url: str,
        dest_path: str,
        expected_sha256: Optional[str] = None,
        resume: bool = True,
    ) -> str:
        """Stream a URL to disk, resuming a partial file where supported."""
        destination = Path(dest_path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        existing_bytes = destination.stat().st_size if resume and destination.exists() else 0
        request = urllib.request.Request(url)
        if existing_bytes:
            request.add_header("Range", f"bytes={existing_bytes}-")

        try:
            response = urllib.request.urlopen(request, timeout=60)
        except urllib.error.HTTPError as exc:
            if exc.code == 416 and existing_bytes:
                digest = self._hash_file(destination)
                if expected_sha256 is None or digest.lower() == expected_sha256.lower():
                    self.downloaded_bytes = existing_bytes
                    self.total_bytes = existing_bytes
                    return str(destination)
            raise DemoDownloadError(f"Unable to download {url}: {exc}") from exc
        except (urllib.error.URLError, OSError) as exc:
            raise DemoDownloadError(f"Unable to download {url}: {exc}") from exc

        with response:
            status = getattr(response, "status", response.getcode())
            content_range = response.headers.get("Content-Range", "")
            append = bool(existing_bytes and status == 206)
            if append:
                match = re.fullmatch(r"bytes\s+(\d+)-(\d+)/(\d+|\*)", content_range.strip(), re.I)
                if not match or int(match.group(1)) != existing_bytes:
                    raise DemoDownloadError(
                        f"Server returned an invalid Content-Range while resuming {url}: {content_range!r}"
                    )
                response_total = None if match.group(3) == "*" else int(match.group(3))
                response_length = int(match.group(2)) - existing_bytes + 1
                self.total_bytes = response_total
                if response.headers.get("Content-Length") is not None and int(response.headers["Content-Length"]) != response_length:
                    raise DemoDownloadError(f"Invalid Content-Length while resuming {url}")
            else:
                # A server that ignores Range returns the full object; replace the
                # old partial instead of appending it to the response.
                existing_bytes = 0
                append = False
                raw_length = response.headers.get("Content-Length")
                self.total_bytes = int(raw_length) if raw_length is not None else None
                response_length = self.total_bytes

            digest = hashlib.sha256()
            if append:
                with destination.open("rb") as previous:
                    for block in iter(lambda: previous.read(self.chunk_size), b""):
                        digest.update(block)
            mode = "ab" if append else "wb"
            self.downloaded_bytes = existing_bytes
            try:
                with destination.open(mode) as output:
                    received = 0
                    while True:
                        block = response.read(self.chunk_size)
                        if not block:
                            break
                        output.write(block)
                        digest.update(block)
                        received += len(block)
                        self.downloaded_bytes += len(block)
                        if self.progress_callback is not None:
                            self.progress_callback(self.downloaded_bytes, self.total_bytes)
                    output.flush()
                    os.fsync(output.fileno())
            except (OSError, http.client.HTTPException, urllib.error.URLError) as exc:
                raise DemoDownloadError(f"Download interrupted; partial data retained at {destination}: {exc}") from exc

            if response_length is not None and received != response_length:
                raise DemoDownloadError(
                    f"Incomplete download from {url}: received {received} of {response_length} response bytes"
                )
            if self.total_bytes is not None and destination.stat().st_size != self.total_bytes:
                raise DemoDownloadError(
                    f"Incomplete download from {url}: file has {destination.stat().st_size} of {self.total_bytes} bytes"
                )
            actual_sha256 = digest.hexdigest()

        if expected_sha256 and actual_sha256.lower() != expected_sha256.lower():
            raise ChecksumVerificationError(
                f"SHA-256 mismatch for {destination}: expected {expected_sha256}, got {actual_sha256}"
            )
        return str(destination)

    def extract_archive(self, archive_path: str, extract_dir: str) -> List[str]:
        """Extract an archive and return all contained .dem paths."""
        archive = Path(archive_path)
        output_dir = Path(extract_dir)
        if not archive.is_file():
            raise ArchiveExtractionError(f"Archive does not exist: {archive}")
        output_dir.mkdir(parents=True, exist_ok=True)
        lower_name = archive.name.lower()
        try:
            if lower_name.endswith(".zip"):
                with zipfile.ZipFile(archive) as zipped:
                    for member in zipped.infolist():
                        if self._is_demo(member.filename):
                            target = self._safe_target(output_dir, member.filename)
                            target.parent.mkdir(parents=True, exist_ok=True)
                            with zipped.open(member) as source, target.open("wb") as dest:
                                shutil.copyfileobj(source, dest, self.chunk_size)
            elif lower_name.endswith(".tar.gz") or lower_name.endswith(".tgz"):
                with tarfile.open(archive, mode="r:gz") as tar:
                    for member in tar.getmembers():
                        if member.isfile() and self._is_demo(member.name):
                            target = self._safe_target(output_dir, member.name)
                            source = tar.extractfile(member)
                            if source is None:
                                continue
                            target.parent.mkdir(parents=True, exist_ok=True)
                            with source, target.open("wb") as dest:
                                shutil.copyfileobj(source, dest, self.chunk_size)
            elif lower_name.endswith(".gz"):
                filename = archive.name[:-3]
                if not filename:
                    raise ArchiveExtractionError(f"Invalid gzip filename: {archive}")
                target = self._safe_target(output_dir, filename)
                target.parent.mkdir(parents=True, exist_ok=True)
                with gzip.open(archive, "rb") as source, target.open("wb") as dest:
                    shutil.copyfileobj(source, dest, self.chunk_size)
            elif lower_name.endswith(".rar"):
                self._extract_rar(archive, output_dir)
            elif lower_name.endswith(".dem"):
                target = self._safe_target(output_dir, archive.name)
                if archive.resolve() != target.resolve():
                    shutil.copyfile(archive, target)
            else:
                raise ArchiveExtractionError(f"Unsupported demo archive type: {archive.name}")
        except ArchiveExtractionError:
            raise
        except (OSError, EOFError, ValueError, RuntimeError, zipfile.BadZipFile, tarfile.TarError, gzip.BadGzipFile, zlib.error) as exc:
            raise ArchiveExtractionError(f"Unable to extract {archive}: {exc}") from exc

        return sorted(str(path) for path in output_dir.rglob("*.dem") if path.is_file())

    def download_and_extract_demo(
        self,
        url: str,
        match_id: str,
        dest_dir: str = "data/raw/demos",
        expected_sha256: Optional[str] = None,
    ) -> List[DemoFileMeta]:
        """Download a demo source and describe every extracted map demo."""
        root = Path(dest_dir)
        root.mkdir(parents=True, exist_ok=True)
        url_name = Path(urllib.parse.unquote(urllib.parse.urlparse(url).path)).name
        if not url_name:
            url_name = f"{match_id}.dem"
        archive_path = root / f"{match_id}-{url_name}"
        self.download_file(url, str(archive_path), expected_sha256=expected_sha256)

        extract_dir = root / match_id
        demo_paths = self.extract_archive(str(archive_path), str(extract_dir))
        if not demo_paths:
            raise ArchiveExtractionError(f"No .dem files were found in {archive_path}")
        metadata: List[DemoFileMeta] = []
        for index, path in enumerate(demo_paths, start=1):
            demo_file = Path(path)
            metadata.append(
                DemoFileMeta(
                    demo_id=f"{match_id}-{index}-{demo_file.stem}",
                    match_id=match_id,
                    file_path=str(demo_file),
                    map_name=demo_file.stem,
                    file_size_bytes=demo_file.stat().st_size,
                    sha256=self._hash_file(demo_file),
                    download_url=url,
                )
            )
        return metadata

    @staticmethod
    def create_dummy_test_demo(output_path: str, map_name: str = "de_mirage") -> str:
        """Write a tiny Source 2 demo container header for parser test fixtures.

        This is a synthetic framing fixture, not a playable match recording.
        """
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        # Encode the map name in the Source 2 header's protobuf framing.
        map_bytes = map_name.encode("utf-8")
        encoded_map_length = bytearray()
        length = len(map_bytes)
        while length >= 0x80:
            encoded_map_length.append((length & 0x7f) | 0x80)
            length >>= 7
        encoded_map_length.append(length)
        header = b"\x2a" + bytes(encoded_map_length) + map_bytes
        header_length = len(header)
        encoded_header_length = bytearray()
        while header_length >= 0x80:
            encoded_header_length.append((header_length & 0x7f) | 0x80)
            header_length >>= 7
        encoded_header_length.append(header_length)
        path.write_bytes(b"PBDEMS2\x00" + bytes(encoded_header_length) + header)
        return str(path)

    def _extract_rar(self, archive: Path, output_dir: Path) -> None:
        executable = shutil.which("unrar") or shutil.which("rar") or shutil.which("7z")
        try:
            import rarfile  # type: ignore[import-not-found]
        except ImportError:
            rarfile = None
        if rarfile is not None:
            try:
                with rarfile.RarFile(archive) as rar:
                    for member in rar.infolist():
                        name = member.filename
                        if not self._is_demo(name):
                            continue
                        target = self._safe_target(output_dir, name)
                        target.parent.mkdir(parents=True, exist_ok=True)
                        with rar.open(member) as source, target.open("wb") as dest:
                            shutil.copyfileobj(source, dest, self.chunk_size)
                return
            except Exception as exc:
                raise ArchiveExtractionError(f"Unable to extract RAR archive {archive}: {exc}") from exc
        if executable and Path(executable).name.lower().startswith("7z"):
            result = subprocess.run(
                [executable, "x", "-y", f"-o{output_dir}", str(archive)],
                capture_output=True,
                text=True,
                check=False,
            )
            if result.returncode == 0:
                return
            raise ArchiveExtractionError(f"7z failed to extract {archive}: {result.stderr.strip()}")
        if executable:
            result = subprocess.run(
                [executable, "x", "-o", str(output_dir), str(archive)],
                capture_output=True,
                text=True,
                check=False,
            )
            if result.returncode == 0:
                return
            raise ArchiveExtractionError(f"{Path(executable).name} failed to extract {archive}: {result.stderr.strip()}")
        raise ArchiveExtractionError(
            f"Cannot extract {archive}: install the Python 'rarfile' package and unrar/unar, rar, or 7z. "
            "The RAR archive has been kept."
        )

    @staticmethod
    def _is_demo(name: str) -> bool:
        return name.lower().endswith(".dem")

    @staticmethod
    def _safe_target(root: Path, member_name: str) -> Path:
        # Normalize archive paths as POSIX paths regardless of host platform.
        relative = PurePosixPath(member_name.replace("\\", "/"))
        if relative.is_absolute() or ".." in relative.parts:
            raise ArchiveExtractionError(f"Unsafe archive path: {member_name}")
        target = root.joinpath(*relative.parts)
        try:
            target.resolve().relative_to(root.resolve())
        except ValueError as exc:
            raise ArchiveExtractionError(f"Unsafe archive path: {member_name}") from exc
        return target

    def _hash_file(self, path: Path) -> str:
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            for block in iter(lambda: stream.read(self.chunk_size), b""):
                digest.update(block)
        return digest.hexdigest()
