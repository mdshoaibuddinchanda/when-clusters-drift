"""Test idempotency of the dataset downloader: second call must skip without redownloading."""

from pathlib import Path
from clusterdrift.data.downloader import DatasetDownloader
from clusterdrift.data.registry import get_dataset_spec
from clusterdrift.data.schemas import DownloadStatus


def test_download_idempotency(tmp_path):
    downloader = DatasetDownloader(data_root=tmp_path, force=False)
    spec = get_dataset_spec("iris")

    # First download
    res1 = downloader.download(spec)
    assert res1.status == DownloadStatus.OK, f"First download failed: {res1.message}"
    assert downloader.is_canonical_present(spec)

    # Second download (idempotent skip)
    res2 = downloader.download(spec)
    assert res2.status == DownloadStatus.ALREADY_EXISTS, f"Expected ALREADY_EXISTS, got {res2.status}"
    assert "[SKIP VERIFIED]" in res2.message
