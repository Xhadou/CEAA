"""Download and extract RCAEval dataset."""

import os
import zipfile
from pathlib import Path

from tqdm import tqdm

RCAEVAL_URLS = {
    "RE1": {
        "online-boutique": "https://zenodo.org/records/14590730/files/RE1-online-boutique.zip",
        "sock-shop": "https://zenodo.org/records/14590730/files/RE1-sock-shop.zip",
        "train-ticket": "https://zenodo.org/records/14590730/files/RE1-train-ticket.zip",
    },
    "RE2": {
        "online-boutique": "https://zenodo.org/records/14590730/files/RE2-online-boutique.zip",
        "sock-shop": "https://zenodo.org/records/14590730/files/RE2-sock-shop.zip",
        "train-ticket": "https://zenodo.org/records/14590730/files/RE2-train-ticket.zip",
    },
}


def download_file(url: str, dest_path: Path, chunk_size: int = 8192) -> None:
    """Download file with progress bar.

    Args:
        url: URL to download.
        dest_path: Local path to save the downloaded file.
        chunk_size: Bytes per chunk for streaming download.
    """
    try:
        import requests
    except ImportError:
        raise ImportError("requests is required for downloading data. Install with: pip install requests")

    response = requests.get(url, stream=True)
    total_size = int(response.headers.get("content-length", 0))

    with open(dest_path, "wb") as f:
        with tqdm(total=total_size, unit="B", unit_scale=True, desc=dest_path.name) as pbar:
            for chunk in response.iter_content(chunk_size=chunk_size):
                f.write(chunk)
                pbar.update(len(chunk))


def extract_zip(zip_path: Path, extract_to: Path) -> None:
    """Extract zip file and remove the archive.

    Args:
        zip_path: Path to the zip file.
        extract_to: Directory to extract contents into.
    """
    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        zip_ref.extractall(extract_to)
    os.remove(zip_path)


def download_rcaeval_dataset(
    dataset: str = "RE1",
    systems: list = None,
    data_dir: str = "data/raw",
) -> None:
    """Download RCAEval dataset.

    Args:
        dataset: ``"RE1"`` or ``"RE2"``.
        systems: List of systems to download. Default: all available.
        data_dir: Directory to save data.
    """
    if systems is None:
        systems = ["online-boutique", "sock-shop", "train-ticket"]

    data_path = Path(data_dir)
    data_path.mkdir(parents=True, exist_ok=True)

    urls = RCAEVAL_URLS.get(dataset, {})

    for system in systems:
        if system not in urls:
            print(f"Warning: {system} not found in {dataset}")
            continue

        url = urls[system]
        zip_path = data_path / f"{dataset}-{system}.zip"

        print(f"Downloading {dataset}/{system}...")
        download_file(url, zip_path)

        print(f"Extracting {dataset}/{system}...")
        extract_zip(zip_path, data_path / dataset / system)

    print("Download complete!")


if __name__ == "__main__":
    download_rcaeval_dataset(dataset="RE1", systems=["online-boutique"])
