"""Utility for downloading Kokoro voices."""

import json
import logging
import shutil
from pathlib import Path
from typing import Any, Dict, Iterable, Set, Tuple, Union
from urllib.error import URLError
from urllib.parse import quote, urlsplit, urlunsplit
from urllib.request import urlopen

URL_FORMAT = "https://huggingface.co/hexgrad/Kokoro-82M/resolve/main/voices/{file}"

_DIR = Path(__file__).parent
_LOGGER = logging.getLogger(__name__)


class ModelNotFoundError(Exception):
    pass


def _quote_url(url: str) -> str:
    """Quote file part of URL in case it contains UTF-8 characters."""
    parts = list(urlsplit(url))
    parts[2] = quote(parts[2])
    return urlunsplit(parts)


def get_voices() -> Dict[str, Any]:
    """Loads available voices from embedded JSON file."""

    voices_embedded = _DIR / "voices.json"
    _LOGGER.debug("Loading %s", voices_embedded)
    with open(voices_embedded, "r", encoding="utf-8") as voices_file:
        voices = json.load(voices_file)

    return voices


def ensure_voice_exists(
    name: str,
    data_dirs: Iterable[Union[str, Path]],
    download_dir: Union[str, Path],
    voices_info: Dict[str, Any],
):
    if name not in voices_info:
        # Try as name or file path to a custom voice.
        #
        # This will raise ModelNotFoundError if the model or config file
        # can't be found.
        find_model_file(name, data_dirs)
        return

    assert data_dirs, "No data dirs"

    voice_info = voices_info[name]
    file_path = voice_info["file"]
    file_name = f"{name}.pt"

    found = False
    try:
        find_model_file(file_name, data_dirs)
        found = True
    except ModelNotFoundError:
        pass

    if not found:
        try:
            # Download missing or update to data files
            download_dir = Path(download_dir)

            file_url = URL_FORMAT.format(file=file_path)
            download_file_path = download_dir / file_name
            download_file_path.parent.mkdir(parents=True, exist_ok=True)

            _LOGGER.debug("Downloading %s to %s", file_url, download_file_path)
            with urlopen(_quote_url(file_url)) as response, open(
                download_file_path, "wb"
            ) as download_file:
                shutil.copyfileobj(response, download_file)

            _LOGGER.info("Downloaded %s (%s)", download_file_path, file_url)
        except URLError:
            _LOGGER.exception("Unexpected error while downloading files for %s", name)


def find_model_file(filename: str, data_dirs: Iterable[Union[str, Path]]) -> Path:
    """Looks for a model file.

    Returns: path
    """
    for data_dir in data_dirs:
        data_dir = Path(data_dir)
        file_path = data_dir / filename

        if file_path.exists():
            return file_path

    raise ModelNotFoundError(filename)
