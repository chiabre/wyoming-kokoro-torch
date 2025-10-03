#!/usr/bin/env python3
import argparse
import asyncio
import signal
import json
import logging
from functools import partial
from pathlib import Path
from typing import Any, Dict, Set

from wyoming.info import Attribution, Info, TtsProgram, TtsVoice, TtsVoiceSpeaker
from wyoming.server import AsyncServer

from . import __version__
from .download import ensure_voice_exists, find_model_file, get_voices
from .handler import KokoroEventHandler

_LOGGER = logging.getLogger(__name__)


async def main() -> None:
    """Main entry point."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--voice",
        required=True,
        help="Default kokoro voice to use (e.g., af_heart)",
    )
    parser.add_argument("--uri", default="stdio://", help="unix:// or tcp://")
    parser.add_argument(
        "--data-dir",
        required=True,
        action="append",
        help="Data directory to store downloaded models",
    )
    parser.add_argument(
        "--download-dir",
        help="Directory to download voices into (default: first data dir)",
    )
    #
    parser.add_argument(
        "--speed", type=float, default=1.0, help="Speed of the voice"
    )
    #
    parser.add_argument(
        "--auto-punctuation", default=".?!", help="Automatically add punctuation"
    )
    parser.add_argument("--samples-per-chunk", type=int, default=1024)
    parser.add_argument(
        "--streaming",
        action="store_true",
        help="Enable audio streaming on sentence boundaries",
    )
    #
    parser.add_argument(
        "--compile",
        action="store_true",
        help="Compile the Torch model",
    )
    parser.add_argument(
        "--device",
        default="cpu",
        help="Torch backend device to use (eg. cpu, cuda, mps, depending on your system and installed runtimes)",
    )
    #
    parser.add_argument("--debug", action="store_true", help="Log DEBUG messages")
    parser.add_argument(
        "--log-format", default=logging.BASIC_FORMAT, help="Format for log messages"
    )
    parser.add_argument(
        "--version",
        action="version",
        version=__version__,
        help="Print version and exit",
    )
    args = parser.parse_args()

    if not args.download_dir:
        # Default to first data directory
        args.download_dir = args.data_dir[0]

    logging.basicConfig(
        level=logging.DEBUG if args.debug else logging.INFO, format=args.log_format
    )
    _LOGGER.debug(args)

    # Load voice info
    voices_info = get_voices()

    # Resolve aliases for backwards compatibility with old voice names
    aliases_info: Dict[str, Any] = {}
    for voice_info in voices_info.values():
        for voice_alias in voice_info.get("aliases", []):
            aliases_info[voice_alias] = {"_is_alias": True, **voice_info}

    voices_info.update(aliases_info)
    voices = [
        TtsVoice(
            name=voice_name,
            description=get_description(voice_info),
            attribution=Attribution(
                name="hexgrad", url="https://github.com/hexgrad/kokoro"
            ),
            installed=True,
            version=None,
            languages=[
                voice_info.get("language", {}).get("code")
            ],
            speakers=None,
        )
        for voice_name, voice_info in voices_info.items()
        if not voice_info.get("_is_alias", False)
    ]

    custom_voice_names: Set[str] = set()
    if args.voice not in voices_info:
        custom_voice_names.add(args.voice)

    for custom_voice_name in custom_voice_names:
        # Add custom voice info
        custom_voice_path = find_model_file(
            custom_voice_name, args.data_dir
        )

        voices.append(
            TtsVoice(
                name=custom_voice_name,
                description="Custom voice",
                version=None,
                attribution=Attribution(name="", url=""),
                installed=True,
                languages=["en_US"],
            )
        )

    wyoming_info = Info(
        tts=[
            TtsProgram(
                name="kokoro",
                description="A fast, local, neural text to speech engine",
                attribution=Attribution(
                    name="hexgrad", url="https://github.com/hexgrad/kokoro"
                ),
                installed=True,
                voices=sorted(voices, key=lambda v: v.name),
                version=__version__,
                supports_synthesize_streaming=args.streaming,
            )
        ],
    )

    # Ensure default voice is downloaded
    voice_info = voices_info.get(args.voice, {})
    voice_name = voice_info.get("key", args.voice)
    assert voice_name is not None

    ensure_voice_exists(voice_name, args.data_dir, args.download_dir, voices_info)

    # Start server
    server = AsyncServer.from_uri(args.uri)

    # Handle OS signals
    loop = asyncio.get_event_loop()
    for s in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(s, lambda: asyncio.create_task(server.stop()))

    _LOGGER.info("Ready")
    await server.run(
        partial(
            KokoroEventHandler,
            wyoming_info,
            args,
            voices_info,
        )
    )


# -----------------------------------------------------------------------------


def get_description(voice_info: Dict[str, Any]):
    """Get a human readable description for a voice."""
    name = voice_info["name"]
    name = " ".join(name.split("_"))
    quality = voice_info["quality"]

    return f"{name} ({quality})"


# -----------------------------------------------------------------------------


def run():
    asyncio.run(main())


if __name__ == "__main__":
    try:
        run()
    except KeyboardInterrupt:
        pass
