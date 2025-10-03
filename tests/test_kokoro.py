"""Tests for wyoming-kokoro-torch"""

import asyncio
import sys
import wave
from asyncio.subprocess import PIPE
from pathlib import Path

import numpy as np
import pytest
import python_speech_features
from wyoming.audio import AudioChunk, AudioStart, AudioStop
from wyoming.event import async_read_event, async_write_event
from wyoming.info import Describe, Info
from wyoming.tts import Synthesize, SynthesizeVoice

from .dtw import compute_optimal_path

_DIR = Path(__file__).parent
_LOCAL_DIR = _DIR.parent / "local"
_TIMEOUT = 60


@pytest.mark.asyncio
async def test_kokoro() -> None:
    proc = await asyncio.create_subprocess_exec(
        sys.executable,
        "-m",
        "wyoming_kokoro_torch",
        # takes about 40s to compile
        #"--compile",
        "--uri",
        "stdio://",
        "--voice",
        "af_heart",
        "--data-dir",
        str(_LOCAL_DIR),
        stdin=PIPE,
        stdout=PIPE,
    )
    assert proc.stdin is not None
    assert proc.stdout is not None

    # Check info
    await async_write_event(Describe().event(), proc.stdin)
    while True:
        event = await asyncio.wait_for(async_read_event(proc.stdout), timeout=_TIMEOUT)
        assert event is not None

        if not Info.is_type(event.type):
            continue

        info = Info.from_event(event)
        assert len(info.tts) == 1, "Expected one tts service"
        tts = info.tts[0]
        assert len(tts.voices) > 0, "Expected at least one voice"
        voice_model = next((v for v in tts.voices if v.name == "af_heart"), None)
        assert voice_model is not None, "Expected af_heart voice"
        break

    # Synthesize text
    await async_write_event(
        Synthesize("This is a test.", voice=SynthesizeVoice("af_heart")).event(),
        proc.stdin,
    )

    event = await asyncio.wait_for(async_read_event(proc.stdout), timeout=_TIMEOUT)
    assert event is not None
    assert AudioStart.is_type(event.type)
    audio_start = AudioStart.from_event(event)

    with wave.open(str(_DIR / "this_is_a_test.wav"), "rb") as wav_file:
        assert audio_start.rate == wav_file.getframerate()
        assert audio_start.width == wav_file.getsampwidth()
        assert audio_start.channels == wav_file.getnchannels()
        expected_audio = wav_file.readframes(wav_file.getnframes())
        expected_array = np.frombuffer(expected_audio, dtype=np.int16)

    actual_audio = bytes()
    while True:
        event = await asyncio.wait_for(async_read_event(proc.stdout), timeout=_TIMEOUT)
        assert event is not None
        if AudioStop.is_type(event.type):
            break

        if AudioChunk.is_type(event.type):
            chunk = AudioChunk.from_event(event)
            assert chunk.rate == audio_start.rate
            assert chunk.width == audio_start.width
            assert chunk.channels == audio_start.channels
            actual_audio += chunk.audio

    actual_array = np.frombuffer(actual_audio, dtype=np.int16)

    # with wave.open(str(_DIR / "this_is_a_test.wav"), "wb") as wav_file:
    #     wav_file.setframerate(audio_start.rate)
    #     wav_file.setnchannels(audio_start.channels)
    #     wav_file.setsampwidth(audio_start.width)
    #     wav_file.writeframes(actual_audio)

    # Less than 20% difference in length
    assert (
        abs(len(actual_array) - len(expected_array))
        / max(len(actual_array), len(expected_array))
        < 0.2
    )

    # Compute dynamic time warping (DTW) distance of MFCC features
    expected_mfcc = python_speech_features.mfcc(expected_array, winstep=0.02)
    actual_mfcc = python_speech_features.mfcc(actual_array, winstep=0.02)
    assert compute_optimal_path(actual_mfcc, expected_mfcc) < 10

    # Need to close stdin for graceful termination
    proc.stdin.close()
    _, stderr = await proc.communicate()

    assert proc.returncode == 0, stderr.decode()
