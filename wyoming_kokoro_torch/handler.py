"""Event handler for clients of the server."""

import argparse
import asyncio
import sys
import logging
import math
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
from kokoro import KModel, KPipeline
import torch
from sentence_stream import SentenceBoundaryDetector
from wyoming.audio import AudioChunk, AudioStart, AudioStop
from wyoming.error import Error
from wyoming.event import Event
from wyoming.info import Describe, Info
from wyoming.server import AsyncEventHandler
from wyoming.tts import (
    Synthesize,
    SynthesizeChunk,
    SynthesizeStart,
    SynthesizeStop,
    SynthesizeStopped,
)

from .download import ensure_voice_exists, find_model_file

_LOGGER = logging.getLogger(__name__)

# Keep the most recently used voice loaded
_VOICE = None # will be a FloatTensor of the voice model
_VOICE_NAME: Optional[str] = None
_VOICE_LOCK = asyncio.Lock()


# class KCompiledModel(KModel):
#     def __init__(self, *args, **kwargs):
#         super().__init__(*args, **kwargs)

#     @torch.compile
#     def forward(self, *args, **kwargs):
#         return super().forward(*args, **kwargs)


class KokoroEventHandler(AsyncEventHandler):
    def __init__(
        self,
        wyoming_info: Info,
        cli_args: argparse.Namespace,
        voices_info: Dict[str, Any],
        *args,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)

        data_dirs = cli_args.data_dir

        self.cli_args = cli_args
        self.wyoming_info_event = wyoming_info.event()
        self.voices_info = voices_info
        self.is_streaming: Optional[bool] = None
        self.sbd = SentenceBoundaryDetector()
        self._synthesize: Optional[Synthesize] = None

        self._model = KModel(model=find_model_file("kokoro-v1_0.pth", data_dirs), config=find_model_file("config.json", data_dirs)).to(self.cli_args.device).eval()
        if self.cli_args.compile:
            self._model.compile(dynamic=True)

        self._pipelines: Dict[str, KPipeline] = {}

    def get_pipeline(self, lang_code: str) -> KPipeline:
        if lang_code not in self._pipelines:
            self._pipelines[lang_code] = KPipeline(lang_code, model=self._model)
        return self._pipelines[lang_code]

    async def handle_event(self, event: Event) -> bool:
        if Describe.is_type(event.type):
            await self.write_event(self.wyoming_info_event)
            _LOGGER.debug("Sent info")
            return True

        try:
            if Synthesize.is_type(event.type):
                if self.is_streaming:
                    # Ignore since this is only sent for compatibility reasons.
                    # For streaming, we expect:
                    # [synthesize-start] -> [synthesize-chunk]+ -> [synthesize]? -> [synthesize-stop]
                    return True

                # Sent outside a stream, so we must process it
                synthesize = Synthesize.from_event(event)
                self._synthesize = Synthesize(text="", voice=synthesize.voice)
                self.sbd = SentenceBoundaryDetector()
                start_sent = False
                for i, sentence in enumerate(self.sbd.add_chunk(synthesize.text)):
                    self._synthesize.text = sentence
                    await self._handle_synthesize(
                        self._synthesize, send_start=(i == 0), send_stop=False
                    )
                    start_sent = True

                self._synthesize.text = self.sbd.finish()
                if self._synthesize.text:
                    # Last sentence
                    await self._handle_synthesize(
                        self._synthesize, send_start=(not start_sent), send_stop=True
                    )
                else:
                    # No final sentence
                    await self.write_event(AudioStop().event())

                return True

            if not self.cli_args.streaming:
                # Streaming is not enabled
                return True

            if SynthesizeStart.is_type(event.type):
                # Start of a stream
                stream_start = SynthesizeStart.from_event(event)
                self.is_streaming = True
                self.sbd = SentenceBoundaryDetector()
                self._synthesize = Synthesize(text="", voice=stream_start.voice)
                _LOGGER.debug("Text stream started: voice=%s", stream_start.voice)
                return True

            if SynthesizeChunk.is_type(event.type):
                assert self._synthesize is not None
                stream_chunk = SynthesizeChunk.from_event(event)
                for sentence in self.sbd.add_chunk(stream_chunk.text):
                    _LOGGER.debug("Synthesizing stream sentence: %s", sentence)
                    self._synthesize.text = sentence
                    await self._handle_synthesize(self._synthesize)

                return True

            if SynthesizeStop.is_type(event.type):
                assert self._synthesize is not None
                self._synthesize.text = self.sbd.finish()
                if self._synthesize.text:
                    # Final audio chunk(s)
                    await self._handle_synthesize(self._synthesize)

                # End of audio
                await self.write_event(SynthesizeStopped().event())

                _LOGGER.debug("Text stream stopped")
                return True

            if not Synthesize.is_type(event.type):
                return True

            synthesize = Synthesize.from_event(event)
            return await self._handle_synthesize(synthesize)
        except Exception as err:
            await self.write_event(
                Error(text=str(err), code=err.__class__.__name__).event()
            )
            raise err

    async def _handle_synthesize(
        self, synthesize: Synthesize, send_start: bool = True, send_stop: bool = True
    ) -> bool:
        global _VOICE, _VOICE_NAME

        _LOGGER.debug(synthesize)

        raw_text = synthesize.text

        # Join multiple lines
        text = " ".join(raw_text.strip().splitlines())

        if self.cli_args.auto_punctuation and text:
            # Add automatic punctuation (important for some voices)
            has_punctuation = False
            for punc_char in self.cli_args.auto_punctuation:
                if text[-1] == punc_char:
                    has_punctuation = True
                    break

            if not has_punctuation:
                text = text + self.cli_args.auto_punctuation[0]

        # Resolve voice
        _LOGGER.debug("synthesize: raw_text=%s, text='%s'", raw_text, text)
        voice_name: Optional[str] = None
        if synthesize.voice is not None:
            voice_name = synthesize.voice.name

        if voice_name is None:
            # Default voice
            voice_name = self.cli_args.voice

        assert voice_name is not None

        # Resolve alias
        voice_info = self.voices_info.get(voice_name, {})
        voice_name = voice_info.get("key", voice_name)
        assert voice_name is not None

        # Kokoro lang_code to fetch the correct pipeline
        lang_code = voice_name[0]

        pipeline = self.get_pipeline(lang_code)

        async with _VOICE_LOCK:
            if voice_name != _VOICE_NAME:
                # Load new voice
                _LOGGER.debug("Loading voice: %s", voice_name)

                ensure_voice_exists(
                    voice_name,
                    self.cli_args.data_dir,
                    self.cli_args.download_dir,
                    self.voices_info,
                )
                voice_model_path = find_model_file(
                    f"{voice_name}.pt", self.cli_args.data_dir
                )

                # TODO: voices can be mixed by seperating them by a comma
                _VOICE = pipeline.load_voice(str(voice_model_path))
                _VOICE_NAME = voice_name

        assert _VOICE is not None

        samplerate = 24000
        width = 2 # in bytes

        if send_start:
            await self.write_event(
                AudioStart(
                    rate=samplerate,
                    width=width,
                    channels=1,
                ).event(),
            )

        bytes_per_chunk = width * self.cli_args.samples_per_chunk

        for i, (graphenes, phonemes, audio) in enumerate(pipeline(text, voice=_VOICE, speed=self.cli_args.speed, split_pattern=None)):
            raw_audio = np.array(audio * 32767.0, dtype=np.int16).tobytes()
            num_chunks = int(math.ceil(len(raw_audio) / bytes_per_chunk))

            # Split into chunks
            for i in range(num_chunks):
                offset = i * bytes_per_chunk
                chunk = raw_audio[offset : offset + bytes_per_chunk]
                await self.write_event(
                    AudioChunk(
                        audio=chunk,
                        rate=samplerate,
                        width=width,
                        channels=1,
                    ).event(),
                )

        if send_stop:
            await self.write_event(AudioStop().event())

        return True
