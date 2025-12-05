"""Event handler for clients of the server."""

import argparse
import asyncio
import logging
import math
import time
from typing import Any, Dict, Optional

import numpy as np
import torch
from kokoro import KModel, KPipeline
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

SAMPLE_RATE = 24000
_LOGGER = logging.getLogger(__name__)

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

        self.cli_args = cli_args
        self.wyoming_info_event = wyoming_info.event()
        self.voices_info = voices_info
        
        # State
        self.is_streaming: Optional[bool] = None
        self.sbd = SentenceBoundaryDetector()
        self._synthesize: Optional[Synthesize] = None
        self._start_time: Optional[float] = None
        self._total_samples = 0
        
        # Voice State (Instance specific to prevent race conditions)
        self._voice: Optional[torch.Tensor] = None
        self._voice_name: Optional[str] = None
        self._voice_lock = asyncio.Lock()

        # Device Selection
        device = "cuda" if torch.cuda.is_available() else "cpu"
        if device == "cpu" and torch.backends.mps.is_available():
            device = "mps"
        
        if self.cli_args.device:
            device = self.cli_args.device
            
        self.device = device
        _LOGGER.info(f"Using inference device: {self.device}")

        # Model Loading
        data_dirs = cli_args.data_dir
        self._model = KModel(
            model=find_model_file("kokoro-v1_0.pth", data_dirs), 
            config=find_model_file("config.json", data_dirs)
        ).to(self.device).eval()

        if self.cli_args.compile:
            self._model.compile(dynamic=True)

        self._pipelines: Dict[str, KPipeline] = {}

    def get_pipeline(self, lang_code: str) -> KPipeline:
        """Get or create a KPipeline for the specific language on the correct device."""
        if lang_code not in self._pipelines:
            self._pipelines[lang_code] = KPipeline(
                lang_code, 
                model=self._model, 
                device=self.device
            )
        return self._pipelines[lang_code]

    async def handle_event(self, event: Event) -> bool:
        if Describe.is_type(event.type):
            await self.write_event(self.wyoming_info_event)
            _LOGGER.debug("Sent info")
            return True

        try:
            if Synthesize.is_type(event.type):
                if self.is_streaming:
                    return True

                # Handle non-streaming synthesis
                synthesize = Synthesize.from_event(event)
                await self._process_and_synthesize(synthesize)
                return True

            if not self.cli_args.streaming:
                return True

            if SynthesizeStart.is_type(event.type):
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
                    await self._handle_synthesize(self._synthesize)

                await self.write_event(SynthesizeStopped().event())
                
                if self._start_time:
                    _LOGGER.debug("Stream stats: Time to last audio: %s s, Total len: %s s", 
                                  time.perf_counter() - self._start_time, 
                                  self._total_samples / SAMPLE_RATE)
                    self._start_time = None
                    
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

    async def _process_and_synthesize(self, synthesize: Synthesize) -> None:
        """Helper to process full text through SBD and synthesize."""
        self._synthesize = Synthesize(text="", voice=synthesize.voice)
        self.sbd = SentenceBoundaryDetector()
        start_sent = False
        
        # Process sentences
        for i, sentence in enumerate(self.sbd.add_chunk(synthesize.text)):
            self._synthesize.text = sentence
            await self._handle_synthesize(
                self._synthesize, send_start=(i == 0), send_stop=False
            )
            start_sent = True

        # Process final buffer
        self._synthesize.text = self.sbd.finish()
        if self._synthesize.text:
            await self._handle_synthesize(
                self._synthesize, send_start=(not start_sent), send_stop=True
            )
        else:
            await self.write_event(AudioStop().event())

        if self._start_time:
            _LOGGER.debug("Non-stream stats: Time to last audio: %s s, Total len: %s s", 
                          time.perf_counter() - self._start_time, 
                          self._total_samples / SAMPLE_RATE)
            self._start_time = None

    async def _handle_synthesize(
        self, synthesize: Synthesize, send_start: bool = True, send_stop: bool = True
    ) -> bool:
        _LOGGER.debug("Synthesizing text for voice '%s': '%s'", synthesize.voice, synthesize.text)

        if self._start_time is None:
            self._start_time = time.perf_counter()
            self._total_samples = 0

        # Text preprocessing
        raw_text = synthesize.text
        text = " ".join(raw_text.strip().splitlines())
        
        if self.cli_args.auto_punctuation and text:
            has_punctuation = any(text.endswith(p) for p in self.cli_args.auto_punctuation)
            if not has_punctuation:
                text = text + self.cli_args.auto_punctuation[0]

        # Resolve voice name
        voice_name: Optional[str] = None
        if synthesize.voice is not None:
            voice_name = synthesize.voice.name
        
        if not voice_name:
            voice_name = self.cli_args.voice
        
        assert voice_name, "Voice name could not be resolved"

        # Resolve alias
        voice_info = self.voices_info.get(voice_name, {})
        voice_name = voice_info.get("key", voice_name)
        assert voice_name is not None

        # Fetch Pipeline
        lang_code = voice_name[0]
        pipeline = self.get_pipeline(lang_code)

        # Load Voice (Instance Safe)
        async with self._voice_lock:
            if voice_name != self._voice_name:
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

                # Load to GPU/Device immediately
                voice_tensor = pipeline.load_voice(str(voice_model_path))
                self._voice = voice_tensor.to(self.device)
                self._voice_name = voice_name

        assert self._voice is not None

        width = 2 # in bytes
        if send_start:
            await self.write_event(
                AudioStart(
                    rate=SAMPLE_RATE, width=width, channels=1
                ).event(),
            )

        bytes_per_chunk = width * self.cli_args.samples_per_chunk

        # Inference
        # Note: pipeline() input 'voice' arg expects the tensor on the correct device
        stream = pipeline(
            text, 
            voice=self._voice, 
            speed=self.cli_args.speed, 
            split_pattern=None
        )

        for _, (_, _, audio) in enumerate(stream):
            max_volume = 0.95 / np.abs(audio).max() if np.abs(audio).max() > 0 else 1.0
            if self.cli_args.volume > max_volume:
                _LOGGER.warning("Volume too high, reducing to %s", max_volume)

            # Audio Normalization & Conversion
            raw_audio = np.array(
                audio * 32767.0 * np.minimum(max_volume, self.cli_args.volume), 
                dtype=np.int16
            ).tobytes()
            
            num_chunks = int(math.ceil(len(raw_audio) / bytes_per_chunk))

            if self._total_samples == 0:
                 _LOGGER.debug("Latency to first audio: %s s", time.perf_counter() - self._start_time)
            
            self._total_samples += audio.shape[0]

            # Split into chunks and send
            for i in range(num_chunks):
                offset = i * bytes_per_chunk
                chunk = raw_audio[offset : offset + bytes_per_chunk]
                await self.write_event(
                    AudioChunk(
                        audio=chunk, rate=SAMPLE_RATE, width=width, channels=1
                    ).event(),
                )

        if send_stop:
            await self.write_event(AudioStop().event())

        return True
