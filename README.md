# Wyoming Kokoro Torch

[Wyoming protocol](https://github.com/rhasspy/wyoming) server for the original [Kokoro](https://github.com/hexgrad/kokoro/) Torch TTS implementation.

Contrary to other Wyoming implementation, [wyoming-kokoro](https://github.com/nordwestt/kokoro-wyoming/), this is one uses Torch instead of ONNX.
As of the time of writing, our implementation also supports `streaming` mode, while the ONNX one doesn't. Streaming is important for LLM-based assistant,
so that it can start speaking before the LLM is finished generating.

## Local Install

Clone the repository and set up Python virtual environment:

``` sh
git clone https://github.com/debackerl/wyoming-kokoro-torch.cpp.git
cd wyoming-kokoro-torch
script/setup
```

Download the base model:

```sh
mkdir /data
wget -O /data/kokoro-v1_0.pth https://huggingface.co/hexgrad/Kokoro-82M/resolve/main/kokoro-v1_0.pth
wget -O /data/config.json https://huggingface.co/hexgrad/Kokoro-82M/resolve/main/config.json
```

Run a server anyone can connect to:

```sh
script/run --voice af_heart --streaming --uri 'tcp://0.0.0.0:10300' --data-dir /data --download-dir /data
```

See [available voices](https://huggingface.co/hexgrad/Kokoro-82M/tree/main/voices).
