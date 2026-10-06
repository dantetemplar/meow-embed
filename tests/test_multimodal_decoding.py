from __future__ import annotations

import base64
import wave
from io import BytesIO
from typing import Any, cast

import av
import numpy as np

from meow_embed.server import EmbedRequest, encode_dense_inputs


def test_real_audio_and_video_decoding() -> None:
    audio = BytesIO()
    with wave.open(audio, "wb") as stream:
        stream.setnchannels(2)
        stream.setsampwidth(2)
        stream.setframerate(32000)
        stream.writeframes(np.zeros((32000, 2), dtype=np.int16).tobytes())

    video = BytesIO()
    with av.open(video, mode="w", format="mp4") as container:
        stream = container.add_stream("mpeg4", rate=2)
        stream.width = 16
        stream.height = 16
        stream.pix_fmt = "yuv420p"
        for _ in range(4):
            frame = av.VideoFrame.from_ndarray(
                np.zeros((16, 16, 3), dtype=np.uint8), format="rgb24"
            )
            for packet in stream.encode(frame):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)

    class Model:
        def encode(self, inputs: list[Any], **kwargs: Any) -> np.ndarray:
            assert len(inputs) == 1
            sound = inputs[0]["audio"][0]
            assert sound["sampling_rate"] == 16000
            assert sound["array"].shape == (16000,)
            clip = inputs[0]["video"][0]
            assert clip["array"].shape == (4, 16, 16, 3)
            assert clip["video_metadata"].fps == 2
            assert clip["video_metadata"].duration == 2
            return np.zeros((1, 768), dtype=np.float32)

        encode_query = encode
        encode_document = encode

    request = EmbedRequest.model_validate(
        {
            "dense_model_id": "d",
            "texts": [
                {
                    "text": "Sound <|audio|> Clip <|video|>",
                    "audio": {"data": base64.b64encode(audio.getvalue()).decode()},
                    "video": {"data": base64.b64encode(video.getvalue()).decode()},
                }
            ],
        }
    )
    assert encode_dense_inputs(cast(Any, Model()), request).shape == (1, 768)
