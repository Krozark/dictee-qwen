from __future__ import annotations

from typing import Callable

import numpy as np

from .base import AsrBackend

Loader = Callable[[str], tuple]  # (model_id) -> (model, processor)


class AsrModelLoadError(RuntimeError):
    pass


def default_loader(model_id: str):
    """Loads the real Qwen3-ASR model via transformers. Requires `torch` and
    a CUDA device; not exercised by unit tests (see README manual checklist)."""
    import torch
    from transformers import AutoModelForMultimodalLM, AutoProcessor

    processor = AutoProcessor.from_pretrained(model_id)
    model = (
        AutoModelForMultimodalLM.from_pretrained(model_id, dtype=torch.bfloat16)
        .to("cuda")
        .eval()
    )
    return model, processor


class QwenBackend(AsrBackend):
    def __init__(self, model_id: str, language: str = "fr", loader: Loader = default_loader):
        try:
            self._model, self._processor = loader(model_id)
        except RuntimeError as exc:
            if "out of memory" in str(exc).lower():
                raise AsrModelLoadError(
                    f"Not enough VRAM to load {model_id}. "
                    "Try the 0.6B model or the whisper backend instead."
                ) from exc
            raise
        self._language = language

    def transcribe(self, audio: np.ndarray, sample_rate: int) -> str:
        inputs = self._processor.apply_transcription_request(
            audio=audio, language=self._language
        ).to(self._model.device, self._model.dtype)
        output_ids = self._model.generate(**inputs, max_new_tokens=256)
        generated_ids = output_ids[:, inputs["input_ids"].shape[1]:]
        text = self._processor.decode(generated_ids, return_format="transcription_only")[0]
        return text.strip()

    def close(self) -> None:
        import torch

        del self._model
        torch.cuda.empty_cache()
