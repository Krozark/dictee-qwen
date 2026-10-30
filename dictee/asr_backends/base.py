from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np


class AsrBackend(ABC):
    @abstractmethod
    def transcribe(self, audio: np.ndarray, sample_rate: int) -> str:
        raise NotImplementedError

    def close(self) -> None:
        pass
