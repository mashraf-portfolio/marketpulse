"""ModelRegistry: loads checkpoints into memory, dispatches to the right model."""

from __future__ import annotations

from pathlib import Path


class ModelRegistry:
    """Holds all loaded models keyed by (head, ticker, model_type)."""

    def __init__(self, skip_pytorch: bool = False):
        self.skip_pytorch = skip_pytorch
        self._models: dict = {}

    @classmethod
    def load_from(
        cls, models_dir: Path, ticker_whitelist: list[str], skip_pytorch: bool = False
    ) -> ModelRegistry:
        raise NotImplementedError("Implemented in Phase 5")

    def predict_price(self, request) -> dict:
        raise NotImplementedError("Implemented in Phase 5")

    def predict_volatility(self, request) -> dict:
        raise NotImplementedError("Implemented in Phase 5")

    def classify_regime(self, request) -> dict:
        raise NotImplementedError("Implemented in Phase 5")
