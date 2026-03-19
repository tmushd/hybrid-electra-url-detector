from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Paths:
    repo_root: Path

    @property
    def data_raw(self) -> Path:
        return self.repo_root / "data" / "raw"

    @property
    def data_processed(self) -> Path:
        return self.repo_root / "data" / "processed"

    @property
    def models(self) -> Path:
        return self.repo_root / "models"

    @property
    def results(self) -> Path:
        return self.repo_root / "results"


DEFAULT_URL_COL = "url"
DEFAULT_LABEL_COL = "label"
RANDOM_SEED = 1337
