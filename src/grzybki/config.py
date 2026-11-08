from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class ObjectiveWeights:
    w1: float = 0.4
    w2: float = 0.4
    w3: float = 0.2

    def __post_init__(self) -> None:
        if self.w1 < 0.0 or self.w2 < 0.0 or self.w3 < 0.0:
            raise ValueError("objective weights must be non-negative")
        if self.w1 + self.w2 + self.w3 <= 0.0:
            raise ValueError("at least one objective weight must be positive")

    def to_dict(self) -> dict[str, float]:
        return {"w1": self.w1, "w2": self.w2, "w3": self.w3}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ObjectiveWeights":
        return cls(
            w1=float(data.get("w1", 0.4)),
            w2=float(data.get("w2", 0.4)),
            w3=float(data.get("w3", 0.2)),
        )


@dataclass(frozen=True)
class FGOConfig:
    population_size: int = 30
    max_evals: int = 20_000
    M: float = 0.6
    Ep: float = 0.7
    R: float = 0.9
    seed: int | None = None
    nutrient_mode: str = "fitness"
    snapshot_interval: int = 100
    checkpoint_interval: int = 1_000
    save_vectors: bool = False

    def __post_init__(self) -> None:
        if self.population_size < 4:
            raise ValueError("population_size must be at least 4")
        if self.max_evals < self.population_size:
            raise ValueError("max_evals must be at least population_size")
        for name in ("M", "Ep", "R"):
            value = getattr(self, name)
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must lie in [0, 1]")
        if self.nutrient_mode not in {"fitness", "inverse"}:
            raise ValueError("nutrient_mode must be 'fitness' or 'inverse'")
        if self.snapshot_interval < 0:
            raise ValueError("snapshot_interval must be non-negative")
        if self.checkpoint_interval < 0:
            raise ValueError("checkpoint_interval must be non-negative")

    def to_dict(self) -> dict[str, Any]:
        return {
            "population_size": self.population_size,
            "max_evals": self.max_evals,
            "M": self.M,
            "Ep": self.Ep,
            "R": self.R,
            "seed": self.seed,
            "nutrient_mode": self.nutrient_mode,
            "snapshot_interval": self.snapshot_interval,
            "checkpoint_interval": self.checkpoint_interval,
            "save_vectors": self.save_vectors,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "FGOConfig":
        return cls(
            population_size=int(data.get("population_size", 30)),
            max_evals=int(data.get("max_evals", 20_000)),
            M=float(data.get("M", 0.6)),
            Ep=float(data.get("Ep", 0.7)),
            R=float(data.get("R", 0.9)),
            seed=None if data.get("seed") is None else int(data["seed"]),
            nutrient_mode=str(data.get("nutrient_mode", "fitness")),
            snapshot_interval=int(data.get("snapshot_interval", 100)),
            checkpoint_interval=int(data.get("checkpoint_interval", 1_000)),
            save_vectors=bool(data.get("save_vectors", False)),
        )
