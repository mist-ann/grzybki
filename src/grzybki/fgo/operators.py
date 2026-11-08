from __future__ import annotations

import numpy as np

EPSILON = 1e-12


def initialize_population(
    rng: np.random.Generator, population_size: int, dimension: int
) -> np.ndarray:
    if population_size < 4:
        raise ValueError("population_size must be at least 4")
    if dimension < 1:
        raise ValueError("dimension must be positive")
    return rng.random((population_size, dimension))


def normalized_fitness(fitness: np.ndarray) -> np.ndarray:
    values = np.asarray(fitness, dtype=float)
    positive = np.clip(values, 0.0, None)
    total = float(positive.sum())
    if total <= EPSILON:
        return np.full(values.shape, 1.0 / values.size, dtype=float)
    return positive / total


def quality_scores(fitness: np.ndarray) -> np.ndarray:
    values = np.asarray(fitness, dtype=float)
    low = float(values.min())
    high = float(values.max())
    return (values - low) / (high - low + EPSILON)


def exploration_threshold(t: int, t_max: int, M: float = 0.6) -> float:
    tau = _tau(t, t_max)
    return M + (1.0 - M) * (1.0 - tau)


def growth_factor(
    fitness: np.ndarray, index: int, tau: float, r1: float
) -> float:
    share = normalized_fitness(fitness)[index]
    return float(share * r1 * (1.0 - tau) ** (1.0 - tau))


def nutrient_strength(
    rng: np.random.Generator,
    fitness: np.ndarray,
    t: int,
    t_max: int,
    mode: str = "fitness",
) -> np.ndarray:
    values = np.asarray(fitness, dtype=float)
    if t <= t_max / 2:
        eta = rng.random(values.size)
    elif mode == "inverse":
        eta = 1.0 / (np.clip(values, 0.0, None) + EPSILON)
    elif mode == "fitness":
        eta = values
    else:
        raise ValueError(f"unsupported nutrient mode: {mode}")
    total = float(eta.sum())
    if total <= EPSILON:
        eta = np.full(values.shape, 1.0 / values.size, dtype=float)
    else:
        eta = eta / total
    return eta + 2.0 * rng.random()


def select_distinct_indices(
    rng: np.random.Generator, population_size: int, exclude: int, count: int = 3
) -> tuple[int, ...]:
    if count > population_size - 1:
        raise ValueError("not enough individuals to select distinct peers")
    candidates = np.delete(np.arange(population_size), exclude)
    selected = rng.choice(candidates, size=count, replace=False)
    return tuple(int(index) for index in selected)


def exploration_candidate(
    rng: np.random.Generator,
    S_i: np.ndarray,
    S_a: np.ndarray,
    S_b: np.ndarray,
    fitness: np.ndarray,
    index: int,
    t: int,
    t_max: int,
) -> np.ndarray:
    tau = _tau(t, t_max)
    r1 = rng.random()
    growth = np.exp(growth_factor(fitness, index, tau, r1))
    candidate = S_i + growth * (S_a - S_b)
    return _crossover_keep_old(rng, S_i, candidate)


def exploitation_candidate(
    rng: np.random.Generator,
    S_i: np.ndarray,
    S_a: np.ndarray,
    S_b: np.ndarray,
    S_best: np.ndarray,
    fitness: np.ndarray,
    index: int,
    t: int,
    t_max: int,
    eta: float,
    M: float = 0.6,
    Ep: float = 0.7,
    R: float = 0.9,
) -> np.ndarray:
    tau = _tau(t, t_max)
    dimension = S_i.size
    Ec = (rng.random(dimension) - 0.5) * rng.random() * (S_a - S_b)
    if rng.random() < rng.random():
        r_j = rng.random(dimension)
        active = rng.random(dimension) > rng.random()
        De2 = r_j * (S_i - S_best) * active
        candidate = S_i + eta * De2 + Ec * float(rng.random() > rng.random())
    else:
        r4 = rng.random()
        r5 = rng.random(dimension)
        beta = 1.0 if rng.random() < 0.5 else -1.0
        toward_best = float(rng.random() > R)
        De = r4 * (S_a - S_i) + r5 * (beta * S_best - S_i) * toward_best
        candidate = S_i + eta * De + Ec * float(rng.random() > Ep)
    return np.clip(candidate, 0.0, 1.0)


def branching_candidate(
    rng: np.random.Generator,
    S_i: np.ndarray,
    S_a: np.ndarray,
    S_b: np.ndarray,
    S_c: np.ndarray,
    S_best: np.ndarray,
    fitness: np.ndarray,
    index: int,
    t: int,
    t_max: int,
) -> np.ndarray:
    tau = _tau(t, t_max)
    share = normalized_fitness(fitness)[index]
    E_L = 1.0 + np.exp(share) * float(rng.random() < rng.random())
    r5 = rng.random()
    D_ep1 = S_a - S_b
    D_ep2 = S_c - S_best
    candidate = S_i + r5 * E_L * D_ep1 + (1.0 - r5) * E_L * D_ep2
    return _crossover_keep_old(rng, S_i, candidate)


def germination_candidate(
    rng: np.random.Generator,
    S_i: np.ndarray,
    S_a: np.ndarray,
    S_b: np.ndarray,
    S_c: np.ndarray,
    S_best: np.ndarray,
    fitness: np.ndarray,
    index: int,
    t: int,
    t_max: int,
) -> np.ndarray:
    tau = _tau(t, t_max)
    S_g = 1.0 if rng.random() < 0.5 else -1.0
    r1 = rng.random()
    growth = np.exp(growth_factor(fitness, index, tau, r1))
    r5 = rng.random()
    mu = S_g * r5 * growth
    spore = (
        (tau * S_best + (1.0 - tau) * S_a + S_b) / 2.0
        + mu * np.abs((S_c + S_a + S_b) / 3.0 - S_i)
    )
    return _crossover_keep_old(rng, S_i, spore)


def clip_vector(vector: np.ndarray) -> np.ndarray:
    return np.clip(np.asarray(vector, dtype=float), 0.0, 1.0)


def _crossover_keep_old(
    rng: np.random.Generator, old: np.ndarray, candidate: np.ndarray
) -> np.ndarray:
    keep_old = rng.random(old.size) < rng.random()
    return np.where(keep_old, old, candidate)


def _tau(t: int, t_max: int) -> float:
    if t_max <= 0:
        return 1.0
    return float(np.clip(t / t_max, 0.0, 1.0))
