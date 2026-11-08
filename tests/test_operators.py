import numpy as np
import pytest

from grzybki.fgo import (
    branching_candidate,
    clip_vector,
    exploration_candidate,
    exploration_threshold,
    exploitation_candidate,
    germination_candidate,
    growth_factor,
    initialize_population,
    make_rng,
    normalized_fitness,
    nutrient_strength,
    quality_scores,
    select_distinct_indices,
)


def test_initialize_population_bounds_and_reproducibility() -> None:
    first = initialize_population(make_rng(1), 8, 5)
    second = initialize_population(make_rng(1), 8, 5)
    assert first.shape == (8, 5)
    assert np.all((first >= 0.0) & (first <= 1.0))
    assert np.array_equal(first, second)
    with pytest.raises(ValueError):
        initialize_population(make_rng(1), 3, 5)


def test_normalized_fitness_and_quality_scores() -> None:
    weights = normalized_fitness(np.asarray([1.0, 2.0, 3.0]))
    assert weights.sum() == pytest.approx(1.0)
    assert np.allclose(weights, [1 / 6, 2 / 6, 3 / 6])

    uniform = normalized_fitness(np.asarray([0.0, 0.0]))
    assert np.allclose(uniform, [0.5, 0.5])

    scores = quality_scores(np.asarray([3.0, 1.0, 2.0]))
    assert scores[1] == 0.0
    assert scores[0] == pytest.approx(1.0)
    assert np.allclose(quality_scores(np.asarray([2.0, 2.0])), [0.0, 0.0])


def test_exploration_threshold_and_growth_factor() -> None:
    assert exploration_threshold(0, 100, 0.6) == pytest.approx(1.0)
    assert exploration_threshold(100, 100, 0.6) == pytest.approx(0.6)
    assert exploration_threshold(50, 100, 0.6) == pytest.approx(0.8)

    fitness = np.asarray([1.0, 3.0])
    factor = growth_factor(fitness, 1, tau=0.5, r1=0.4)
    expected = (3.0 / 4.0) * 0.4 * 0.5**0.5
    assert factor == pytest.approx(expected)


def test_nutrient_strength_modes() -> None:
    fitness = np.asarray([1.0, 2.0, 3.0])
    first = nutrient_strength(make_rng(4), fitness, t=1, t_max=10)
    second = nutrient_strength(make_rng(4), fitness, t=1, t_max=10)
    assert np.array_equal(first, second)
    assert first.shape == (3,)
    assert np.all(first > 0.0)

    eta = nutrient_strength(make_rng(4), fitness, t=9, t_max=10, mode="fitness")
    assert eta[2] > eta[0]
    inverse = nutrient_strength(
        make_rng(4), fitness, t=9, t_max=10, mode="inverse"
    )
    assert inverse[0] > inverse[2]
    with pytest.raises(ValueError):
        nutrient_strength(make_rng(4), fitness, 9, 10, mode="bad")


def test_select_distinct_indices() -> None:
    rng = make_rng(5)
    selected = select_distinct_indices(rng, population_size=10, exclude=3, count=3)
    assert len(set(selected)) == 3
    assert 3 not in selected
    assert all(0 <= index < 10 for index in selected)
    assert selected == select_distinct_indices(make_rng(5), 10, 3, 3)
    with pytest.raises(ValueError):
        select_distinct_indices(make_rng(5), 3, 0, 3)


def test_candidate_operators_are_reproducible_and_bounded() -> None:
    rng = make_rng(7)
    population = initialize_population(rng, 8, 6)
    fitness = np.arange(8, dtype=float) + 1.0
    a, b, c = select_distinct_indices(rng, 8, 0, 3)

    for factory in [
        lambda r: exploration_candidate(
            r, population[0], population[a], population[b], fitness, 0, 10, 100
        ),
        lambda r: exploitation_candidate(
            r, population[0], population[a], population[b], population[1],
            fitness, 0, 10, 100, 0.2,
        ),
        lambda r: branching_candidate(
            r, population[0], population[a], population[b], population[c],
            population[1], fitness, 0, 10, 100,
        ),
        lambda r: germination_candidate(
            r, population[0], population[a], population[b], population[c],
            population[1], fitness, 0, 10, 100,
        ),
    ]:
        first = factory(make_rng(11))
        second = factory(make_rng(11))
        assert first.shape == (6,)
        assert np.array_equal(first, second)
        clipped = clip_vector(first)
        assert np.all((clipped >= 0.0) & (clipped <= 1.0))
