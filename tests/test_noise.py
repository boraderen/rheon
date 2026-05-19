from __future__ import annotations

from dataclasses import replace

from rheon.composer import generate_log
from rheon.config import ACTIVITY_KEY, CASE_ID_KEY, make_rng
from rheon.noise import inject_noise


def test_fully_random_noise_preserves_schema_and_marks_all_traces(small_config):
    base_config = replace(small_config, noise_probability=0.0)
    generated = generate_log(base_config, [], rng=make_rng(11))
    noisy_config = replace(small_config, noise_probability=1.0, noise_similar_vs_random=0.0)

    noisy, metadata = inject_noise(
        generated.dataframe,
        noisy_config,
        small_config.activity_pool,
        make_rng(12),
    )

    assert list(noisy.columns) == list(generated.dataframe.columns)
    assert metadata["fully_random_noise"] == generated.dataframe[CASE_ID_KEY].nunique()
    assert noisy[CASE_ID_KEY].nunique() == generated.dataframe[CASE_ID_KEY].nunique()
    assert noisy[ACTIVITY_KEY].isin(small_config.activity_pool).all()


def test_similar_trace_noise_keeps_trace_lengths_in_bounds(small_config):
    base_config = replace(small_config, noise_probability=0.0)
    generated = generate_log(base_config, [], rng=make_rng(21))
    noisy_config = replace(small_config, noise_probability=1.0, noise_similar_vs_random=1.0)
    noisy, metadata = inject_noise(generated.dataframe, noisy_config, small_config.activity_pool, make_rng(22))

    lengths = noisy.groupby(CASE_ID_KEY).size()
    assert metadata["similar_trace_noise"] == generated.dataframe[CASE_ID_KEY].nunique()
    assert lengths.between(small_config.min_trace_length, small_config.max_trace_length).all()
