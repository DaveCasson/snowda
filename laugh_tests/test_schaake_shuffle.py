"""Laugh tests for the Schaake Shuffle.

Clark, Gangopadhyay, Hay, Rajagopalan, and Wilby (2004), J. Hydrometeorol.
5(1):243-262. doi:10.1175/1525-7541(2004)005<0243:TSSAMF>2.0.CO;2
Section 3b, equations 5-12, gives a 10-member worked example. The shuffle
keeps each site's values and copies the template's rank correlation.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

from laugh_tests._figures import save_figure
from snowda.schaake_shuffle.core import compute_shuffle_permutation, schaake_shuffle

_CLARK = "Clark et al. (2004), section 3b, equations 5-12"


def spearman_matrix(samples: np.ndarray) -> np.ndarray:
    """Spearman correlation across columns, from ranks. Independent of the shuffle."""
    ranks = np.vstack([stats.rankdata(samples[:, j]) for j in range(samples.shape[1])])
    return np.corrcoef(ranks)


def test_clark_2004_worked_example(laugh) -> None:
    """The vectors printed in Clark et al. (2004), including B and X_ss."""
    forecast = np.array(
        [15.3, 11.2, 8.8, 11.9, 7.5, 9.7, 8.3, 12.5, 10.3, 10.1],
        dtype=float,
    ).reshape(-1, 1)
    template = np.array(
        [10.7, 9.3, 6.8, 11.3, 12.2, 13.6, 8.9, 9.9, 11.8, 12.9],
        dtype=float,
    ).reshape(-1, 1)
    shuffled, permutation = schaake_shuffle(forecast, template)
    laugh.check(
        "rank index B, 1-based as in the paper",
        np.array([3, 7, 2, 8, 1, 4, 9, 5, 10, 6]),
        permutation.ravel() + 1,
        exact=True,
        reference=_CLARK,
    )
    laugh.check(
        "shuffled forecast X_ss",
        np.array([10.1, 8.8, 7.5, 10.3, 11.9, 15.3, 8.3, 9.7, 11.2, 12.5]),
        shuffled.ravel(),
        exact=True,
        reference=_CLARK,
    )


def test_marginal_distributions_are_unchanged(laugh) -> None:
    rng = np.random.default_rng(0)
    forecast = rng.normal(size=(8, 3))
    template = rng.normal(size=(8, 3))
    shuffled, _permutation = schaake_shuffle(forecast, template)
    laugh.check(
        "values at each site, sorted",
        np.sort(forecast, axis=0),
        np.sort(shuffled, axis=0),
        exact=True,
        reference="Clark et al. (2004): the shuffle reorders members and does not change values",
    )


def test_ranks_match_the_template(laugh) -> None:
    rng = np.random.default_rng(1)
    forecast = rng.normal(size=(12, 4))
    template = rng.normal(size=(12, 4))
    shuffled, _permutation = schaake_shuffle(forecast, template)
    rank_gap = 0.0
    for site in range(forecast.shape[1]):
        gap = np.max(
            np.abs(
                stats.rankdata(shuffled[:, site]) - stats.rankdata(template[:, site])
            )
        )
        rank_gap = max(rank_gap, float(gap))
    laugh.check(
        "largest difference between shuffled ranks and template ranks",
        0.0,
        rank_gap,
        atol=0.0,
        reference="Clark et al. (2004), equation 10: member q receives the value of rank r = B[q]",
    )


def test_spearman_correlation_matches_the_template(laugh) -> None:
    rng = np.random.default_rng(2)
    forecast = rng.normal(size=(15, 3))
    template = rng.normal(size=(15, 3))
    shuffled, _permutation = schaake_shuffle(forecast, template)
    laugh.check(
        "Spearman correlation matrix",
        spearman_matrix(template),
        spearman_matrix(shuffled),
        atol=1e-12,
        reference="Clark et al. (2004): the shuffle preserves the Spearman rank correlation",
    )


def test_comonotone_template_gives_perfect_positive_correlation(laugh) -> None:
    member = np.arange(1, 9, dtype=float)
    template = np.column_stack([member, 3.0 * member + 1.0, member**2])
    forecast = np.random.default_rng(3).normal(size=template.shape)
    shuffled, _permutation = schaake_shuffle(forecast, template)
    off_diagonal = spearman_matrix(shuffled)[np.triu_indices(3, k=1)]
    laugh.check(
        "pairwise Spearman correlation",
        np.ones(3),
        off_diagonal,
        atol=1e-12,
        reference="Identical ranks at every site give correlation +1",
    )


def test_antimonotone_template_gives_perfect_negative_correlation(laugh) -> None:
    member = np.arange(1, 9, dtype=float)
    template = np.column_stack([member, -member])
    forecast = np.random.default_rng(4).normal(size=template.shape)
    shuffled, _permutation = schaake_shuffle(forecast, template)
    laugh.check(
        "Spearman correlation",
        -1.0,
        float(spearman_matrix(shuffled)[0, 1]),
        atol=1e-12,
        reference="Reversed ranks give correlation -1",
    )


def test_shuffling_a_field_by_itself_is_identity(laugh) -> None:
    forecast = np.array(
        [
            [1.0, 4.0],
            [3.0, 1.0],
            [2.0, 3.0],
            [4.0, 2.0],
        ]
    )
    shuffled, _permutation = schaake_shuffle(forecast, forecast)
    laugh.check(
        "schaake_shuffle(X, X)",
        forecast,
        shuffled,
        exact=True,
        reference="Ranking a field by its own values puts every member back",
    )


def test_ties_break_in_member_order(laugh) -> None:
    """Equal template values are ranked by member index, and the rank is repeatable."""
    template = np.ones((4, 2))
    first = compute_shuffle_permutation(template)
    second = compute_shuffle_permutation(template)
    laugh.check(
        "permutation at a tied site",
        np.arange(4),
        first[:, 0],
        exact=True,
        reference="A deterministic jitter increasing with member index breaks ties",
    )
    laugh.check(
        "same order at the second site",
        np.arange(4),
        first[:, 1],
        exact=True,
        reference="The same tie break applies at every site",
    )
    laugh.check(
        "second call",
        first,
        second,
        exact=True,
        reference="Tie breaking does not use a random number",
    )


def test_figure_clark_worked_example(laugh) -> None:
    """The 10-member vectors printed in Clark et al. (2004), section 3b."""
    forecast = np.array(
        [15.3, 11.2, 8.8, 11.9, 7.5, 9.7, 8.3, 12.5, 10.3, 10.1],
        dtype=float,
    ).reshape(-1, 1)
    template = np.array(
        [10.7, 9.3, 6.8, 11.3, 12.2, 13.6, 8.9, 9.9, 11.8, 12.9],
        dtype=float,
    ).reshape(-1, 1)
    published_ss = np.array([10.1, 8.8, 7.5, 10.3, 11.9, 15.3, 8.3, 9.7, 11.2, 12.5])
    published_b = np.array([3, 7, 2, 8, 1, 4, 9, 5, 10, 6])
    shuffled, permutation = schaake_shuffle(forecast, template)
    code_b = permutation.ravel() + 1
    members = np.arange(1, 11)

    fig, axes = plt.subplots(1, 2, figsize=(7.6, 3.6))
    axes[0].plot(members, forecast.ravel(), "o-", color="C0", ms=5, label="forecast X")
    axes[0].plot(members, published_ss, "--", color="0.35", lw=1.0, label="X_ss in the paper")
    axes[0].plot(members, shuffled.ravel(), "s", color="C1", ms=5, label="X_ss from the code")
    axes[0].set_xlabel("member")
    axes[0].set_ylabel("value")
    axes[0].set_xticks(members)
    axes[0].set_title("Values reordered")
    axes[0].legend(fontsize=7)

    axes[1].plot(members, published_b, "--", color="0.35", lw=1.0, label="B in the paper")
    axes[1].plot(members, code_b, "s", color="C1", ms=5, label="B from the code")
    axes[1].set_xlabel("member")
    axes[1].set_ylabel("1-based rank index")
    axes[1].set_xticks(members)
    axes[1].set_ylim(0.5, 10.5)
    axes[1].set_title("Rank index B")
    axes[1].legend(fontsize=7)
    fig.suptitle("Clark et al. (2004) worked example")
    fig.tight_layout()
    save_figure(fig, "schaake_clark_example.png")

    laugh.check(
        "largest gap from the printed X_ss",
        0.0,
        float(np.max(np.abs(shuffled.ravel() - published_ss))),
        atol=0.0,
        reference=_CLARK,
    )
    laugh.check(
        "largest gap from the printed rank index B",
        0.0,
        float(np.max(np.abs(code_b - published_b))),
        atol=0.0,
        reference=_CLARK,
    )


def test_figure_spatial_dependence(laugh) -> None:
    """Site 3 is the reverse of site 1. The shuffled forecast should inherit that."""
    rng = np.random.default_rng(5)
    n_members = 40
    shared = rng.normal(size=n_members)
    template = np.column_stack(
        [shared, shared + 0.15 * rng.normal(size=n_members), -shared]
    )
    forecast = rng.normal(size=(n_members, 3))
    shuffled, _permutation = schaake_shuffle(forecast, template)
    fig, axes = plt.subplots(1, 3, figsize=(9.2, 3.3), sharex=True, sharey=True)
    panels = (
        (forecast, "forecast, before"),
        (template, "historical template"),
        (shuffled, "forecast, after the shuffle"),
    )
    for ax, (field, title) in zip(axes, panels):
        ax.scatter(field[:, 0], field[:, 1], s=18, c="C0", label="sites 1 and 2")
        ax.scatter(field[:, 0], field[:, 2], s=18, c="C3", label="sites 1 and 3")
        ax.set_title(title)
        ax.set_xlabel("site 1")
        ax.axhline(0.0, color="0.85", lw=0.6)
        ax.axvline(0.0, color="0.85", lw=0.6)
    axes[0].set_ylabel("other site")
    axes[2].legend(fontsize=8, loc="best")
    fig.suptitle("Schaake Shuffle copies the template's rank dependence")
    fig.tight_layout()
    save_figure(fig, "schaake_rank_dependence.png")
    correlation = spearman_matrix(shuffled)
    laugh.check(
        "Spearman correlation of reversed sites",
        -1.0,
        float(correlation[0, 2]),
        atol=1e-12,
        reference="Clark et al. (2004), Figure 2: ranks follow the template",
    )
    laugh.check(
        "Spearman matrix equals the template",
        spearman_matrix(template),
        correlation,
        atol=1e-12,
        reference=_CLARK,
    )
