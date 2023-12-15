"""Tests for the AtlasBrowser class."""

import pytest
import numpy as np
import matplotlib.pyplot as plt

from mni_to_atlas import AtlasBrowser
from mni_to_atlas.atlases import _SUPPORTED_ATLASES

# Do not show plots when testing
plt.switch_backend("Agg")


@pytest.mark.parametrize("name", _SUPPORTED_ATLASES)
def test_init_runs(name: str):
    """Test AtlasBrowser initialisation."""
    AtlasBrowser(name)


def test_init_error_catch():
    """Test AtlasBrowser initialisation errors caught."""
    incorrect_name = "Not an atlas"
    with pytest.raises(
        ValueError,
        match=f"The requested atlas '{incorrect_name}' is not recognised.",
    ):
        AtlasBrowser(incorrect_name)


@pytest.mark.parametrize(
    "inputs",
    [
        ["AAL", np.array([40, 0, 60]), ["Frontal_Mid_R"]],
        ["AAL3", np.array([40, 0, 60]), ["Frontal_Mid_2_R"]],
        ["HCPEx", np.array([40, 0, 60]), ["Inferior_6-8_Transitional_Area_R"]],
        [
            "AAL",
            np.array([[40, 0, 60], [0, 0, 0]]),
            ["Frontal_Mid_R", "Undefined"],
        ],
        [
            "AAL3",
            np.array([[40, 0, 60], [0, 0, 0]]),
            ["Frontal_Mid_2_R", "Undefined"],
        ],
        [
            "HCPEx",
            np.array([[40, 0, 60], [0, 0, 0]]),
            ["Inferior_6-8_Transitional_Area_R", "Undefined"],
        ],
    ],
)
@pytest.mark.parametrize("plot", [False, True])
def test_find_regions_runs(
    inputs: tuple[str, np.ndarray, list[str]], plot: bool
):
    """Test that `find_regions` returns the correct region(s).

    Parameters
    ----------
    inputs : tuple
        The atlas name, coordinates, and expected region(s), respectively.

    plot : bool
        Whether to plot the results.

    Notes
    -----
    Correct regions were found using MRIcron
    (https://www.nitrc.org/projects/mricron).
    """
    atlas = AtlasBrowser(inputs[0])
    assert atlas.find_regions(inputs[1], plot=plot) == inputs[2]


def test_find_regions_error_cach():
    """Test that errors are caught for `find_regions`."""
    atlas = AtlasBrowser("AAL")  # atlas type is irrelevant

    coords_list = [[40, 0, 60]]
    with pytest.raises(
        TypeError,
        match="`coordinates` must be a NumPy array.",
    ):
        atlas.find_regions(coords_list)

    coords_3d = np.array([[40, 0, 60]])[:, np.newaxis]
    with pytest.raises(
        ValueError,
        match=(
            "`coordinates` must have two dimensions, but it has "
            f"{coords_3d.ndim} dimensions."
        ),
    ):
        atlas.find_regions(coords_3d)

    coords_n_by_4 = np.array([[40, 0, 60, 0]])
    with pytest.raises(
        ValueError,
        match=(
            r"'coordinates' must have shape \(n, 3\), but it has shape \(n, "
            rf"{coords_n_by_4.shape[1]}\)."
        ),
    ):
        atlas.find_regions(coords_n_by_4)
