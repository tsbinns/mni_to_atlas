"""Tests for the AtlasBrowser class."""

import pytest
import numpy as np
from mni_to_atlas import AtlasBrowser


class TestAtlasBrowser:
    def test_invalid_atlas(self):
        with pytest.raises(ValueError, match="The requested atlas"):
            AtlasBrowser("Not An Atlas")

    def test_invalid_coordinates_type(self):
        with pytest.raises(TypeError, match="'coordinates' should be a numpy"):
            AtlasBrowser("AAL").find_regions([44, 44, 44])

    def test_invalid_coordinates_ndim(self):
        with pytest.raises(ValueError, match="'coordinates' should have two"):
            AtlasBrowser("AAL").find_regions(np.zeros((1, 3, 1)))

    def test_invalid_coordinates_shape(self):
        with pytest.raises(
            ValueError, match=r"'coordinates' should be an \[n x 3\]"
        ):
            AtlasBrowser("AAL").find_regions(np.full((1, 4), 44))

    def test_vector_coordinates(self):
        AtlasBrowser("AAL").find_regions(np.full((3,), 44))

    def test_multiple_coordinates(self):
        coordinates = np.full((2, 3), 44)
        regions = AtlasBrowser("AAL").find_regions(coordinates)
        assert len(regions) == coordinates.shape[0], (
            "The number of returned regions does not match the number of "
            "supplied coordinates"
        )

    def test_defined_region(self):
        regions = AtlasBrowser("AAL").find_regions(np.array([40, 0, 60]))
        # Correct region found using MRIcron
        # (https://www.nitrc.org/projects/mricron)
        assert regions == ["Frontal_Mid_R"], "The region is not correct."

    def test_undefined_region(self):
        regions = AtlasBrowser("AAL").find_regions(np.full((1, 3), 0))
        assert regions == ["undefined"], "The region is not undefined."

    def test_supported_atlases(self):
        for atlas_name in AtlasBrowser.supported_atlases:
            AtlasBrowser(atlas_name).find_regions(np.full((1, 3), 44))

    def test_plotting(self):
        AtlasBrowser("AAL").find_regions(np.full((1, 3), 44), plot=True)
