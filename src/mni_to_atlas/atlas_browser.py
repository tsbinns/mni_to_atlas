"""Class for converting MNI coordinates to atlas regions."""

import os

import nibabel as nib
import numpy as np
import matplotlib
from matplotlib import pyplot as plt

from mni_to_atlas.atlases import _ATLASES_PATH, _SUPPORTED_ATLASES


class AtlasBrowser:  # noqa: D414
    """Class for converting MNI coordinates to brain atlas regions.

    Parameters
    ----------
    atlas : str
        The name of the atlas to use. Supported atlases are:
        - "AAL" (automated anatomical labelling atlas [1])
        - "AAL3" (automated anatomical labeling atlas 3 [2]; 1mm^3 voxel
          version)
        - "HCPEx" (Human Connectome Project extended parcellation atlas [3])

    Methods
    -------
    find_regions
        Find the regions associated with MNI coordinates for the atlas.

    Notes
    -----
    References:
    [1] Tzourio-Mazoyer et al. (2002) DOI: 10.1006/nimg.2001.0978
    [2] Rolls et al. (2020) DOI: 10.1016/j.neuroimage.2019.116189
    [3] Huang et al. (2022) DOI: 10.1007/s00429-021-02421-6
    """

    _plotting_ready: bool = False

    _image: np.ndarray = None
    _plotting_image: np.ndarray = None
    _affine: np.ndarray = None
    _region_names: dict = None

    def __init__(self, atlas: str) -> None:  # noqa: D107
        self.atlas_name = atlas

        self._check_atlas()
        self._load_data()

    def _check_atlas(self) -> None:
        """Check that the requested atlas is supported."""
        if self.atlas_name not in _SUPPORTED_ATLASES:
            raise ValueError(
                f"The requested atlas '{self.atlas_name}' is not recognised.\n"
                f"Supported atlases are: {_SUPPORTED_ATLASES}"
            )

    def _load_data(self) -> None:
        """Load the atlas and accompanying information."""
        self._path = os.path.join(_ATLASES_PATH, self.atlas_name)
        self._load_atlas()
        self._load_regions()

    def _load_atlas(self) -> None:
        """Load the atlas' nifti file."""
        atlas = nib.load(self._path + ".nii")
        self._image = np.array(atlas.get_fdata(), dtype=np.int32)
        self._affine = atlas.affine

    def _load_regions(self) -> None:
        """Load the regions' IDs, names, and colour groups for the atlas."""
        self._region_names = {0: "Undefined"}

        with open(self._path + ".txt", encoding="utf8", mode="r") as file:
            for line in file:
                columns = line.split()
                self._region_names[int(columns[0])] = columns[1]

    def project_to_nearest(self, coordinates: np.ndarray) -> np.ndarray:
        """Project MNI coordinates to the nearest defined region in the atlas.

        Parameters
        ----------
        coordinates : numpy.ndarray, shape (3, ) or (n, 3)
            MNI coordinates (in mm) to find the associated atlas regions for,
            given as an (n x 3) matrix where n is the number of coordinate
            sets to find regions for, and 3 the x-, y-, and z-axis coordinates,
            respectively.

        Returns
        -------
        projected_coordinates : numpy.ndarray, shape (n, 3)
            The projected MNI coordinates.
        """
        coordinates = self._sort_coordinates(coordinates)
        mni_coords = coordinates.astype(np.int32)  # round to ints
        atlas_coords = self._convert_mni_to_atlas_space(mni_coords)

        projected_atlas_coords = []
        defined_coords = np.nonzero(self._image)
        for coord in atlas_coords:
            if self._image[coord[0], coord[1], coord[2]] != 0:
                projected_atlas_coords.append(coord)
            else:
                nearest_coord_idx = np.argmin(
                    np.linalg.norm(coord - defined_coords, axis=1)
                )
                projected_atlas_coords.append(defined_coords[nearest_coord_idx])
        projected_atlas_coords = np.array(projected_atlas_coords)

        return self._convert_atlas_to_mni_space(projected_atlas_coords)

    def find_regions(self, coordinates: np.ndarray, plot: bool = False) -> list[str]:
        """Find the regions associated with MNI coordinates for the atlas.

        Parameters
        ----------
        coordinates : numpy.ndarray, shape (3, ) or (n, 3)
            MNI coordinates (in mm) to find the associated atlas regions for,
            given as an (n x 3) matrix where n is the number of coordinate
            sets to find regions for, and 3 the x-, y-, and z-axis coordinates,
            respectively.

        plot : bool (default False)
            Whether or not to plot a sagittal, coronal, and axial view of the
            coordinate on the atlas.

        Returns
        -------
        regions : list of str
            Names of the regions in the atlas corresponding to `coordinates`.

        Notes
        -----
        If an entry of `coordinates` does not correspond to a mapped region,
        "Undefined" is returned. To avoid this, use the `project_to_nearest`
        method to first project coordinates to the nearest defined region.
        """
        coordinates = self._sort_coordinates(coordinates)
        if plot and not self._plotting_ready:
            self._prepare_plotting()

        mni_coords = coordinates.astype(np.int32)  # round to ints
        atlas_coords = self._convert_mni_to_atlas_space(mni_coords)
        regions = self._find_regions(atlas_coords)
        if plot:
            for atlas_coord, mni_coord, region in zip(
                atlas_coords, mni_coords, regions
            ):
                self._plot_region(atlas_coord, mni_coord, region)

        return regions

    def _sort_coordinates(self, coordinates: np.ndarray) -> np.ndarray:
        """Check that coordinates are in the correct format.

        Parameters
        ----------
        coordinates

        Returns
        -------
        coordinates : numpy.ndarray, shape (n, 3)
            The coordinates as an (n, 3) matrix.
        """
        if not isinstance(coordinates, np.ndarray):
            raise TypeError("`coordinates` must be a NumPy array.")

        if coordinates.ndim == 1 and coordinates.shape == (3,):
            coordinates = coordinates[np.newaxis, :]

        if coordinates.ndim != 2:
            raise ValueError(
                "`coordinates` must have two dimensions, but it has "
                f"{coordinates.ndim} dimensions."
            )

        if coordinates.shape[1] != 3:
            raise ValueError(
                "'coordinates' must have shape (n, 3), but it has shape (n, "
                f"{coordinates.shape[1]})."
            )

        return coordinates

    def _prepare_plotting(self) -> None:
        """Prepare the atlas for being plotted."""
        plotting_image = self._image.copy().astype(np.float32)

        # set white background to the empty regions of the plot
        plotting_image[plotting_image == 0] = np.nan

        self._plotting_image = self._assign_colour_ids(plotting_image)
        self._plotting_ready = True

    def _assign_colour_ids(self, image: np.ndarray) -> np.ndarray:
        """Assign colours to the atlas regions.

        Parameters
        ----------
        image : numpy.ndarray

        Returns
        -------
        coloured_image : numpy.ndarray
            The image with region IDs reassigned based on the shuffled colour
            IDs.

        Notes
        -----
        Shuffling the colour group values (mostly) ensures that neighbouring
        regions have IDs which are not just one value apart (which when plotted
        leads to neighbouring regions having a similar colour that is difficult
        to distinguish between).

        Additionally, grouping the same regions in different hemispheres with
        the same colour IDs means the colour is shared across both hemispheres.
        """
        colour_assignment = self._define_colour_assignment()

        return self._assign_colours(image, colour_assignment)

    def _define_colour_assignment(self) -> dict:
        """Define how the colour group IDs of regions will be assigned.

        Returns
        -------
        colour_assignement : dict
            Dictionary where the keys are the region IDs and the values are the
            colour IDs which will be used as the new region IDs.

        Notes
        -----
        The matching of regions across the hemispheres assumes that their names
        are identical except for the final two characters which end either with
        "_L" or "_R".
        """
        region_names = list(self._region_names.values())
        region_names = [name for name in region_names if name != "Undefined"]

        unique_region_names = np.unique([name[:-2] for name in region_names])
        colour_ids = np.arange(len(unique_region_names)) + 1
        np.random.seed(40)
        shuffled_colour_ids = colour_ids[
            np.random.randint(0, len(colour_ids), len(colour_ids))
        ]

        colour_assignment = {}
        for unique_name, colour in zip(unique_region_names, shuffled_colour_ids):
            for region_id, region_name in self._region_names.items():
                if region_name.startswith(unique_name):
                    colour_assignment[region_id] = colour

        return colour_assignment

    def _assign_colours(self, image: np.ndarray, assignment: dict) -> np.ndarray:
        """Assign colour ID values as the region IDs in the image.

        Parameters
        ----------
        image : numpy.ndarray

        assignment : dict
            Dictionary where the keys are the original region IDs and the
            values are the new region IDs (i.e. the colours).

        Returns
        -------
        coloured_image : numpy.ndarray
            The image with the new region IDs.
        """
        coloured_image = image.copy()
        for value in assignment.keys():
            coloured_image[image == value] = assignment[value]

        return coloured_image

    def _convert_mni_to_atlas_space(self, mni_coords: np.ndarray) -> np.ndarray:
        """Convert MNI coordinates to atlas coordinates.

        Parameters
        ----------
        mni_coords : numpy.ndarray, shape (n, 3)
            The MNI coordinates rounded to the nearest integer.

        Returns
        -------
        atlas_coords : numpy.ndarray, shape (n, 3)
            The corresponding atlas coordinates.
        """
        # Add column of ones to allow for affine transformation
        extended_mni_coords = np.hstack(
            (mni_coords, np.ones((mni_coords.shape[0], 1), dtype=np.int32))
        )
        # np.linalg.solve faster than taking inverse of affine and multiplying
        atlas_coords = np.linalg.solve(self._affine, extended_mni_coords.T)

        return atlas_coords[:3, :].astype(np.int32).T

    def _convert_atlas_to_mni_space(self, atlas_coords: np.ndarray) -> np.ndarray:
        """Convert atlas coordinates to MNI coordinates.

        Parameters
        ----------
        atlas_coords : numpy.ndarray, shape (n, 3)
            The atlas coordinates rounded to the nearest integer.

        Returns
        -------
        mni_coords : numpy.ndarray, shape (n, 3)
            The corresponding MNI coordinates.
        """
        # Add column of ones to allow for affine transformation
        extended_mni_coords = np.hstack(
            (atlas_coords, np.ones((atlas_coords.shape[0], 1), dtype=np.int32))
        )
        # np.linalg.solve faster than taking inverse of affine and multiplying
        mni_coords = np.linalg.solve(self._affine, extended_mni_coords.T)

        return mni_coords[:3, :].astype(np.int32).T

    def _find_regions(self, atlas_coords: np.ndarray) -> str:
        """Find the name of the regions in the atlas.

        Parameters
        ----------
        atlas_coords : numpy.ndarray
            Coordinates as an (n, 3) matrix in the atlas space.

        Returns
        -------
        regions : list of str
            The name of the regions corresponding to the coordinates.
        """
        region_ids = self._image[
            atlas_coords[:, 0], atlas_coords[:, 1], atlas_coords[:, 2]
        ]

        return [self._region_names[region_id] for region_id in region_ids]

    def _plot_region(
        self, atlas_coords: np.ndarray, mni_coords: np.ndarray, region: str
    ) -> None:
        """Plot a single set of x-, y-, and z-coordinates.

        Parameters
        ----------
        atlas_coords : numpy.ndarray
            (1, 3) atlas coordinates.

        mni_coords : numpy.ndarray
            (1, 3) MNI coordinates.

        region : str
            The name of the region corresponding to the coordinates.
        """
        fig, axes = plt.subplots(1, 3)
        self._plot_views(axes, atlas_coords)
        self._style_plot(fig, axes, atlas_coords, mni_coords, region)
        plt.show()

    def _plot_views(self, axes: list[plt.Axes], atlas_coords: np.ndarray) -> None:
        """Plot a sagittal, coronal, and axial view of atlas coordinates.

        Parameters
        ----------
        axes : list of matplotlib.pyplot.Axes
            A list of three axes to plot the sagittal, coronal, and axial view
            of the coordinates, respectively.

        atlas_coords : numpy.ndarray
            Atlas coordinates, consisting of an x-, y-, and z-coordinate.
        """
        sagittal, coronal, axial = self._get_plot_views(atlas_coords)
        corrected_y, corrected_z = self._get_corrected_coords(atlas_coords)
        axes[0].imshow(sagittal, interpolation="none", vmin=0)
        axes[0].scatter(atlas_coords[1], corrected_z, marker="X", s=100, color="r")
        axes[1].imshow(coronal, interpolation="none", vmin=0)
        axes[1].scatter(atlas_coords[0], corrected_z, marker="X", s=100, color="r")
        axes[2].imshow(axial, interpolation="none", vmin=0)
        axes[2].scatter(atlas_coords[0], corrected_y, marker="X", s=100, color="r")

    def _get_plot_views(
        self, atlas_coords: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Get the sagittal, coronal, and axial views of the atlas.

        Parameters
        ----------
        atlas_coords : numpy.ndarray
            Atlas coordinates, consisting of an x-, y-, and z-coordinate.

        Returns
        -------
        sagittal : numpy.ndarray
            A 2D array containing the sagittal atlas view.

        coronal : numpy.ndarray
            A 2D array containing the coronal atlas view.

        axial : numpy.ndarray
            A 2D array containing the axial atlas view.
        """
        sagittal = np.rot90(self._plotting_image[atlas_coords[0], :, :])
        coronal = np.rot90(self._plotting_image[:, atlas_coords[1], :])
        axial = np.rot90(self._plotting_image[:, :, atlas_coords[2]])

        return sagittal, coronal, axial

    def _get_corrected_coords(self, atlas_coords: np.ndarray) -> tuple[int, int]:
        """Get the corrected y- and z-coordinates for marking the coordinate.

        Parameters
        ----------
        atlas_coords : numpy.ndarray
            Atlas coordinates, consisting of an x-, y-, and z-coordinate.

        Returns
        -------
        corrected_y : int
            The y-coordinate, corrected for the size of the plotted y-axis.

        corrected_z : int
            The z-coordinate, corrected for the size of the plotted z-axis.
        """
        corrected_y = self._plotting_image.shape[1] - atlas_coords[1]
        corrected_z = self._plotting_image.shape[2] - atlas_coords[2]

        return corrected_y, corrected_z

    def _style_plot(
        self,
        fig: matplotlib.figure.Figure,
        axes: list[plt.Axes],
        atlas_coords: np.ndarray,
        mni_coords: np.ndarray,
        region: str,
    ) -> None:
        """Add additional information to the subplots.

        Parameters
        ----------
        fig : matplotlib.figure.Figure
            The figure to plot on.

        axes : list of matplotlib.pyplot.Axes
            A list of three axes to plot the sagittal, coronal, and axial view
            of the coordinates, respectively.

        atlas_coords : numpy.ndarray
            The coordinates in the atlas space.

        mni_coords : numpy.ndarray
            The coordinates in MNI space.

        region : str
            The name of the region corresponding to the coordinates.
        """
        self._add_title(fig, region, mni_coords, atlas_coords)
        self._remove_subplot_axes(axes)
        self._add_orientation_labels(fig, axes)

    def _add_orientation_labels(
        self, fig: matplotlib.figure.Figure, axes: list[plt.Axes]
    ) -> None:
        """Add labels to show the orientation of the subplots.

        Parameters
        ----------
        fig : matplotlib.figure.Figure
            The figure on which the subplots are plotted.

        axes : list of matplotlib.pyplot.Axes
            A list of three axes to plot the sagittal, coronal, and axial view
            of the coordinates, respectively.
        """
        x_left, y_left, x_top, y_top = self._get_orientation_label_positions(axes)

        text_left = ["Posterior", "Left", "Left"]
        text_top = ["Dorsal", "Dorsal", "Anterior"]

        for axis_i in range(len(axes)):
            # text to the left of subplots
            fig.text(
                x_left[axis_i],
                y_left,
                text_left[axis_i],
                verticalalignment="center",
                horizontalalignment="center",
            )
            # text above subplots
            fig.text(
                x_top[axis_i],
                y_top,
                text_top[axis_i],
                verticalalignment="center",
                horizontalalignment="center",
            )

    def _get_orientation_label_positions(
        self, axes: list[plt.Axes]
    ) -> tuple[np.ndarray, float, np.ndarray, float]:
        """Get the figure coordinates for the orientation labels.

        Parameters
        ----------
        axes : list of matplotlib.pyplot.Axes
            A list of three axes to plot the sagittal, coronal, and axial view
            of the coordinates, respectively.

        Returns
        -------
        x_left : numpy.ndarray
            Three x-coordinate figure positions, one for each subplot,
            corresponding to the left of each subplot.

        y_left : float
            The average y-coordinate figure position of the center of each
            subplot.

        x_top : numpy.ndarray
            Three x-coordinate figure positions, one for each subplot,
            corresponding to the center of each subplot.

        y_top : float
            The average y-coordinate figure position of the top of each
            subplot.
        """
        x_left = np.zeros((3,))
        y_left = np.zeros((3,))
        x_top = np.zeros((3,))
        y_top = np.zeros((3,))
        for axis_i, axis in enumerate(axes):
            bounds = axis.get_position().bounds
            if axis_i == 0:
                x_shift = 0.2
            else:
                x_shift = 0
            x_left[axis_i] = bounds[0] - bounds[0] * x_shift  # x position
            y_left[axis_i] = bounds[1] + (bounds[3] / 2)  # y position + half
            # height
            x_top[axis_i] = bounds[0] + (bounds[2] / 2)  # x position + half
            # width
            y_top[axis_i] = bounds[1] + bounds[3]  # y position + height
        y_left = y_left.mean()
        y_top = y_top.mean()

        return x_left, y_left, x_top, y_top + (y_top / 20)

    def _add_title(
        self,
        fig: matplotlib.figure.Figure,
        region: str,
        mni_coords: np.ndarray,
        atlas_coords: np.ndarray,
    ) -> None:
        """Generate a title for the figure.

        Parameters
        ----------
        fig : matplotlib.figure.Figure
            The figure on which the subplots are plotted.

        region : str
            The name of the region corresponding to the coordinates.

        mni_coords : numpy ndarray
            The coordinates in MNI space.

        atlas_coords : numpy ndarray
            The coordinates in the atlas space.
        """
        title = (
            f"Region: {region}     "
            f"Atlas: {self.atlas_name}\n"
            f"MNI coords.: {mni_coords.tolist()}     "
            f"Atlas coords.: {(atlas_coords + 1).tolist()}"
        )
        fig.suptitle(title)

    def _remove_subplot_axes(self, axes: list[plt.Axes]) -> None:
        """Remove the subplot axes from the figure.

        Parameters
        ----------
        axes : list of matplotlib.pyplot.Axes
            A list of three axes to plot the sagittal, coronal, and axial view
            of the coordinates, respectively.
        """
        for axis in axes:
            axis.axis("off")
