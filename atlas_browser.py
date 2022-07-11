"""A class and methods for converting MNI coordinates to atlas regions."""

from os.path import join as pathjoin
import random
from typing import Union
import json
import csv
import nibabel as nib
import numpy as np
from numpy.typing import NDArray
from matplotlib.figure import Figure
from matplotlib import pyplot as plt


class AtlasBrowser:
    """A class for converting MNI coordinates to brain atlas regions, with basic
    plotting functionality.

    PARAMETERS
    ----------
    atlas : str
    -   The name of the atlas to use.
    -   Supported atlases are: "AAL" (automated anatomical labelling atlas [1]);
        and "AAL3" (automated anatomical labeling atlas 3 [2]; 1x1x1 mm voxel
        version).
    
    METHODS
    -------
    find_regions
    -   Finds the regions associated with MNI coordinates for the atlas. If a
        set of coordinates does not correspond to a mapped region, 'undefined'
        is returned.

    NOTES
    -----
    References:
        [1] Tzourio-Mazoyer et al. (2002), NeuroImage, 10.1006/nimg.2001.0978.
        [2] Rolls et al. (2020), NeuroImage, 10.1016/j.neuroimage.2019.116189.
    """

    def __init__(self, atlas: str) -> None:
        ### Initialise inputs
        self.atlas_name = atlas

        ### Initialise status attributes
        self._plotting_ready = False

        ### Initialise attributes that will be filled with information
        self._atlas = None
        self._plotting_atlas = None
        self._conversion = None
        self._region_names = None
        self._region_colours = None

        ### Begin work
        self._check_atlas()
        self._load_data()

    def _check_atlas(self) -> None:
        """Checks that the requested atlas is supported.

        RAISES
        ------
        NotImplementedError
        -   Raised if the requested atlas is not supported.
        """
        supported_atlases = ["AAL", "AAL3"]
        if self.atlas_name not in supported_atlases:
            raise NotImplementedError(
                f"The requested atlas '{self.atlas_name}' is not recognised. "
                f"The supported atlases are {supported_atlases}."
            )

    def _load_data(self) -> None:
        """Loads the atlas and accompanying information into the object."""
        self._path = pathjoin("atlases", self.atlas_name)
        self._load_atlas()
        self._load_conversion()
        self._load_regions()

    def _load_atlas(self) -> None:
        """Loads the atlas .nii file."""
        self._atlas = nib.load(pathjoin(self._path, "atlas.nii")).get_fdata()

    def _load_conversion(self) -> None:
        """Loads the MNI to atlas coordinates conversion information."""
        with open(
            pathjoin(self._path, "conversion.json"), "r", encoding="utf8"
        ) as file:
            self._conversion = json.load(file)

    def _load_regions(self) -> None:
        """Loads the regions' IDs, names, and colour groups for the atlas."""
        self._region_names = {}
        self._region_colours = {}
        with open(
            pathjoin(self._path, "regions.csv"), encoding="utf8", mode="r"
        ) as file:
            contents = csv.reader(file, delimiter="\t")
            for region_id, region_name, colour_group in contents:
                region_id = int(region_id)
                colour_group = int(colour_group)
                self._region_names[region_id] = region_name
                self._region_colours[region_id] = colour_group

    def find_regions(
        self, coordinates: list[list[Union[int, float]]], plot: bool = False
    ) -> list[str]:
        """Finds the regions associated with MNI coordinates for the atlas. If a
        set of coordinates does not correspond to a mapped region, 'undefined'
        is returned.

        PARAMETERS
        ----------
        coordinates : list[list[int | float]]
        -   MNI coordinates (in mm) to find the associated atlas regions for.
        -   Must be a list of sublists in which each sublist contains an x-, y-,
            and z-coordinate. A corresponding region will be returned for each
            sublist of coordinates.

        plot : bool; default False
        -   Whether or not to plot a sagittal, coronal, and axial view of the
            coordinate on the atlas.

        RETURNS
        -------
        regions : list[str]
        -   Names of the regions in the atlas corresponding to the MNI
            coordinates.
        """
        self._check_coordinates(coordinates)
        if plot and not self._plotting_ready:
            self._prepare_plotting()

        regions = []
        for coords in coordinates:
            mni_coords = self._round_coords(coords)
            atlas_coords = self._convert_coords(mni_coords)
            region = self._find_region(atlas_coords)
            regions.append(region)
            if plot:
                self._plot(atlas_coords, mni_coords, region)

        return regions

    def _check_coordinates(
        self, coordinates: list[list[Union[int, float]]]
    ) -> None:
        """Checks that each set of coordinates contains three values,
        corresponding to the x-, y-, and z-coordinates.

        PARAMETERS
        ----------
        coordinates : list[list[int | float]]
        -   A list of sublists in which each sublist contains an x-, y-, and
            z-coordinate.

        RAISES
        ------
        ValueError
        -   Raised if a set of coordinates does not have three values.
        """
        for coord_i, coords in enumerate(coordinates):
            if len(coords) != 3:
                raise ValueError(
                    "Each set of coordinates must have three values, "
                    "corresponding to the x-, y-, and z-coordinates, "
                    "respectively, however this is not the case for the "
                    f"coordinates {coords} in position {coord_i}."
                )

    def _prepare_plotting(self) -> None:
        """Prepares the atlas for being plotted."""
        plotting_atlas = self._atlas.copy()
        plotting_atlas[plotting_atlas == 0] = np.nan  # provides a white...
        # background to the empty regions of the plot
        self._plotting_atlas = self._assign_colour_ids(plotting_atlas)
        self._plotting_ready = True

    def _assign_colour_ids(self, atlas: NDArray) -> NDArray:
        """Shuffles the colour group values of regions belonging to the same
        colour groups and assigns these new values to the plotting atlas as the
        region IDs.

        PARAMETERS
        ----------
        atlas : numpy ndarray
        -   A copy of the atlas.

        RETURNS
        -------
        numpy ndarray
        -   The atlas with region IDs reassigned based on the shuffled colour
            IDs.

        NOTES
        -----
        -   Shuffling the colour group values (mostly) ensures that neighbouring
            regions have region IDs which are not just one value apart, which
            when plotted leads to neighbouring regions having a very similar
            colour which is difficult to distinguish between. Additionally,
            grouping regions with belonging to the same regions, but in
            different hemispheres, with the same colour group IDs means regions
            share the same colour across both hemispheres.
        """
        transformation = self._define_colour_reassignment()
        return self._reassign_colours(atlas, transformation)

    def _define_colour_reassignment(self) -> dict:
        """Creates a dictionary that defines how the colour group IDs of regions
        will be switched.

        RETURNS
        -------
        transformation : dict
        -   Dictionary where the keys are the region IDs and the values are the
            new colour IDs which will be used as the new region IDs..
        """
        random.seed(44)
        colour_ids = np.unique(list(self._region_colours.values())).tolist()
        shuffled_ids = random.sample(colour_ids, len(colour_ids))
        transformation = {}
        for region_id, colour_id in self._region_colours.items():
            transformation[region_id] = shuffled_ids[colour_id - 1]
        return transformation

    def _reassign_colours(
        self, atlas: NDArray, transformation: dict
    ) -> NDArray:
        """Assigns new colour ID values as the region IDs in the atlas.

        PARAMETERS
        ----------
        atlas : numpy ndarray
        -   A copy of the atlas.

        transformation : dict
        -   Dictionary where the keys are the original region IDs and the values
            are the new region IDs.

        RETURNS
        -------
        numpy ndarray
        -   The atlas with the new region IDs.
        """
        atlas_1d = atlas.ravel()
        for val_i, val in enumerate(atlas_1d):
            if not np.isnan(val):
                atlas_1d[val_i] = transformation[val]
        return atlas_1d.reshape(atlas.shape)

    def _convert_coords(self, mni_coords: list[int]) -> list[int]:
        """Converts MNI coordinates (in mm) to atlas coordinates.

        PARAMETERS
        ----------
        mni_coords : list[int]
        -   MNI coordinates (in mm) as integers, consisting of an x-, y-, and
            z-coordinate.

        RETURNS
        -------
        atlas_coords : list[int]
        -   Atlas coordinates consiting of an x-, y-, and z-coordinate.

        NOTES
        -----
        -   The atlas only accepts coordinates in integers, and so MNI
            coordinates must be rounded in the case of non-integer values before
            finding the corresponding region.
        """
        atlas_coords = []
        for offset, coord in zip(self._conversion, mni_coords):
            atlas_coords.append(coord + offset)
        return atlas_coords

    def _round_coords(self, coords: list[Union[int, float]]) -> list[int]:
        """Rounds coordinates to the nearest integer.

        PARAMETERS
        ----------
        coords : list[int | float]
        -   A list of coordinates, consisting of an x-, y-, and z-coordinate.

        RETURNS
        -------
        rounded_coords : list[int]
        -   A list of coordinates, consisting of an x-, y-, and z-coordinate
            rounded to their nearest integer.
        """
        rounded_coords = []
        for coord in coords:
            if coord < 0:
                rounded = int(coord - 0.5)
            else:
                rounded = int(coord + 0.5)
            rounded_coords.append(rounded)
        return rounded_coords

    def _find_region(self, atlas_coords: list[int]) -> str:
        """Finds the name of the region in the atlas belonging to a set of
        coordinates. If the coordinates do not correspond to a mapped region,
        'undefined' is returned.

        PARAMETERS
        ----------
        atlas_coords : list[int]
        -   Atlas coordinates, consisting of an x-, y-, and z-coordinate.

        RETURNS
        -------
        region : str
        -   The name of the region corresponding to the coordinates.
        """
        region_id = int(
            self._atlas[atlas_coords[0], atlas_coords[1], atlas_coords[2]]
        )
        if region_id != 0:
            region = self._region_names[region_id]
        else:
            region = "undefined"
        return region

    def _plot(
        self, atlas_coords: list[int], mni_coords: list[int], region: str
    ) -> None:
        """Plots coordinates on the atlas in the sagittal, coronal, and axial
        views.

        PARAMETERS
        ----------
        atlas_coords : list[int]
        -   Atlas coordinates, consisting of an x-, y-, and z-coordinate.

        mni_coords : list[int]
        -  The atlas coordinates in MNI space.

        region : str
        -   The name of the region corresponding to the coordinates.
        """
        fig, axes = plt.subplots(1, 3)
        self._plot_views(axes, atlas_coords)
        self._style_plot(fig, axes, atlas_coords, mni_coords, region)
        plt.show()

    def _plot_views(
        self, axes: list[plt.Axes], atlas_coords: list[int]
    ) -> None:
        """Plots a sagittal, coronal, and axial view of atlas coordinates on a
        set of axes.

        PARAMETERS
        ----------
        axes : list[matplotlib pyplot Axes]
        -   A list of three axes to plot the sagittal, coronal, and axial view
            of the coordinates, respectively.

        atlas_coords : list[int]
        -   Atlas coordinates, consisting of an x-, y-, and z-coordinate.
        """
        sagittal, coronal, axial = self._get_plot_views(atlas_coords)
        corrected_y, corrected_z = self._get_corrected_coords(atlas_coords)
        axes[0].imshow(sagittal, interpolation="none", vmin=0)
        axes[0].scatter(
            atlas_coords[1], corrected_z, marker="X", s=100, color="r"
        )
        axes[1].imshow(coronal, interpolation="none", vmin=0)
        axes[1].scatter(
            atlas_coords[0], corrected_z, marker="X", s=100, color="r"
        )
        axes[2].imshow(axial, interpolation="none", vmin=0)
        axes[2].scatter(
            atlas_coords[0], corrected_y, marker="X", s=100, color="r"
        )

    def _get_plot_views(
        self, atlas_coords: list[int]
    ) -> tuple[NDArray, NDArray, NDArray]:
        """Gets the sagittal, coronal, and axial views of the atlas at the
        specified coordinates.

        PARAMETERS
        ----------
        atlas_coords : list[int]
        -   Atlas coordinates, consisting of an x-, y-, and z-coordinate.

        RETURNS
        -------
        sagittal : numpy ndarray
        -   A 2D array containing the sagittal atlas view.

        coronal : numpy ndarray
        -   A 2D array containing the coronal atlas view.

        axial : numpy ndarray
        -   A 2D array containing the axial atlas view.
        """
        sagittal = np.rot90(self._plotting_atlas[atlas_coords[0], :, :])
        coronal = np.rot90(self._plotting_atlas[:, atlas_coords[1], :])
        axial = np.rot90(self._plotting_atlas[:, :, atlas_coords[2]])
        return sagittal, coronal, axial

    def _get_corrected_coords(self, atlas_coords: list[int]) -> tuple[int, int]:
        """Gets the corrected y- and z-coordinates used for plotting a marker of
        the coordinate location on the plots.

        PARAMETERS
        ----------
        atlas_coords : list[int]
        -   Atlas coordinates, consisting of an x-, y-, and z-coordinate.

        RETURNS
        -------
        corrected_y : int
        -   The y-coordinate, corrected for the size of the plotted y-axis.

        corrected_z : int
        -   The z-coordinate, corrected for the size of the plotted z-axis.
        """
        corrected_y = self._plotting_atlas.shape[1] - atlas_coords[1]
        corrected_z = self._plotting_atlas.shape[2] - atlas_coords[2]
        return corrected_y, corrected_z

    def _style_plot(
        self,
        fig: Figure,
        axes: list[plt.Axes],
        atlas_coords: list[int],
        mni_coords: list[int],
        region: str,
    ) -> None:
        """Adds additional information to the subplots.

        PARAMETERS
        ----------
        fig : matplotlib Figure
        -   The figure on which the subplots are plotted.

        axes : list[matplotlib pyplot Axes]
        -   A list of three axes to plot the sagittal, coronal, and axial view
            of the coordinates, respectively.

        atlas_coords : list[int]
        -   The coordinates in the atlas space.

        mni_coords : list[int]
        -   The coordinates in MNI space.

        region : str
        -   The name of the region corresponding to the coordinates.
        """
        self._add_title(fig, region, mni_coords, atlas_coords)
        self._remove_subplot_axes(axes)
        self._add_orientation_labels(fig, axes)

    def _add_orientation_labels(
        self, fig: Figure, axes: list[plt.Axes]
    ) -> None:
        """Adds labels to the sagittal, coronal, and axial views to show the
        orientation of the subplots.

        PARAMETERS
        ----------
        fig : matplotlib Figure
        -   The figure on which the subplots are plotted.

        axes : list[matplotlib pyplot Axes]
        -   A list of three axes to plot the sagittal, coronal, and axial view
            of the coordinates, respectively.
        """
        x_left, y_left, x_top, y_top = self._get_orientation_label_positions(
            axes
        )
        text_left = ["Posterior", "Left", "Left"]
        text_top = ["Dorsal", "Dorsal", "Anterior"]
        for axis_i in range(len(axes)):
            fig.text(
                x_left[axis_i],
                y_left,
                text_left[axis_i],
                verticalalignment="center",
                horizontalalignment="center",
            )  # text to the left of subplots
            fig.text(
                x_top[axis_i],
                y_top,
                text_top[axis_i],
                verticalalignment="center",
                horizontalalignment="center",
            )  # text above subplots

    def _get_orientation_label_positions(
        self, axes: list[plt.Axes]
    ) -> tuple[list[int], int, list[int], int]:
        """Gets the figure coordinates for the orientation labels that will be
        added to the figure.

        PARAMETERS
        ----------
        axes : list[matplotlib pyplot Axes]
        -   A list of three axes to plot the sagittal, coronal, and axial view
            of the coordinates, respectively.

        RETURNS
        -------
        x_left : list[int]
        -   Three x-coordinate figure positions, one for each subplot,
            corresponding to the left of each subplot.

        y_left : int
        -   The average y-coordinate figure position of the center of each
            subplot.

        x_top : list[int]
        -   Three x-coordinate figure positions, one for each subplot,
            corresponding to the center of each subplot.

        y_top : int
        -   The average y-coordinate figure position of the top of each subplot.
        """
        x_left = []
        y_left = []
        x_top = []
        y_top = []
        for axis in axes:
            bounds = axis.get_position().bounds
            x_left.append(bounds[0])  # x position
            y_left.append(
                bounds[1] + (bounds[3] / 2)
            )  # y position + half height
            x_top.append(bounds[0] + (bounds[2] / 2))  # x position + half width
            y_top.append(bounds[1] + bounds[3])  # y position + height
        y_left = np.mean(y_left)
        y_top = np.mean(y_top)
        return x_left, y_left, x_top, y_top + (y_top / 20)

    def _add_title(
        self,
        fig: Figure,
        region: str,
        mni_coords: list[int],
        atlas_coords: list[int],
    ) -> None:
        """Generates a title for the figure containing the MNI and atlas
        coordinates, and the corresponding region.

        PARAMETERS
        ----------
        fig : matplotlib Figure
        -   The figure on which the subplots are plotted.

        region : str
        -   The name of the region corresponding to the coordinates.

        mni_coords : list[int]
        -   The coordinates in MNI space.

        atlas_coords : list[int]
        -   The coordinates in the atlas space.
        """
        title = (
            f"Region: {region}     Atlas: {self.atlas_name}\nMNI coords.: "
            f"{mni_coords}     Atlas coords.: {atlas_coords}"
        )
        fig.suptitle(title)

    def _remove_subplot_axes(self, axes: list[plt.Axes]) -> None:
        """Removes the subplot axes from the figure.

        PARAMETERS
        ----------
        axes : list[matplotlib pyplot Axes]
        -   A list of three axes to plot the sagittal, coronal, and axial view
            of the coordinates, respectively.
        """
        for axis in axes:
            axis.axis("off")
