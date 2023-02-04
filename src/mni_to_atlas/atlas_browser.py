"""A class for converting MNI coordinates to atlas regions."""

from os.path import join as pathjoin
import random
import json
import csv
import nibabel as nib
import numpy as np
from matplotlib.figure import Figure
from matplotlib import pyplot as plt
from mni_to_atlas import atlases


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
        set of coordinates does not correspond to a mapped region, "undefined"
        is returned.

    NOTES
    -----
    References:
        [1] Tzourio-Mazoyer et al. (2002), NeuroImage, 10.1006/nimg.2001.0978.
        [2] Rolls et al. (2020), NeuroImage, 10.1016/j.neuroimage.2019.116189.
    """

    ## Initialise class information
    supported_atlases = ["AAL", "AAL3"]
    _atlases_path = atlases.__path__[0]

    ### Initialise status attributes
    _plotting_ready = False

    ### Initialise attributes that will be filled with information
    _atlas = None
    _plotting_atlas = None
    _conversion = None
    _region_names = None
    _region_colours = None

    def __init__(self, atlas: str) -> None:
        ### Initialise inputs
        self.atlas_name = atlas

        ### Begin work
        self._check_atlas()
        self._load_data()

    def _check_atlas(self) -> None:
        """Checks that the requested atlas is supported.

        RAISES
        ------
        ValueError
        -   Raised if the requested atlas is not supported.
        """
        if self.atlas_name not in self.supported_atlases:
            raise ValueError(
                f"The requested atlas '{self.atlas_name}' is not recognised. "
                f"The supported atlases are {self.supported_atlases}."
            )

    def _load_data(self) -> None:
        """Loads the atlas and accompanying information into the object."""
        self._path = pathjoin(self._atlases_path, self.atlas_name)
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
        self, coordinates: np.ndarray, plot: bool = False
    ) -> list[str]:
        """Finds the regions associated with MNI coordinates for the atlas. If a
        set of coordinates does not correspond to a mapped region, "undefined"
        is returned.

        PARAMETERS
        ----------
        coordinates : numpy ndarray
        -   MNI coordinates (in mm) to find the associated atlas regions for,
            given as an [n x 3] matrix where n is the number of coordinates to
            find regions for, and 3 the x-, y-, and z-axis coordinates,
            respectively.

        plot : bool; default False
        -   Whether or not to plot a sagittal, coronal, and axial view of the
            coordinate on the atlas.

        RETURNS
        -------
        regions : list[str]
        -   Names of the regions in the atlas corresponding to the MNI
            coordinates.
        """
        coordinates = self._check_coordinates(coordinates)
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

    def _check_coordinates(self, coordinates: np.ndarray) -> np.ndarray:
        """Checks that each set of coordinates contains three values,
        corresponding to the x-, y-, and z-coordinates.

        PARAMETERS
        ----------
        coordinates : numpy ndarray
        -   An [n x 3] or [3, 0] array where n is the number of coordinates to
            find regions for, and 3 the x-, y-, and z-axis coordinates,
            respectively.
        
        RETURNS
        -------
        coordinates : numpy ndarray
        -   An [n x 3] array where n is the number of coordinates to find
            regions for, and 3 the x-, y-, and z-axis coordinates, respectively.

        RAISES
        ------
        ValueError
        -   Raised if 'coordinates' is not a numpy ndarray.
        -   Raised if 'coordinates' has more than two dimensions.
        -   Raised if the second dimension of 'coordinates' does not have a
            length of 3.
        """
        if not isinstance(coordinates, np.ndarray):
            raise TypeError("'coordinates' should be a numpy ndarray.")
        
        if coordinates.ndim == 1 and coordinates.shape == (3,):
            coordinates = coordinates.copy()[np.newaxis, :]

        if coordinates.ndim != 2:
            raise ValueError(
                "'coordinates' should have two dimensions, but it has "
                f"{coordinates.ndim} dimensions."
            )

        if coordinates.shape[1] != 3:
            raise ValueError(
                "'coordinates' should be an [n x 3] array, but it is an "
                f"[n x {coordinates.shape[1]}] array.")
        
        return coordinates

    def _prepare_plotting(self) -> None:
        """Prepares the atlas for being plotted."""
        plotting_atlas = self._atlas.copy()
        plotting_atlas[plotting_atlas == 0] = np.nan  # provides a white...
        # background to the empty regions of the plot
        self._plotting_atlas = self._assign_colour_ids(plotting_atlas)
        self._plotting_ready = True

    def _assign_colour_ids(self, atlas: np.ndarray) -> np.ndarray:
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
        self, atlas: np.ndarray, transformation: dict
    ) -> np.ndarray:
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

    def _convert_coords(self, mni_coords: np.ndarray) -> np.ndarray:
        """Converts MNI coordinates (in mm) to atlas coordinates.

        PARAMETERS
        ----------
        mni_coords : numpy ndarray
        -   MNI coordinates (in mm) as integers, consisting of an x-, y-, and
            z-coordinate.

        RETURNS
        -------
        atlas_coords : numpy ndarray
        -   Atlas coordinates consiting of an x-, y-, and z-coordinate.

        NOTES
        -----
        -   The atlas only accepts coordinates in integers, and so MNI
            coordinates must be rounded in the case of non-integer values before
            finding the corresponding region.
        """
        return mni_coords + self._conversion

    def _round_coords(self, coords: np.ndarray) -> np.ndarray:
        """Rounds coordinates to the nearest integer.

        PARAMETERS
        ----------
        coords : numpy ndarray
        -   A [1 x 3] array, consisting of an x-, y-, and z-coordinate.

        RETURNS
        -------
        rounded_coords : numpy ndarray
        -   An array of length three, consisting of an x-, y-, and z-coordinate
            rounded to their nearest integer.
        """
        rounded_coords = np.zeros((3,), dtype=int)
        for coord_i, coord in enumerate(coords):
            if coord < 0:
                rounded = int(coord - 0.5)
            else:
                rounded = int(coord + 0.5)
            rounded_coords[coord_i] = rounded

        return rounded_coords

    def _find_region(self, atlas_coords: np.ndarray) -> str:
        """Finds the name of the region in the atlas belonging to a set of
        coordinates. If the coordinates do not correspond to a mapped region,
        'undefined' is returned.

        PARAMETERS
        ----------
        atlas_coords : numpy ndarray
        -   A [1 x 3] array, consisting of an x-, y-, and z-coordinate.

        RETURNS
        -------
        region : str
        -   The name of the region corresponding to the coordinates.
        """
        region_id = int(self._atlas[*atlas_coords])
        if region_id != 0:
            region = self._region_names[region_id]
        else:
            region = "undefined"

        return region

    def _plot(
        self, atlas_coords: np.ndarray, mni_coords: np.ndarray, region: str
    ) -> None:
        """Plots coordinates on the atlas in the sagittal, coronal, and axial
        views.

        PARAMETERS
        ----------
        atlas_coords : numpy ndarray
        -   Atlas coordinates, consisting of an x-, y-, and z-coordinate.

        mni_coords : numpy ndarray
        -  The atlas coordinates in MNI space.

        region : str
        -   The name of the region corresponding to the coordinates.
        """
        fig, axes = plt.subplots(1, 3)
        self._plot_views(axes, atlas_coords)
        self._style_plot(fig, axes, atlas_coords, mni_coords, region)
        plt.show()

    def _plot_views(
        self, axes: list[plt.Axes], atlas_coords: np.ndarray
    ) -> None:
        """Plots a sagittal, coronal, and axial view of atlas coordinates on a
        set of axes.

        PARAMETERS
        ----------
        axes : list[matplotlib pyplot Axes]
        -   A list of three axes to plot the sagittal, coronal, and axial view
            of the coordinates, respectively.

        atlas_coords : numpy ndarray
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
        self, atlas_coords: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Gets the sagittal, coronal, and axial views of the atlas at the
        specified coordinates.

        PARAMETERS
        ----------
        atlas_coords : numpy ndarray
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

    def _get_corrected_coords(
        self, atlas_coords: np.ndarray
    ) -> tuple[int, int]:
        """Gets the corrected y- and z-coordinates used for plotting a marker of
        the coordinate location on the plots.

        PARAMETERS
        ----------
        atlas_coords : numpy ndarray
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
        atlas_coords: np.ndarray,
        mni_coords: np.ndarray,
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

        atlas_coords : numpy ndarray
        -   The coordinates in the atlas space.

        mni_coords : numpy ndarray
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
    ) -> tuple[np.ndarray, float, np.ndarray, float]:
        """Gets the figure coordinates for the orientation labels that will be
        added to the figure.

        PARAMETERS
        ----------
        axes : list[matplotlib pyplot Axes]
        -   A list of three axes to plot the sagittal, coronal, and axial view
            of the coordinates, respectively.

        RETURNS
        -------
        x_left : numpy ndarray
        -   Three x-coordinate figure positions, one for each subplot,
            corresponding to the left of each subplot.

        y_left : float
        -   The average y-coordinate figure position of the center of each
            subplot.

        x_top : numpy ndarray
        -   Three x-coordinate figure positions, one for each subplot,
            corresponding to the center of each subplot.

        y_top : float
        -   The average y-coordinate figure position of the top of each subplot.
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
            x_left[axis_i] = bounds[0] - bounds[0] * x_shift # x position
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
        fig: Figure,
        region: str,
        mni_coords: np.ndarray,
        atlas_coords: np.ndarray,
    ) -> None:
        """Generates a title for the figure containing the MNI and atlas
        coordinates, and the corresponding region.

        PARAMETERS
        ----------
        fig : matplotlib Figure
        -   The figure on which the subplots are plotted.

        region : str
        -   The name of the region corresponding to the coordinates.

        mni_coords : numpy ndarray
        -   The coordinates in MNI space.

        atlas_coords : numpy ndarray
        -   The coordinates in the atlas space.
        """
        title = (
            f"Region: {region}     Atlas: {self.atlas_name}\nMNI coords.: "
            f"{mni_coords.tolist()}     Atlas coords.: {atlas_coords.tolist()}"
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
