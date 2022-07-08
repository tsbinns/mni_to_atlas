# mni_to_region
A simple tool for finding brain atlas regions based on MNI coordinates, with basic plotting abilities to show the sagittal, coronal, and axial views of the coordinates on the atlas.

Currently, the automated anatomical labelling (AAL) atlas [1] and AAL3 atlas [2] are supported.

![image](https://user-images.githubusercontent.com/56922019/178039475-998e077b-482f-4fbe-94af-88e1891b493b.png)

## Use Example:
1. Create an instance of the `AtlasBrowser` class and specify an atlas to use, e.g. `atlas = AtlasBrowser("AAL3")`.
2. Provide MNI coordinates to the `AtlasBrowser` object to find the corresponding atlas regions, e.g. `regions = atlas.find_regions(coordinates)`, where coordinates is a list of lists, where each nested list contains an x-, y-, and z-axis MNI coordinate. The `regions` output is a list of strings containing the region names for each set of coordinates.
   - By default, plotting the coordinates is not performed, however this can be changed by setting `plot = True` in the `find_regions` method, e.g. `atlas.find_regions(coordinates, plot=True)`. In this case, a figure will be generated for each set of coordinates.

## References:
1. [Tzourio-Mazoyer et al. (2002), NeuroImage, 10.1006/nimg.2001.0978](https://www.sciencedirect.com/science/article/pii/S1053811901909784).
2. [Rolls et al. (2020), NeuroImage, 10.1016/j.neuroimage.2019.116189](https://www.sciencedirect.com/science/article/pii/S1053811919307803).
