from atlas_browser import AtlasBrowser
import numpy as np

atlas = AtlasBrowser("AAL")

regions = atlas.find_regions(np.array([[30, 0, 60], [35, 0, 55]]), plot=True)