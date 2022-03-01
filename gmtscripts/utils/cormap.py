"""
cormap.py

Helper functions to handle GMT axis ticks and coordinate mapping.
"""
from typing import Tuple, List

import numpy as np
import pyproj
from scipy.spatial import KDTree
from numpy.typing import NDArray


def gmt_project(startlon: float, startlat: float, endlon: float, endlat: float, thetype: str, npts: int = 1001) -> Tuple[NDArray]:
    """Given two ends of the great circle line, generate evenly distributed points. When use dist as the input type, similar to pygmt.project.
    When use lon or lat, the points are still along the great circle, but either lon or lat will be evently distributed.

    Args:
        startlon (float): starting longitude
        startlat (float): starting latitude
        endlon (float): ending longitude
        endlat (float): ending latitude
        thetype (str): type of the even interpolation, including dist, lon, and lat.
        npts (int, optional): number of returned points. Defaults to 1001.

    Raises:
        Exception: The Input type {thetype} is not supported! Supported types include dist, lon, and lat. Use any of the supported inputs.

    Returns:
        _type_: (Tuple[np.ndarray]): two element tuple of numpy array, as (lon,lat)
    """
    g = pyproj.Geod(ellps='WGS84')
    if(thetype == "dist"):
        result = g.npts(startlon, startlat, endlon, endlat, npts)
        result = np.array(result)
        # result_lons, result_lats
        return result[:, 0], result[:, 1]
    elif(thetype == "lon"):
        # we divide 10 more times points using npts and find nearest lon
        test_points = g.npts(startlon, startlat, endlon, endlat, (npts-1)*10+1)
        test_points = np.array(test_points)
        tree = KDTree(test_points[:, 0].reshape(test_points.shape[0], -1))
        # evenly distributed lons
        result_lons = np.linspace(startlon, endlon, npts)
        _, pos = tree.query(result_lons.reshape(npts, -1))
        result_lats: NDArray = test_points[:, 1][pos]
        return result_lons, result_lats
    elif(thetype == "lat"):
        # we divide 10 more times points using npts and find nearest lat
        test_points = g.npts(startlon, startlat, endlon, endlat, (npts-1)*10+1)
        test_points = np.array(test_points)
        tree = KDTree(test_points[:, 1].reshape(test_points.shape[0], -1))
        # evenly distributed lons
        result_lats = np.linspace(startlat, endlat, npts)
        _, pos = tree.query(result_lats.reshape(npts, -1))
        result_lons: NDArray = test_points[:, 0][pos]
        return result_lons, result_lats
    else:
        raise Exception(
            f"The Input type {thetype} is not supported! Supported types include dist, lon, and lat. Use any of the supported inputs.")
