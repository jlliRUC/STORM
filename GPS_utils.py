# GPS utils

import numpy as np
import random
import math
from datetime import datetime
from pytz import timezone
from itertools import tee
random.seed(2023)

R_EARTH = 6371000  # unit: meter

tz_utc = timezone("UTC")  # Time convert


def pairwise(iterable):
    a, b = tee(iterable)
    next(b, None)
    return zip(a, b)


def str2ts(timestring, format, tz=tz_utc):
    return datetime.timestamp(datetime.strptime(timestring, format).replace(tzinfo=tz))


def ts2str(timestamp, format, tz=tz_utc):
    return datetime.fromtimestamp(timestamp).astimezone(tz).strftime(format)


def lonlat2meters(lon, lat):
    """
    Convert location point from GPS coordinate to meters
    :param lon:
    :param lat:
    :return:
    """
    semimajoraxis = 6378137.0
    east = lon * 0.017453292519943295
    north = lat * 0.017453292519943295
    t = np.sin(north)
    return semimajoraxis * east, 3189068.5 * math.log((1 + t) / (1 - t))


def meters2lonlat(x, y):
    """
    Convert location point from meters to GPS coordinate
    :param lon:
    :param lat:
    :return:
    """
    semimajoraxis = 6378137.0
    lon = x / semimajoraxis / 0.017453292519943295
    t = math.exp(y / 3189068.5)
    lat = math.asin((t - 1) / (t + 1)) / 0.017453292519943295
    #import pyproj
    #proj = pyproj.Transformer.from_crs(3857, 4326, always_xy=True)
    #lon, lat = proj.transform(x, y)
    return lon, lat


def haversine(lon1, lat1, lon2, lat2):
    lon1, lat1, lon2, lat2 = map(math.radians, [lon1, lat1, lon2, lat2])
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = math.sin(dlat / 2.0) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2.0) ** 2
    c = 2 * math.asin(math.sqrt(a))
    return round(c * R_EARTH, 4)


def radian(lon1, lat1, lon2, lat2):
    """
    The radian of one segment
    :param lon1:
    :param lat1:
    :param lon2:
    :param lat2:
    :return:
    """
    dy = lat2 - lat1
    dx = lon2 - lon1
    r = 0.0
    if dx == 0:
        if dy >= 0:
            r = 1.5707963267948966  # math.pi / 2
        else:
            r = 4.71238898038469  # math.pi * 1.5
        return round(r, 3)

    r = math.atan(dy / dx)
    # angle_in_degrees = math.degrees(angle_in_radians)
    if dx < 0:
        r = r + 3.141592653589793
    else:
        if dy < 0:
            r = r + 6.283185307179586
        else:
            pass
    return round(r, 4)


def angle2radian(angle):
    """
    convert longitude/latitude from an angle to a radian
    :param angle: (float)
    :return: radian (float)
    """
    return math.radians(angle)


def radian2angle(radian):
    """
    convert longitude/latitude from a radian to an angle
    :param radian:
    :return:
    """

    return math.degrees(radian)


def get_city_boundaries(city_name):
    """
    For those cities that cannot get the boundaries easily from the Internet, use this function.
    Note that the obtained boundaries seem be the central area of the old days, which may shrink the domain.
    :param city_name:
    :return:
    """

    from geopy.geocoders import Nominatim
    geolocator = Nominatim(user_agent="city_boundaries")
    location = geolocator.geocode(city_name, exactly_one=True)

    if location:
        # min_lat, max_lat, min_lon, max_lon
        return location.raw.get('boundingbox', None)
    else:
        return None


def init_bearing(phi1, lambda1, phi2, lambda2):
    """
    initial bearing of a great circle route
    :return: 0~360
    """
    y = math.sin(lambda2 - lambda1) * math.cos(phi2)
    x = math.cos(phi1) * math.sin(phi2) - math.sin(phi1) * math.cos(phi2) * math.cos(lambda2 - lambda1)
    theta = math.atan2(y, x)
    brng = (theta * 180 / math.pi + 360) % 360
    return brng


if __name__ == "__main__":
    lon1 = 20.3
    lat1 = 40.333
    lon2 = 15.66
    lat2 = 33.25
    print(haversine(lon1, lat1, lon2, lat2))

