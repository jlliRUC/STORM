# For those large areas, you cannot download .osm file from OpenStreetMap directly.
# Instead, using osmosis to extract the area from larger (country-level) open-source file

import os


datasets = {
    'porto': {'country': 'portugal', 'boundaries': {'min_lat': 41.1128, 'max_lat': 41.1928, 'min_lon': -8.6868, 'max_lon': -8.5611}},
    'tdrive': {'country': 'china', 'boundaries': {'min_lat': 39.7542, 'max_lat': 40.0465, 'min_lon': 116.1509, 'max_lon': 116.7133}}
}


dataset_name = 'porto'
boundary = datasets[dataset_name]['boundaries']

# """
cmd = f"osmosis --read-pbf " \
      f"file=../data/{dataset_name}/meta/{datasets[dataset_name]['country']}-latest.osm.pbf " \
      f"--bounding-box top={boundary['max_lat']} left={boundary['min_lon']} bottom={boundary['min_lat']} right={boundary['max_lon']} " \
      f"--write-xml " \
      f"file=../data/{dataset_name}/meta/{dataset_name}.osm"
"""
cmd = f"/Users/jiali/osmosis-0.48.3/bin/osmosis --read-pbf " \
      f"file=../data/{dataset_name}/meta/{datasets[dataset_name]['country']}-latest.osm.pbf " \
      f"--bounding-box top={boundary['max_lat']} left={boundary['min_lon']} bottom={boundary['min_lat']} right={boundary['max_lon']} " \
      f"--write-xml " \
      f"file=../data/{dataset_name}/meta/{dataset_name}.osm"
"""
os.system(cmd)