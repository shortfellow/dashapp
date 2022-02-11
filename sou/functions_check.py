# -*- coding: utf-8 -*-
"""
Created on Wed Jan  5 11:30:25 2022

@author: Krithika 
"""


import sys
from pathlib import Path
path_root = Path(__file__).parents[2]
sys.path.append(str(path_root))


# Reading array info, general info, inverter data, weather data

                               

horizon_list = [
    {"label": "7 days", "value": "7"},
    {"label": "15 days", "value": "15"},
    {"label": "30 days", "value": "30"},
]

