'''
This script is used to simulate sequencing data for a given set of variants.
'''

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from tqdm import tqdm
from multiprocessing import Pool, cpu_count
from numpy.random import default_rng
import math