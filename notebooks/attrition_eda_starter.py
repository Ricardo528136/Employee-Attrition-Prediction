# Imports & Settings
import os
import sys
import json
import textwrap
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

try:
    import plotly.express as px
    _HAS_PLOTLY = True
except Exception:
    _HAS_PLOTLY = False

from scipy import stats
from sklearn.preprocessing import LabelEncoder

warnings.filterwarnings("ignore", category=FutureWarning)
pd.set_option('display.max_columns', 100)

# Ensure the script is run from the correct directory -- EDIT THESE PATHS
# Path to the CSV file
CSV_PATH = Path(data\WA_Fn-UseC_-HR-Employee-Attrition.csv)
if not CSV_PATH.exists():
    print(f"CSV file not found at {CSV_PATH}. Please check the path.")
    sys.exit(1)

