import sys
import os
sys.path.append(os.path.abspath(".."))
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from src.data import load_raw_data, TARGET_COL

train, test, sample = load_raw_data()
print(train.head())
print(train[TARGET_COL].value_counts())

sns.countplot(train[TARGET_COL])
plt.show()
