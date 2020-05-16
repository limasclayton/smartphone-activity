import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()

# ml
from sklearn.model_selection import train_test_split

# variables
dataset_path = "input/smartphone_activity_dataset.csv"
RANDOM_STATE = 0

# reading data
df = pd.read_csv(dataset_path)

# PREPROCESS
# FEATURE ENGENIRING
# FEATURE SELECTION
features = ['feature_316', 'feature_139', 'feature_156', 'feature_248', 'feature_321', 'feature_144', 'feature_357', 'feature_154', 'feature_41', 'feature_30', 'feature_103', 'feature_475', 'feature_378', 'feature_469', 'feature_34', 'feature_43', 'feature_359', 'feature_27', 'feature_99', 'feature_376', 'feature_318', 'feature_358']
X = df[features]
y = df.activity
X_train, x_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.3, random_state=RANDOM_STATE)

# MODEL

# METRICS