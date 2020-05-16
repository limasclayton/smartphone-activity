import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()

from sklearn.preprocessing import LabelEncoder
# caminhos
dataset_path = "input/smartphone_activity_dataset.csv"

# lendo
df = pd.read_csv(dataset_path)

# initial EDA
# info(), head(), tail() e describe() from data
print(df.info())
print(df.shape)
#print(df.head())
#print(df.tail())
#print(df.describe())

# Correlation betweem features
corr = df.corr()
# drops the activity row, keeps the columns
corr.drop('activity', inplace=True)
#corr.to_csv('corr.csv')
#sns.heatmap(corr, xticklabels=corr.columns, yticklabels=corr.columns, annot=True, fmt='.2f')
#plt.show()

# Select features with corr > threshold and see if they are correlated betweem themselves
threshold_activity = 0.5
features = corr[abs(corr.activity) > threshold_activity].index.values
max_corr_feature = corr[(abs(corr.activity) == max(abs(corr.activity)))].index.values
#print(max_corr_feature)
#print(features)
corr2 = df[features].corr()
print(len(features))
#sns.heatmap(corr2, xticklabels=corr2.columns, yticklabels=corr2.columns, annot=True, fmt='.2f')
#plt.show()

threshold_features = 0.7
# Already selects the feature that is most correlated with activity
selected_features = set(max_corr_feature)
#print(selected_features)
print(selected_features)
# Select features with high correlation with target but not with each other
for f1 in features:
    correlations = [abs(corr2.loc[f1,f2]) for f2 in selected_features]
    if all(c < threshold_features for c in correlations):
        selected_features.add(f1)
        print(f1)    
    
print(selected_features)
print(len(selected_features))

corr_selected_features = df[selected_features].corr()
sns.heatmap(corr_selected_features, xticklabels=corr_selected_features.columns, yticklabels=corr_selected_features.columns, annot=True, fmt='.2f')
plt.show()

# Conclusions
# Aparrently some feature have high correlation with activity
# But they have high correlation between themselves
# Selecting features with > 0.5 correlation with target and < 0.7 between themselves