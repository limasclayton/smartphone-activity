import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()

# ml
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split, RandomizedSearchCV

# tensorflow
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.callbacks import ModelCheckpoint, EarlyStopping

# variables
dataset_path = "input/smartphone_activity_dataset.csv"
RANDOM_STATE = 0

# reading data
df = pd.read_csv(dataset_path)

# PREPROCESSING
# selecting only float type features and scaling them
columns = df.select_dtypes(include=['float']).columns
print(columns)
for c in columns:
    df[c] = StandardScaler().fit_transform(df[c].values.reshape(-1,1))

# transforming the target in labels for sparse categorical loss
df.activity = LabelEncoder().fit_transform(df.activity)

# FEATURE SELECTION
X = df.drop(['activity'], axis=1)
y = df['activity']
#X_dummies = pd.get_dummies(X)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=RANDOM_STATE)
print(X_train.shape, y_train.shape)
print(X_test.shape, y_test.shape)
print(y.unique())
print(len(y.unique()))
print(type(y))

# callbalcks
callback = EarlyStopping(monitor='val_loss', patience=3)

# MODEL
model = Sequential()
model.add(Dense(128, activation='relu', input_shape=(561,)))
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.25))
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.25))
model.add(Dense(6, activation='softmax'))

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.summary()

model.fit(X_train, y_train, epochs=50, validation_split=0.1, callbacks=[callback])

model.evaluate(X_test, y_test)