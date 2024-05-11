import pandas as pd
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import RandomOverSampler
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.regularizers import l2
from tensorflow.keras.metrics import Precision, Recall
from tensorflow.keras.callbacks import LearningRateScheduler
import math

def scheduler(epoch, lr):
    if epoch < 10:
        return lr * math.exp(-0.1)
    else:
        return lr * math.exp(-0.2)  # More aggressive reduction

lr_scheduler = LearningRateScheduler(scheduler)

# Load data
df = pd.read_csv("../data/heart_2022_transformed_with_outliers.csv")
print(df.head())

# Split data into features and target
X = df.drop('HadHeartDisease', axis=1)
y = df['HadHeartDisease']

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Oversample training data
ros = RandomOverSampler(random_state=42)
X_train_resampled, y_train_resampled = ros.fit_resample(X_train, y_train)

# Model definition
model = Sequential([
    Dense(128, activation='relu', input_shape=(X_train_resampled.shape[1],)),
    BatchNormalization(),
    Dropout(0.1),
    Dense(128, activation='relu'),
    BatchNormalization(),
    Dropout(0.1),
    Dense(1, activation='sigmoid')
])

optimizer = tf.keras.optimizers.Adam(
    learning_rate=0.001,
    beta_1=0.9,
    beta_2=0.999,
    epsilon=1e-07
)
model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy', Recall(), Precision()])



# Callbacks for learning rate adjustment and early stopping
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=0.00001, verbose=1)
early_stopping = EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True, verbose=1)


# Training the model
callbacks = [reduce_lr, early_stopping, lr_scheduler]
history = model.fit(X_train_resampled, y_train_resampled, epochs=100,
                    validation_data=(X_test, y_test),
                    callbacks=callbacks, batch_size=64, verbose=1)