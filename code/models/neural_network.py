import math
import pandas as pd
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import RandomOverSampler
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.metrics import Precision, Recall
from tensorflow.keras.callbacks import LearningRateScheduler
from sklearn.metrics import classification_report


# Here we make a custom learning rate scheduler function:
def scheduler(epoch, lr):
    if epoch < 10:
        return lr * math.exp(-0.1)
    else:
        return lr * math.exp(-0.2)  # More aggressive reduction after 10 epochs.


# Initializing the learning rate scheduler:
lr_scheduler = LearningRateScheduler(scheduler)

# Loading the dataset:
df = pd.read_csv("../../data/processed/heart_2022_transformed.csv")
df_nn = df.copy()
df_nn = df_nn.replace({"Yes": 1, "No": 0})

# Split data into features and target variables:
X = df_nn.drop("HadHeartDisease", axis=1)
y = df_nn["HadHeartDisease"]

# Splitting the dataset into training and test sets:
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Applying Random Oversampling to the training data:
ros = RandomOverSampler(random_state=42)
X_train_resampled, y_train_resampled = ros.fit_resample(X_train, y_train)

# Defining the structure of the neural network:
model = Sequential(
    [
        Dense(128, activation="relu", input_shape=(X_train_resampled.shape[1],)),
        # Adjusts the outputs of the previous layer to stabilize
        BatchNormalization(),
        # Randomly ignores 10% of the neurons during training - to reduce overfitting.
        Dropout(0.1),
        Dense(128, activation="relu"),
        BatchNormalization(),
        Dropout(0.1),
        Dense(1, activation="sigmoid"),
    ]
)

# Compiling the model with Adam optimizer and setting the loss and metrics:
optimizer = tf.keras.optimizers.Adam(
    learning_rate=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-07
)
model.compile(
    optimizer=optimizer,
    loss="binary_crossentropy",
    metrics=["accuracy", Recall(), Precision()],
)

# Initializing callbacks for reducing learning rate and early stopping:
reduce_lr = ReduceLROnPlateau(
    monitor="val_loss", factor=0.2, patience=5, min_lr=0.00001, verbose=1
)
early_stopping = EarlyStopping(
    monitor="val_loss", patience=15, restore_best_weights=True, verbose=1
)

# Training the model with specified callbacks and hyperparameters:
callbacks = [reduce_lr, early_stopping, lr_scheduler]
history = model.fit(
    X_train_resampled,
    y_train_resampled,
    epochs=5,
    validation_data=(X_test, y_test),
    callbacks=callbacks,
    batch_size=64,
    verbose=1,
)

y_pred = (model.predict(X_test) > 0.5).astype("int32")

# Generate the classification report
report = classification_report(
    y_test, y_pred, target_names=["No Heart Disease", "Heart Disease"]
)
print(report)

# Save the TensorFlow model to a directory
# model.save("../models/saved_models/model_tensorflow.keras")


""" All results can be found inside the directory "./results/model_results.py". """
