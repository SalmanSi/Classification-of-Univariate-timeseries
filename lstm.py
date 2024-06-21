import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, precision_recall_fscore_support
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.utils import to_categorical

# Load dataset
df = pd.read_csv('Dataset_timeseries.csv')

# Data Preprocessing
# Handle missing values
df = df.dropna()

# Standardize data
scaler = StandardScaler()
df['SOPAS'] = scaler.fit_transform(df[['SOPAS']])

# Assuming 'Number' column contains the class labels
# Convert class labels to categorical (one-hot encoding)
y = to_categorical(df['Number'])

# Reshape input data for LSTM (samples, timesteps, features)
# Here each time series is treated as a separate sample
X = df['SOPAS'].values.reshape(-1, 1, 1)  # -1 infers the number of samples, 1 timestep, 1 feature

# Split data
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Model Definition
model = Sequential()
model.add(LSTM(50, activation='relu', input_shape=(1, 1), return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(50, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(4, activation='softmax'))  # Assuming 4 classes

# Compile model with a smaller learning rate and add callbacks
optimizer = Adam(learning_rate=0.001)
model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

# Callbacks for early stopping and reducing learning rate
early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=0.0001)

# Model Training
history = model.fit(X_train, y_train, epochs=100, batch_size=32, validation_data=(X_val, y_val),
                    callbacks=[early_stopping, reduce_lr])

# Model Evaluation
y_pred = model.predict(X_val)
y_pred_classes = np.argmax(y_pred, axis=1)
y_val_classes = np.argmax(y_val, axis=1)

print(f'Accuracy: {accuracy_score(y_val_classes, y_pred_classes)}')
print(classification_report(y_val_classes, y_pred_classes))

# Calculate precision, recall, f1-score
precision, recall, f1, _ = precision_recall_fscore_support(y_val_classes, y_pred_classes, average=None)

# Plotting precision, recall, and f1-score
labels = np.unique(y_val_classes)  # Get unique class labels

plt.figure(figsize=(14, 6))

# Precision
plt.subplot(1, 3, 1)
sns.barplot(x=labels, y=precision)
plt.title('Precision per Class')
plt.xlabel('Class')
plt.ylabel('Precision')

# Recall
plt.subplot(1, 3, 2)
sns.barplot(x=labels, y=recall)
plt.title('Recall per Class')
plt.xlabel('Class')
plt.ylabel('Recall')

# F1 Score
plt.subplot(1, 3, 3)
sns.barplot(x=labels, y=f1)
plt.title('F1 Score per Class')
plt.xlabel('Class')
plt.ylabel('F1 Score')

plt.tight_layout()
plt.show()
