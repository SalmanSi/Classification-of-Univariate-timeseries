import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, precision_recall_fscore_support

# Load dataset
df = pd.read_csv('./Dataset_timeseries.csv')

# Data Preprocessing
# Handle missing values
df = df.dropna()

# Standardize data
scaler = StandardScaler()
df['SOPAS'] = scaler.fit_transform(df[['SOPAS']])

# Split data
X = df[['Time (s)', 'SOPAS']]
y = df['Number']  # Ensure this column has the class labels
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Model Selection and Training
model = RandomForestClassifier()
model.fit(X_train, y_train)

# Model Evaluation
y_pred = model.predict(X_val)
print(f'Accuracy: {accuracy_score(y_val, y_pred)}')
print(classification_report(y_val, y_pred))

# Calculate precision, recall, f1-score
precision, recall, f1, _ = precision_recall_fscore_support(y_val, y_pred, average=None)

# Plotting precision, recall, and f1-score
labels = np.unique(y_val)  # Get unique class labels

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
