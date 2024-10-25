import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.feature_extraction.text import CountVectorizer

# Load dataset
df = pd.read_csv(r''C:\Users\Lenovo\Downloads\python\financial market\Financial Market News.csv', encoding="ISO-8859-1")

# Check the dataset
df.info()  # Ensure the columns are correct and the data is loaded

# Combine text columns from index 2 to 27 into a single string for each row
news = []
for row in range(len(df.index)):
    news.append(' '.join(str(x) for x in df.iloc[row, 2:27]))  # Use row index properly

# Check the type and content of 'news'
print(type(news))
print(news[0])  # Check the first entry to ensure it's collected correctly

# Feature extraction using CountVectorizer
cv = CountVectorizer(lowercase=True, ngram_range=(1, 1))  # You can modify ngram_range if needed
X = cv.fit_transform(news)

# Target variable
y = df['Label']

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, stratify=y, random_state=2529)

# Initialize and train RandomForestClassifier
rf = RandomForestClassifier(n_estimators=200)
rf.fit(X_train, y_train)

# Make predictions on the test set
y_pred = rf.predict(X_test)

# Evaluate the model
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))
print("\nClassification Report:")
print(classification_report(y_test, y_pred))
