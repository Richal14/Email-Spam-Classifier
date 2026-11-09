
import pandas as pd
import pickle

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# 1. Load dataset
data = pd.read_csv(
    "dataset/spam.csv",
    encoding="latin-1"
)

# 2. Select required columns
data = data[["v1", "v2"]]
data.columns = ["label", "message"]

# 3. Remove missing values
data = data.dropna(subset=["label", "message"])

# 4. Convert labels into numbers
data["label"] = data["label"].map({
    "ham": 0,
    "spam": 1
})

# 5. Separate input and output
X = data["message"]
y = data["label"]

# 6. Split dataset into training and testing
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# 7. Convert text into numerical features
vectorizer = TfidfVectorizer(
    stop_words="english",
    lowercase=True
)

X_train = vectorizer.fit_transform(X_train)
X_test = vectorizer.transform(X_test)

# 8. Train Naive Bayes model
model = MultinomialNB()
model.fit(X_train, y_train)

print("Model training completed!")

# 9. Make predictions
y_pred = model.predict(X_test)

# 10. Evaluate model
print("\nAccuracy:")
print(accuracy_score(y_test, y_pred) * 100, "%")

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# 11. Save model
with open("spam_model.pkl", "wb") as file:
    pickle.dump(model, file)

# 12. Save vectorizer
with open("vectorizer.pkl", "wb") as file:
    pickle.dump(vectorizer, file)

print("\nModel saved successfully!")
