#Spam Detection (Naive Bayes)
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
texts = [
 "I love sunny days",
 "Free money now",
 "Earn money fast",
 "I hate rainy days",
 "I enjoy sunny and warm weather",
 "Fast cash now",
 "Win money instantly",
 "The sun is bright",
 "Warm weather makes me happy",
 "Get rich quick"
]
labels = ["ham", "spam", "spam", "ham", "ham", "spam", "spam", "ham", "ham", "spam"]
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(texts)
X_train, X_test, y_train, y_test = train_test_split(X, labels, test_size=0.3, random_state=42)
model = MultinomialNB()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy}")
print("Classification Report:\n", classification_report(y_test, y_pred))
new_text = ["Win money with this new opportunity"]
new_text_transformed = vectorizer.transform(new_text)
prediction = model.predict(new_text_transformed)

print("Prediction for new text:", prediction)
