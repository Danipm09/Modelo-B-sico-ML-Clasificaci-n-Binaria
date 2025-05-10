import nltk
from nltk.corpus import movie_reviews
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score

# Descargar datos
nltk.download('movie_reviews')

# Preparar datos
reviews = [(movie_reviews.raw(fileid), category)
           for category in movie_reviews.categories()
           for fileid in movie_reviews.fileids(category)]

texts, labels = zip(*reviews)

# Convertir etiquetas a formato binario (0-negativo, 1-positivo)
labels_bin = [1 if label == 'pos' else 0 for label in labels]

# Dividir datos en entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(
    texts, labels_bin, test_size=0.2, random_state=42)

# Vectorización (Convertir textos en vectores numéricos)
vectorizer = CountVectorizer(stop_words='english', max_features=5000)
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# Entrenar el modelo
model = MultinomialNB()
model.fit(X_train_vec, y_train)

# Realizar predicciones
y_pred = model.predict(X_test_vec)

# Evaluar el modelo
accuracy = accuracy_score(y_test, y_pred)
print(f'Precisión del modelo: {accuracy * 100:.2f}%')
