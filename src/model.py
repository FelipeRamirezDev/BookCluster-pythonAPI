import pandas as pd
import numpy as np
import pickle
from scipy.sparse import csr_matrix
from sklearn.neighbors import NearestNeighbors

# Cargar y limpiar datos
books = pd.read_csv('data/BX-Books.csv', sep=";", error_bad_lines=False, encoding='latin-1')
books = books[['ISBN', 'Book-Title', 'Book-Author', 'Year-Of-Publication', 'Publisher', 'Image-URL-L']]
books.rename(columns={
    "Book-Title": 'title',
    'Book-Author': 'author',
    "Year-Of-Publication": 'year',
    "Publisher": "publisher",
    "Image-URL-L": "image_url"
}, inplace=True)

# Guardar como pickle para usarlo en la API
books.to_pickle('artifacts/books.pkl')

users = pd.read_csv('data/BX-Users.csv', sep=";", error_bad_lines=False, encoding='latin-1')
users.rename(columns={"User-ID": 'user_id', 'Location': 'location', "Age": 'age'}, inplace=True)

ratings = pd.read_csv('data/BX-Book-Ratings.csv', sep=";", error_bad_lines=False, encoding='latin-1')
ratings.rename(columns={"User-ID": 'user_id', 'Book-Rating': 'rating'}, inplace=True)

# Filtrar usuarios con más de 200 calificaciones
active_users = ratings['user_id'].value_counts() > 200
active_users = active_users[active_users].index
ratings = ratings[ratings['user_id'].isin(active_users)]

# Combinar ratings con libros
ratings_with_books = ratings.merge(books, on='ISBN')

# Filtrar libros con al menos 50 calificaciones
number_rating = ratings_with_books.groupby('title')['rating'].count().reset_index()
number_rating.rename(columns={'rating': 'num_of_rating'}, inplace=True)
final_rating = ratings_with_books.merge(number_rating, on='title')
final_rating = final_rating[final_rating['num_of_rating'] >= 10]
final_rating.drop_duplicates(['user_id', 'title'], inplace=True)

# Crear tabla pivote
book_pivot = final_rating.pivot_table(columns='user_id', index='title', values='rating')
book_pivot.fillna(0, inplace=True)

# Modelo de entrenamiento
book_sparse = csr_matrix(book_pivot)
model = NearestNeighbors(algorithm='brute')
model.fit(book_sparse)

# Guardar modelo y datos procesados
pickle.dump(model, open('artifacts/model.pkl', 'wb'))
pickle.dump(book_pivot.index, open('artifacts/book_names.pkl', 'wb'))
pickle.dump(final_rating, open('artifacts/final_rating.pkl', 'wb'))
pickle.dump(book_pivot, open('artifacts/book_pivot.pkl', 'wb'))

# Función para recomendar libros
def recommend_book(book_name):
    book_id = np.where(book_pivot.index == book_name)[0][0]
    distance, suggestion = model.kneighbors(book_pivot.iloc[book_id, :].values.reshape(1, -1), n_neighbors=6)
    
    print(f"You searched '{book_name}'\n")
    print("The suggested books are:\n")
    for i in range(1, len(suggestion[0])):  # Empezamos en 1 para omitir el libro buscado
        print(book_pivot.index[suggestion[0][i]])

# Ejemplo de recomendación
book_name = "Harry Potter and the Chamber of Secrets (Book 2)"
recommend_book(book_name)
