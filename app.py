from flask import Flask, request, jsonify
from flask_cors import CORS
import pickle
import numpy as np
import os

# Cargar el modelo y los datos
model = pickle.load(open('artifacts/model.pkl', 'rb'))
book_names = pickle.load(open('artifacts/book_names.pkl', 'rb'))
final_rating = pickle.load(open('artifacts/final_rating.pkl', 'rb'))
book_pivot = pickle.load(open('artifacts/book_pivot.pkl', 'rb'))
books = pickle.load(open('artifacts/books.pkl', 'rb'))  # Asegúrate de cargar todos los datos de BX-Books

# Inicializar Flask
app = Flask(__name__)
CORS(app)  # Permitir CORS


# Endpoint para obtener recomendaciones
@app.route('/recommend', methods=['GET'])
def recommend():
    book_name = request.args.get('book_name')
    if not book_name:
        return jsonify({"error": "Missing 'book_name' parameter"}), 400
    
    try:
        # Buscar el índice del libro
        book_id = np.where(book_pivot.index == book_name)[0][0]
        distance, suggestion = model.kneighbors(book_pivot.iloc[book_id, :].values.reshape(1, -1), n_neighbors=6)
        
        # Generar la lista de recomendaciones
        recommendations = []
        for i in range(1, len(suggestion[0])):  # Omitimos el libro buscado (posición 0)
            recommended_book_title = book_pivot.index[suggestion[0][i]]
            book_info = final_rating[final_rating['title'] == recommended_book_title].iloc[0]
            
            # Buscar información completa del libro en el dataset original
            full_book_info = books[books['title'] == recommended_book_title].iloc[0].to_dict()
            recommendations.append(full_book_info)
        
        return jsonify({
            "searched_book": book_name,
            "recommendations": recommendations
        })
    except IndexError:
        return jsonify({"error": f"Book '{book_name}' not found in the dataset"}), 404


@app.route('/search', methods=['GET'])
def search_books():
    query = request.args.get('query', '').lower()  # Obtener el texto de búsqueda y convertirlo a minúsculas
    if not query:
        return jsonify({"error": "Missing 'query' parameter"}), 400

    try:
        # Filtrar los libros que coinciden con la consulta en el título
        filtered_books = books[books['title'].str.lower().str.contains(query, na=False)]
        
        # Limitar los resultados a 5 libros
        results = filtered_books.head(5).to_dict(orient='records')

        return jsonify({
            "query": query,
            "results": results
        })
    except Exception as e:
        return jsonify({"error": "An error occurred while searching for books", "details": str(e)}), 500


# Ruta para verificar el estado de la API
@app.route('/status', methods=['GET'])
def status():
    return jsonify({"status": "API is running"}), 200

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=int(os.environ.get("PORT", 5000)))