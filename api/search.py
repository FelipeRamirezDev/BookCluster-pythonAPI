import pickle
from flask import jsonify, request
import os

# Cargar los libros
books = pickle.load(open('artifacts/books.pkl', 'rb'))

# Endpoint de búsqueda
def handler(request):
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
