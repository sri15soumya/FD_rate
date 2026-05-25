from flask import Flask, request, jsonify, render_template
from rag import initailize_rag, get_fd_answer

app = Flask(__name__)

df = initailize_rag()

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/query', methods=['POST'])  # Changed from '/ask' to '/query' to match your HTML
def query():
    data = request.get_json()
    query = data.get("query", "")
    if not query.strip():
        return jsonify({"error": "query is required"}), 400
    answer = get_fd_answer(df, query)
    return jsonify({"answer": answer})

if __name__ == '__main__':
    print("The backend is running on port 5000")
    app.run(debug=True)
