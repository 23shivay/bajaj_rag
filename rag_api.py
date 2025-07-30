from flask import Flask, request, jsonify
from flask_cors import CORS
from enhanced_rag_pipeline import run_ultra_optimized_pipeline

app = Flask(__name__)
CORS(app)

# ✅ Use the route expected by the HackRx API
@app.route('/api/v1/hackrx/run', methods=['POST'])
def run_rag():
    try:
        # Optional: handle Authorization header (for HackRx)
        auth_header = request.headers.get('Authorization')
        team_token = "eaa1d26662bda11af3de797cdad9b7308559e1e1e96dc683b3b8e54e21fcd99c"
        if auth_header != f"Bearer {team_token}":
            return jsonify({"error": "Unauthorized"}), 401

        data = request.get_json()
        documents_url = data.get("documents")
        questions = data.get("questions")

        if not documents_url or not questions:
            return jsonify({"error": "Both 'documents' and 'questions' fields are required."}), 400

        result =run_ultra_optimized_pipeline(questions=questions, documents_url=documents_url)
        return jsonify(result), 200

    except Exception as e:
        return jsonify({"error": f"An error occurred: {str(e)}"}), 500

# ✅ HOST and PORT for Render deployment
if __name__ == '__main__':
    import os
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)
