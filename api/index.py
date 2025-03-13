from flask import Flask, request, jsonify
from typing import Optional
from flask_cors import CORS

from process import pred as p
app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})  # Allow all origins

@app.route('/pred', methods=['POST', 'OPTIONS'])  # Handle preflight requests
def pred():
    if request.method == 'OPTIONS':  # Preflight request
        response = jsonify({"message": "CORS preflight"})
        response.headers.add("Access-Control-Allow-Origin", "*")
        response.headers.add("Access-Control-Allow-Methods", "POST, OPTIONS")
        response.headers.add("Access-Control-Allow-Headers", "Content-Type")
        return response, 204  # No content response

    data = request.get_json()
    opt: Optional[str] = request.args.get('opt', None)
    if not opt:
        opt = 'trn'
    opt = opt.upper()

    nn: list[float] = data['nn']
    
    response = jsonify({"predict": p(nn, opt)[0]})
    response.headers.add("Access-Control-Allow-Origin", "*")  # Explicitly allow cross-origin
    return response

if __name__ == '__main__':
    app.run(debug=False, host="0.0.0.0")
