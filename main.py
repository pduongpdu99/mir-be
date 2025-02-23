from flask import Flask, request, jsonify, render_template
from process import pred as p
from typing import Optional
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

@app.route('/pred', methods=['POST'])
def pred():
    data = request.get_json()
    opt:Optional[str] = request.args.get('opt', None)
    if not opt:
        opt = 'trn'
    opt = opt.upper()
    
    nn: list[float] = data['nn']
    return jsonify({"predict": p(nn, opt)[0]})

@app.route('/', methods=['GET'])
def home():
    return render_template("templates/index.html", title="Flask Example", message="Hello world")

if __name__ == '__main__':
    app.run(debug=True)
