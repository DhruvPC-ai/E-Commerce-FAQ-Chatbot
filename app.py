import pickle
import json

from flask import Flask, request, jsonify, render_template
from tensorflow.keras.models import load_model
from Chatbot import process_input

app = Flask(__name__, template_folder='templates')

with open('data.json', 'r') as f:
    Data = json.load(f)
with open('tokenizer.pkl', 'rb') as f:
    tokenizer = pickle.load(f)
with open('classes.pkl', 'rb') as f:
    classes = pickle.load(f)

model = load_model("Chatbot.keras")

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/chatbot', methods=['POST'])
def chatbot():
    try:
        message = request.json.get('message', '')
        if not message:
            return jsonify({'error': 'Empty message'}), 400

        response = process_input(message, tokenizer, 100, model, Data, classes)
        return jsonify({'response': response})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=False)