import os

from flask import Flask, request, jsonify, render_template

# inference.py runs the model with NumPy only (no TensorFlow), so the web app
# starts in about a second and uses ~60 MB of RAM. It reads chatbot_model.npz,
# which Chatbot.py writes after training.
from inference import IntentModel, load_responses, process_input

app = Flask(__name__, template_folder='templates')

responses = load_responses()   # intent -> list of responses from data.json
model = IntentModel()          # loads chatbot_model.npz


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/chatbot', methods=['POST'])
def chatbot():
    try:
        payload = request.get_json(silent=True) or {}
        message = str(payload.get('message', '')).strip()
        if not message:
            return jsonify({'error': 'Empty message'}), 400

        response = process_input(message, model, responses)
        return jsonify({'response': response})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    # Hosting platforms such as Render tell the app which port to use via $PORT
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)
