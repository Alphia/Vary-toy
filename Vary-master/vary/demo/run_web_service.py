from flask import Flask, request, jsonify
from model import eval_model
app = Flask(__name__)


@app.route('/', methods=['POST'])
def captioning():
    data = request.get_json()
    image_url = data['image_url']
    prompt = data['prompt']
    caption = eval_model(image_url, prompt)
    return jsonify({caption: caption})


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080)
