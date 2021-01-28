from flask import Flask, request, redirect, render_template, jsonify

from services.process import process

app = Flask(__name__)


@app.route('/')
def main():
    return render_template('index.html')


@app.route('/upload', methods=['POST'])
def upload():
    if not request.files:
        return "Image is required", 400
    image = request.files["car"]
    #if not image.content_length:
       # return "Image is empty", 400

    result = process(image)
    return jsonify(result)


if __name__ == '__main__':
    app.run()
