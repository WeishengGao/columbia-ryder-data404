from flask import Flask, request, jsonify, render_template, url_for
from method1 import perform_analysis

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('5900index.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    method = data['method']

    # Call perform_analysis to generate all images
    if method == 'method1':
        image_path_1, image_path_2, image_path_3, image_path_4, image_path_nn = perform_analysis()
    else:
        image_path_1 = image_path_2 = image_path_3 = image_path_4 = image_path_nn = None  # Placeholder for other methods

    if all([image_path_1, image_path_2, image_path_3, image_path_4, image_path_nn]):
        return jsonify({
            'image_url_1': url_for('static', filename=image_path_1.split('static/')[-1]),
            'image_url_2': url_for('static', filename=image_path_2.split('static/')[-1]),
            'image_url_3': url_for('static', filename=image_path_3.split('static/')[-1]),
            'image_url_4': url_for('static', filename=image_path_4.split('static/')[-1]),
            'image_url_nn': url_for('static', filename=image_path_nn.split('static/')[-1])
        })
    else:
        return jsonify({'error': 'Failed to generate prediction images.'}), 500

if __name__ == '__main__':
    app.run(port=5500, debug=True)