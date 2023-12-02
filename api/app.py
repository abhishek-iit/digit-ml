from flask import Flask, request, jsonify
from joblib import load

app = Flask(__name__)
model_path = "models/svm_gamma:0.01_C:1.joblib"
model = load(model_path)


def load_model():
    svm_model = load("models/svm_gamma:0.01_C:1.joblib")
    lr_model = load("models/m22aie228_lr_solver_lbfgs.joblib")
    dt_model = load("models/m22aie228_tree_max_depth_10.joblib")
    return svm_model, lr_model, dt_model

# Example usage
svm_model, lr_model, dt_model = load_model()

@app.route("/")
def hello_world():
    return "<b>Hello, World!</b>"

@app.route("/", methods=["POST"])
def hello_world_post():    
    return {"op" : "Hello, World POST " + request.json["suffix"]}

@app.route("/sum", methods=['POST'])
def sum_numbers():
    data = request.json
    x = data.get('x')
    y = data.get('y')
    if x is None or y is None:
        return jsonify({"error": "Missing x or y parameter"}), 400
    z = x + y
    return jsonify({'sum': z})

@app.route("/predict/<model_type>", methods=['POST'])
def predict(model_type):
    image = request.json.get('image')
    if image is None:
        return jsonify({"error": "Missing image data"}), 400

    # Load the appropriate model
    if model_type == 'svm':
        model = svm_model
    elif model_type == 'tree':
        model = dt_model
    elif model_type == 'lr':
        model = lr_model
    else:
        return jsonify({"error": "Invalid model type"}), 400

    # Perform prediction
    predicted = model.predict([image])
    return jsonify({"y_predicted": int(predicted[0])})

# @app.route("/predict", methods=['POST'])
# def predict():
#     image = request.json.get('image')
#     if image is None:
#         return jsonify({"error": "Missing image data"}), 400
#     predicted = model.predict([image])
#     return jsonify({"y_predicted": int(predicted[0])})

# @app.route("/compare", methods=['POST'])
# def compare():
#     data = request.json
#     image1 = data.get('image1')
#     image2 = data.get('image2')
#     if image1 is None or image2 is None:
#         return jsonify({"error": "Missing image1 or image2 data"}), 400
#     predicted1 = model.predict([image1])
#     predicted2 = model.predict([image2])
#     return jsonify({"are_same_digit": int(predicted1[0]) == int(predicted2[0])})

# if __name__ == '__main__':
#     app.run(debug=True)
