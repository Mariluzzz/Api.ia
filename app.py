from flask import Flask, request, jsonify
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
import io
from PIL import Image

# Carrega o modelo salvo
model = tf.keras.models.load_model('TCC_3_classes_vgg16_model_tl.h5')

app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({"error": "No file provided"}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No file selected"}), 400

    # Carrega e processa a imagem
    img = Image.open(io.BytesIO(file.read()))  # Carrega a imagem diretamente do conteúdo do arquivo
    img = img.resize((224, 224))  # Redimensiona para o tamanho esperado pelo modelo
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)  # Adiciona uma dimensão para o batch
    img_array /= 255.0  # Normaliza os pixels para o intervalo [0, 1]

    # Faz a predição
    predictions = model.predict(img_array)

    # Encontra a classe mais provável
    predicted_class = np.argmax(predictions, axis=1)[0]

    # Retorna a previsão
    response = {
        "predictions": predictions.tolist(),
        "predicted_class": int(predicted_class)
    }
    return jsonify(response)

if __name__ == '__main__':
    app.run(debug=True)
