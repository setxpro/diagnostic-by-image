from tensorflow.keras.models import load_model
import numpy as np
from tensorflow.keras.preprocessing import image

# Carregar o modelo treinado
model = load_model('cancer_classifier_model.h5')


def predict_image(img_detected_path):
    img = image.load_img(img_detected_path, target_size=(150, 150))
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    prediction = model.predict(img_array)
    return 'Cancer' if prediction[0] > 0.5 else 'No Cancer'


img_path = 'dataset/to/skin_cancer/train/benign/02.jpg'
result = predict_image(img_path)
print(f"Prediction: {result}")

