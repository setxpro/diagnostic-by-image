import os
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Certifique-se de que o Pillow está instalado
try:
    from PIL import Image
except ImportError:
    import pip

    pip.main(['install', 'pillow'])
    from PIL import Image

# Diretórios de treinamento e validação
train_dir = 'dataset/to/skin_cancer/train'
validation_dir = 'dataset/to/skin_cancer/val'


def load_image(image_path):
    try:
        img = Image.open(image_path)
        img = img.convert('RGB')
        img = img.resize((150, 150))
        img_array = tf.keras.preprocessing.image.img_to_array(img) / 255.0
        return img_array
    except Exception as e:
        print(f"Erro ao carregar imagem {image_path}: {e}")


# Verificar se os diretórios existem e contêm imagens
def verify_directories(directory):
    if not os.path.exists(directory):
        print(f"Diretório {directory} não existe.")
        return
    for class_dir in os.listdir(directory):
        class_path = os.path.join(directory, class_dir)
        if os.path.isdir(class_path):
            print(f"Classe {class_dir} contém {len(os.listdir(class_path))} imagens")


# Configurar os geradores de dados com aumento de dados para o conjunto de treinamento
train_datagen = ImageDataGenerator(
    rescale=1. / 255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True
)

validation_datagen = ImageDataGenerator(rescale=1. / 255)

# Geradores de dados
train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(150, 150),
    batch_size=2,
    class_mode='binary'
)

validation_generator = validation_datagen.flow_from_directory(
    validation_dir,
    target_size=(150, 150),
    batch_size=2,
    class_mode='binary'
)

# MODELO

# Criar o modelo
model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(150, 150, 3)),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(512, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Treinar o modelo
history = model.fit(
    train_generator,
    epochs=20,
    validation_data=validation_generator
)

# Avaliar o modelo
loss, accuracy = model.evaluate(validation_generator)
# print(f"Accuracy: {accuracy * 100:.2f}%")

# Salvar o modelo treinado
model.save('cancer_classifier_model.h5')

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    print(f"Imagens de treinamento: {train_generator.samples}")
    print(f"Imagens de validação: {validation_generator.samples}")
