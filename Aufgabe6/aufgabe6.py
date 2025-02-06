import numpy as np
import os
import cv2
import tensorflow as tf
from tensorflow import keras
from keras import layers
from PIL import Image

image_size = (180, 180)
num_classes = 3
input_shape = (180, 180, 3)
model_path = "Aufgabe6/model.keras"

# mit gleich vielen Daten trainieren
def balance_dataset(data_directory):
    class_counts = {cls: len(os.listdir(os.path.join(data_directory, cls)))
                    for cls in os.listdir(data_directory) if os.path.isdir(os.path.join(data_directory, cls))}
    
    min_samples = min(class_counts.values())
    print("Bilder pro Klasse:", class_counts)
    print("Angleichung auf:", min_samples, "Bilder pro Klasse")

    return min_samples

def create_model():
    data_augmentation = keras.Sequential([
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(0.3),
        layers.RandomZoom(0.3),
        layers.RandomBrightness(0.4),
        layers.RandomContrast(0.4),
        layers.RandomTranslation(height_factor=0.3, width_factor=0.3)
    ])

    base_model = keras.applications.MobileNetV2(input_shape=input_shape, include_top=False, weights='imagenet')
    base_model.trainable = False  

    model = keras.Sequential([
        keras.Input(shape=input_shape),
        data_augmentation,
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.Dense(512, activation="relu"),
        layers.Dropout(0.3),
        layers.Dense(256, activation="relu"),
        layers.Dropout(0.2),
        layers.Dense(128, activation="relu"),  
        layers.Dense(num_classes, activation="softmax")
    ])

    model.compile(optimizer=keras.optimizers.Adam(learning_rate=1e-4), 
                  loss="categorical_crossentropy", metrics=["accuracy"])
    
    return model

def fine_tune_model(model, dataset, val_dataset, initial_epochs=50, fine_tune_epochs=30):
    history = model.fit(dataset, validation_data=val_dataset, epochs=initial_epochs)

    base_model = model.layers[1]  
    base_model.trainable = True

    fine_tune_at = 80
    for layer in base_model.layers[:fine_tune_at]:
        layer.trainable = False

    model.compile(optimizer=keras.optimizers.Adam(learning_rate=1e-6), 
                  loss="categorical_crossentropy", metrics=["accuracy"])

    history_fine_tune = model.fit(dataset, validation_data=val_dataset, epochs=fine_tune_epochs)

    return history, history_fine_tune

def train_model(data_directory, epochs=80, batch_size=32):
    min_samples = balance_dataset(data_directory)

    dataset = keras.preprocessing.image_dataset_from_directory(
        data_directory, image_size=image_size, label_mode="categorical",
        batch_size=batch_size, validation_split=0.2, subset="training", seed=42)

    val_dataset = keras.preprocessing.image_dataset_from_directory(
        data_directory, image_size=image_size, label_mode="categorical",
        batch_size=batch_size, validation_split=0.2, subset="validation", seed=42)

    print("Daten erfolgreich geladen.")

    model = create_model()
    
    early_stopping = keras.callbacks.EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True)
    lr_scheduler = keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=3, min_lr=1e-6)

    history, history_fine_tune = fine_tune_model(model, dataset, val_dataset, initial_epochs=epochs, fine_tune_epochs=20)

    model.save(model_path)
    print(f"Modell wurde unter {model_path} gespeichert.")
    
    return model

def load_model():
    return keras.models.load_model(model_path)

def predict_image(model, image_path):
    image = Image.open(image_path).resize(image_size)
    im_arr = np.array(image) / 255.0  
    im_arr = np.reshape(im_arr, (1, *image_size, 3))
    
    pred = model.predict(im_arr)
    
    class_names = ["Schere", "Stein", "Papier"]
    predicted_class = class_names[np.argmax(pred)]
    confidence = np.max(pred)
    
    print(f"Vorhersage: {predicted_class} (Wahrscheinlichkeit: {confidence:.2f})")
    return predicted_class

def real_time_prediction():
    model = load_model()
    class_names = ["Schere", "Stein", "Papier"]
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Kamera konnte nicht geöffnet werden.")
        return

    print("Drücke 'q', um das Programm zu beenden.")
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Kein Frame verfügbar.")
            break

        resized_frame = cv2.resize(frame, image_size)
        im_arr = np.array(resized_frame) / 255.0
        im_arr = np.reshape(im_arr, (1, *image_size, 3))

        pred = model.predict(im_arr)
        max_prob = np.max(pred)
        predicted_class = class_names[np.argmax(pred)]

        if max_prob > 0.2:  
            print(f"Erkannt: {predicted_class} (Wahrscheinlichkeit: {max_prob:.2f})")
            cv2.putText(frame, f"{predicted_class} ({max_prob:.2f})", (10, 50), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
        else:
            print("Keine eindeutige Entscheidung")
            cv2.putText(frame, "Unklar", (10, 50), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        cv2.imshow("Schere, Stein, Papier - Vorhersage", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    train_model("Aufgabe6/img/rps", epochs=80)

    #real_time_prediction()
