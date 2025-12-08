import tensorflow as tf
from tensorflow.keras import layers, models, callbacks, optimizers
from dataset_loader import DatasetLoader
from constants import *
from os.path import join

print("TensorFlow:", tf.__version__)

# load data
dl = DatasetLoader()
dl.load_from_save()
X_train, y_train = dl.images, dl.labels
X_val, y_val = dl.images_test, dl.labels_test

def build_model():
    model = models.Sequential([
        layers.Input(shape=(SIZE_FACE, SIZE_FACE, 1)),

        layers.Conv2D(64, 5, padding='same', activation='relu'),
        layers.BatchNormalization(),
        layers.MaxPooling2D(3, strides=2),

        layers.Conv2D(64, 5, padding='same', activation='relu'),
        layers.BatchNormalization(),
        layers.MaxPooling2D(3, strides=2),

        layers.Conv2D(128, 4, padding='same', activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.4),

        layers.Flatten(),
        layers.Dense(3072, activation='relu'),
        layers.Dropout(0.4),
        layers.Dense(len(EMOTIONS), activation='softmax')
    ])

    model.compile(
        optimizer=optimizers.Adam(learning_rate=5e-4),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    return model

model = build_model()
model.summary()

# callbacks: safe long training
save_path = join(SAVE_DIRECTORY, SAVE_MODEL_FILENAME + "_keras.h5")
cb_list = [
    callbacks.ModelCheckpoint(save_path, monitor='val_loss', save_best_only=True, verbose=1),
    callbacks.EarlyStopping(monitor='val_loss', patience=8, restore_best_weights=True, verbose=1),
    callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-7, verbose=1),
    callbacks.TensorBoard(log_dir="tflearn_logs_keras")
]

# train (up to 100 epochs; EarlyStopping will stop earlier if no improvement)
history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    batch_size=64,
    epochs=100,
    shuffle=True,
    callbacks=cb_list
)

# save (ModelCheckpoint already saves best; this ensures file exists)
model.save(save_path)
print("[+] Model saved to:", save_path)
