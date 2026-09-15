import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import MobileNetV2
import os
import json



DATASET_PATH = "../dataset/Garbage classification"

IMAGE_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 10


train_ds = tf.keras.utils.image_dataset_from_directory(
    DATASET_PATH,
    validation_split=0.2,
    subset="training",
    seed=123,
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE
)

val_ds = tf.keras.utils.image_dataset_from_directory(
    DATASET_PATH,
    validation_split=0.2,
    subset="validation",
    seed=123,
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE
)


class_names = train_ds.class_names

print("\nClasses:")
print(class_names)



os.makedirs("../model", exist_ok=True)

with open("../model/class_names.json", "w") as f:
    json.dump(class_names, f)


AUTOTUNE = tf.data.AUTOTUNE

train_ds = train_ds.prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.prefetch(buffer_size=AUTOTUNE)


data_augmentation = tf.keras.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.1),
    layers.RandomZoom(0.1)
])


base_model = MobileNetV2(
    input_shape=(224, 224, 3),
    include_top=False,
    weights="imagenet"
)

base_model.trainable = False


model = models.Sequential([

    layers.Input(shape=(224, 224, 3)),

    data_augmentation,

    layers.Rescaling(
        1.0 / 127.5,
        offset=-1
    ),

    base_model,

    layers.GlobalAveragePooling2D(),

    layers.Dropout(0.3),

    layers.Dense(
        len(class_names),
        activation="softmax"
    )
])


model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)


model.summary()


print("\n====================================")
print("STARTING WASTE CLASSIFICATION TRAINING")
print("====================================")

history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS
)



model.save("../model/waste_model.keras")


print("\n====================================")
print("TRAINING COMPLETED!")
print("====================================")

print("Model saved at:")
print("../model/waste_model.keras")