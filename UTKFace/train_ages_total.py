import tensorflow as tf
from tqdm import tqdm
import numpy as np
import aisquared
import click
import cv2
import os

def build_model():
    input_layer = tf.keras.layers.Input((256, 256, 3))
    x = tf.keras.layers.Conv2D(
        32,
        3,
        padding = 'same',
        activation = 'relu'
    )(input_layer)
    x = tf.keras.layers.Conv2D(
        32,
        3,
        padding = 'same',
        activation = 'relu'
    )(x)
    x = tf.keras.layers.MaxPool2D()(x)
    x = tf.keras.layers.Conv2D(
        64,
        3,
        padding = 'same',
        activation = 'relu'
    )(x)
    x = tf.keras.layers.Conv2D(
        64,
        3,
        padding = 'same',
        activation = 'relu'
    )(x)
    x = tf.keras.layers.MaxPool2D()(x)
    x = tf.keras.layers.Conv2D(
        128,
        3,
        padding = 'same',
        activation = 'relu'
    )(x)
    x = tf.keras.layers.Conv2D(
        256,
        3,
        padding = 'same',
        activation = 'relu'
    )(x)
    x = tf.keras.layers.MaxPool2D()(x)
    x = tf.keras.layers.Conv2D(
        256,
        3,
        padding = 'same',
        activation = 'relu'
    )(x)
    x = tf.keras.layers.Conv2D(
        256,
        3,
        padding = 'same',
        activation = 'relu'
    )(x)
    x = tf.keras.layers.MaxPool2D()(x)
    x = tf.keras.layers.Flatten()(x)
    for _ in range(3):
        x = tf.keras.layers.Dense(256, activation = 'relu')(x)
    output_layer = tf.keras.layers.Dense(1, activation = 'relu')(x)

    model = tf.keras.models.Model(input_layer, output_layer)
    model.compile(
        loss = 'mse',
        optimizer = 'adam'
    )
    return model

@click.command()
@click.argument('directory', type = click.Path(exists = True, file_okay = False, dir_okay = True))
def main(directory):
    model = build_model()
    model.summary()

    files = os.listdir(directory)
    images = []
    labels = []
    for f in tqdm(files):
        try:
            label = int(f.split('_')[0])
            img = cv2.imread(os.path.join(directory, f))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = img.tolist()
            images.append(img)
            labels.append(label)
        except Exception as e:
            pass

    images = tf.image.resize(np.asarray(images)/255, (256, 256))
    labels = np.asarray(labels)
    
    callback = tf.keras.callbacks.EarlyStopping(
        monitor = 'val_loss',
        patience = 5,
        min_delta = 0.02
    )
    model.fit(images, labels, epochs = 100, validation_split = 0.2, callbacks = [callback])
    model.save('age_model.h5')

if __name__ == '__main__':
    main()
    
