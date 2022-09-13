import tensorflow as tf
import numpy as np
import aisquared
import click
import os

def training_generator(directory, batch_size = 512):
    files = os.listdir(directory)
    np.random.shuffle(files)

    ages = [int(f.split('_')[0]) for f in files]
    m = min(ages)
    M = max(ages)
    del ages

    cutoffs = list(range(10, 100, 10))
    print(len(cutoffs))
    
    i = 0
    while True:
        batch, labels = [], []
        for _ in range(batch_size):
            if i >= len(files):
                np.random.shuffle(files)
                i = 0
            img = tf.keras.preprocessing.image.load_img(
                os.path.join(directory, files[i]),
                target_size = (256, 256)
            )
            img = np.array(img)/255
            age = int(files[i].split('_')[0])
            label = sum([age > cutoff for cutoff in cutoffs])
            batch.append(img)
            labels.append(label)
            i += 1
        yield np.asarray(batch), np.asarray(labels)

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
    output_layer = tf.keras.layers.Dense(10, activation = 'softmax')(x)

    model = tf.keras.models.Model(input_layer, output_layer)
    model.compile(
        loss = 'sparse_categorical_crossentropy',
        optimizer = 'adam',
        metrics = ['accuracy']
    )
    return model

@click.command()
@click.argument('directory', type = click.Path(exists = True, file_okay = False, dir_okay = True))
@click.option('--batch-size', '-b', type = int, default = 64)
def main(directory, batch_size):
    model = build_model()
    model.summary()
    
    train_dataset = training_generator(directory, batch_size)
    callback = tf.keras.callbacks.EarlyStopping(
        monitor = 'loss',
        patience = 5,
        min_delta = 0.001
    )
    model.fit(train_dataset, epochs = 100, callbacks = [callback], steps_per_epoch = len(os.listdir(directory))//batch_size)
    model.save('age_model.h5')

if __name__ == '__main__':
    main()
    
