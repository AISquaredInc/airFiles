import tensorflow as tf
import aisquared
import click

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
    )(input_layer)
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
    )(input_layer)
    x = tf.keras.layers.Conv2D(
        256,
        3,
        padding = 'same',
        activation = 'relu'
    )(x)
    x = tf.keras.layers.MaxPool2D()(x)
    x = tf.keras.layers.Flatten()(x)
    for _ in range(3):
        x = tf.keras.layers.Dense(512, activation = 'relu')(x)
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
    train_dataset = tf.keras.utils.image_dataset_from_directory(
        directory,
        batch_size = 256,
        validation_split = 0.2,
        seed = 12,
        subset = 'training'
    )
    test_dataset = tf.keras.utils.image_dataset_from_directory(
        directory,
        batch_size = 256,
        validation_split = 0.2,
        seed = 12,
        subset = 'validation'
    )
    callback = tf.keras.callbacks.EarlyStopping(
        monitor = 'val_loss',
        patience = 5,
        min_delta = 0.02
    )
    model.fit(train_dataset, epochs = 100, validation_data = test_dataset, callbacks = [callback])

if __name__ == '__main__':
    main()
    
