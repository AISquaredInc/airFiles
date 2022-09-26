import tensorflow as tf
import aisquared

train_dataset = tf.keras.utils.image_dataset_from_directory('brain_tumor_dataset', batch_size = 64, seed = 32, subset = 'training', validation_split = 0.2)
val_dataset = tf.keras.utils.image_dataset_from_directory('brain_tumor_dataset', batch_size = 64, seed = 32, subset = 'validation', validation_split = 0.2)

train_input = tf.keras.layers.Input((256, 256, 3))
train_rescale = tf.keras.layers.Rescaling(1./255)(train_input)
train_head = tf.keras.models.Model(train_input, train_rescale)

true_input = tf.keras.layers.Input((256, 256, 3))
x = tf.keras.layers.Conv2D(16, kernel_size = 3, padding = 'same', activation = 'relu')(true_input)
x = tf.keras.layers.Conv2D(16, kernel_size = 3, padding = 'same', activation = 'relu')(x)
x = tf.keras.layers.MaxPool2D()(x)

x = tf.keras.layers.Conv2D(32, kernel_size = 3, padding = 'same', activation = 'relu')(x)
x = tf.keras.layers.Conv2D(32, kernel_size = 3, padding = 'same', activation = 'relu')(x)
x = tf.keras.layers.MaxPool2D()(x)

x = tf.keras.layers.Flatten()(x)
x = tf.keras.layers.Dense(32, activation = 'relu')(x)
x = tf.keras.layers.Dense(32, activation = 'relu')(x)
x = tf.keras.layers.Dense(2, activation = 'softmax')(x)
true_model = tf.keras.models.Model(true_input, x)

train_model = tf.keras.models.Sequential()
train_model.add(train_head)
train_model.add(true_model)
train_model.compile(
    loss = 'sparse_categorical_crossentropy',
    optimizer = 'adam',
    metrics = ['accuracy']
)

callback = tf.keras.callbacks.EarlyStopping(monitor = 'val_accuracy', min_delta = 0.01, patience = 3)

train_model.fit(
    train_dataset,
    batch_size = 64,
    epochs = 100,
    callbacks = [callback],
    validation_data = val_dataset
)
true_model.save('mri_classifier.h5')

harvester = aisquared.config.harvesting.ImageHarvester()
preprocesser = aisquared.config.preprocessing.image.ImagePreprocessor(
    [
        aisquared.config.preprocessing.image.Resize([256, 256]),
        aisquared.config.preprocessing.image.DivideValue(255)
    ]
)
analytic = aisquared.config.analytic.LocalModel('mri_classifier.h5', 'cv')
postprocesser = aisquared.config.postprocessing.BinaryClassification(['tumor', 'healthy'])
renderer = aisquared.config.rendering.ImageRendering(thickness = '5', include_probability = False, font_size = '20')

config = aisquared.config.ModelConfiguration(
    'MRIClassifier',
    harvester,
    preprocesser,
    analytic,
    postprocesser,
    renderer
).compile(dtype = 'float16')
