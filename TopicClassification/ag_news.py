import tensorflow_datasets as tfds
import tensorflow as tf
from tqdm import tqdm
import numpy as np
import aisquared
import mann
import json

data = tfds.as_numpy(
    tfds.load('ag_news_subset', as_supervised = True, split = 'train')
)
x_train, y_train = [], []
for value in tqdm(data):
    x_train.append(str(value[0]))
    y_train.append(value[1])

data = tfds.as_numpy(
    tfds.load('ag_news_subset', as_supervised = True, split = 'test')
)
x_test, y_test = [], []
for value in tqdm(data):
    x_test.append(str(value[0]))
    y_test.append(value[0])

x_train = np.asarray(x_train)
x_test = np.asarray(x_test)
y_train = np.asarray(y_train)
y_test = np.asarray(y_test)
    
tokenizer = tf.keras.preprocessing.text.Tokenizer(
    num_words = 10000,
    oov_token = '[oov]'
)
tokenizer.fit_on_texts(x_train)
vocab = json.loads(tokenizer.get_config()['word_index'])
print(vocab['[oov]'])
del vocab['[oov]']

train_sequences = tokenizer.texts_to_sequences(x_train)
train_sequences = tf.keras.preprocessing.sequence.pad_sequences(
    train_sequences,
    maxlen = 16,
    padding = 'pre',
    truncating = 'post'
)

test_sequences = tokenizer.texts_to_sequences(x_test)
test_sequences = tf.keras.preprocessing.sequence.pad_sequences(
    test_sequences,
    maxlen = 16,
    padding = 'pre',
    truncating = 'post'
)

input_layer = tf.keras.layers.Input(16)
x = tf.keras.layers.Embedding(
    10000,
    8
)(input_layer)
x = tf.keras.layers.Flatten()(x)
for _ in range(10):
    x = tf.keras.layers.Dense(256, activation = 'relu')(x)
output_layer = tf.keras.layers.Dense(4, activation = 'softmax')(x)
model = tf.keras.models.Model(input_layer, output_layer)
model.compile(loss = 'sparse_categorical_crossentropy', metrics = ['accuracy'], optimizer = 'adam')
model.summary()

callback = tf.keras.callbacks.EarlyStopping(
    monitor = 'val_loss',
    patience = 3,
    min_delta = 0.01,
    restore_best_weights = True
)

model.fit(
    train_sequences,
    y_train,
    batch_size = 1024,
    epochs = 100,
    validation_split = 0.2,
    callbacks = [callback]
)

model.save('topic_classifier.h5')

harvester = aisquared.config.harvesting.TextHarvester()
preprocesser = aisquared.config.preprocessing.TextPreprocessor(
    [
        aisquared.config.preprocessing.RemoveCharacters(),
        aisquared.config.preprocessing.ConvertToCase(),
        aisquared.config.preprocessing.Tokenize(),
        aisquared.config.preprocessing.ConvertToVocabulary(
            vocab,
            0,
            1
        ),
        aisquared.config.preprocessing.PadSequences(
            0,
            16,
            'pre',
            'post'
        )
    ]
)
model = aisquared.config.analytic.LocalModel('topic_classifier.h5', 'text')
postprocesser = aisquared.config.postprocessing.MulticlassClassification(
    [
        'World',
        'Sports',
        'Business',
        'Science/Technology'
    ]
)
renderer = aisquared.config.rendering.DocumentRendering(include_probability = True)
model_feedback = aisquared.config.feedback.ModelFeedback()
model_feedback.add_question('Is this model useful?', choices = ['yes', 'no'])
model_feedback.add_question('Please elaborate', answer_type = 'text')
prediction_feedback = aisquared.config.feedback.MulticlassFeedback(
    [
        'World',
        'Sports',
        'Business',
        'Science/Technology'
    ]
)

config = aisquared.config.ModelConfiguration(
    'NewsTopicClassifier',
    harvester,
    preprocesser,
    model,
    postprocesser,
    renderer,
    [model_feedback, prediction_feedback]
).compile(dtype = 'float16')



