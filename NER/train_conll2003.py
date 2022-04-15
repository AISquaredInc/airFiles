import tensorflow as tf
import aisquared
import datasets
import json

if __name__ == '__main__':

    dataset = datasets.load_dataset('conll2003')

    train_tokens = dataset['train']['tokens']
    train_tags = dataset['train']['ner_tags']
    
    val_tokens = dataset['validation']['tokens']
    val_tags = dataset['validation']['ner_tags']

    test_tokens = dataset['test']['tokens']
    test_tags = dataset['test']['ner_tags']

    train_texts = [
        ' '.join(tokens) for tokens in train_tokens
    ]
    val_texts = [
        ' '.join(tokens) for tokens in val_tokens
    ]
    test_texts = [
        ' '.join(tokens) for tokens in test_tokens
    ]

    tokenizer = tf.keras.preprocessing.text.Tokenizer(
        num_words = 10000,
        oov_token = '[oov]'
    )

    tokenizer.fit_on_texts(train_texts)
    vocab = json.loads(tokenizer.get_config()['word_index'])
    print(vocab['[oov]'])
    del vocab['[oov]']

    train_sequences = tokenizer.texts_to_sequences(train_texts)
    train_sequences = tf.keras.preprocessing.sequence.pad_sequences(
        train_sequences,
        maxlen = 32,
        padding = 'pre',
        truncating = 'post'
    )
    train_labels = tf.keras.preprocessing.sequence.pad_sequences(
        train_tags,
        maxlen = 32,
        padding = 'pre',
        truncating = 'post'
    )
    train_labels = (train_labels != 0).astype(int)

    val_sequences = tokenizer.texts_to_sequences(val_texts)
    val_sequences = tf.keras.preprocessing.sequence.pad_sequences(
        val_sequences,
        maxlen = 32,
        padding = 'pre',
        truncating = 'post'
    )
    val_labels = tf.keras.preprocessing.sequence.pad_sequences(
        val_tags,
        maxlen = 32,
        padding = 'pre',
        truncating = 'post'
    )
    val_labels = (val_labels != 0).astype(int)
    
    test_sequences = tokenizer.texts_to_sequences(test_texts)
    test_sequences = tf.keras.preprocessing.sequence.pad_sequences(
        test_sequences,
        maxlen = 32,
        padding = 'pre',
        truncating = 'post'
    )
    test_labels = tf.keras.preprocessing.sequence.pad_sequences(
        test_tags,
        maxlen = 32,
        padding = 'pre',
        truncating = 'post'
    )
    test_labels = (test_labels != 0).astype(int)

    input_layer = tf.keras.layers.Input(32)
    x = tf.keras.layers.Embedding(
        10000,
        32
    )(input_layer)
    x = tf.keras.layers.LSTM(
        512,
        return_sequences = True
    )(x)
    x = tf.keras.layers.LSTM(
        512,
        return_sequences = True
    )(x)
    x = tf.keras.layers.LSTM(
        512,
        return_sequences = True
    )(x)
    for _ in range(3):
        x = tf.keras.layers.Dense(512, activation = 'relu')(x)
    output_layer = tf.keras.layers.Dense(1, activation = 'sigmoid')(x)

    model = tf.keras.models.Model(
        input_layer,
        output_layer
    )
    model.compile(
        loss = 'binary_crossentropy',
        optimizer = 'adam',
        metrics = ['accuracy']
    )

    callback = tf.keras.callbacks.EarlyStopping(
        monitor = 'val_loss',
        patience = 10,
        min_delta = 0.001
    )
    
    model.fit(
        train_sequences,
        train_labels,
        epochs = 100,
        batch_size = 256,
        validation_data = (val_sequences, val_labels),
        callbacks = [callback]
    )
    preds = model.predict(test_sequences).argmax(axis = -1)
    print(preds[:10])
    model.save('NERModel.h5')
    with open('vocab.json', 'w') as f:
        json.dump(vocab, f)    
