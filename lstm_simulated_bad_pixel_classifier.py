from itertools import product as iterproduct

from pylab import *
ion()

import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, Dense
from tensorflow.keras.layers.embeddings import Embedding
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.datasets import imdb

from sklearn.externals import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

from xgboost import XGBClassifier


def build_lstm_autoencoder(n_nodes=128):
    print('[INFO]: Establishing LSTM Classifier')

    # Define an input sequence and process it.
    encoder_inputs = Input(shape=(None, num_encoder_tokens))
    encoder = LSTM(latent_dim, return_state=True)
    encoder_outputs, state_h, state_c = encoder(encoder_inputs)
    # We discard `encoder_outputs` and only keep the states.
    encoder_states = [state_h, state_c]

    # Set up the decoder, using `encoder_states` as initial state.
    decoder_inputs = Input(shape=(None, num_decoder_tokens))
    # We set up our decoder to return full output sequences,
    # and to return internal states as well. We don't use the
    # return states in the training model, but we will use them in inference.
    decoder_lstm = LSTM(latent_dim, return_sequences=True, return_state=True)
    decoder_outputs, _, _ = decoder_lstm(decoder_inputs,
                                         initial_state=encoder_states)
    decoder_dense = Dense(num_decoder_tokens, activation='softmax')
    decoder_outputs = decoder_dense(decoder_outputs)

    # Define the model that will turn
    # `encoder_input_data` & `decoder_input_data` into `decoder_target_data`
    return Model([encoder_inputs, decoder_inputs], decoder_outputs)


def build_lstm_inference()
    encoder_model = Model(encoder_inputs, encoder_states)

    decoder_state_input_h = Input(shape=(latent_dim,))
    decoder_state_input_c = Input(shape=(latent_dim,))
    decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]
    decoder_outputs, state_h, state_c = decoder_lstm(
        decoder_inputs, initial_state=decoder_states_inputs)
    decoder_states = [state_h, state_c]
    decoder_outputs = decoder_dense(decoder_outputs)
    decoder_model = Model(
        [decoder_inputs] + decoder_states_inputs,
        [decoder_outputs] + decoder_states)

    return decoder_model


def decode_sequence(input_seq):
    # Encode the input as state vectors.
    states_value = encoder_model.predict(input_seq)

    # Generate empty target sequence of length 1.
    target_seq = np.zeros((1, 1, num_decoder_tokens))
    # Populate the first character of target sequence with the start character.
    target_seq[0, 0, target_token_index['\t']] = 1.

    # Sampling loop for a batch of sequences
    # (to simplify, here we assume a batch of size 1).
    stop_condition = False
    decoded_sentence = ''
    while not stop_condition:
        output_tokens, h, c = decoder_model.predict(
            [target_seq] + states_value)

        # Sample a token
        sampled_token_index = np.argmax(output_tokens[0, -1, :])
        sampled_char = reverse_target_char_index[sampled_token_index]
        decoded_sentence += sampled_char

        # Exit condition: either hit max length
        # or find stop character.
        if (sampled_char == '\n' or
                len(decoded_sentence) > max_decoder_seq_length):
            stop_condition = True

        # Update the target sequence (of length 1).
        target_seq = np.zeros((1, 1, num_decoder_tokens))
        target_seq[0, 0, sampled_token_index] = 1.

        # Update states
        states_value = [h, c]

    return decoded_sentence


def load_and_process_data():
    # fix random seed for reproducibility
    np.random.seed(7)

    # load the dataset but only keep the top n words, zero the rest
    top_words = 5000
    (X_train, y_train), (X_test, y_test) = imdb.load_data(num_words=top_words)
    # truncate and pad input sequences
    max_review_length = 500
    X_train = sequence.pad_sequences(X_train, maxlen=max_review_length)
    X_test = sequence.pad_sequences(X_test, maxlen=max_review_length)

    return (X_train, y_train), (X_test, y_test)


def build_lstm_classifier():
    # create the model
    embedding_vecor_length = 32
    model = Sequential()
    model.add(Embedding(top_words, embedding_vecor_length,
                        input_length=max_review_length))
    model.add(LSTM(100))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy',
                  optimizer='adam', metrics=['accuracy'])
    print(model.summary())
    model.fit(X_train, y_train, epochs=3, batch_size=64)
    # Final evaluation of the model
    scores = model.evaluate(X_test, y_test, verbose=0)
    print("Accuracy: %.2f%%" % (scores[1] * 100))


def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in iterproduct(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()
print('[INFO]: Loading data')
data_dict = joblib.load('simulated_bad_pixels_df.joblib.save')

print('[INFO]: Selecting data')
features = data_dict['features']
labels = data_dict['labels']

print('[INFO]: Train Test Splitting')
idx_train, idx_test = train_test_split(np.arange(labels.size),
                                       test_size=0.2,
                                       stratify=labels)

print('[INFO]: Transforming with StandardScaler')
std_sclr = StandardScaler()
features_std_train = std_sclr.fit_transform(features[idx_train])
features_std_test = std_sclr.transform(features[idx_test])

print('[INFO]: Building LSTM Autoencoder')
lstm_autoencoder = build_lstm(n_nodes=128)

print('[INFO]: Compiling LSTM Autoencoder')
lstm_autoencoder.compile(optimizer='adam', loss='categorical_crossentropy')

print('[INFO]: Training LSTM Autoencoder')
lstm_autoencoder.fit([encoder_input_data, decoder_input_data],
                     decoder_target_data,
                     batch_size=batch_size,
                     epochs=epochs,
                     validation_split=0.2)

print('[INFO]: Finished LSTM Training')

print('[INFO]: Predicting Random Forest Classifier')
predict = lstm_autoencoder.predict(features_std_test)

print('[INFO]: Computing Quality Metrics')
accuracy = accuracy_score(labels[idx_test], predict)

num_per_class = [(labels[idx_test] == label).sum() for label in set(labels)]
confusionMatrix = confusion_matrix(labels[idx_test], predict) / num_per_class


###
class_names = ['Clean', 'Hot Pixel', 'Cold Pixel', 'Sat Hot Pixel', 'Sat Cold Pixel',
               'Cosmic Ray', 'Popcorn Pixel', 'Noisy']
plot_confusion_matrix(confusionMatrix, classes=class_names, normalize=True,
                      title='Normalized confusion matrix')

fig = gcf()
fig.savefig('LSTM128_ConfusionMatrix.png')

gotItRight = labels[idx_test] == predict

class_GIR = {classnow: np.where((labels[idx_test] == classnow) * gotItRight)[0][0]
             for classnow in set(labels)
             if np.any((labels[idx_test] == classnow) * gotItRight)}

class_nGIR = {classnow: np.where((labels[idx_test] == classnow) * (~gotItRight))[0][0]
              for classnow in set(labels)
              if np.any((labels[idx_test] == classnow) * (~gotItRight))}

# Got It Right
for key, val in class_GIR.items():
    # fig = figure()
    plt.plot(features[idx_test][val], label=class_names[predict[val]])

    legend(loc=4, fontsize=20, framealpha=0.9)

    xlim(0, 110)
    ylabel('Electrons Read Off Detetor', fontsize=30)
    xlabel('Group Number [0,107]', fontsize=30)
    legend(loc=4, fontsize=15, framealpha=.9)
    title('Correctly Predicted', fontsize=30)

    ax = gcf().get_axes()[0]
    for tick in ax.xaxis.get_major_ticks():
        tick.label.set_fontsize(15)
    for tick in ax.yaxis.get_major_ticks():
        tick.label.set_fontsize(15)

    fig.savefig('Raw_data_correct_prediction.png')
    # fig.savefig('Raw_data_correct_prediction_{}.png'.format(class_names[predict[val]]))

# not Got It Right
for key, val in class_nGIR.items():
    fig = figure()
    plt.plot(features[idx_test][val], label=class_names[predict[val]])
    legend(loc=0, fontsize=20, framealpha=0.9)

    xlim(0, 110)
    ylabel('Electrons Read Off Detetor', fontsize=30)
    xlabel('Group Number [0,107]', fontsize=30)
    legend(loc=4, fontsize=15, framealpha=.9)
    title('Incorrectly Predicted: {}'.format(
        class_names[predict[val]]), fontsize=30)

    ax = gcf().get_axes()[0]
    for tick in ax.xaxis.get_major_ticks():
        tick.label.set_fontsize(15)
    for tick in ax.yaxis.get_major_ticks():
        tick.label.set_fontsize(15)

    fig.savefig('Raw_data_wrong_prediction_{}.png'.format(
        class_names[predict[val]]))
