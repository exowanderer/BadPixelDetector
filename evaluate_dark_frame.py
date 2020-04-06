import tensorflow as tf
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
            logical_gpus = tf.config.experimental.list_logical_devices('GPU')
            print(len(gpus), "Physical GPUs,", len(
                logical_gpus), "Logical GPUs")
    except RuntimeError as e:
        print(e)

from itertools import product as iterproduct

from matplotlib import pyplot as plt

import joblib
import numpy as np
import pandas as pd
import time

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, confusion_matrix
from sklearn.preprocessing import StandardScaler
from statsmodels.robust import scale as sc

from tensorflow.keras.models import Sequential
from tensorflow.keras.models import Model
from tensorflow.keras import layers


def build_lstm_autoencoder(train_x, n_units=128, num_decoder_tokens=4,
                           latent_dim=2, dropout_rate=0.2, fit_now=False,
                           epochs=10, batch_size=4096, validation_split=0.2,
                           shuffle=True, loss='mae', optimizer='adam'):

    model = Sequential()
    model.add(layers.LSTM(
        units=n_units,
        input_shape=(train_x.shape[1], train_x.shape[2])
    ))
    model.add(layers.Dropout(rate=dropout_rate))
    model.add(layers.RepeatVector(n=train_x.shape[1]))
    model.add(layers.LSTM(units=n_units, return_sequences=True))
    model.add(layers.Dropout(rate=dropout_rate))
    model.add(
        layers.TimeDistributed(
            layers.Dense(units=train_x.shape[2])
        )
    )

    model.compile(loss=loss, optimizer=optimizer)

    return model


def evaluate_lstm_autoencoder(model, new_data_x, THRESHOLD=0.1):
    # New Data
    start = time.time()
    print('[INFO] Starting `new_data_x` predict step')
    new_data_pred = model.predict(new_data_x)
    print('[INFO] Completed `new_data_x` predict step: '
          f'{time.time() - start} sec')

    new_data_mae_loss = np.mean(np.abs(new_data_pred - new_data_x), axis=1)
    new_data_mae_loss = new_data_mae_loss.squeeze()

    new_data_score_df = pd.DataFrame()
    new_data_score_df['loss'] = new_data_mae_loss
    new_data_score_df['threshold'] = THRESHOLD
    new_data_score_df['anomaly'] = new_data_score_df.loss > THRESHOLD

    print(f'[INFO] Created `new_data_x` Dataframe: {time.time() - start} sec')

    return new_data_score_df, new_data_pred


def plot_confusion_matrix(confusionMatrix, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues,
                          float_fmt='.1f',
                          figsize=(12, 12),
                          rotation=0):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        confusionMatrix = confusionMatrix.astype('float')
        confusionMatrix /= confusionMatrix.sum(axis=1)[:, np.newaxis]
        confusionMatrix = confusionMatrix * 100
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    plt.imshow(confusionMatrix, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()

    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=rotation)
    plt.yticks(tick_marks, classes)

    fmt = float_fmt if normalize else 'd'
    thresh = confusionMatrix.max() / 2.

    range0 = range(confusionMatrix.shape[0])
    range1 = range(confusionMatrix.shape[1])
    for i, j in iterproduct(range0, range1):
        plt.text(j, i, format(confusionMatrix[i, j], fmt) + '%',
                 horizontalalignment="center",
                 color="white" if confusionMatrix[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument('-fn', '--fits_filename', required=True, type=str)
    parser.add_argument('-nu', '--n_units', type=int, default=128)
    parser.add_argument('-e', '--epochs', type=int, default=10)
    parser.add_argument('-sn', '--save_now', action='store_true')
    parser.add_argument('-ln', '--load_name', type=str,
                        default='simulated_55k_bad_pixels_df.joblib.save')
    parser.add_argument('-ns', '--n_sig', type=float, default=4.5)
    parser.add_argument('-bsn', '--base_name', type=str, default='JWST_Dark')
    parser.add_argument('-pv', '--plot_verbose', action='store_true')

    clargs = parser.parse_args()

    fits_filename = clargs.fits_filename
    n_units = clargs.n_units
    epochs = clargs.epochs
    save_now = clargs.save_now
    load_name = clargs.load_name
    n_sig = clargs.n_sig
    base_name = clargs.base_name
    plot_verbose = clargs.plot_verbose

    if not plot_verbose:
        plt.ion()

    # history = joblib.load(
    #     f'LSTM{n_units}_{base_name}_history_{epochs}epochs.joblib.save')
    lstm_autoencoder = lstm_autoencoder = build_lstm_autoencoder(
        train_x=train_x, n_units=n_units,
        num_decoder_tokens=4, latent_dim=2,
        epochs=epochs, batch_size=batch_size)

    lstm_autoencoder.load(
        f'LSTM{n_units}_{base_name}_history_{epochs}epochs.h5'
    )
    # lstm_autoencoder_weights = keras.load_weights(
    #     f'LSTM{n_units}_{base_name}_history_{epochs}epochs_weights.h5')

    print('[INFO] Loading Train Score Dataframe')
    train_score_df = pd.read_csv(
        f'LSTM{n_units}_{base_name}_train_score_df.csv'
    )

    print('[INFO] Loading Test Score Dataframe')
    test_score_df = pd.read_csv(f'LSTM{n_units}_{base_name}_test_score_df.csv')

    n_sig = 4.5
    THRESHOLD = np.median(train_score_df.loss) + \
        n_sig * sc.mad(train_score_df.loss)

    train_score_df.anomaly = train_score_df.loss > THRESHOLD
    test_score_df.anomaly = test_score_df.loss > THRESHOLD

    # Open JWST dark current file and reshape to AE input shape
    new_data_x = fits.open(fits_filename)['SCI'].data
    n_rows, n_cols, n_timesteps = new_data_x.shape
    new_data_x.reshape((n_rows * n_cols, n_timesteps))

    if scaler_name is not None:
        scaler = joblib.load(scaler_name)
        new_data_x = scaler.transform(new_data_x)

    new_data_x = new_data_x.reshape(new_data_x.shape + (1,))

    new_data_score_df, new_data_pred = evaluate_lstm_autoencoder(
        lstm_autoencoder, new_data_x, THRESHOLD=THRESHOLD)

    if save_now:
        new_data_score_df.to_csv(
            f'LSTM{n_units}_{base_name}_new_data_score_df.csv'
        )

    if plot_now:
        print('[INFO] Creating KDE Figure')

        fig = plt.figure()

        new_data_score_df.loss.plot.kde()
        train_score_df.loss.plot.kde()
        test_score_df.loss.plot.kde()

        plt.axvline(THRESHOLD, ls='--', lw=3)
        plt.xlabel('MAE', fontsize=20)
        plt.ylabel('Probability Density')
        plt.title('Compare MAE vs Anomalies in Train and Test Sets')
        plt.legend(('New Data Loss', 'Train Loss',
                    'Test Loss', 'Threshold MAE'),
                   loc=0, fontsize=20)
        plt.show()

    if save_now:
        print('[INFO] Saving KDE Figure')
        fig.savefig(f'LSTM{n_units}_{base_name}_MAE_KDE.pdf')
