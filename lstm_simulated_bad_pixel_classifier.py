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

from tqdm import tqdm


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

    # model.add_metric(value, aggregation=None, name=None)
    if fit_now:
        history = model.fit(
            train_x, train_x,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=validation_split,
            shuffle=shuffle
        )

        return model, history

    return model


def evaluate_lstm_autoencoder(model, train_x, test_x, THRESHOLD=0.18):
    # Train Data
    start = time.time()
    print('[INFO] Starting `train_x` predict step')
    train_pred = model.predict(train_x)
    print('[INFO] Completed `train_x` predict step: '
          f'{time.time() - start} sec')

    train_mae_loss = np.mean(np.abs(train_pred - train_x), axis=1).squeeze()

    train_score_df = pd.DataFrame()
    train_score_df['loss'] = train_mae_loss
    train_score_df['threshold'] = THRESHOLD
    train_score_df['anomaly'] = train_score_df.loss > train_score_df.threshold

    print(f'[INFO] Created `train_x` Dataframe: {time.time() - start} sec')

    # Test Data
    start = time.time()
    print('[INFO] Starting `test_x` predict step')
    test_pred = model.predict(test_x)
    print(f'[INFO] Completed `test_x` predict step: {time.time() - start} sec')
    test_mae_loss = np.mean(np.abs(test_pred - test_x), axis=1).squeeze()

    test_score_df = pd.DataFrame()
    test_score_df['loss'] = test_mae_loss
    test_score_df['threshold'] = THRESHOLD
    test_score_df['anomaly'] = test_score_df.loss > test_score_df.threshold
    print(f'[INFO] Created `train_x` Dataframe: {time.time() - start} sec')

    return train_score_df, test_score_df, train_pred, test_pred
    # anomalies = test_score_df[test_score_df.anomaly == True]


def load_and_process_data(load_name=None, test_size=0.2,
                          base_name='55k', scaler=None,
                          save_now=False, make_3d_tensor=True):

    if load_name is None:
        load_name = f'simulated_{base_name}_bad_pixels_df.joblib.save'

    features, labels = joblib.load(load_name).values()

    n_samples = len(labels)
    idx_train, idx_test = train_test_split(np.arange(n_samples),
                                           test_size=test_size)

    train_x = features[idx_train]
    test_x = features[idx_test]
    train_y = labels[idx_train]
    test_y = labels[idx_test]

    if scaler is not None:
        train_x = scaler.fit_transform(train_x)
        test_x = scaler.transform(test_x)
        if save_now:
            joblib.dump(scaler, f'StandardScaler_{base_name}.joblib.save')

    if make_3d_tensor:
        train_x = train_x.reshape(train_x.shape + (1,))
        test_x = test_x.reshape(test_x.shape + (1,))

    return (train_x, train_y), (test_x, test_y)


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
    parser.add_argument('-nu', '--n_units', type=int, default=128)
    parser.add_argument('-e', '--epochs', type=int, default=10)
    parser.add_argument('-bs', '--batch_size', type=int, default=4096)
    parser.add_argument('-vs', '--validation_split', type=float, default=0.2)
    parser.add_argument('-s', '--shuffle', action='store_true')
    parser.add_argument('-sn', '--save_now', action='store_true')
    parser.add_argument('-fit', '--fit_on_the_go', action='store_true')
    parser.add_argument('-ln', '--load_name', type=str,
                        default='simulated_55k_bad_pixels_df.joblib.save')
    parser.add_argument('-ns', '--n_sig', type=float, default=4.5)
    parser.add_argument('-bsn', '--base_name', type=str, default='55k')
    parser.add_argument('-pv', '--plot_verbose', action='store_true')

    # load_name ='simulated_4M_bad_pixels_df.joblib.save',

    clargs = parser.parse_args()
    n_units = clargs.n_units
    epochs = clargs.epochs
    batch_size = clargs.batch_size
    validation_split = clargs.validation_split
    shuffle = clargs.shuffle
    save_now = clargs.save_now
    fit_on_the_go = clargs.fit_on_the_go
    load_name = clargs.load_name
    n_sig = clargs.n_sig
    base_name = clargs.base_name
    plot_verbose = clargs.plot_verbose

    if not plot_verbose:
        plt.ion()

    if 'train_x' not in locals().keys():
        print(f'[INFO] Loading Data from {load_name}')
        (train_x, train_y), (test_x, test_y) = load_and_process_data(
            load_name=load_name,
            test_size=0.2,
            scaler=StandardScaler(),
            base_name=base_name,
            save_now=save_now)
        print(f'[INFO] Completed Loading and Scaling from {load_name}')

    print('[INFO]: Building LSTM Autoencoder')
    if fit_on_the_go:
        lstm_autoencoder, history = build_lstm_autoencoder(
            train_x=train_x, fit_now=fit_on_the_go,
            n_units=n_units, num_decoder_tokens=4, latent_dim=2,
            dropout_rate=0.2,
            epochs=epochs, batch_size=batch_size,
            validation_split=validation_split,
            shuffle=shuffle, loss='mae', optimizer='adam')
    else:
        lstm_autoencoder = build_lstm_autoencoder(
            train_x=train_x, fit_now=fit_on_the_go,
            n_units=n_units, num_decoder_tokens=4, latent_dim=2,
            dropout_rate=0.2,
            epochs=epochs, batch_size=batch_size,
            validation_split=validation_split,
            shuffle=shuffle, loss='mae', optimizer='adam')

    if not fit_on_the_go:
        history = lstm_autoencoder.fit(
            train_x, train_x,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=validation_split,
            shuffle=shuffle
        )

    if save_now:
        joblib.dump(
            history.history,
            f'LSTM{n_units}_{base_name}_history_{epochs}epochs.joblib.save'
        )
        lstm_autoencoder.save(
            f'LSTM{n_units}_{base_name}_history_{epochs}epochs.h5'
        )
        lstm_autoencoder.save_weights(
            f'LSTM{n_units}_{base_name}_history_{epochs}epochs_weights.h5'
        )

    (train_score_df, test_score_df,
        train_pred, test_pred) = evaluate_lstm_autoencoder(
        lstm_autoencoder, train_x, test_x, THRESHOLD=0.1)

    THRESHOLD = np.median(train_score_df.loss) + \
        n_sig * sc.mad(train_score_df.loss)

    train_score_df.anomaly = train_score_df.loss > THRESHOLD
    test_score_df.anomaly = test_score_df.loss > THRESHOLD

    if save_now:
        print('[INFO] Saving Train Score Dataframe')
        train_score_df.to_csv(f'LSTM{n_units}_{base_name}_train_score_df.csv')

        print('[INFO] Saving Test Score Dataframe')
        test_score_df.to_csv(f'LSTM{n_units}_{base_name}_test_score_df.csv')

    if plot_verbose:
        print('[INFO] Creating KDE Figure')
        fig = plt.figure()
        train_score_df.loss.plot.kde()
        test_score_df.loss.plot.kde()
        plt.axvline(THRESHOLD)
        plt.xlabel('MAE', fontsize=20)
        plt.ylabel('Probability Density')
        plt.title('Compare MAE vs Anomalies in Train and Test Sets')
        plt.legend(('Train Loss', 'Test Loss'), loc=0, fontsize=20)
        plt.show()

    if save_now:
        print('[INFO] Saving KDE Figure')
        fig.savefig(f'LSTM{n_units}_{base_name}_MAE_KDE.pdf')

    # print('[INFO]: Computing Quality Metrics')
    # test_mse = mean_squared_error(test_y, test_pred)
    if plot_verbose:
        print('[INFO] Creating Confusion Matrix Figure')
        train_y_ = np.int8(train_y != 0)
        test_y_ = np.int8(test_y != 0)
        train_num_per_class = np.array([(train_y_ == label).sum()
                                        for label in set(np.unique(train_y_))])
        test_num_per_class = np.array([(test_y_ == label).sum()
                                       for label in set(np.unique(test_y_))])

        confusionMatrix = confusion_matrix(
            np.int8(test_y) != 0,
            np.int8(test_score_df.anomaly)
        )
        confusionMatrix = confusionMatrix / test_num_per_class

        # class_names = ['Clean', 'Hot Pixel', 'Cold Pixel', 'Sat Hot Pixel',
        #                'Sat Cold Pixel', 'Cosmic Ray', 'Popcorn Pixel',
        #                'Noisy']
        class_names = ['Clean', 'Bad Pixel']
        fig = plt.figure(figsize=figsize)
        plot_confusion_matrix(confusionMatrix, classes=class_names,
                              figsize=(12, 12), normalize=True,
                              title='Normalized confusion matrix')

    if save_now:
        print('[INFO] Saving Confusion Matrix Figure')
        fig.savefig(f'LSTM{n_units}_{base_name}_ConfusionMatrix.pdf')

    plot_got_it_right = False
    if plot_got_it_right:
        gotItRight = test_y == test_pred

        class_GIR = {classnow: np.where((test_y == classnow) * gotItRight)[0][0]
                     for classnow in set(labels)
                     if np.any((test_y == classnow) * gotItRight)}

        class_nGIR = {classnow: np.where((test_y == classnow) * (~gotItRight))[0][0]
                      for classnow in set(labels)
                      if np.any((test_y == classnow) * (~gotItRight))}

        # Got It Right
        for key, val in class_GIR.items():
            # fig = figure()
            plt.plot(features[idx_test][val], label=class_names[predict[val]])

            plt.xlim(0, 110)
            plt.ylabel('Electrons Read Off Detetor', fontsize=30)
            plt.xlabel('Group Number [0,107]', fontsize=30)
            plt.legend(loc=4, fontsize=15, framealpha=.9)
            plt.title('Correctly Predicted', fontsize=30)

            ax = plt.gcf().get_axes()[0]
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
            plt.legend(loc=0, fontsize=20, framealpha=0.9)

            plt.xlim(0, 110)
            plt.ylabel('Electrons Read Off Detetor', fontsize=30)
            plt.xlabel('Group Number [0,107]', fontsize=30)
            plt.legend(loc=4, fontsize=15, framealpha=.9)
            plt.title('Incorrectly Predicted: {}'.format(
                class_names[predict[val]]), fontsize=30)

            ax = plt.gcf().get_axes()[0]
            for tick in ax.xaxis.get_major_ticks():
                tick.label.set_fontsize(15)
            for tick in ax.yaxis.get_major_ticks():
                tick.label.set_fontsize(15)

            fig.savefig('Raw_data_wrong_prediction_{}.png'.format(
                class_names[predict[val]]))
