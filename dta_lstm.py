# pylint: disable=wrong-import-position
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"

import numpy as np
import pandas as pd

import keras
from keras.models import Sequential
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import TimeDistributed

from config import DEFAULTS

PADDING_SYMBOL_INT = 0


class DTA_LSTM():

    name = 'dta_lstm'
    version = '0.1'
    default_config = DEFAULTS

    model_attributes = DEFAULTS.keys()

    def __init__(self, threshold=DEFAULTS['threshold'], top_k=DEFAULTS['top_k'],
                 batch_size=DEFAULTS['batch_size'], epochs=DEFAULTS['epochs'],
                 validation_split=DEFAULTS['validation_split'],
                 explain=DEFAULTS['explain'], gpus=DEFAULTS['gpus']):
        """ Creates an LSTM model (sequence to sequence mapping)
        """
        self.threshold = threshold
        self.top_k = top_k
        self.batch_size = batch_size
        self.validation_split = validation_split
        self.epochs = epochs
        self.explain = explain
        self.gpus = gpus

        self.max_n_span_types = None
        self.trace_size = None
        self.model = None
        self.rank_prob = []

        self.input_dim = 0

    def _check_parameters(self, X):

        if not isinstance(X, pd.DataFrame):
            raise TypeError('X should be of type pandas.DataFrame')

        if len(X.shape) != 2 or X.shape[1] != 1:
            raise ValueError(f'Input array has invalid dimensions: {X.shape}. Should be (n, 1)')

    def _build_model(self, input_dim=0):
        """ Creates an LSTM model (sequence to sequence mapping)

        Parameters
        ----------
        trace_seg_size : int
            The segment size of a trace to analyze to build the model
        input_dim : int
            Dimensionality of the output space.

        Returns
        -------
        model : Sequential
            a Sequential model representing the LSTM
        """
        model = Sequential()
        model.add(LSTM(100, dropout=0.2, recurrent_dropout=0.2, return_sequences=True,
                       input_shape=(self.trace_size, input_dim)))
        model.add(LSTM(100, dropout=0.2, recurrent_dropout=0.2, return_sequences=True))
        model.add(LSTM(100, dropout=0.2, recurrent_dropout=0.2, return_sequences=True))

        # A Dense layer is used as the output for the network.
        model.add(TimeDistributed(Dense(input_dim, activation='softmax')))
        if self.gpus > 1:
            model = keras.utils.multi_gpu_model(model, gpus=self.gpus)
        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

        return model

    def _get_lstm_parameters(self, X_train):
        """
            Calculates how many distinct symbols exist in the sequences
            Calculates the longest sequence
        """
        # Add one extra element for the special symbol zero
        self.max_n_span_types = np.max(X_train) + 1
        self.trace_size = max(len(i) for i in X_train)

    def fit(self, X_train):  # pylint: disable=arguments-differ
        """ Train the model
        """
        self._check_parameters(X_train)

        # X_train has 2 columns: traceId, span_list. Here we select the span_list
        X_train = X_train.loc[:, DEFAULTS['span_list_col_name']].values

        X = self._pad_traces(X_train)
        y = self._shift_traces_left(X)

        # We can get the parameters only after padding the span lists
        self._get_lstm_parameters(X)

        X = self._traces_to_binary(X)
        y = self._traces_to_binary(y)

        # The X_train.shape is (n_traces, self.trace_size, self.max_n_span_types)
        # e.g. (21251, 20, 33)
        self.model = self._build_model(input_dim=X.shape[2])

        self.model.fit(X, y, epochs=self.epochs, batch_size=self.batch_size, shuffle=True,
                       validation_split=self.validation_split)

    def _pad_traces(self, X):
        """Pad traces with the special symbol PADDING_SYMBOL_INT up to self.trace_size cells

        _pad_traces([[1, 2, 3], [3, 4, 5, 6], [7, 8]])

        array([[0, 1, 2, 3],
               [3, 4, 5, 6],
               [0, 0, 7, 8]], dtype=int16)

        Parameters
        ----------
        X : {array-like}, shape=[n_traces, span_list]
            The n_traces traces with span_list to pad.

        Returns
        -------
        Padded traces: array, shape=[n_traces, padded_span_list]
            The traces padded
        """
        # Padding is done using using keras.preprocessing.sequence.pad_sequences(sequences, ...)
        # maxlen: Int, maximum length of all sequences.
        # truncating='pre' remove values at the beginning from sequences larger than maxlen
        # padding='pre' pads each trace at the beginning with a special integer (e.g., 0)
        X = pad_sequences(X, maxlen=self.trace_size, dtype=np.int16,
                          truncating='pre', padding='pre', value=PADDING_SYMBOL_INT)
        # Shape X is (n_traces, self.trace_size)
        # e.g., (25002, 20)

        return X

    @staticmethod
    def _shift_traces_left(X):
        """
        Shift traces by one position to the left. Fill new cell with PADDING_SYMBOL_INT

        Parameters
        ----------
        X : {array-like}, shape=[n_traces, span_list]
            Traces with span_list to shift.

        Returns
        -------
        Shifted traces: array, shape=[n_traces, shifted_span_list]
            Traces shifted to the left
        """
        if not isinstance(X, np.ndarray):
            raise ValueError('Invalid parameter type: {}'.format(type(X)))

        if X.size == 0:
            return X

        if len(X.shape) != 2:
            raise ValueError('Requires a 2-dim array: {}'.format(X))

        # X.shape and y.shape are (n_traces, self.trace_size)
        y = np.roll(X, -1, axis=1)

        # Pad j array where X array has zeros
        y[X == 0] = PADDING_SYMBOL_INT

        # Write the PADDING_SYMBOL_INT at the last position of the shifted array
        y[:, -1] = PADDING_SYMBOL_INT

        return y

    def _traces_to_binary(self, X):
        """
            Converts a class vector (integers) to binary class matrix.
            X.shape = (n_traces, self.trace_size)
            returns shape = (n_traces, self.trace_size, self.max_n_span_types)

            e.g.,

            [[[1. 0. 0. ... 0. 0. 0.]
              [1. 0. 0. ... 0. 0. 0.]
              [1. 0. 0. ... 0. 0. 0.]

        """
        return keras.utils.to_categorical(X, num_classes=self.max_n_span_types, dtype='int32')

    def predict(self, X_test):  # pylint: disable=arguments-differ
        """Predict if the structure of a trace is anomalous.

        Parameters
        ----------
        X_test : {array-like}, shape=[n_trace, span_list]
                 Traces to evaluate

        Returns
        -------
        Anomalies : array, shape [n_traces]
            True/False indicating if a trace is anomalous.
        """

        self._check_parameters(X_test)

        trace_ids = X_test.index.values

        # X_train has two columns: traceId and span list, we select only the span list
        X_test = X_test.loc[:, DEFAULTS['span_list_col_name']].values

        if not self.model:
            raise ValueError('Model not available. Did you called fit(...)?')

        X_test = self._pad_traces(X_test)
        X_test_bin = self._traces_to_binary(X_test)
        # e.g., X_test_bin[0][0] = array([1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0....], dtype=int32)

        yhat = self.model.predict(X_test_bin)
        # e.g.,
        # yhat.shape = (n_traces, self.trace_size, self.max_n_span_types)
        # yhat[0][0] = [9.9969006e-01, 2.2644450e-05, 3.9419938e-06, 2.8681773e-09, ... ]

        self._identify_anomalies(X_test, yhat, prob=self.threshold)  # idx -> True/False

        return self.rank_prob

    def _identify_anomalies(self, X, yhat, prob):
        """
            mark anomalies based on the difference between X, y, and yhat

            i.e.,

            top_k = 5
            top_k_yhat = np.argsort(yhat, axis=2)[:, :, -top_k:]
            top_yhat = np.argmax(yhat, axis=2)

            y = _shift_traces_left(X)

            X[i]: Input:        [ 0  0  0  0  0  0  0  0  0  0  0 10 10 17  5  5  7  7  6  6]
            y[i]: Output:       [ 0  0  0  0  0  0  0  0  0  0  0 10 17  5  5  7  7  6  6  0]
            yhat[i]: Predicted: [ 0  0  0  0  0  0  0  0  0  0  0 10 17  5  5  7  7  6  6  0]

        """

        y = DTA_LSTM._shift_traces_left(X)

        for i in range(len(X)):

            rp = self._compare_traces(y[i], yhat[i])
            self.rank_prob.append(rp)

            if self.explain:
                print('X[{}]: {} = {}'.format(i, X[i], rp))

        return self.rank_prob


    def _compare_traces(self, y, yhat):
        """
            Calculate the distance between y and yhat
        """

        distance = []
        for (i,), y_i in np.ndenumerate(y):
            # Get the (reverse) sorted predictions yhat for the symbol at position i
            # e.g., [4, 2, 1, 3]
            prediction = np.argsort(-yhat[i])

            # What is the ranking of the predictions for y_i
            # e.g., assuming prediction = [4, 2, 1, 3] and y_i = 2, then rank = 1
            if y_i == -1:
                continue  # ignore unknown span types - todo: review this
            rank = np.where(prediction == y_i)[0][0]

            # Get the threshold associated with the 'rank'
            prob = yhat[i][prediction[rank]]

            distance.append([rank, prob])

        return distance

    def predict_proba(self):
        """Retrieve the probabilities of a trace being an anomaly.
           The threshold is calculated as the average probabilities of each symbol being anomalous

        Returns
        -------
        threshold : array-like, shape = (n_traces, threshold)
            The threshold of each trace processes with predict(...) of being an anomaly
        """
        if self.rank_prob is None:
            raise ValueError('No results available. Did you already call predict(...)?')

        return np.array([sum(map(lambda x: x[1], result)) / len(result) for result in self.rank_prob])

