import numpy as np
import PriceQuenchAnnotation as ann
from keras.models import Sequential
from keras.layers.core import Dense, Activation
from keras.layers.recurrent import LSTM

def wrangling_pricevector(prices,
                          annotation,
                          window_size=20,
                          future_window=5
                         ):
    prices_vectors = np.array([prices[timeidx:(timeidx+window_size)] for timeidx in range(len(prices)-window_size-future_window)])
    wrangled_annotations = np.array([annotation[timeidx+window_size-1] for timeidx in range(len(prices)-window_size-future_window)])
    return prices_vectors, wrangled_annotations

def train_prediction_model(prices,
                           window_size=20,
                           future_window=5,  # number of days to include
                           drop_threshold=0.01,  # drop threshold
                           drop_window=2,  # drop window
                           percentage=True,  # use percentage if True; otherwise, absolute number
                           batch_size=32
                           ):
    annotations = ann.annotate_sharpdrop(prices,
                                         future_window=future_window,
                                         drop_threshold=drop_threshold,
                                         drop_window=drop_window,
                                         percentage=percentage)
    prices_vectors, wrangled_annotations = wrangling_pricevector(prices,
                                                                 annotations,
                                                                 window_size=window_size,
                                                                 future_window=future_window)
    prices_vectors = np.reshape(prices_vectors, prices_vectors.shape+(1,))
    model = Sequential()
    model.add(LSTM(window_size, input_shape=prices_vectors.shape[1:], dropout_W=0.2, dropout_U=0.2))
    model.add(Dense(1))
    model.add(Activation('sigmoid'))

    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    model.fit(prices_vectors, wrangled_annotations,
              batch_size=batch_size, nb_epoch=15)

    return model