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
                           future_window=10,  # number of days to include
                           drop_threshold=0.01,  # drop threshold
                           drop_window=5,  # drop window
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

def pricequench_predict(prices, model):
    pr = np.array([prices])
    pr = np.reshape(pr, pr.shape+(1,))
    return model.predict(pr)

def evaluate(prices, model,
             divide_threshold=0.5,
             window_size=20,
             future_window=10,  # number of days to include
             drop_threshold=0.01,  # drop threshold
             drop_window=5,  # drop window
             percentage=True,  # use percentage if True; otherwise, absolute number
             ):
    annotations = ann.annotate_sharpdrop(prices,
                                         future_window=future_window,
                                         drop_threshold=drop_threshold,
                                         drop_window=drop_window,
                                         percentage=percentage)
    vectors, annots = wrangling_pricevector(prices,
                                            annotations,
                                            window_size=window_size,
                                            future_window=future_window
                                           )
    predprobs = map(lambda pr: pricequench_predict(pr, model), vectors)
    predlabels = map(int, map(lambda elem: elem[0][0]>divide_threshold, predprobs))

    tp = 0
    fp = 0
    fn = 0
    tn = 0
    for expannot, predannot in zip(annots, predlabels):
        if expannot==1 and predannot==1:
            tp += 1
        elif expannot==1 and predannot==0:
            fn += 1
        elif expannot==0 and predannot==1:
            fp += 1
        else:
            tn += 1
    print "tp = ", tp, "fp = ", fp, "fn = ", fn, "tn = ", tn

    try:
        recall = float(tp)/(tp+fn)
    except ZeroDivisionError:
        recall = np.inf
    print "recall = ", recall

    try:
        precision = float(tp)/(tp+fp)
    except ZeroDivisionError:
        precision = np.inf
    print "precision = ", precision

    try:
        fscore = 2*recall*precision/(recall+precision)
    except ZeroDivisionError:
        fscore = np.inf
    print "F-score = ", fscore
