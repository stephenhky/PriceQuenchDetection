import numpy as np
from keras.layers.core import Dense, Activation
from keras.layers.recurrent import LSTM
from keras.models import Sequential
from evaluation import IRutils

# default parameters
def_annotation_params = {'future_window': 10,        # number of days to include
                         'drop_threshold': 0.01,     # drop threshold
                         'drop_window': 5,           # drop window
                         'use_percentage': True}     # use percentage if True; otherwise, absolute number
def_prediction_params = {'window_size': 20,
                         'divide_threshold': 0.5}


def annotate_sharpdrop(prices,  # ndarray
                       future_window=def_annotation_params['future_window'],
                       drop_threshold=def_annotation_params['drop_threshold'],
                       drop_window=def_annotation_params['drop_window'],
                       percentage=def_annotation_params['use_percentage']
                       ):
    "Return 1 if there will be a sharp drop; 0 if not; -1 if invalid."
    flags = np.repeat(-1, len(prices))
    for idx in range(0, len(prices) - future_window):
        flag = 0
        for kdx in range(idx, idx + future_window - drop_window):
            absdrop = prices[idx] - prices[idx + drop_window - 1]
            if absdrop / (prices[idx] if percentage else 1) > drop_threshold:
                flag = 1
            if flag > 0: break
        flags[idx] = flag
    return flags


def wrangling_pricevector(prices,
                          annotation,
                          window_size=def_prediction_params['window_size'],
                          future_window=def_annotation_params['future_window']
                         ):
    prices_vectors = np.array([prices[timeidx:(timeidx+window_size)] for timeidx in range(len(prices)-window_size-future_window)])
    wrangled_annotations = np.array([annotation[timeidx+window_size-1] for timeidx in range(len(prices)-window_size-future_window)])
    return prices_vectors, wrangled_annotations

def train_prediction_model(prices,
                           window_size=def_prediction_params['window_size'],
                           future_window=def_annotation_params['future_window'],
                           drop_threshold=def_annotation_params['drop_threshold'],
                           drop_window=def_annotation_params['drop_window'],
                           percentage=def_annotation_params['use_percentage'],
                           batch_size=32
                           ):
    annotations = annotate_sharpdrop(prices,
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

def counts_pn(prices, model,
              divide_threshold=def_prediction_params['divide_threshold'],
              window_size=def_prediction_params['window_size'],
              future_window=def_annotation_params['future_window'],
              drop_threshold=def_annotation_params['drop_threshold'],
              drop_window=def_annotation_params['drop_window'],
              percentage=def_annotation_params['use_percentage']
              ):
    annotations = annotate_sharpdrop(prices,
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
    return {'tp': tp, 'fp': fp, 'fn': fn, 'tn': tn}

def evaluate(prices, model,
             divide_threshold=def_prediction_params['divide_threshold'],
             window_size=def_prediction_params['window_size'],
             future_window=def_annotation_params['future_window'],
             drop_threshold=def_annotation_params['drop_threshold'],
             drop_window=def_annotation_params['drop_window'],
             percentage=def_annotation_params['use_percentage']):
    ir_counts = counts_pn(prices, model,
                          divide_threshold=divide_threshold,
                          window_size=window_size,
                          future_window=future_window,
                          drop_threshold=drop_threshold,
                          drop_window=drop_window,
                          percentage=percentage)
    print "tp = ", ir_counts['tp'], "fp = ", ir_counts['fp'], "fn = ", ir_counts['fn'], "tn = ", ir_counts['tn']
    ir_measures = IRutils.calculate_IR(**ir_counts)
    print "recall = ", ir_measures['recall']
    print "precision = ", ir_measures['precision']
    print "F-score = ", ir_measures['fscore']