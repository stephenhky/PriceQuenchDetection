import numpy as np
from keras.layers.core import Dense, Activation
from keras.layers.recurrent import LSTM, GRU
from keras.models import Sequential
from sklearn.preprocessing import normalize
from evaluation import irutils

# default parameters
def_annotation_params = {'future_window': 5,         # number of days to include
                         'drop_threshold': 0.01,     # drop threshold
                         'drop_window': 2,           # drop window
                         'use_percentage': True}     # use percentage if True; otherwise, absolute number
def_prediction_params = {'window_size': 30,
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
                          future_window=def_annotation_params['future_window'],
                          to_normalize=True
                         ):
    prices_vectors = np.array([prices[timeidx:(timeidx+window_size)] for timeidx in range(len(prices)-window_size-future_window)])
    if to_normalize:
        prices_vectors = normalize(prices_vectors)
    wrangled_annotations = np.array([annotation[timeidx+window_size-1] for timeidx in range(len(prices)-window_size-future_window)])
    return prices_vectors, wrangled_annotations

def train_prediction_model(prices,
                           window_size=def_prediction_params['window_size'],
                           future_window=def_annotation_params['future_window'],
                           drop_threshold=def_annotation_params['drop_threshold'],
                           drop_window=def_annotation_params['drop_window'],
                           percentage=def_annotation_params['use_percentage'],
                           batch_size=32,
                           to_normalize=True
                           ):
    annotations = annotate_sharpdrop(prices,
                                     future_window=future_window,
                                     drop_threshold=drop_threshold,
                                     drop_window=drop_window,
                                     percentage=percentage)
    prices_vectors, wrangled_annotations = wrangling_pricevector(prices,
                                                                 annotations,
                                                                 window_size=window_size,
                                                                 future_window=future_window,
                                                                 to_normalize=to_normalize)
    prices_vectors = np.reshape(prices_vectors, prices_vectors.shape+(1,))
    model = Sequential()
    model.add(GRU(window_size, input_shape=prices_vectors.shape[1:], dropout_W=0.2, dropout_U=0.2))
    model.add(Dense(1))
    model.add(Activation('sigmoid'))
    # model.add(Activation('softmax'))

    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    model.fit(prices_vectors, wrangled_annotations,
              batch_size=batch_size, nb_epoch=15)

    return model

def pricequench_predict(prices, model):
    pr = np.array([prices])
    pr = np.reshape(pr, pr.shape+(1,))
    return model.predict(pr)

def batch_pricequench_predict_int(prices, model,
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
    return {'vectors': vectors, 'wrangled_annotations': annots,
            'predicted_score': map(lambda pr: pricequench_predict(pr, model), vectors)}


def batch_pricequench_predict(*args, **kwargs):
    return batch_pricequench_predict_int(*args, **kwargs)['predicted_score']

def counts_pn(prices, model,
              divide_threshold=def_prediction_params['divide_threshold'],
              window_size=def_prediction_params['window_size'],
              future_window=def_annotation_params['future_window'],
              drop_threshold=def_annotation_params['drop_threshold'],
              drop_window=def_annotation_params['drop_window'],
              percentage=def_annotation_params['use_percentage']
              ):
    predpack  = batch_pricequench_predict_int(prices, model,
                                              window_size=window_size,
                                              future_window=future_window,
                                              drop_threshold=drop_threshold,
                                              drop_window=drop_window,
                                              percentage=percentage)
    predprobs = predpack['predicted_score']
    predlabels = map(int, map(lambda elem: elem[0][0]>divide_threshold, predprobs))

    tp = fp = fn = tn = 0
    annots = predpack['wrangled_annotations']
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
    ir_measures = irutils.calculate_IR(**ir_counts)
    print "recall = ", ir_measures['recall']
    print "precision = ", ir_measures['precision']
    print "F-score = ", ir_measures['fscore']

def evaluate_AUC(prices, model, score_thresholds=None,
                 window_size=def_prediction_params['window_size'],
                 future_window=def_annotation_params['future_window'],
                 drop_threshold=def_annotation_params['drop_threshold'],
                 drop_window=def_annotation_params['drop_window'],
                 percentage=def_annotation_params['use_percentage']):
    if score_thresholds==None:
        score_thresholds = np.linspace(0.0, 1.0, 101)
    recalls = []
    precisions = []
    for threshold in score_thresholds:
        ir_measures = irutils.calculate_IR(**counts_pn(prices, model,
                                                       divide_threshold=threshold,
                                                       window_size=window_size,
                                                       future_window=future_window,
                                                       drop_threshold=drop_threshold,
                                                       drop_window=drop_window,
                                                       percentage=percentage
                                                       )
                                           )
        recalls.append(ir_measures['recall'])
        precisions.append(ir_measures['precision'])

    recalls = np.array(recalls)
    precisions = np.array(precisions)
    evaluation_results = {'score_thresholds': score_thresholds,
                          'recalls': recalls,
                          'precisions': precisions,
                          'AUC': irutils.calculate_AUC(recalls, precisions)}
    return evaluation_results