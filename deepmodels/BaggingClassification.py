import PriceQuenchPrediction as pqp
import numpy as np
from sklearn import linear_model

def produce_bagging_models(prices, numbags=11, minority=1, batch_size=21, **preprocess_params):
    # preprocess the training data
    prices_vectors, wrangled_annotations = pqp.preprocess_trainingdata(prices, **preprocess_params)

    # separate into minority and majority
    minority_pairs = filter(lambda item: item[1]==minority, zip(prices_vectors, wrangled_annotations))
    majority_pairs = filter(lambda item: item[1]!=minority, zip(prices_vectors, wrangled_annotations))

    # sampling and bagging
    window_size = preprocess_params['window_size'] if preprocess_params.has_key('window_size') else pqp.def_prediction_params['window_size']
    len_minority = len(minority_pairs)
    len_majority = len(majority_pairs)
    models = []
    choices = []
    for i in range(numbags):
        selected_majidx = np.random.randint(0, len_majority, len_minority)
        sampled_pairs = minority_pairs + map(lambda i: majority_pairs[i], selected_majidx)
        sampled_prvecs = np.array(map(lambda pair: pair[0], sampled_pairs))
        sampled_wranns = np.array(map(lambda pair: pair[1], sampled_pairs))
        model = pqp.train_prediction_model_int(sampled_prvecs, sampled_wranns,
                                               window_size=window_size,
                                               batch_size=batch_size)
        models.append(model)
        choices.append(selected_majidx)

    # return
    return {'models': models, 'selected_indices': choices}

def batch_pricequenchmodels_prediction(prices, models,
                                       window_size=pqp.def_prediction_params['window_size'],
                                       future_window=pqp.def_annotation_params['future_window'],
                                       to_normalize=True
                                       ):
    prices_vectors = pqp.wrangling_pricevector(prices,
                                               window_size=window_size,
                                               future_window=future_window,
                                               to_normalize=to_normalize)
    prediction_matrix = np.array(map(lambda model: map(lambda price: pqp.pricequench_predict(price, model), prices_vectors), models))
    prediction_matrix = np.reshape(prediction_matrix, prediction_matrix.shape[:2])
    return prediction_matrix

def pricequenchmodels_logistic_model(prediction_matrix, annotations, **logregparam):
    logreg = linear_model.LogisticRegression(**logregparam)
    logreg.fit(np.transpose(prediction_matrix), annotations)
    return logreg

