import PriceQuenchPrediction as pqp
import numpy as np

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