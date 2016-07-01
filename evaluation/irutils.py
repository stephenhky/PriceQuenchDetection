import numpy as np

def calculate_IR(tp, fp, fn):
    # recall
    try:
        recall = float(tp)/(tp+fn)
    except ZeroDivisionError:
        recall = np.inf

    # precision
    try:
        precision = float(tp)/(tp+fp)
    except ZeroDivisionError:
        precision = np.inf

    # F-score
    try:
        fscore = 2*recall*precision/(recall+precision)
    except ZeroDivisionError:
        fscore = np.inf

    return {'recall': recall, 'precision': precision, 'fscore': fscore}

def calculate_AUC(precisions, recalls):
    if len(precisions) != len(recalls):
        raise Exception('Precisions & Recalls lengths different.')
    # horizontal axis: precisions, vertical axis: recalls
    sorted_order = np.argsort(np.append(precisions, [0., 1.]))
    sorted_precisions = np.append(precisions, [0., 1.])[sorted_order]
    sorted_recalls = np.append(recalls, [1., 0.])[sorted_order]

    # trapezoidal rule
    precdiff = np.ediff1d(sorted_precisions)
    recadjsum = np.array(map(lambda idx: sorted_recalls[idx] + sorted_recalls[idx + 1],
                             range(len(sorted_recalls) - 1)
                             )
                         )
    return np.sum(0.5*precdiff*recadjsum)