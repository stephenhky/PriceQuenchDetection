import numpy as np

def wrangling_pricevector(prices,
                          annotation,
                          window_size=20,
                          future_window=5
                         ):
    prices_vectors = np.array([prices[timeidx:(timeidx+window_size)] for timeidx in range(len(prices)-window_size-future_window)])
    wrangled_annotations = np.array([annotation[timeidx+window_size-1] for timeidx in range(len(prices)-window_size-future_window)])
    return prices_vectors, wrangled_annotations

