import numpy as np

def annotate_sharpdrop(prices,                 # ndarray
                       future_window=10,        # number of days to include
                       drop_threshold=0.01,    # drop threshold
                       drop_window=5,          # drop window
                       percentage=True         # use percentage if True; otherwise, absolute number
                      ):
    "Return 1 if there will be a sharp drop; 0 if not; -1 if invalid."
    flags = np.repeat(-1, len(prices))
    for idx in range(0, len(prices)-future_window):
        flag = 0
        for kdx in range(idx, idx+future_window-drop_window):
            absdrop = prices[idx]-prices[idx+drop_window-1]
            if absdrop / (prices[idx] if percentage else 1) > drop_threshold:
                flag = 1
            if flag > 0: break
        flags[idx] = flag
    return flags