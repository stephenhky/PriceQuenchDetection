import Quandl
import pandas as pd
import numpy as np

reference_timept = np.datetime64('1900-01-01T00:00:00.0000000Z')

class QuanDLRetriever:
    def __init__(self, code='YAHOO/INDEX_GSPC'):
        self.data = Quandl.get(code)

    def getseconds(self, transpose=True):
        delta_t = np.array(self.data.index)-reference_timept
        delta_t = np.array(map(lambda t: t.item()/1e+9, delta_t))
        return delta_t if not transpose else np.transpose(np.matrix(delta_t))

    def getyears(self):
        return self.getseconds() / (3600*24*365.24)

    def getclose(self):
        return np.array(self.data['Close'])

    def numdatapts(self):
        return self.data.size