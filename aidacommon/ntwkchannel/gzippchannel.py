import pickle;
from collections import OrderedDict;

import numpy as np;

import gzip;

def transmit(result, sock):
    pickler = pickle.Pickler(sock);
    cols = list(result.keys());
    pickler.dump(cols);

    for col in cols:
        if(result[col].dtype == object):
            colz = gzip.compress(pickle.dumps(result[col]))
        else:
            colz = gzip.compress(result[col]);
        pickler.dump(result[col].dtype);
        pickler.dump(colz);



def receive(sock):
    unpickler = pickle.Unpickler(sock);
    result = OrderedDict([]);
    keylist = unpickler.load();

    for col in keylist:
        dt = unpickler.load();
        if(dt == object):
            result[col] = pickle.loads(gzip.decompress(unpickler.load()));
        else:
            result[col] = np.frombuffer(gzip.decompress(unpickler.load()), dtype=dt)

    return result;