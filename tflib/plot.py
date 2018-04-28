import numpy as np

import matplotlib
from utils.misc import ensure_dir

matplotlib.use('Agg')
import matplotlib.pyplot as plt

import collections
import cPickle as pickle

_since_beginning = collections.defaultdict(lambda: {})
_since_last_flush = collections.defaultdict(lambda: {})

_iter = [0]

import os

def tick():
    _iter[0] += 1


def plot(name, value):
    _since_last_flush[name][_iter[0]] = value


def flush():
    prints = []

    for name, vals in _since_last_flush.items():
        prints.append("{}\t{}".format(name, np.mean(vals.values())))
        _since_beginning[name].update(vals)

        x_vals = np.sort(_since_beginning[name].keys())
        y_vals = [_since_beginning[name][x] for x in x_vals]

        plt.clf()
        plt.plot(x_vals, y_vals)
        plt.xlabel('iteration')
        plt.ylabel(name)
        fpath=os.path.join('debug',name.replace(' ', '_') + '.jpg')
        base_dir = os.path.dirname(fpath)
        ensure_dir(base_dir)
        plt.savefig(fpath)

    print "iter {}\t{}".format(_iter[0], "\t".join(prints))
    _since_last_flush.clear()

    with open('debug/log.pkl', 'wb') as f:
        pickle.dump(dict(_since_beginning), f, pickle.HIGHEST_PROTOCOL)
