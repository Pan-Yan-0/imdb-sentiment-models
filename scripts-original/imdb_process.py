import logging
import os
import re
import sys
from itertools import chain

import gensim
import pandas as pd
import torch
from bs4 import BeautifulSoup

from sklearn.model_selection import train_test_split

import pickle

embed_size = 300
max_len = 512

# Read data from files
train = pd.read_csv("./corpus/imdb/labeledTrainData.tsv", header=0,
                    delimiter="\t", quoting=3)
test = pd.read_csv("./corpus/imdb/testData.tsv", header=0, delimiter="\t", quoting=3)
unlabeled_train = pd.read_csv("./corpus/imdb/unlabeledTrainData.tsv", header=0,
                              delimiter="\t", quoting=3)
