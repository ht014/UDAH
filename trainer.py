import scipy.sparse as sp
from model import Graph2Gauss
from utils import load_dataset, score_link_prediction, score_node_classification
import os
import scipy
from utils import *
import scipy.io
from netmf import netmf_large
import copy
from predict import predict_cv
g = load_dataset('.data_path')
A, X, l = g['A'], g['X'], g['z']
model = UDAH(A=A, X=X, sim=A.toarray(),z=l,L=128, K=1,verbose=True)
sess = model.train('0')

