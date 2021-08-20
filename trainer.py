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
A, X, z = g['A'], g['X'], g['z']
model = UDAH(A=A, X=X, sim=A.toarray(),z=z,L=128, K=1,verbose=True, p_val=0.1, p_test=0.1, p_nodes=0.0)
sess = model.train('0')

