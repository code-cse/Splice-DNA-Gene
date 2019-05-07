import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit as s_split
import Encoder as edr
import os


class CrossValDataPreparation:
    
    def __init__(self, n_folds=5):
        self.n_folds = n_folds
    

    def shuffle_data(self, X, y):
        '''
        X: Contains the sequences 
        y: Contains the labels
        '''
        split_ratio = 1.0/self.n_folds
        X = np.asarray(X)
        split = s_split(n_splits= 1, test_size=split_ratio, random_state=18)
        X_train = []
        y_train = []
        X_test = []
        y_test = []
        for train_id, test_id in split.split(X, y):
            X_train.append(X[train_id])
            y_train.append(y[train_id])
            X_test.append(X[test_id])
            y_test.append(y[test_id])
        del X
        del y
        return X_train[0], y_train[0], X_test[0], y_test[0]
    
    def statified_crossVal_split(self, X, y):
        '''
        X: Contains the sequences 
        y: Contains the labels
        '''
        
        X_train = []
        y_train = []
        X_test = []
        y_test = []
        
        X_tr, y_tr, X_te, y_te = self.shuffle_data(X,y)
        X_train.append(X_tr)
        y_train.append(y_tr)
        X_test.append(X_te)
        y_test.append(y_te)
        
        l = len(X_te)

        for i in range(0,self.n_folds-1):
            X_t = X_tr[0:l]
            y_t = y_tr[0:l]
            X_tr = np.concatenate((X_tr[l:],X_te),axis=0)
            y_tr = np.concatenate((y_tr[l:],y_te),axis=0)
            X_te = X_t
            y_te = y_t
            X_train.append(X_tr)
            y_train.append(y_tr)
            X_test.append(X_te)
            y_test.append(y_te)
            
        X_train = np.asarray(X_train, dtype="float32")
        X_test = np.asarray(X_test, dtype="float32")
        y_train = np.asarray(y_train, dtype="float32")
        y_test = np.asarray(y_test, dtype="float32")
        
        return X_train, y_train, X_test, y_test

def data_set():
    print("In Encoder class")
    ob_enc = edr.Encoder()
    filePath = "../data/spliceData.txt"
    X, Y = ob_enc.get_all_possible_codon_list(filePath, 39, 99, 0)
    crs = CrossValDataPreparation()
    X_train, y_train, X_test, y_test = crs.statified_crossVal_split(X, Y)
    return X_train, y_train, X_test, y_test   

def data_set_1():
    print("In Encoder class")
    ob_enc = edr.Encoder()
    filePath = "../data/spliceData.txt"
    X, Y = ob_enc.get_all_possible_codon_list(filePath, 39, 99, 0)
    return X, Y


def single_pred():
    print("In single data prediction...!")
    ob_enc = edr.Encoder()
    input_2 = "IE,    HUMALPI-ACCEPTOR-2698,          TCAACTACAGGGACCCGCATCTCCCTACAGGGTTGGCCCCCAGCAAGGCTCAGGACAGCA"
    os.remove("../data/test.txt")
    with open("../data/test.txt",'w',encoding = 'utf-8') as f:
        f.write(input_2)
    fil.close() 
    filePath = "../data/test.txt"
    X, Y = ob_enc.get_all_possible_codon_list(filePath, 39, 99, 0)

    return X, Y

def _pred(data_g):
#    print("In single data prediction...!")
    ob_enc = edr.Encoder()
    X= ob_enc.get_all_possible_codon_list_predict(data_g)

    return X
