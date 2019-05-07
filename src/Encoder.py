import os
import numpy as np
from sklearn.preprocessing import OneHotEncoder, LabelEncoder

class Encoder:

    def getSeqList(self, filePath, seqSt, seqEd, labSt):
        '''
        filePath: path to the sequence file
        seqSt: the common starting posistion of the sequences in the file. For the dataset we use seqSt=39
        seqEd: the common ending position of the sequences in the file + 1. For the dataset we use seqEd=99
        labSt: the common starting position of the labels in the file. End of the labels are detected automatically for this dataset(commas are used to detect the end).
        '''
        ##Reading the sequences and labels##
        seqlen = seqEd - seqSt
        with open(filePath) as file:
            content = file.readlines()
            file.close()
        dataList = [x.strip() for x in content]
        X = []
        y= []
        for item in dataList:
            flag=1
            if len(item) > seqlen:
                _seq = item[seqSt:seqEd]
                _label = item[labSt:item.find(",")]
                ##Removing incompletely specified sequences##
                for ch in _seq:
                    if ch in ['D','N','S','R']:
                        flag=0
                if flag == 1:
                    X.append(_seq)
                    y.append(_label)
        return X,y
    
    def grouping(self, seq):
        '''
        This method groups the sequences in batches of 3

        seq: the sequence to be converted into a sequence of codons
        '''
        seqL = len(seq)
        wordList = []
        st = 0
        for i in range(0,seqL):
            if (i+1)%3 == 0:
                wordList.append(seq[st:i+1])
                st = i+1
        return wordList
    
    def getCodoneList(self, X):
        '''
        This method obtains the shifted codon sequences: 0-shift, 1-shift and 2-shift
        '''
        shiftX0 = []
        shiftX1 = []
        shiftX2 = []
        for item in X:
            shiftX0.append(self.grouping(item))
            shiftX1.append(self.grouping(item[1:-2]))
            shiftX2.append(self.grouping(item[2:-1]))

        return shiftX0, shiftX1, shiftX2
    
    def encodeSeq(self, X):
        '''
        One hot encoding the sequences based on a DNA codon table
        '''
        codonDict=[]
        base=['A','C','G','T']
        for c1 in base:
            for c2 in base:
                for c3 in base:
                    codonDict.append(c1+c2+c3)
        labelEncoder = LabelEncoder()
        labelEncoder.fit(codonDict)
        codonEncoded = labelEncoder.transform(codonDict)
        oneHotEncoder = OneHotEncoder(sparse=False)
        oneHotEncoder.fit(codonEncoded.reshape(-1,1))
        encodedList = []
        for item in X:
            encodedList.append(oneHotEncoder.transform(labelEncoder.transform(item).reshape(-1,1)))
        return encodedList
    
    def encodeLab(self, y):
        '''
        One hot encoding the labels
        '''
        labelEncoder = LabelEncoder()
        y = labelEncoder.fit_transform(y)
        oneHotEncoder = OneHotEncoder(sparse=False)
        y = oneHotEncoder.fit_transform(y.reshape(-1,1))
        return y
        
    def padSeq(self, X, padVectorLen=64):
        '''
        This method is used to pad the uneven sequences
        padvectorLen: is equal to the length of the one hot encoded vector
        '''
        c=0
        for item in X:
            X[c] = np.append(item, [[0]*padVectorLen], axis = 0)
            c+=1
        return X
    
    def encodeAndPad(self, X, vectorLen):
        '''
        This method provides a call to encodeLab() and padSeq() to avoid the user making separate calls

        vectorLen: is the length of the codon dictionary
        '''
        encoded_X = self.encodeSeq(X)
        padded_X = self.padSeq(encoded_X, vectorLen)
        return padded_X
    
    def join_lists(self, *args):
        '''
        Number of arguments = 3(ABSOLUTE CONDITION)
        '''
        l = len(args)
        listGroup = []
        for item in enumerate(args[0]):
            idx = item[0]
            listGroup.append([args[0][idx], args[1][idx], args[2][idx]])
        return listGroup
    
    def get_all_possible_codon_list(self, filePath, seqSt, seqEd, labSt):
        '''
        filePath: path to the sequence file
        seqSt: the common starting posistion of the sequences in the file. For the dataset we use seqSt=39
        seqEd: the common ending position of the sequences in the file + 1. For the dataset we use seqEd=99
        labSt: the common starting position of the labels in the file. End of the labels are detected automatically for this dataset(commas are used to detect the end).
        padvectorLen: is equal to the length of the one hot encoded vector
        '''
        X, y = self.getSeqList(filePath,39,99,0)
        print("**********************")
        shift0, shift1, shift2 = self.getCodoneList(X)
        enShift0 = self.encodeSeq(shift0)
        enShift1 = self.encodeAndPad(shift1,64)
        enShift2 = self.encodeAndPad(shift2,64)
        y = self.encodeLab(y)

        return self.join_lists(enShift0, enShift1, enShift2), y

    def get_all_possible_codon_list_predict(self, seq_data):
        '''
        seq_data: gene squence data
        '''
        X = seq_data
        print("**********************")
        shift0, shift1, shift2 = self.getCodoneList(X)
        enShift0 = self.encodeSeq(shift0)
        enShift1 = self.encodeAndPad(shift1,64)
        enShift2 = self.encodeAndPad(shift2,64)
        
        return self.join_lists(enShift0, enShift1, enShift2)
