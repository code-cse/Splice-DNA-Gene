DNA-Level Splice Junction Prediction.

Dataset:

The domain consists of 60 variables, representing a sequence of DNA bases an additional class Variable.
The task is to determine if the middle of the sequence is a splice junction and what is its type:
Splice junctions are of two types:

1. exon-intron (EI): represents the end of an exon and the beginning of an intron
2. intron-exon (IE):  represents where the intron ends and the next exon, or coding section, begins.

So the class variable contains 3 values:
1. exon-intron (EI)
2. intron-exon (IE)
3. No-Junction

Other 4 values corrosponding to the 4 possible DNA bases (C, A, G, T)
C : Cytosine
A : Adenine
G : Guanine
T : Thymine

Lenght of gene sequence must be : multiple of 60

The dataset consists of 3,175 labeled examples. We use 5 fold cross validation to train and validate our model.
Here, the data set is split into 5 folds. In the first iteration, the first fold is used to test the model and the rest are used to train the model. In the second iteration, 2nd fold is used as the testing set while the rest serve as the training set. This process is repeated until each fold of the 5 folds have been used as the testing set.

3-words: codons
 Let’s again assume a sequence L of independent bases
 L = TCCCTCTGTTGCCCTCTGGTTTCTCCCCAGGCTCCCGGACGTCCCTGCTCCTGGCTTTTG
 Probability of 3-word r1,r2,r3 at position i,i+1,i+2 in sequence L is
<<<<<<< HEAD
 P(Li = r1, Li+1 = r2, Li+2 = r3) = P(Li = r1)P(Li+1 = r2)P(Li+2 = r3)


=======
 P(Li = r1, Li+1 = r2, Li+2 = r3) = P(Li = r1)P(Li+1 = r2)P(Li+2 = r3)
>>>>>>> 2cdd3241e3c7fb12db6843ceb75bd796bea8dd38
