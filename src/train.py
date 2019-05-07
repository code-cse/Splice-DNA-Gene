####Importing necessary libraries####
import tensorflow as tf
import numpy as np
from scipy import interp
from matplotlib import pyplot as plt
from sklearn.metrics import roc_curve, auc
import CrossValDataPreparation as cvp
import os
import base as bs


def model_train(X_train, y_train, X_test, y_test, train_steps=10000, weight_path="", 
                n_folds=5, esPatience=15, lrPatience=10, epsilon=3, lr_decay = 0.1,
                per_process_gpu_memory_fraction=0.925, log_path_train=""):
    '''
    X_train and y_train: Sets of cross validation training sets
    X_test and y_test: Sets of corresponding cross validation test sets
    
    train_steps: number of training epochs
    weight_path: path to save the weights
    n_folds: the number of cross validation folds
    esPatience: patience for early stopping
    lrPatience: patience for learning rate reduction
    epsilon: number of places after decimal to which the loss is scalled
    lr_decay: learning rate decay factor
    per_process_gpu_memory_fraction: percentage of gpu memory allowed
    log_path_train: path to which log files are saved
    
    We used 5-fold cross validation for our contribution 
    i.e., X_train and X_test contain 5 sets of training and validation sets representing every possible combination of the 5 folds
    '''
    final_output, cross_entropy, train_step, correct_prediction, \
                    accuracy, input_data, target = bs.BaseModel.base_model() 

    taccList = []
    tlossList = []
    weight_path = "../weight/"
    
    gpuOpt = tf.GPUOptions(per_process_gpu_memory_fraction=per_process_gpu_memory_fraction)
    saver = tf.train.Saver()
    
    loss_summary = tf.summary.scalar(name='loss', tensor=cross_entropy)
    accuracy_summary = tf.summary.scalar(name="accuracy", tensor=accuracy)
    
    with tf.Session(config = tf.ConfigProto(gpu_options = gpuOpt)) as sess:
        sess.run(tf.global_variables_initializer())
        train_writer = tf.summary.FileWriter(log_path_train, sess.graph)
        summaries_train = tf.summary.merge_all()
        ##Dummy initial accuracy and loss values##
        mAcc=0
        lLoss = 999
        ##Patience counter##
        lCounter=0
        for step in range(train_steps):
            ##validation loss and accuracy lists##
            lList=[]
            aList=[]
            ##training loss and accuracy list##
            ta = []
            tl = []
            
            for fold in range(0,n_folds):
                _, tacc, tloss = sess.run([train_step, accuracy, cross_entropy], feed_dict = {input_data:X_train[fold],
                                                      target: y_train[fold]})
                ta.append(tacc)
                tl.append(tloss)
                summary_str, acc, loss = sess.run([summaries_train, accuracy, cross_entropy], 
                                            feed_dict = {input_data:X_test[fold], target: y_test[fold]})
                aList.append(acc)
                lList.append(loss)
            taccList.append(np.mean(ta))
            tlossList.append(np.mean(tl))
            train_writer.add_summary(summary_str, global_step=step)
            if np.mean(aList) > mAcc:
                mAcc = np.mean(aList)
                save_path = saver.save(sess, "../weight/model.ckpt")
                print(save_path)
                # saver.save(sess, os.path.join(weight_path, str(step), "model.ckpt"))#, global_step=step)
                print("Step %d:" % (step), aList, lList)
                print("-"*90)
                print("Accuracy and loss at %d: %f and %f" % (step,np.mean(aList),np.mean(lList)))
                print("."*90)
            if round(np.mean(lList),epsilon) != lLoss:
                lLoss = round(np.mean(lList),epsilon)
                lCounter = 0
            else:
                lCounter += 1
            if lCounter >= esPatience:
                break   
            if lCounter >= lrPatience:
                lrate *= lr_decay
        train_writer.close()
    taccList = np.asarray(taccList, dtype="float32")
    tlossList = np.asarray(tlossList, dtype="float32")
    np.save("accuracy.npy",taccList)
    np.save("loss.npy",tlossList)

    

print("Dataset prepration...................")
X_train, y_train, X_test, y_test = cvp.data_set()

print("Start training and save the checkpoint....")
model_train(X_train, y_train, X_test, y_test)
