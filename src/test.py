import tensorflow as tf
import numpy as np
from scipy import interp
from matplotlib import pyplot as plt
from sklearn.metrics import roc_curve, auc
import CrossValDataPreparation as cvp
import os
import base as bs

def model_roc_visualize(X, y, model_path, per_process_gpu_memory_fraction=0.925):
        '''
        X: training encoded codone lists
        y: respective labels
        model_path: path to the saved best model parameters 
        '''

        final_output, cross_entropy, train_step, correct_prediction, \
                        accuracy, input_data, target = bs.BaseModel.base_model() 

        saver = tf.train.Saver()

        gpuOpt = tf.GPUOptions(per_process_gpu_memory_fraction=per_process_gpu_memory_fraction)
        with tf.Session(config = tf.ConfigProto(gpu_options = gpuOpt)) as sess:
            sess.run(tf.global_variables_initializer())
            saver.restore(sess, model_path)

            ####Getting prediction scores####
            y_score = sess.run(final_output, feed_dict={input_data:X})
            acc, loss = sess.run([accuracy, cross_entropy], feed_dict={input_data:X, target:y})
        
        fpr = dict()
        tpr = dict()
        roc_auc = dict()

        for i in range(3):
            fpr[i], tpr[i], _ = roc_curve(y[:, i], y_score[:, i])
            roc_auc[i] = auc(fpr[i], tpr[i])

        fig = plt.figure(figsize=(8,4))
        plt.axes([0.00,0.00,1.90,0.90])
        plt.xlabel('False Positive Rate', fontsize=20)
        plt.ylabel('True Positive Rate', fontsize=20)
        plt.title('ROC curves for our multi-class classification model',fontsize=18)
        ax1 = plt.axes([0.10, 0.10, 0.5, 0.7])
        ax2 = plt.axes([0.70, 0.10, 0.5, 0.7])
        ax3 = plt.axes([1.30, 0.10, 0.5, 0.7])

        colors = ['blue','orange','green']
        classes = ['Exon-Intron','Intron-Exon','Neither']
        ax1.plot(fpr[0], tpr[0], color=colors[0], linewidth=2,
                    label='(area = {1:0.4f})'
                    ''.format(classes[0], roc_auc[0]), linestyle="--")
        l1 = ax1.legend(loc = "lower right", prop={'size':16})
        l1.set_title(classes[0],prop={'size':16})
        ax2.plot(fpr[1], tpr[1], color=colors[1], linewidth=3,
                    label='(area = {1:0.4f})'
                    ''.format(classes[1], roc_auc[1]), linestyle=":")
        l2 = ax2.legend(loc = "lower right", prop={'size':16})
        l2.set_title(classes[1],prop={'size':16})
        ax3.plot(fpr[2], tpr[2], color=colors[2], linewidth=2,
                    label='(area = {1:0.4f})'
                    ''.format(classes[2], roc_auc[2]))
        l3 = ax3.legend(loc = "lower right", prop={'size':16})
        l3.set_title(classes[2],prop={'size':16})
        if not os.path.exists('visualization'):
            os.makedirs('visualization')
        fig.savefig(os.path.join("visualization/ROC.png"), transparent=True, dpi=fig.dpi, bbox_inches='tight')
        print("Accuracy: %f, Loss: %f" % (acc,loss))
    

def model_predict(X, model_path, per_process_gpu_memory_fraction=0.925):
        '''
        X: training encoded codone lists
        y: respective labels
        model_path: path to the saved best model parameters 
        '''

        final_output, cross_entropy, train_step, correct_prediction, \
                        accuracy, input_data, target = bs.BaseModel.base_model() 

        saver = tf.train.Saver()

        gpuOpt = tf.GPUOptions(per_process_gpu_memory_fraction=per_process_gpu_memory_fraction)
        with tf.Session(config = tf.ConfigProto(gpu_options = gpuOpt)) as sess:
            sess.run(tf.global_variables_initializer())
            saver.restore(sess, model_path)

            ####Getting prediction scores####
            y_score = sess.run(final_output, feed_dict={input_data:X})
            print(y_score)
        return y_score


