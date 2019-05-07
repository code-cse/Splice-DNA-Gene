import test as tst
# import tensorflow as tf
import CrossValDataPreparation as cvp

def predict_(seq_data):
	seq_d = cvp._pred(seq_data)
	result_set = ["Donor-EI", "Acceptor-IE", "No-Junction"]
	# print(result)
	model_path = "../weight/model.ckpt"
	prediction = tst.model_predict(seq_d,model_path)
	index = prediction[0].argmax()
	res = result_set[index]
	su = prediction[0][0] + prediction[0][1] + prediction[0][2]
	print(res)
	
	return res



# print("Model prediction and ROC visualization")
# # X,Y = cvp.data_set_1()
# # X, Y = cvp.single_pred()
# data_g = ["TCAACTACAGGGACCCGCATCTCCCTACAGGGTTGGCCCCCAGCAAGGCTCAGGACAGCA"]
# data5 = ["CTAAGTTGTCCTTTTCTGGTTTCGTGTTCACCATGGAACATTTTGATTATAGTTAATCCT"]
# # print(len(data_g[0]),"=================")
# pred = predict_(data3)
