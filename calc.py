import torch
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score
from sklearn.metrics import recall_score
import numpy as np

def torch_to_numpy(y):
	if torch.cuda.is_available():
		return y.detach().cpu().numpy()
	return y.detach().numpy()

def cont_to_binary(y):	
	return [1 if x >= 0.5 else 0 for x in y]

def recall(y_hat, y):
	y = torch_to_numpy(y)
	y_hat = cont_to_binary(torch_to_numpy(y_hat))
	return recall_score(y, y_hat)

def f1(y_hat, y):
	y = torch_to_numpy(y)
	y_hat = cont_to_binary(torch_to_numpy(y_hat))		
	return f1_score(y, y_hat)

def accuracy(y_hat, y):	
	final_y_hat = []		
	if torch.cuda.is_available():
		y_hat = y_hat.detach().cpu().numpy()
		y = y.detach().cpu().numpy()
	else:
		y_hat = y_hat.detach().numpy()
		y = y.detach().numpy()		
	final_y_hat += [1 if x > 0.5 else 0 for x in y_hat]		
	return (sum(1 for a, b in zip(final_y_hat, y) if a == b) / float(len(final_y_hat)))*100

def cm(y_hat, y):
	final_y_hat = []
	final_y = []
	if torch.cuda.is_available():
		y_hat = y_hat.detach().cpu().numpy()
		y = y.detach().cpu().numpy()
	else:
		y_hat = y_hat.detach().numpy()
		y = y.detach().numpy()		
	final_y_hat += [1 if x > 0.5 else 0 for x in y_hat]
	final_y += [1 if x > 0.5 else 0 for x in y]
	tn, fp, fn, tp = confusion_matrix(final_y, final_y_hat).ravel()
	# False Positive, False, negative, True positive, true negative 
	return [tp, tn, fp, fn]	

def average_array(data):
	if data == []: return 0
	return sum(data)/len(data)

def average_arrays(data):
	return np.average(data, axis=0)