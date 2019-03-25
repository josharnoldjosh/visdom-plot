import visdom
import subprocess
import torch
from data import Epoch
from sklearn.metrics import confusion_matrix
import numpy as np
import calc

LINE_TYPE = "window_type_line"
CM_TYPE = "window_confusion_matrix"

class Plot:		
	"""
	Use the follow methods to initialize a plot:
	- line (for line plot)
	- cm (for confusion matrix plot)

	Use the follow method to update data sources (note, this does not display changes)
	- update

	Use the follow method to visually display new results:
	- plot
	"""
	def __init__(self, model_name="Model"):		
		self.vis = visdom.Visdom() # must run python -m visdom.server			
		self.model_name = model_name
		self.plots = {}
		return

	def clear(self):
		self.vis.close()

	def line(self, plot_name, xlabel, ylabel, legend_ext):		
		window = self.vis.line(
			Y=torch.zeros((1)).cpu(), 
			X=torch.zeros((1)).cpu(), 
			opts=dict(xlabel=xlabel,ylabel=ylabel,title=self.model_name+' '+plot_name,
				legend=['Train '+legend_ext, 'Validation '+legend_ext, 'Test '+legend_ext]))	

		self.plots[plot_name] = {"type":LINE_TYPE, "window":window, "idx":0, "train":[], "val":[], "test":[], "legend_ext":legend_ext}

	def cm(self, plot_name):
		window = self.vis.heatmap(
		X=[[0, 0], [0, 0]],
		opts=dict(
		columnnames=['Positive', 'Negative'],
		rownames=['True', 'False'],
		colormap='Electric', title=self.model_name+' '+plot_name
		))
		self.plots[plot_name] = {"type":CM_TYPE, "window":window, "data":[]}

	def update(self, plot_name, data=[], train=0, val=0, test=0):
		"""
		Appends data to the plot.
		"""
		if self.plots[plot_name]["type"] == LINE_TYPE:
			if train != 0:
				self.plots[plot_name]["train"] += [train]
			if val != 0:
				self.plots[plot_name]["val"] += [val]
			if test != 0:
				self.plots[plot_name]["test"] += [test]

		if self.plots[plot_name]["type"] == CM_TYPE:
			self.plots[plot_name]["data"].append(data)

	def plot_line_update(self, plot, key):
		window = plot["window"]
		x = plot["idx"]

		train_y = calc.average_array(plot["train"])		
		if train_y != 0:						
			self.vis.line(X=torch.ones((1,1)).cpu()*x,
			Y=torch.Tensor([train_y]).unsqueeze(0).cpu(),
			win=window, update='append', name='Train '+plot["legend_ext"])

		val_y = calc.average_array(plot["val"])
		if val_y != 0:						
			self.vis.line(X=torch.ones((1,1)).cpu()*x,
			Y=torch.Tensor([val_y]).unsqueeze(0).cpu(),
			win=window, update='append', name='Val '+plot["legend_ext"])

		test_y = calc.average_array(plot["test"])
		if test_y != 0:						
			self.vis.line(X=torch.ones((1,1)).cpu()*x,
			Y=torch.Tensor([test_y]).unsqueeze(0).cpu(),
			win=window, update='append', name='Test '+plot["legend_ext"])

		plot["idx"] = x+1
		plot["train"] = []
		plot["val"] = []
		plot["test"] = []
		self.plots[key] = plot
		return

	def plot_cm_update(self, plot, key):
		window = plot["window"]
		(tp, tn, fp, fn) = calc.average_arrays(plot["data"])
		window = self.vis.heatmap(win=window,
		X=[[tp, fn], 
		[fp, tn]],
		opts=dict(
		columnnames=['Positive', 'Negative'],
		rownames=['True', 'False'],
		colormap='Electric', title=self.model_name+' '+key, update="replace", name=self.model_name+' '+key
		))

	def plot(self):
		"""
		Updates the plot data.
		"""
		for key in self.plots.keys():
			plot = self.plots[key]

			if plot["type"] == LINE_TYPE:
				self.plot_line_update(plot, key)

			if plot["type"] == CM_TYPE:
				self.plot_cm_update(plot, key)
		return