from data import DataLoader
from data import Epoch
from model import Model
from model import Loss
from model import Optimizer
from visual import Plot
import calc

data = DataLoader()
model = Model()
loss = Loss()
optimizer = Optimizer(model)

plot = Plot("Baseline")
plot.clear()
plot.line("Loss", "Epoch", "Loss", "loss")
plot.line("Accuracy", "Epoch", "Accuracy (%)", "acc")
plot.line("F1 Score", "Epoch", "F1 Score", "score")
plot.line("Recall", "Epoch", "Recall", "recall")

plot.cm("Confusion Matrix (Train)")
plot.cm("Confusion Matrix (Val)")

for epoch in Epoch():

	print("Epoch", epoch)
	
	# train model
	for X, y in data.train:	
		optimizer.zero_grad()
		y_hat = model(X)
		cost = loss(y_hat, y)
		cost.backward()
		optimizer.step()
		plot.update("Loss", train=Epoch.get_cost(cost))
		plot.update("Accuracy", train=calc.accuracy(y_hat, y))
		plot.update("F1 Score", train=calc.f1(y_hat, y))
		plot.update("Recall", train=calc.recall(y_hat, y))
		plot.update("Confusion Matrix (Train)", data=calc.cm(y_hat, y))

	# test model on validation set
	for X, y in data.val:
		y_hat = model(X)
		cost = loss(y_hat, y)
		plot.update("Loss", val=Epoch.get_cost(cost))
		plot.update("Accuracy", val=calc.accuracy(y_hat, y))
		plot.update("F1 Score", val=calc.f1(y_hat, y))
		plot.update("Recall", val=calc.recall(y_hat, y))
		plot.update("Confusion Matrix (Val)", data=calc.cm(y_hat, y))

	# update plot
	plot.plot()

print("Script done.")