# Visualizing Your Neural Network in Real Time with Visdom

Here's a script that I wrote that provides a simple framework for plotting your data.
It provides built in support for line plots and confusion matrices.

![](https://media.giphy.com/media/4NiFnKc1Kr1KqPq984/giphy.gif)

## Install

First, we have to install visdom and torch:

```
pip install visdom
pip3 install torch torchvision
```

## Usage

```python
from visual import Plot
plot = Plot("Baseline") # Initialize your plot with a name, e.g, "Baseline"
plot.clear() # optionally clear all the previous plots
plot.line("Loss", "Epoch", "Loss", "loss") # define a line plot: plot_name, xlabel, ylabel, legend name
```
Each time you get a new value you want to plot, e.g, after each batch you calculate the back propagation error,
you simply call "update" and pass in a value. Note: the first value, in this case, "Loss", must be the "plot_name"
of your plot. This doesn't update the plot visually, but just appends data to later be updated!

```python
for epoch in Epoch():		
	for X, y in data.train:	
		# train your model
		# ...
		plot.update("Loss", train=cost)
```

When you want to evaluate your model, you can do the same thing. Notice we pass the cost in as "val" instead
of "train".

```python
for epoch in Epoch():		
	for X, y in data.test:	
		# test your model
		# ...
		plot.update("Loss", val=cost)
```

Lastly, after each epoch, update your plot! Simply call: 

```python
plot.plot()
```

And thats it. All your data will be beautifully plotted. 

In summary, the steps are:

```python
plot = Plot("Baseline") # define plot
plot.clear() # optionally clear it
plot.line("Loss", "Epoch", "Loss", "loss") # create a line plot: : plot_name, xlabel, ylabel, legend name
plot.update("Loss", train=cost_train, val=cost_val, test=cost_test) # use update to add data to the plot
plot.plot() # use this to visually SHOW the data you added to the plot. You must call this. 
```

Finally, before running your script, you must execute in another terminal:

```
python -m visdom.server
```

In order to start running a Visdom server. Navigate to `http://localhost:8097`

Here are the results:

I'm really excited about visdom, because with a simply class extension,
we can make plotting and visualizing our data incredibly easy.
That means more time running experiments and less time debugging plots!

Note, the plot class just plots numbers, it doesn't calculate them from the raw PyTorch tensors. 
I wrote an additional class, calc.py, that supports functions for calculating values to directly go into your plots.
Notice the "calc.accuracy" for accuracy score, "calc.f1" for f1 and "calc.recall" for recall, all from
raw y_hat and y PyTorch tensors.

Example:

```python
plot.update("Accuracy", train=calc.accuracy(y_hat, y))
plot.update("F1 Score", train=calc.f1(y_hat, y))
plot.update("Recall", train=calc.recall(y_hat, y)
```

For a full script example, check out `run.py` in this repository.
