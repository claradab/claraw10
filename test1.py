from claraw10 import clustering
from claraw10 import plotting
from sklearn import datasets
import numpy as np

iris = datasets.load_iris()

x=iris.data
y=iris.target
numberofclusters=3
clustering.create_clusters(x,y,3)

boston = datasets.load_boston()
boston['MEDV']=boston.target

y=boston['MEDV']
numberofbins=30
plottitle="Test Title"
plotting.plotdistribution(y,numberofbins,plottitle)

x = np.array(boston.data[:,5])
y = np.array(boston.target)
xtitle = "Number of Rooms"
ytitle = "Median Value of Owner-Occupied Homes ($1000's)"
graphtitle = "Effect of Number of Rooms on Home Values"
outlier_treatment = 'color'
outlier_sensitivity = 1.4
plotting.scattergraph(x,y,xtitle,ytitle,graphtitle,outlier_treatment,outlier_sensitivity)
