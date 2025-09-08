import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from  main import *

#We will chwck for class imbalances and the distrubution of columns in our dataset

data = create_data()
drop_columns(data)



def scatter_plot():
    for col in data.columns:
        sns.scatterplot(x=data[col],y = data.iloc[:,4])
        plt.xlabel(col)
        plt.ylabel(data.iloc[:,4])
        plt.show()



def correlation_plot():
    """"
       To see if some features effect other features and who effects
       our y values(life expectency) the most.
    """

    corr = data.corr(numeric_only=True)
    sns.heatmap(corr,annot=True,cmap="crest",annot_kws={"size": 4} )
    plt.show()



def label_imb():
    """"
    To show if there are any class imbalances
    """

    y = data.iloc[:,4]
    y.hist(bins= 20,figsize= (15,10))
    plt.title('Distribution of Y classes')
    plt.show()

label_imb()
#correlation_plot()








