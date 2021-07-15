# importing required modules
import warnings
warnings.filterwarnings("ignore")
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import numpy as np
from tabulate import tabulate



def heatmap(data, method):
    """

    :param data: dataframe object containing data and features
    :param method: method for heat maps ("pearson", "kendall", "spearman")
    :return: None
    """
    sns.set_theme()
    ax = sns.heatmap(data.corr(method= method))
    plt.title(method+" Correlation")
    plt.show()




def histogram(data):
    """
    :param data: dataframe object containing data and features
    :return: None
    """
    for i in data.columns:
        sns.displot(data, x=i)
        plt.title("Histogram between "+str(i)+" feature and frequency")
        plt.show()



def boxplot(data):
    """

    :param data: dataframe object containing data and features
    :return: None
    """
    for i in data.columns:
        f, (ax_hist, ax_box) = plt.subplots(2, sharex=True, gridspec_kw={"height_ratios": (.5, .5)})
        plt.title(str(i) + ' feature')
        sns.boxplot(data[i], ax=ax_box)
        sns.distplot(data[i], ax=ax_hist)
        plt.show()


def scatterplot(data):
    """
    :param data: dataframe object containing data and features
    :return: None
    """
    for i in data.columns:
        for j in data.columns:
            sns.scatterplot(data=data, x=i, y=j, hue="temp")
            plt.show()



if __name__ == '__main__':


    data = pd.read_csv("./datasets/bike-sharing/hour.csv")
    pd.set_option('display.max_rows', 1000)
    print(tabulate(data.agg(['min', 'max','mean',np.std,'skew',pd.DataFrame.kurt]).T, headers='keys', tablefmt='psql'))


    data = data.drop(columns=['dteday'])
    heatmap(data,"pearson")
    heatmap(data,"kendall")
    heatmap(data, "spearman")
    histogram(data)
    boxplot(data)
    scatterplot(data)

