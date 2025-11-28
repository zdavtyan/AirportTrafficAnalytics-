import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

start_year = 2015
end_year = 2024
start_col = (0,255,0)
end_col = (255,0,0)
gradient = [f"#{int(i*end_col[0]+(1-i)*start_col[0]):0{2}X}{int(i*end_col[1]+(1-i)*start_col[1]):0{2}X}{int(i*end_col[2]+(1-i)*start_col[2]):0{2}X}" for i in np.linspace(0,1,num = (end_year-start_year+1))]
print(gradient)

def onecol(filename, title, ylabel, xlabel, chartname):
    plt.cla()
    plt.clf()
    in_file = pd.read_csv(f"{filename}.csv", header=None,dtype=float)
    y_data=[]
    for index, row in in_file.iterrows():
        y_data.append(float(row.iloc[0]))
    plt.plot(y_data, color=gradient[-1])
    plt.xticks(np.arange(end_year-start_year+1),np.arange(start_year,end_year+1))
    plt.grid()
    plt.title(title)
    plt.ylabel(ylabel)
    plt.xlabel(xlabel)
    plt.savefig(f"{chartname}.png")

def multicol(filename, title, ylabel, xlabel, chartname):
    plt.cla()
    plt.clf()
    in_file = pd.read_csv(f"{filename}.csv", header=None, dtype=float)
    plt.grid()
    for index, row in in_file.iterrows():
        y_data = [float(x) for x in list(row)]
        plt.scatter(y=y_data, x=np.arange(len(y_data)), label=f"{index+start_year}", color=gradient[index], s=2)
    plt.loglog()
    plt.title(title)
    plt.ylabel(ylabel)
    plt.xlabel(xlabel)
    plt.legend()
    plt.savefig(f"{chartname}.png")


if __name__=="__main__":
    onecol("generated_data/avg_spl", "Average Shortest Path Length Per Year", "Average Shortest Path Length", "Year", "generated_charts/avg_spl")
    onecol("generated_data/cluster", "Average Clustering Coefficient Per Year", "Average Clustering Coefficient (C)", "Year","generated_charts/cluster")
    onecol("generated_data/density", "Average Density Per Year", "Average Clustering Density","Year","generated_charts/density")
    multicol("generated_data/CCDF","Strength Distribution (CCDF)","CCDF", "Node Degree","generated_charts/CCDF")
