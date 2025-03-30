# Analysis
# First analysis of Iris data set
# By Ignacio Riboldi

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt 

def generate_summary():
    
    #Load Iris Data Set
    iris = sns.load_dataset('iris')
    
    # create a resume of dataset
    summary = iris.describe()

    # Creates a new txt file
    with open('iris_summary.txt', 'w') as file:
        file.write("Summary of Iris dataset:")
        
        # Statistics
        file.write(summary.to_string())
        
    print("Iris summary was created on iris_summary.txt")

if __name__ == "__main__":
    generate_summary()
