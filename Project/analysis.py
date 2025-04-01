# Analysis
# First analysis of Iris data set
# By Ignacio Riboldi

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import datasets

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

# HISTOGRAMS

data = datasets.load_iris()

df = pd.DataFrame(data.data, columns=data.feature_names)


# Create histogram for each variable
for i, feature in enumerate(df.columns):
    plt.subplot(2, 2, i+1) # Organice the graphics in 2 files and 2 columns. 
    sns.histplot(df[feature], bins=20, kde=True, color='skyblue') # Using "feature" to do the graphics with all the variables in the data set.
    plt.title(f'Histogram of {feature}')
    plt.xlabel(feature)
    plt.ylabel('frequency')

plt.savefig('histogramas_variables.png', format='png')
plt.close()

sns.pairplot(df)
plt.suptitle("Scatter plot") 
plt.show()
