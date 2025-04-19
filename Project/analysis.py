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

    # Display feature names
    print("Feature names:", iris.columns[:-1].tolist())

    # Display target names (target classes)
    print("Target classes:", iris['species'].unique().tolist())
    
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
    # Title and labels
    plt.title(f'Histogram of {feature}')
    plt.xlabel(feature)
    plt.ylabel('frequency')
# Saving in PNG
plt.savefig('histogramas_variables.png', format='png')
plt.close()

sns.pairplot(df)
plt.suptitle("Scatter plot") 
plt.show()


# Create the line plot for each feature

from sklearn.datasets import load_iris

iris_data = load_iris()
iris = pd.DataFrame(iris_data.data, columns=iris_data.feature_names)
iris['species'] = pd.Categorical.from_codes(iris_data.target, iris_data.target_names)

# Rename columns to simpler names (optional)
iris.columns = [col.replace(' (cm)', '').replace(' ', '_') for col in iris.columns]

# List of features to plot
features = iris.columns[:-1]  # exclude 'species'

# Create line plots for each feature
for feature in features:
    plt.figure(figsize=(10, 6))
    sns.lineplot(data=iris, x=iris.index, y=feature, hue='species', palette='Set2')

    # Customize the plot
    plt.title(f'{feature.replace("_", " ").title()} in the Iris Dataset')
    plt.xlabel('Index')
    plt.ylabel(f'{feature.replace("_", " ").title()} (cm)')
    plt.legend(title='Species')
    plt.grid(True)
    plt.show()
