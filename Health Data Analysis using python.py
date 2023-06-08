import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load the health data from a CSV file
data = pd.read_csv('health_data.csv')

# Explore the data
print('--- Data Summary ---')
print(data.head())
print('---------------------')

# Analyze basic statistics of the numerical variables
print('--- Basic Statistics ---')
print(data.describe())
print('-----------------------')

# Calculate correlations between variables
correlation_matrix = data.corr()

# Visualize correlations using a heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
plt.title('Correlation Matrix')
plt.show()

# Analyze distributions of numerical variables
plt.figure(figsize=(12, 6))
plt.subplot(2, 2, 1)
sns.histplot(data['Age'], kde=True)
plt.title('Age Distribution')

plt.subplot(2, 2, 2)
sns.histplot(data['BMI'], kde=True)
plt.title('BMI Distribution')

plt.subplot(2, 2, 3)
sns.histplot(data['Glucose'], kde=True)
plt.title('Glucose Distribution')

plt.subplot(2, 2, 4)
sns.histplot(data['BloodPressure'], kde=True)
plt.title('Blood Pressure Distribution')

plt.tight_layout()
plt.show()

# Analyze categorical variables
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
sns.countplot(data['Gender'])
plt.title('Gender Distribution')

plt.subplot(1, 2, 2)
sns.countplot(data['Smoker'])
plt.title('Smoker Distribution')

plt.tight_layout()
plt.show()