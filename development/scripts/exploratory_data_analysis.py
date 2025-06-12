import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

file_path = 'AI4I 2020 Predictive Maintenance Dataset.csv'
df = pd.read_csv(file_path)

print(df.head())
print(df.shape)
print(df.isnull().sum())
print(df.describe())

#Create a box plot
df_shortened = df.rename(columns={'Air temperature [K]': 'Air Temp', 'Process temperature [K]': 'Process Temp', 'Rotational speed [rpm]': 'Speed', 'Torque [Nm]': 'Torque', 'Tool wear [min]': 'Wear'})
sns.boxplot(data=df_shortened[['Air Temp', 'Process Temp', 'Speed', 'Torque', 'Wear']])
plt.title('Boxplot of Features')
plt.show()

#Speed outliers
feature = 'Rotational speed [rpm]'
Q1 = df[feature].quantile(0.25)
Q3 = df[feature].quantile(0.75)
IQR = Q3 - Q1
outliers = df[((df[feature] < (Q1 - 1.5 * IQR)) | (df[feature] > (Q3 + 1.5 * IQR)))]#.any(axis=1)]
print(outliers.shape[0])
print(outliers.head(10))

#Torque outliers
feature = 'Torque [Nm]'
Q1 = df[feature].quantile(0.25)
Q3 = df[feature].quantile(0.75)
IQR = Q3 - Q1
outliers = df[((df[feature] < (Q1 - 1.5 * IQR)) | (df[feature] > (Q3 + 1.5 * IQR)))]#.any(axis=1)]
print(outliers.shape[0])
print(outliers.head(10))

# Compute correlation matrix of numerical variables
corr_value = df[['Air temperature [K]', 'Process temperature [K]', 'Rotational speed [rpm]', 'Torque [Nm]', 'Tool wear [min]']].corr()

# Create a heatmap of the correlation matrix
sns.heatmap(corr_value, annot=True, cmap='coolwarm')
plt.title('Correlation Matrix Heatmap')
plt.show()

# Create a scatter plot matrix of numerical variables.
df1 = df.sample(frac=0.025)
selected_columns = ['Air temperature [K]', 'Process temperature [K]', 'Rotational speed [rpm]', 'Torque [Nm]', 'Tool wear [min]']
sns.pairplot(df1[selected_columns])
plt.show()

#Proportion of failures
failure_condition = df['Target'] == 1
failure_count = df[failure_condition].shape[0]
total_count = df.shape[0]
proportion_of_failures = failure_count / total_count
print(f'Proportion of failures: {proportion_of_failures:.2%}')

#Types of failures
failure_types = df['Failure Type'].unique()
print(failure_types)

#Feature Statistic Difference
failures = df[df['Target'] == 1]
no_failures = df[df['Target'] == 0]

failures = failures.drop(columns=['Target'])
no_failures = no_failures.drop(columns=['Target'])

failure_stats = failures.describe()
no_failure_stats = no_failures.describe()

print("Failure Cases Statistics:\n", failure_stats)
print("\nNo-Failure Cases Statistics:\n", no_failure_stats)
