import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
data = pd.read_csv('benin-malanville.csv')

# Display the first five rows of the data
print(data.describe())
# Data Quality Check
# Check for missing values
print(data.isnull().sum())

# for col in ['GHI', 'DNI', 'DHI']:
#     plt.figure(figsize=(10, 6))
#     sns.boxplot(x=data[col])
#     plt.title(f'Boxplot of {col}')
#     plt.show()

# # Check for outliers in sensor readings and wind speed
# for col in ['ModA', 'ModB', 'WS', 'WSgust']:
#     plt.figure(figsize=(10, 6))
#     sns.boxplot(x=data[col])
#     plt.title(f'Boxplot of {col}')
#     plt.show()

# # Check for incorrect entries (e.g., negative values)
# print(data[data['GHI'] < 0])  # Example: Check for negative GHI values
# print(data[data['DNI'] < 0])  # Example: Check for negative DNI values
# print(data[data['DHI'] < 0])  # Example: Check for negative DHI values

# Time Series Analysis:
# Plot GHI, DNI, DHI, and Tamb over time
plt.figure(figsize=(12, 8))
plt.plot(data['Timestamp'], data['GHI'], label='GHI')
plt.plot(data['Timestamp'], data['DNI'], label='DNI')
plt.plot(data['Timestamp'], data['DHI'], label='DHI')
plt.plot(data['Timestamp'], data['Tamb'], label='Tamb')
plt.xlabel('Timestamp')
plt.ylabel('Value')
plt.legend()
plt.title('Time Series of Solar Radiation and Temperature')
plt.show()

# Evaluate the impact of cleaning on sensor readings
plt.figure(figsize=(12, 8))
plt.plot(data['Timestamp'], data['ModA'], label='ModA (Raw)')
plt.plot(data['Timestamp'], data[data['Cleaning'] == 1]['ModA'], label='ModA (Cleaned)')
plt.xlabel('Timestamp')
plt.ylabel('ModA Value')
plt.legend()
plt.title('Impact of Cleaning on ModA Sensor Readings')
plt.show()

# Correlation Analysis:
# Correlation matrix
corr_matrix = data[['GHI', 'DNI', 'DHI', 'TModA', 'TModB']].corr()
sns.heatmap(corr_matrix, annot=True)
plt.title('Correlation Matrix')
plt.show()

# Pair plot
sns.pairplot(data[['GHI', 'DNI', 'DHI', 'TModA', 'TModB']])
plt.show()

# Scatter matrix for wind conditions and solar irradiance
sns.pairplot(data[['WS', 'WSgust', 'WD', 'GHI']])
plt.show()

# Wind Analysis:
# Wind rose
sns.jointplot(x=data['WS'], y=data['WD'], kind="hex", color="k")
plt.show()

# Temperature Analysis:
# Relationship between RH and temperature
sns.scatterplot(x=data['RH'], y=data['Tamb'])
plt.xlabel('Relative Humidity')
plt.ylabel('Temperature')
plt.title('Relationship between RH and Tamb')
plt.show()

# Histograms:
plt.figure(figsize=(12, 8))
plt.subplot(2, 3, 1)
plt.hist(data['GHI'], bins=20)
plt.title('GHI Histogram')
# ... similar plots for DNI, DHI, WS, and temperatures

# Z-Score Analysis:
z_scores = (data - data.mean()) / data.std()
outliers = data[z_scores.abs() > 3]  # Adjust threshold as needed

# Bubble Chart:
plt.figure(figsize=(10, 8))
plt.scatter(data['GHI'], data['Tamb'], s=data['WS']*10, c=data['RH'], alpha=0.5)
plt.xlabel('GHI')
plt.ylabel('Tamb')
plt.title('GHI vs. Tamb with WS and RH')
plt.colorbar()
plt.show()

# Data Cleaning:
# Handle missing values
data.fillna(method='ffill', inplace=True)  # Replace missing values with previous value

# Handle anomalies
# ... (e.g., remove outliers based on z-scores or domain knowledge)

# Drop the 'Comments' column
data.drop('Comments', axis=1, inplace=True)
