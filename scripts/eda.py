import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns



# Function to calculate central tendency (mean, median, mode)
def calculate_central_tendency(df, column):
    mean = df[column].mean()
    median = df[column].median()
    mode = df[column].mode().tolist()  # Handling multiple modes by converting to list
    return {"mean": mean, "median": median, "mode": mode}

# Function to calculate dispersion (range, variance, standard deviation)
def calculate_dispersion(df, column):
    data_range = df[column].max() - df[column].min()
    variance = df[column].var()
    std_dev = df[column].std()
    return {"range": data_range, "variance": variance, "std_dev": std_dev}

# Function to calculate shape of distribution (skewness, kurtosis)
def calculate_distribution_shape(df, column):
    skewness = df[column].skew()
    kurtosis = df[column].kurtosis()
    return {"skewness": skewness, "kurtosis": kurtosis}

# Main function to print all statistical information
def print_statistics(df, column):
    central_tendency = calculate_central_tendency(df, column)
    dispersion = calculate_dispersion(df, column)
    distribution_shape = calculate_distribution_shape(df, column)

    print(f"Statistics for column: {column}")
    print("Central Tendency:")
    print("Mean:", central_tendency["mean"])
    print("Median:", central_tendency["median"])
    print("Mode:", central_tendency["mode"])
    
    print("Dispersion:")
    print("Range:", dispersion["range"])
    print("Variance:", dispersion["variance"])
    print("Standard Deviation:", dispersion["std_dev"])
    
    print("Shape of Distribution:")
    print("Skewness:", distribution_shape["skewness"])
    print("Kurtosis:", distribution_shape["kurtosis"])




# Function to visualize the histogram and density plot
def plot_distribution(df, column):
    plt.figure(figsize=(10, 6))
    
    # Plotting Histogram and KDE (Kernel Density Estimation)
    sns.histplot(df[column], kde=True, bins=30, color='blue')
    
    plt.title(f"Distribution of {column}")
    plt.xlabel(column)
    plt.ylabel('Frequency')
    plt.show()

# Function to visualize boxplot (for outliers detection)
def plot_boxplot(df, column):
    plt.figure(figsize=(8, 4))
    
    # Plotting Boxplot
    sns.boxplot(x=df[column], color='orange')
    
    plt.title(f"Boxplot of {column}")
    plt.show()

# Main function to visualize both the histogram and boxplot
def visualize_distribution(df, column):
    print(f"Visualizing the distribution of column: {column}")
    
    # Plotting distribution and boxplot
    plot_distribution(df, column)
    plot_boxplot(df, column)



# Function to generate a heatmap for the correlation between two columns
def plot_correlation_heatmap(df, column1, column2):
    plt.figure(figsize=(6, 4))
    
    # Calculating the correlation
    correlation_matrix = df[[column1, column2]].corr()
    
    # Creating heatmap
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', vmin=-1, vmax=1, cbar=True)
    
    plt.title(f"Correlation Heatmap between {column1} and {column2}")
    plt.show()

# Function to generate scatter plot to visualize relationship
def plot_scatter(df, column1, column2):
    plt.figure(figsize=(8, 6))
    
    # Scatter plot with regression line
    sns.regplot(x=column1, y=column2, data=df, scatter_kws={'color':'blue'}, line_kws={'color':'red'})
    
    plt.title(f"Scatter Plot of {column1} vs {column2}")
    plt.xlabel(column1)
    plt.ylabel(column2)
    plt.show()

# Main function to visualize both the correlation heatmap and scatter plot
def visualize_correlation(df, column1='Amount', column2='Value'):
    print(f"Visualizing correlation between {column1} and {column2}")
    
    # Plot heatmap and scatter plot
    plot_correlation_heatmap(df, column1, column2)
    plot_scatter(df, column1, column2)
