import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy import stats
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import warnings
warnings.filterwarnings('ignore')

# Set style for better visualizations
plt.style.use('seaborn-v0_8')  # Using a valid style name
sns.set_theme(style="whitegrid")  # Setting seaborn theme

# Read data from CSV file
print("Loading and preprocessing data...")
df = pd.read_csv('Global_AI_Content_Impact_Dataset.csv')

# Create a comprehensive analysis report
def create_analysis_report():
    print("\n" + "="*80)
    print("COMPREHENSIVE DATA ANALYSIS REPORT".center(80))
    print("="*80)
    
    # Basic Information
    print("\n1. DATASET OVERVIEW")
    print("-"*40)
    print(f"Number of observations: {df.shape[0]:,}")
    print(f"Number of features: {df.shape[1]:,}")
    print("\nFeature Information:")
    print(df.info())
    
    # Descriptive Statistics
    print("\n2. DESCRIPTIVE STATISTICS")
    print("-"*40)
    print(df.describe().round(2))
    
    # Missing Values Analysis
    print("\n3. MISSING VALUES ANALYSIS")
    print("-"*40)
    missing_data = pd.DataFrame({
        'Missing Values': df.isnull().sum(),
        'Percentage': (df.isnull().sum() / len(df) * 100).round(2)
    })
    print(missing_data[missing_data['Missing Values'] > 0])
    
    return df

# Enhanced Visualizations
def create_enhanced_visualizations(df):
    print("\nGenerating enhanced visualizations...")
    
    # 1. Distribution Analysis
    numeric_columns = df.select_dtypes(include=[np.number]).columns
    n_cols = len(numeric_columns)
    n_rows = (n_cols + 1) // 2
    
    plt.figure(figsize=(15, 5*n_rows))
    for i, col in enumerate(numeric_columns, 1):
        plt.subplot(n_rows, 2, i)
        
        # Histogram with KDE
        sns.histplot(data=df, x=col, kde=True, stat='density')
        
        # Add normal distribution curve
        x = np.linspace(df[col].min(), df[col].max(), 100)
        p = stats.norm.pdf(x, df[col].mean(), df[col].std())
        plt.plot(x, p, 'k', linewidth=2, label='Normal Distribution')
        
        plt.title(f'Distribution of {col}\nSkewness: {df[col].skew():.2f}, Kurtosis: {df[col].kurtosis():.2f}')
        plt.legend()
    plt.tight_layout()
    plt.savefig('enhanced_distributions.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. Correlation Analysis
    plt.figure(figsize=(12, 10))
    correlation_matrix = df[numeric_columns].corr()
    mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))
    sns.heatmap(correlation_matrix, mask=mask, annot=True, fmt='.2f', cmap='coolwarm',
                center=0, square=True, linewidths=.5, cbar_kws={"shrink": .5})
    plt.title('Correlation Matrix Heatmap', pad=20)
    plt.tight_layout()
    plt.savefig('correlation_matrix.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 3. Box Plots with Outlier Analysis
    plt.figure(figsize=(15, 6))
    df[numeric_columns].boxplot()
    plt.title('Box Plot Analysis with Outliers')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('box_plots.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 4. PCA Analysis (if enough numeric columns)
    if len(numeric_columns) > 1:
        # Standardize the data
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(df[numeric_columns])
        
        # Perform PCA
        pca = PCA()
        pca_result = pca.fit_transform(scaled_data)
        
        # Create PCA visualization
        plt.figure(figsize=(12, 5))
        
        # Scree plot
        plt.subplot(1, 2, 1)
        plt.plot(range(1, len(pca.explained_variance_ratio_) + 1),
                pca.explained_variance_ratio_.cumsum(), 'bo-')
        plt.xlabel('Number of Components')
        plt.ylabel('Cumulative Explained Variance')
        plt.title('Scree Plot')
        
        # First two components
        plt.subplot(1, 2, 2)
        plt.scatter(pca_result[:, 0], pca_result[:, 1], alpha=0.5)
        plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2%} variance)')
        plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2%} variance)')
        plt.title('PCA: First Two Components')
        
        plt.tight_layout()
        plt.savefig('pca_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    # 5. Categorical Analysis (if any)
    categorical_columns = df.select_dtypes(include=['object']).columns
    if len(categorical_columns) > 0:
        for col in categorical_columns:
            plt.figure(figsize=(12, 6))
            value_counts = df[col].value_counts()
            ax = sns.barplot(x=value_counts.index, y=value_counts.values)
            plt.title(f'Distribution of {col}')
            plt.xticks(rotation=45)
            
            # Add value labels on top of bars
            for i, v in enumerate(value_counts.values):
                ax.text(i, v, f'{v:,}', ha='center', va='bottom')
            
            plt.tight_layout()
            plt.savefig(f'{col}_distribution.png', dpi=300, bbox_inches='tight')
            plt.close()

# Main execution
if __name__ == "__main__":
    try:
        # Create analysis report
        df = create_analysis_report()
        
        # Generate enhanced visualizations
        create_enhanced_visualizations(df)
        
        print("\nAnalysis complete! Generated visualizations:")
        print("1. enhanced_distributions.png - Detailed distribution analysis")
        print("2. correlation_matrix.png - Correlation heatmap")
        print("3. box_plots.png - Box plot analysis with outliers")
        print("4. pca_analysis.png - Principal Component Analysis (if applicable)")
        print("5. [column_name]_distribution.png - Categorical variable distributions (if any)")
        
    except Exception as e:
        print(f"\nAn error occurred during analysis: {str(e)}") 