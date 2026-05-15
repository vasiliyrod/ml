"""
Exploratory Data Analysis (EDA) for Car Price Prediction.

This notebook performs:
- Data loading and initial inspection
- Univariate analysis
- Bivariate analysis
- Correlation analysis
- Visualization of key relationships
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Set style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")

# Create directories for saving plots
Path("reports/images").mkdir(parents=True, exist_ok=True)


def load_data(raw_path: str = "data/raw/cars_dataset_raw.csv") -> pd.DataFrame:
    """Load raw dataset."""
    df = pd.read_csv(raw_path)
    return df


def clean_column_names(df: pd.DataFrame) -> pd.DataFrame:
    """Standardize column names."""
    column_mapping = {
        'Company Names': 'company',
        'Cars Names': 'model',
        'Engines': 'engine_type',
        'CC/Battery Capacity': 'engine_capacity',
        'HorsePower': 'horsepower',
        'Total Speed': 'top_speed',
        'Performance(0 - 100 )KM/H': 'acceleration_0_100',
        'Cars Prices': 'price',
        'Fuel Types': 'fuel_type',
        'Seats': 'seats',
        'Torque': 'torque'
    }
    return df.rename(columns=column_mapping)


def extract_numeric_value(value) -> float:
    """Extract numeric value from string."""
    import re
    if pd.isna(value):
        return np.nan
    if isinstance(value, (int, float)):
        return float(value)

    value_str = str(value).strip()
    value_str = re.sub(r'[hp|km/h|sec|cc|Nm|\$|,]', '', value_str, flags=re.IGNORECASE)

    if '-' in value_str:
        try:
            parts = value_str.split('-')
            if len(parts) == 2:
                return (float(parts[0].strip()) + float(parts[1].strip())) / 2
        except ValueError:
            pass

    match = re.search(r'[\d.]+', value_str)
    if match:
        try:
            return float(match.group())
        except ValueError:
            return np.nan

    return np.nan


def create_eda_report(df: pd.DataFrame, save_dir: str = "reports/images"):
    """Generate comprehensive EDA report with visualizations."""

    print("="*60)
    print("EXPLORATORY DATA ANALYSIS REPORT")
    print("="*60)

    # 1. Basic Information
    print("\n1. DATASET OVERVIEW")
    print("-"*40)
    print(f"Shape: {df.shape[0]} rows, {df.shape[1]} columns")
    print(f"\nColumns: {df.columns.tolist()}")
    print(f"\nData Types:\n{df.dtypes}")
    print(f"\nMissing Values:\n{df.isnull().sum()}")
    print(f"\nDuplicate Rows: {df.duplicated().sum()}")

    # 2. Target Variable Distribution
    print("\n2. TARGET VARIABLE DISTRIBUTION (Price)")
    print("-"*40)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Histogram
    axes[0].hist(df['price'].dropna(), bins=50, edgecolor='black', alpha=0.7)
    axes[0].set_xlabel('Price (USD)')
    axes[0].set_ylabel('Frequency')
    axes[0].set_title('Distribution of Car Prices')
    axes[0].ticklabel_format(style='plain', axis='x')

    # Boxplot
    axes[1].boxplot(df['price'].dropna())
    axes[1].set_ylabel('Price (USD)')
    axes[1].set_title('Boxplot of Car Prices')
    axes[1].ticklabel_format(style='plain', axis='y')

    plt.tight_layout()
    plt.savefig(f"{save_dir}/01_price_distribution.png", dpi=150)
    plt.show()

    print(f"Price Statistics:\n{df['price'].describe()}")

    # 3. Categorical Variables Analysis
    print("\n3. CATEGORICAL VARIABLES ANALYSIS")
    print("-"*40)

    categorical_cols = ['company', 'fuel_type', 'engine_type']

    for col in categorical_cols:
        if col in df.columns:
            print(f"\n{col.upper()}:")
            print(df[col].value_counts().head(10))

            # Plot top 10 categories
            top_10 = df[col].value_counts().head(10)
            fig, ax = plt.subplots(figsize=(12, 5))
            ax.bar(range(len(top_10)), top_10.values)
            ax.set_xticks(range(len(top_10)))
            ax.set_xticklabels(top_10.index, rotation=45, ha='right')
            ax.set_xlabel(col)
            ax.set_ylabel('Count')
            ax.set_title(f'Top 10 {col}')
            plt.tight_layout()
            plt.savefig(f"{save_dir}/02_{col}_distribution.png", dpi=150)
            plt.show()

    # 4. Numeric Variables Distribution
    print("\n4. NUMERIC VARIABLES DISTRIBUTION")
    print("-"*40)

    numeric_cols = ['engine_capacity', 'horsepower', 'top_speed', 
                    'acceleration_0_100', 'torque', 'seats']

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()

    for idx, col in enumerate(numeric_cols):
        if col in df.columns:
            axes[idx].hist(df[col].dropna(), bins=30, edgecolor='black', alpha=0.7)
            axes[idx].set_xlabel(col)
            axes[idx].set_ylabel('Frequency')
            axes[idx].set_title(f'Distribution of {col}')

    plt.tight_layout()
    plt.savefig(f"{save_dir}/03_numeric_distributions.png", dpi=150)
    plt.show()

    # 5. Correlation Analysis
    print("\n5. CORRELATION ANALYSIS")
    print("-"*40)

    # Calculate correlations with price
    corr_with_price = df[numeric_cols + ['price']].corr()['price'].sort_values(ascending=False)
    print("Correlation with Price:")
    print(corr_with_price)

    # Correlation heatmap
    plt.figure(figsize=(10, 8))
    corr_matrix = df[numeric_cols + ['price']].corr()
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0, fmt='.2f')
    plt.title('Correlation Heatmap of Numeric Features')
    plt.tight_layout()
    plt.savefig(f"{save_dir}/04_correlation_heatmap.png", dpi=150)
    plt.show()

    # 6. Key Relationships with Price
    print("\n6. KEY RELATIONSHIPS WITH PRICE")
    print("-"*40)

    key_features = ['horsepower', 'engine_capacity', 'top_speed', 'acceleration_0_100']

    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    axes = axes.flatten()

    for idx, feature in enumerate(key_features):
        if feature in df.columns:
            axes[idx].scatter(df[feature], df['price'], alpha=0.5, s=20)
            axes[idx].set_xlabel(feature)
            axes[idx].set_ylabel('Price (USD)')
            axes[idx].set_title(f'{feature} vs Price')
            axes[idx].ticklabel_format(style='plain', axis='y')

    plt.tight_layout()
    plt.savefig(f"{save_dir}/05_key_relationships.png", dpi=150)
    plt.show()

    # 7. Price by Fuel Type
    print("\n7. PRICE BY FUEL TYPE")
    print("-"*40)

    if 'fuel_type' in df.columns:
        plt.figure(figsize=(10, 6))
        df.boxplot(column='price', by='fuel_type', rot=45)
        plt.suptitle('Price Distribution by Fuel Type')
        plt.tight_layout()
        plt.savefig(f"{save_dir}/06_price_by_fuel_type.png", dpi=150)
        plt.show()

        print("Price statistics by fuel type:")
        print(df.groupby('fuel_type')['price'].describe())

    # 8. Price by Company (Top 10)
    print("\n8. PRICE BY COMPANY (TOP 10)")
    print("-"*40)

    if 'company' in df.columns:
        top_companies = df['company'].value_counts().head(10).index

        plt.figure(figsize=(12, 6))
        df[df['company'].isin(top_companies)].boxplot(column='price', by='company', rot=45)
        plt.suptitle('Price Distribution by Company (Top 10)')
        plt.tight_layout()
        plt.savefig(f"{save_dir}/07_price_by_company.png", dpi=150)
        plt.show()

    # 9. Pairplot for key features
    print("\n9. PAIRPLOT FOR KEY FEATURES")
    print("-"*40)

    pairplot_cols = ['price', 'horsepower', 'engine_capacity', 'top_speed', 'acceleration_0_100']
    pairplot_df = df[pairplot_cols].dropna()

    if len(pairplot_df) > 0:
        sns.pairplot(pairplot_df, diag_kind='hist')
        plt.suptitle('Pairplot of Key Features', y=1.02)
        plt.savefig(f"{save_dir}/08_pairplot.png", dpi=150)
        plt.show()

    print("\n" + "="*60)
    print("EDA COMPLETE - All plots saved to reports/images/")
    print("="*60)


def main():
    """Main EDA pipeline."""
    # Load data
    df = load_data()

    # Clean column names
    df = clean_column_names(df)

    # Clean numeric columns
    numeric_cols = ['engine_capacity', 'horsepower', 'top_speed', 
                    'acceleration_0_100', 'price', 'torque']

    for col in numeric_cols:
        if col in df.columns:
            df[col] = df[col].apply(extract_numeric_value)

    # Handle missing values for visualization
    df_clean = df.copy()
    for col in numeric_cols:
        if col in df_clean.columns:
            median_val = df_clean[col].median()
            df_clean[col] = df_clean[col].fillna(median_val)

    # Fill categorical with mode
    for col in ['company', 'fuel_type', 'engine_type']:
        if col in df_clean.columns:
            mode_val = df_clean[col].mode()[0] if len(df_clean[col].mode()) > 0 else 'Unknown'
            df_clean[col] = df_clean[col].fillna(mode_val)

    # Generate EDA report
    create_eda_report(df_clean)

    return df_clean


if __name__ == "__main__":
    df_eda = main()
