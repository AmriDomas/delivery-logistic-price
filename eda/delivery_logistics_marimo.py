import marimo

__generated_with = "0.19.1"
app = marimo.App()


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Delivery Logistics Price Prediction

    Muh Amri Sidiq
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Introduction

    This project analyzes a comprehensive delivery logistics dataset from India's multi-partner logistics ecosystem.
    The goal is to uncover insights that can optimize delivery operations, reduce costs, and improve efficiency
    in the rapidly growing Indian logistics market.

    Business Context:
    - Indian logistics market is expected to reach $380 billion by 2025
    - E-commerce growth driving demand for efficient delivery solutions
    - Multi-partner logistics require sophisticated coordination
    - Cost optimization is crucial for profitability

    Project Objectives:
    1. Perform comprehensive EDA to understand delivery patterns
    2. Segment delivery routes and partners using clustering
    3. Build predictive models for delivery costs
    4. Provide actionable business recommendations

    Dataset Features:
    - Delivery route information
    - Partner performance metrics
    - Cost and time variables
    - Geographical and operational data

    Expected Outcomes:
    - Identification of cost drivers
    - Partner performance segmentation
    - Predictive cost modeling
    - Strategic recommendations for logistics optimization
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Install Library
    """)
    return


@app.cell
def _():
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    import seaborn as sns
    from scipy import stats
    from sklearn.preprocessing import LabelEncoder, OneHotEncoder
    from sklearn.impute import SimpleImputer
    from sklearn.feature_selection import SelectKBest, f_regression, mutual_info_regression
    from sklearn.cluster import KMeans
    from sklearn.metrics import silhouette_score, silhouette_samples
    from sklearn.decomposition import PCA
    from sklearn.manifold import TSNE
    from sklearn.preprocessing import StandardScaler
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.linear_model import LinearRegression
    from sklearn.svm import SVR
    from xgboost import XGBRegressor
    from lightgbm import LGBMRegressor
    from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
    from sklearn.compose import ColumnTransformer
    from sklearn.pipeline import Pipeline
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
    from sklearn.inspection import permutation_importance
    import shap
    import joblib
    import warnings
    warnings.filterwarnings('ignore')
    plt.style.use('seaborn-v0_8')
    # Set style for better visualizations
    sns.set_palette('husl')
    return (
        ColumnTransformer,
        GridSearchCV,
        KMeans,
        LGBMRegressor,
        LabelEncoder,
        OneHotEncoder,
        PCA,
        Pipeline,
        RandomForestRegressor,
        SimpleImputer,
        StandardScaler,
        TSNE,
        XGBRegressor,
        joblib,
        mean_absolute_error,
        mean_squared_error,
        np,
        pd,
        permutation_importance,
        plt,
        r2_score,
        shap,
        silhouette_samples,
        silhouette_score,
        sns,
        train_test_split,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Read and Data Understanding
    """)
    return


@app.cell
def _(pd):
    # magic command not supported in marimo; please file an issue to add support
    # %%time
    df = pd.read_csv("../data/Delivery_Logistics.csv")
    return (df,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Read preview data
    """)
    return


@app.cell
def _(df):
    df.head()
    return


@app.cell
def _(df):
    df.info()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Insight: data have 15 column and 25000 row without missing values
    """)
    return


@app.cell
def _(df):
    df.duplicated().sum()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Exploratory Data Analysis
    Exploratory Data Analysis (EDA) is the process of analyzing and summarizing datasets to understand their main characteristics before applying machine learning models or statistical tests. EDA helps identify patterns, detect anomalies, check assumptions, and gain insights using visualizations and statistical methods
    """)
    return


@app.cell
def _(df):
    df.describe()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Insight:

    1. `delivery_id`
       - Count: There are 25,000 unique delivery records in the dataset.
       - Mean: The average delivery_id is 12500.5, which is expected for an evenly distributed identifier series.
       - Range: The IDs range from a minimum (implied by 25%) around 6250 up to a maximum (implied by 75%) around 18750. (The actual minimum/maximum values 250.99 and 24750.01 are likely just examples of the lower and upper bounds of the ID range).
    2. `distance_km`
       - Mean & Median: The average delivery distance is approximately 150.39 km.
       - The median (50th percentile) is very close at 151.00 km, suggesting the distance distribution is fairly symmetrical without extreme outliers skewing the average significantly.
       - Range: Distances vary widely, from a minimum of 3.60 km to a maximum of 297.10 km.
       - Spread (Standard Deviation): With a standard deviation of 86.41 km, the distances are quite spread out around the mean, indicating a high variability in delivery routes.
       - Quartiles: 50% of deliveries (between the 25th and 75th percentiles) fall within the range of 75.90 km to 224.90 km.
    3. `package_weight_kg`
       - Mean & Median: The average package weight is around 25.15 kg, very close to the median of 25.145 kg. This indicates a symmetrical distribution of package weights.
       - Range: Package weights span from a minimum of 0.67 kg to a maximum of 49.52 kg.
       - Spread: The standard deviation is 14.37 kg, which is high relative to the mean, suggesting a moderate to high variability in package weights.
       - Quartiles: The middle 50% of packages weigh between 12.68 kg and 37.66 kg.
    4. `delivery_rating`
       - Mean: The average delivery rating is 3.67.
       - Range: Ratings range from a minimum of 1.00 to a perfect maximum of 5.00.
       - Spread: The standard deviation is 1.15, showing a reasonable spread of ratings.
       - Distribution: The median (50th percentile) is 4.00, which is higher than the mean (3.67). This difference between the median and mean suggests the ratings distribution is skewed towards the lower ratings (left-skewed), or that there are more low ratings pulling the average down, even though half the ratings are 4 or 5.
       - Quartiles: 75% of all deliveries receive a rating of 3.00 or higher.
    5. `delivery_cost`
       - Mean: The average delivery cost is 864.94.
       - Median: The median cost is 867.54, which is very close to the mean, indicating a symmetrical cost distribution.
       - Range: Costs range from a minimum of 95.67 to a maximum of 1632.72.
       - Spread: The standard deviation of 435.71 is quite high, indicating a very high variability in delivery costs across the records. This is expected as cost is likely influenced by the highly variable distance and package weight.
       - Quartiles: 50% of the delivery costs fall between 490.80 and 1237.91.
    """)
    return


@app.cell
def _(df):
    df.describe(include = 'object')
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Insight:

    1. `delivery_partner`
       - Unique Values: There are 9 different delivery partners in the dataset.
       - Top Partner: The most frequent partner is xpressbees.
       - Frequency: xpressbees handled 2,826 deliveries, which is the most common single partner, though it represents a small portion of the total (11.3%). This suggests the delivery workload is divided among many partners.
    2. `package_type`
       - Unique Values: There are 9 distinct package types (e.g., fragile, non-fragile, perishable).
       - Top Package Type: The most common type is fragile items.
       - Frequency: 2,848 packages were classified as 'fragile items', making it the single most frequent type.
    3. `vehicle_type`
       - Unique Values: There are 6 different vehicle types used for delivery.
       - Top Vehicle Type: The most frequent vehicle is the e-bike (electric bicycle).
       - Frequency: The e-bike was used for 4,218 deliveries, indicating a significant reliance on eco-friendly or smaller, agile vehicles.
    4. `delivery_mode`
       - Unique Values: There are 4 different delivery modes (e.g., standard, express, same-day, two-day).
       - Top Delivery Mode: The most common mode is two day delivery.
       - Frequency: 6,302 deliveries were categorized as 'two day', suggesting that this is the standard or most preferred delivery mode for customers or sellers.
    5. `region`
       - Unique Values: There are 5 different geographical regions covered by the deliveries.
       - Top Region: The most serviced region is west.
       - Frequency: The 'west' region accounts for 9,542 deliveries. This high frequency (approximately 38% of all deliveries) indicates that the west region is the primary area of operation or has the highest volume of orders.
    6. `weather_condition`
       - Unique Values: There are 6 reported weather conditions.
       - Top Weather Condition: The most frequent condition encountered is foggy.
       - Frequency: Deliveries under 'foggy' conditions occurred 2,193 times. This is unusual, and might suggest that the deliveries often take place in a location prone to fog or during seasons when fog is common.
    7. `delivery_time_hours` & `expected_time_hours`
       - Unique Values (both): There are 20 and 9 unique time values, respectively.
       - Top Value (both): The data for top shows a repeated placeholder value (likely indicating a conversion error or standard timestamp used for the mode), which is 1970-01-01 00:00:00.000000006 and 1970-01-01 00:00:00.000000016.
       - Frequency (both): Both columns have an identical top frequency of 1,863 and 2,183 respectively.
       - Interpretation: The time data provided here is not directly interpretable due to the placeholder timestamp format. It is likely that the original data had an hourly value (e.g., 6 hours and 16 hours) that was erroneously converted to a standard epoch time. Further cleaning or transformation of these time columns is required to extract meaningful insights about actual delivery times.
    8. `delayeddelivery_status`
       - Unique Values: There are 2 unique statuses (Delayed or Not Delayed).
       - Top Status: The most common status is no. (Meaning: Not Delayed).
       - Frequency: 21,833 deliveries were 'Not Delayed'. This indicates that the vast majority (87.3%) of deliveries were on time, suggesting a high level of operational efficiency.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Distribution of Numerical Variables
    """)
    return


@app.cell
def _(df, plt):
    # 1. UNIVARIATE ANALYSIS - NUMERICAL VARIABLES
    _fig, _axes = plt.subplots(2, 2, figsize=(15, 10))
    _numerical_cols = ['distance_km', 'package_weight_kg', 'delivery_rating', 'delivery_cost']
    for _i, _col in enumerate(_numerical_cols):
    # Distribution plots
        row, col_idx = (_i // 2, _i % 2)
        _axes[row, col_idx].hist(df[_col], bins=30, alpha=0.7, edgecolor='black')
        _axes[row, col_idx].set_title(f'Distribution of {_col}')
        _axes[row, col_idx].set_xlabel(_col)
        _axes[row, col_idx].set_ylabel('Frequency')
    plt.tight_layout()
    plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Categorical Variables Analysis
    """)
    return


@app.cell
def _(df, plt):
    # UNIVARIATE ANALYSIS - CATEGORICAL VARIABLES
    categorical_cols = ['delivery_partner', 'package_type', 'vehicle_type', 'delivery_mode', 'region', 'weather_condition', 'delayed', 'delivery_status']
    _fig, _axes = plt.subplots(3, 3, figsize=(18, 15))
    _axes = _axes.ravel()
    for _i, _col in enumerate(categorical_cols):
        if _i < 9:
            value_counts = df[_col].value_counts()
            if len(value_counts) > 10:
                value_counts = value_counts.head(10)  # Ensure that you do not exceed the number of subplots.
            _axes[_i].bar(value_counts.index, value_counts.values)
            _axes[_i].set_title(f'Distribution of {_col}')  # If there are too many categories, take the top 10
            _axes[_i].tick_params(axis='x', rotation=45)
            _axes[_i].set_ylabel('Count')
    for _i in range(len(categorical_cols), 9):
        _axes[_i].set_visible(False)
    plt.tight_layout()
    # Hide unused subplots
    plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Correlation Analysis
    """)
    return


@app.cell
def _(df, np, plt, sns):
    # Correlation matrix
    plt.figure(figsize=(12, 8))
    _numerical_cols = ['distance_km', 'package_weight_kg', 'delivery_rating', 'delivery_cost']
    # Select only available numerical columns
    corr_matrix = df[_numerical_cols].corr()
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
    sns.heatmap(corr_matrix, mask=mask, annot=True, cmap='coolwarm', center=0, square=True, fmt='.2f', cbar_kws={'shrink': 0.8})
    # Create heatmap
    plt.title('Correlation Matrix of Logistics Delivery Variables', fontsize=16, fontweight='bold', pad=20)
    plt.tight_layout()
    plt.show()
    _fig, _axes = plt.subplots(2, 2, figsize=(15, 12))
    _axes[0, 0].scatter(df['distance_km'], df['delivery_cost'], alpha=0.6, c=df['package_weight_kg'], cmap='viridis')
    _axes[0, 0].set_xlabel('Delivery Distance (km)')
    _axes[0, 0].set_ylabel('Delivery Cost')
    _axes[0, 0].set_title('Delivery Distance vs Delivery Cost\n(Color: Package Weight)')
    # Scatter plots untuk hubungan penting dalam dataset logistics
    plt.colorbar(_axes[0, 0].collections[0], ax=_axes[0, 0])
    _axes[0, 1].scatter(df['package_weight_kg'], df['delivery_cost'], alpha=0.6, c=df['distance_km'], cmap='plasma')
    # 1. Delivery Distance vs Delivery Cost
    _axes[0, 1].set_xlabel('Package Weight (kg)')
    _axes[0, 1].set_ylabel('Delivery Cost')
    _axes[0, 1].set_title('Package Weight vs Delivery Cost\n(Color: Delivery Distance)')
    plt.colorbar(_axes[0, 1].collections[0], ax=_axes[0, 1])
    _axes[1, 0].scatter(df['distance_km'], df['delivery_rating'], alpha=0.6, c=df['delivery_cost'], cmap='cool')
    _axes[1, 0].set_xlabel('Delivery Distance (km)')
    _axes[1, 0].set_ylabel('Delivery Rating')
    # 2. Package Weight vs Delivery Cost
    _axes[1, 0].set_title('Delivery Distance vs Delivery Rating\n(Color: Delivery Cost)')
    plt.colorbar(_axes[1, 0].collections[0], ax=_axes[1, 0])
    _axes[1, 1].scatter(df['delivery_cost'], df['delivery_rating'], alpha=0.6, c=df['package_weight_kg'], cmap='hot')
    _axes[1, 1].set_xlabel('Delivery Cost')
    _axes[1, 1].set_ylabel('Delivery Rating')
    _axes[1, 1].set_title('Delivery Cost vs Delivery Rating\n(Color: Package Weight)')
    plt.colorbar(_axes[1, 1].collections[0], ax=_axes[1, 1])
    # 3. Delivery Distance vs Delivery Rating
    plt.tight_layout()
    # 4. Delivery Cost vs Delivery Rating
    plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Numerical Vs Categorical
    """)
    return


@app.cell
def _(df, plt, sns):
    # BIVARIATE ANALYSIS - NUMERICAL VS CATEGORICAL
    _fig, _axes = plt.subplots(2, 2, figsize=(16, 12))
    sns.boxplot(data=df, x='package_type', y='delivery_cost', ax=_axes[0, 0])
    # Delivery cost by package type
    _axes[0, 0].set_title('Delivery Cost by Package Type')
    _axes[0, 0].tick_params(axis='x', rotation=45)
    sns.boxplot(data=df, x='vehicle_type', y='package_weight_kg', ax=_axes[0, 1])
    _axes[0, 1].set_title('Package Weight by Vehicle Type (kg)')
    # Replace delivery_time_hours with another available numeric variable.
    _axes[0, 1].tick_params(axis='x', rotation=45)
    sns.boxplot(data=df, x='region', y='distance_km', ax=_axes[1, 0])
    _axes[1, 0].set_title('Delivery Distance by Region')
    _axes[1, 0].tick_params(axis='x', rotation=45)
    # Distance by region
    sns.boxplot(data=df, x='delivery_status', y='delivery_rating', ax=_axes[1, 1])
    _axes[1, 1].set_title('Delivery Rating by Status')
    _axes[1, 1].tick_params(axis='x', rotation=45)
    plt.tight_layout()
    # Rating by delivery status
    plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Delivery Performance Analysis
    """)
    return


@app.cell
def _(df, pd, plt, sns):
    # DELIVERY PERFORMANCE ANALYSIS
    _fig, _axes = plt.subplots(2, 2, figsize=(15, 10))
    status_counts = df['delivery_status'].value_counts()
    # Delivery status distribution
    _axes[0, 0].pie(status_counts.values, labels=status_counts.index, autopct='%1.1f%%')
    _axes[0, 0].set_title('Delivery Status Distribution')
    delay_counts = df['delayed'].value_counts()
    _axes[0, 1].bar(delay_counts.index, delay_counts.values)
    # Delay analysis
    _axes[0, 1].set_title('Delivery Delay Status')
    _axes[0, 1].set_ylabel('Count')
    sns.countplot(data=df, x='delivery_rating', hue='delivery_status', ax=_axes[1, 0])
    _axes[1, 0].set_title('Rating Distribution by Delivery Status')
    _axes[1, 0].legend(title='Delivery Status')
    # Rating distribution by delivery status
    weather_delay = pd.crosstab(df['weather_condition'], df['delayed'])
    weather_delay.plot(kind='bar', ax=_axes[1, 1])
    _axes[1, 1].set_title('Weather Impact on Delivery Delays')
    _axes[1, 1].set_ylabel('Count')
    # Weather impact on delays
    _axes[1, 1].tick_params(axis='x', rotation=45)
    plt.tight_layout()
    plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Partner Performance Analysis
    """)
    return


@app.cell
def _(df, plt):
    # PARTNER PERFORMANCE ANALYSIS
    plt.figure(figsize=(12, 6))

    # Average rating by delivery partner
    partner_rating = df.groupby('delivery_partner')['delivery_rating'].mean().sort_values(ascending=False)
    plt.subplot(1, 2, 1)
    partner_rating.plot(kind='bar')
    plt.title('Average Rating by Delivery Partner')
    plt.ylabel('Average Rating')
    plt.xticks(rotation=45)

    # Success rate by delivery partner
    partner_success = df.groupby('delivery_partner')['delivery_status'].apply(
        lambda x: (x == 'Delivered').mean() * 100
    ).sort_values(ascending=False)

    plt.subplot(1, 2, 2)
    partner_success.plot(kind='bar')
    plt.title('Success Rate by Delivery Partner (%)')
    plt.ylabel('Success Rate (%)')
    plt.xticks(rotation=45)

    plt.tight_layout()
    plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Key Insight Summary
    """)
    return


@app.cell
def _(df):
    # 10. KEY INSIGHTS SUMMARY
    print("\n" + "="*50)
    print("KEY INSIGHTS SUMMARY")
    print("="*50)

    # Delivery success rate
    success_rate = (df['delivery_status'] == 'Delivered').mean() * 100
    print(f"Delivery Success Rate: {success_rate:.2f}%")

    # Average delivery rating
    avg_rating = df['delivery_rating'].mean()
    print(f"Average Delivery Rating: {avg_rating:.2f}/5")

    # Delay rate
    delay_rate = (df['delayed'] == 'Yes').mean() * 100
    print(f"Delay Rate: {delay_rate:.2f}%")

    # Most common package type
    common_package = df['package_type'].value_counts().index[0]
    print(f"Most Common Package Type: {common_package}")

    # Most used vehicle type
    common_vehicle = df['vehicle_type'].value_counts().index[0]
    print(f"Most Used Vehicle Type: {common_vehicle}")

    # Correlation between distance and cost
    corr_distance_cost = df['distance_km'].corr(df['delivery_cost'])
    print(f"Correlation (Distance vs Cost): {corr_distance_cost:.3f}")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Feature Engineering
    """)
    return


@app.cell
def _(LabelEncoder, SimpleImputer, StandardScaler, df, np, pd):
    class AdvancedFeatureEngineering:

        def __init__(self, df):
            self.df = df.copy()  # Remove unused features immediately
            self.remove_unused_features()
            self.numerical_features = []
            self.categorical_features = []

        def remove_unused_features(self):
            """Remove features that won't be used in modeling"""
            unused_features = ['delivery_id', 'delivery_time_hours', 'expected_time_hours']
            for feature in unused_features:
                if feature in self.df.columns:
                    self.df.drop(feature, axis=1, inplace=True)
                    print(f'Removed unused feature: {feature}')

        def handle_missing_values(self):
            """Handle missing values with advanced strategies"""
            print('Handling missing values...')
            if self.df.isnull().sum().sum() == 0:
                print('No missing values found.')  # Check if there are any missing values first
                return self
            _numerical_cols = self.df.select_dtypes(include=[np.number]).columns
            if len(_numerical_cols) > 0:
                num_imputer = SimpleImputer(strategy='median')
                self.df[_numerical_cols] = num_imputer.fit_transform(self.df[_numerical_cols])  # Numerical features - use median imputation
            categorical_cols = self.df.select_dtypes(include=['object']).columns
            if len(categorical_cols) > 0:
                cat_imputer = SimpleImputer(strategy='most_frequent')
                self.df[categorical_cols] = cat_imputer.fit_transform(self.df[categorical_cols])
            return self
      # Categorical features - use mode imputation
        def create_date_features(self, date_columns=None):
            """Extract features from date columns"""
            print('Creating date-based features...')
            if date_columns is None:
                date_columns = self.df.select_dtypes(include=['datetime64']).columns
            if len(date_columns) == 0:
                print('No date columns found.')
                return self
            for _col in date_columns:
                if _col in self.df.columns:
                    self.df[f'{_col}_year'] = self.df[_col].dt.year
                    self.df[f'{_col}_month'] = self.df[_col].dt.month
                    self.df[f'{_col}_day'] = self.df[_col].dt.day  # Auto-detect date columns
                    self.df[f'{_col}_dayofweek'] = self.df[_col].dt.dayofweek
                    self.df[f'{_col}_quarter'] = self.df[_col].dt.quarter
                    self.df[f'{_col}_is_weekend'] = self.df[_col].dt.dayofweek.isin([5, 6]).astype(int)
            return self

        def encode_categorical_features(self, method='auto'):
            """Encode categorical features using various methods"""
            print('Encoding categorical features...')
            categorical_cols = self.df.select_dtypes(include=['object']).columns
            if len(categorical_cols) == 0:
                print('No categorical features found.')
                return self
            for _col in categorical_cols:
                if method == 'auto':
                    if self.df[_col].nunique() <= 10:
                        dummies = pd.get_dummies(self.df[_col], prefix=_col)
                        self.df = pd.concat([self.df, dummies], axis=1)
                        self.df.drop(_col, axis=1, inplace=True)
                    else:
                        le = LabelEncoder()
                        self.df[_col] = le.fit_transform(self.df[_col].astype(str))
                elif method == 'onehot':
                    dummies = pd.get_dummies(self.df[_col], prefix=_col)
                    self.df = pd.concat([self.df, dummies], axis=1)
                    self.df.drop(_col, axis=1, inplace=True)
                elif method == 'label':
                    le = LabelEncoder()
                    self.df[_col] = le.fit_transform(self.df[_col].astype(str))
            return self
      # Use one-hot for low cardinality, label encoding for high cardinality
        def create_interaction_features(self):
            """Create interaction features between important variables"""  # One-hot encoding
            print('Creating interaction features...')
            _numerical_cols = self.df.select_dtypes(include=[np.number]).columns
            if 'distance_km' in self.df.columns and 'package_weight_kg' in self.df.columns:
                self.df['distance_weight_interaction'] = self.df['distance_km'] * self.df['package_weight_kg']
                print('Created distance_weight_interaction')  # Label encoding
            if 'region' in self.df.columns and 'distance_km' in self.df.columns:
                pass
            return self

        def create_polynomial_features(self, degree=2):
            """Create polynomial features for important numerical variables"""
            print('Creating polynomial features...')
            _numerical_cols = self.df.select_dtypes(include=[np.number]).columns
            if 'delivery_cost' in _numerical_cols:
                _numerical_cols = _numerical_cols.drop('delivery_cost')
            if 'delivery_cost' in self.df.columns and len(_numerical_cols) > 0:
                try:
                    corr_with_target = self.df[_numerical_cols.tolist() + ['delivery_cost']].corr()['delivery_cost'].abs().sort_values(ascending=False)
                    top_features = corr_with_target[1:4].index.tolist()
                    for feature in top_features:
                        for deg in range(2, degree + 1):
                            self.df[f'{feature}_power_{deg}'] = self.df[feature] ** deg
                            print(f'Created {feature}_power_{deg}')
                except:  # Create interaction features based on available columns
                    for feature in _numerical_cols[:2]:
                        for deg in range(2, degree + 1):
                            self.df[f'{feature}_power_{deg}'] = self.df[feature] ** deg
                            print(f'Created {feature}_power_{deg}')
            return self  # Create region-specific features if available

        def create_aggregation_features(self, groupby_columns=None):  # This will be handled in aggregation features
            """Create aggregation features based on groups"""
            print('Creating aggregation features...')
            if groupby_columns is None:
                categorical_cols = self.df.select_dtypes(include=['object']).columns
                groupby_columns = categorical_cols.tolist()
            if len(groupby_columns) > 0 and 'delivery_cost' in self.df.columns:
                for _col in groupby_columns:
                    if _col in self.df.columns:
                        mean_encoded = self.df.groupby(_col)['delivery_cost'].transform('mean')
                        self.df[f'{_col}_mean_delivery_cost'] = mean_encoded
                        print(f'Created {_col}_mean_delivery_cost')  # Remove target variable from polynomial features
                        count_encoded = self.df.groupby(_col)[_col].transform('count')
                        self.df[f'{_col}_count'] = count_encoded
                        print(f'Created {_col}_count')
            return self  # Select top correlated features with target if available

        def feature_scaling(self, method='standard'):
            """Scale numerical features (excluding target variable)"""
            print('Scaling features...')  # Top 3 features excluding target itself
            _numerical_cols = self.df.select_dtypes(include=[np.number]).columns
            if 'delivery_cost' in _numerical_cols:
                _numerical_cols = _numerical_cols.drop('delivery_cost')
            if len(_numerical_cols) == 0:
                print('No numerical features found for scaling.')
                return self
            if method == 'standard':  # Fallback: use first few numerical features
                scaler = StandardScaler()
                self.df[_numerical_cols] = scaler.fit_transform(self.df[_numerical_cols])
            elif method == 'minmax':
                from sklearn.preprocessing import MinMaxScaler
                scaler = MinMaxScaler()
                self.df[_numerical_cols] = scaler.fit_transform(self.df[_numerical_cols])
            print(f'Scaled {len(_numerical_cols)} numerical features using {method} scaling')
            return self

        def execute_all_steps(self, target_column='delivery_cost'):
            """Execute all feature engineering steps"""
            print('STARTING ADVANCED FEATURE ENGINEERING')
            print('=' * 50)  # Use categorical columns for grouping
            print(f'Initial dataset shape: {self.df.shape}')
            if target_column in self.df.columns:
                self.target = self.df[target_column].copy()
            self.handle_missing_values()
            self.create_date_features()
            self.encode_categorical_features()
            self.create_interaction_features()  # Mean encoding for target variable
            self.create_polynomial_features()
            self.create_aggregation_features()
            self.feature_scaling()
            if hasattr(self, 'target'):
                self.df[target_column] = self.target  # Count encoding
            print('Feature engineering completed!')
            print(f'Final dataset shape: {self.df.shape}')
            print(f'Features: {list(self.df.columns)}')
            return self.df
    feature_engineer = AdvancedFeatureEngineering(df)
    # Execute feature engineering
    df_processed = feature_engineer.execute_all_steps()  # Don't scale the target variable  # Store target column separately if it exists  # Add target back if it was separated
    return (df_processed,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Clustering
    """)
    return


@app.cell
def _(
    KMeans,
    PCA,
    StandardScaler,
    TSNE,
    df_processed,
    np,
    plt,
    silhouette_samples,
    silhouette_score,
    sns,
):
    class ComprehensiveClustering:

        def __init__(self, df, target_col='delivery_cost'):
            self.df = df.copy()
            self.target_col = target_col
            self.features_for_clustering = None
            self.scaler = StandardScaler()
            self.kmeans_model = None
            self.optimal_k = None

        def prepare_clustering_data(self):
            """Prepare data for clustering"""
            print('Preparing data for clustering...')
            if self.target_col in self.df.columns:  # Remove target variable for clustering
                self.features_for_clustering = self.df.drop(columns=[self.target_col])
            else:
                self.features_for_clustering = self.df.copy()
            numerical_features = self.features_for_clustering.select_dtypes(include=[np.number])
            self.scaled_features = self.scaler.fit_transform(numerical_features)
            print(f'Features for clustering: {numerical_features.shape}')  # Select only numerical features
            return numerical_features.columns.tolist()

        def determine_optimal_clusters(self, max_k=15):  # Scale the features
            """Determine optimal number of clusters using elbow method and silhouette analysis"""
            print('Determining optimal number of clusters...')
            wcss = []
            silhouette_scores = []
            k_range = range(2, max_k + 1)
            plt.figure(figsize=(15, 5))
            plt.subplot(1, 3, 1)
            for k in k_range:
                kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
                kmeans.fit(self.scaled_features)  # Within-cluster sum of squares
                wcss.append(kmeans.inertia_)
                if len(set(kmeans.labels_)) > 1:
                    silhouette_avg = silhouette_score(self.scaled_features, kmeans.labels_)
                    silhouette_scores.append(silhouette_avg)
            plt.plot(k_range, wcss, 'bo-')
            plt.xlabel('Number of Clusters')
            plt.ylabel('WCSS')  # Elbow method
            plt.title('Elbow Method')
            plt.subplot(1, 3, 2)
            plt.plot(k_range[:len(silhouette_scores)], silhouette_scores, 'ro-')
            plt.xlabel('Number of Clusters')
            plt.ylabel('Silhouette Score')
            plt.title('Silhouette Analysis')
            optimal_k_elbow = self.find_elbow_point(wcss) + 2  # Ensure we have more than one cluster
            optimal_k_silhouette = k_range[np.argmax(silhouette_scores)]
            self.optimal_k = optimal_k_silhouette
            print(f'Optimal number of clusters (Elbow method): {optimal_k_elbow}')
            print(f'Optimal number of clusters (Silhouette method): {optimal_k_silhouette}')
            print(f'Selected optimal k: {self.optimal_k}')
            plt.subplot(1, 3, 3)
            self.silhouette_analysis_for_optimal_k()
            plt.tight_layout()
            plt.show()  # Silhouette scores
            return self.optimal_k

        def find_elbow_point(self, wcss):
            """Find the elbow point in WCSS curve"""
            deltas = np.diff(wcss)
            deltas2 = np.diff(deltas)
            elbow_point = np.argmax(deltas2) + 1  # Find optimal k (you can choose based on elbow or silhouette)
            return elbow_point  # +2 because range starts from 2

        def silhouette_analysis_for_optimal_k(self):
            """Detailed silhouette analysis for optimal k"""  # Choose the optimal k (prioritizing silhouette score)
            if self.optimal_k is None:
                self.determine_optimal_clusters()
            _fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
            ax1.set_xlim([-0.1, 1])
            ax1.set_ylim([0, len(self.scaled_features) + (self.optimal_k + 1) * 10])
            clusterer = KMeans(n_clusters=self.optimal_k, random_state=42, n_init=10)
            cluster_labels = clusterer.fit_predict(self.scaled_features)  # Silhouette analysis for optimal k
            silhouette_avg = silhouette_score(self.scaled_features, cluster_labels)
            print(f'For n_clusters = {self.optimal_k}, average silhouette_score is: {silhouette_avg:.3f}')
            sample_silhouette_values = silhouette_samples(self.scaled_features, cluster_labels)
            y_lower = 10
            for _i in range(self.optimal_k):
                ith_cluster_silhouette_values = sample_silhouette_values[cluster_labels == _i]
                ith_cluster_silhouette_values.sort()
                size_cluster_i = ith_cluster_silhouette_values.shape[0]
                y_upper = y_lower + size_cluster_i
                color = plt.cm.nipy_spectral(float(_i) / self.optimal_k)
                ax1.fill_betweenx(np.arange(y_lower, y_upper), 0, ith_cluster_silhouette_values, facecolor=color, edgecolor=color, alpha=0.7)
                ax1.text(-0.05, y_lower + 0.5 * size_cluster_i, str(_i))
                y_lower = y_upper + 10
            ax1.set_title('Silhouette plot for various clusters')
            ax1.set_xlabel('Silhouette coefficient values')
            ax1.set_ylabel('Cluster label')
            ax1.axvline(x=silhouette_avg, color='red', linestyle='--')
            ax1.set_yticks([])
            colors = plt.cm.nipy_spectral(cluster_labels.astype(float) / self.optimal_k)
            ax2.scatter(self.scaled_features[:, 0], self.scaled_features[:, 1], marker='.', s=30, lw=0, alpha=0.7, c=colors)
            centers = clusterer.cluster_centers_  # Create a subplot with 1 row and 2 columns
            ax2.scatter(centers[:, 0], centers[:, 1], marker='o', c='white', alpha=1, s=200, edgecolor='k')
            for _i, c in enumerate(centers):
                ax2.scatter(c[0], c[1], marker='$%d$' % _i, alpha=1, s=50, edgecolor='k')  # The 1st subplot is the silhouette plot
            ax2.set_title('Clustered data')
            ax2.set_xlabel('Feature 1')
            ax2.set_ylabel('Feature 2')
            plt.suptitle(f'Silhouette analysis for KMeans clustering with n_clusters = {self.optimal_k}', fontsize=14, fontweight='bold')  # Initialize the clusterer with optimal k
            plt.show()

        def apply_kmeans(self):
            """Apply K-means clustering with optimal k"""  # The silhouette_score gives the average value for all the samples.
            print(f'Applying K-means clustering with k={self.optimal_k}...')
            self.kmeans_model = KMeans(n_clusters=self.optimal_k, random_state=42, n_init=10)
            self.cluster_labels = self.kmeans_model.fit_predict(self.scaled_features)
            self.df['cluster'] = self.cluster_labels  # Compute the silhouette scores for each sample
            print('K-means clustering completed!')
            return self.cluster_labels

        def dimensionality_reduction_visualization(self):
            """Apply dimensionality reduction for visualization"""  # Aggregate the silhouette scores for samples belonging to cluster i
            print('Applying dimensionality reduction for visualization...')
            pca = PCA(n_components=2, random_state=42)
            features_pca = pca.fit_transform(self.scaled_features)
            tsne = TSNE(n_components=2, random_state=42, perplexity=30)
            features_tsne = tsne.fit_transform(self.scaled_features)
            _fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
            scatter1 = ax1.scatter(features_pca[:, 0], features_pca[:, 1], c=self.cluster_labels, cmap='viridis', alpha=0.6)
            ax1.set_title(f'PCA - Clusters Visualization\n(PCA Variance: {pca.explained_variance_ratio_.sum():.2%})')
            ax1.set_xlabel('Principal Component 1')
            ax1.set_ylabel('Principal Component 2')
            plt.colorbar(scatter1, ax=ax1)
            scatter2 = ax2.scatter(features_tsne[:, 0], features_tsne[:, 1], c=self.cluster_labels, cmap='viridis', alpha=0.6)  # Label the silhouette plots with their cluster numbers at the middle
            ax2.set_title('t-SNE - Clusters Visualization')
            ax2.set_xlabel('t-SNE Component 1')
            ax2.set_ylabel('t-SNE Component 2')  # Compute the new y_lower for next plot
            plt.colorbar(scatter2, ax=ax2)  # 10 for the 0 samples
            plt.tight_layout()
            plt.show()
            return (features_pca, features_tsne)

        def cluster_profiling_analysis(self):
            """Perform comprehensive cluster profiling and analysis"""  # The vertical line for average silhouette score of all the values
            print('Performing cluster profiling and analysis...')
            cluster_summary = self.df.groupby('cluster').agg({'delivery_cost': ['count', 'mean', 'std', 'min', 'max'] if 'delivery_cost' in self.df.columns else 'count'}).round(2)  # Clear the yaxis labels / ticks
            print('Cluster Summary:')
            print(cluster_summary)  # 2nd Plot showing the actual clusters formed
            self.feature_importance_per_cluster()
            return cluster_summary

        def feature_importance_per_cluster(self):
            """Analyze feature importance for each cluster"""  # Labeling the clusters
            print('Analyzing feature importance per cluster...')
            numerical_features = self.df.select_dtypes(include=[np.number]).columns
            numerical_features = [_col for _col in numerical_features if _col != 'cluster' and _col != self.target_col]
            cluster_means = self.df.groupby('cluster')[numerical_features].mean()
            normalized_means = (cluster_means - cluster_means.mean()) / cluster_means.std()
            plt.figure(figsize=(12, 8))
            sns.heatmap(normalized_means.T, annot=True, cmap='RdYlBu', center=0, fmt='.2f', linewidths=0.5)
            plt.title('Feature Importance per Cluster (Normalized)')
            plt.xlabel('Cluster')
            plt.ylabel('Features')
            plt.tight_layout()
            plt.show()
            return normalized_means

        def detailed_cluster_interpretation(self):
            """Provide detailed interpretation and naming for each cluster"""
            print('Providing detailed cluster interpretation...')
            cluster_profiles = {}
            for cluster_id in range(self.optimal_k):
                cluster_data = self.df[self.df['cluster'] == cluster_id]
                profile = {'size': len(cluster_data), 'size_percentage': len(cluster_data) / len(self.df) * 100}
                if 'delivery_cost' in self.df.columns:
                    profile['avg_delivery_cost'] = cluster_data['delivery_cost'].mean()  # Add cluster labels to original dataframe
                numerical_features = self.df.select_dtypes(include=[np.number]).columns
                numerical_features = [_col for _col in numerical_features if _col != 'cluster']
                feature_means = cluster_data[numerical_features].mean()
                overall_means = self.df[numerical_features].mean()
                significant_features = (feature_means - overall_means).abs().nlargest(3)
                profile['distinguishing_features'] = significant_features.to_dict()
                cluster_profiles[cluster_id] = profile
            for cluster_id, profile in cluster_profiles.items():
                print(f'\nCluster {cluster_id}:')
                print(f"  Size: {profile['size']} ({profile['size_percentage']:.1f}%)")  # PCA for dimensionality reduction
                if 'avg_delivery_cost' in profile:
                    print(f"  Average Delivery Cost: {profile['avg_delivery_cost']:.2f}")
                print('  Distinguishing Features:')
                for feature, value in profile['distinguishing_features'].items():  # t-SNE for non-linear dimensionality reduction
                    print(f'    - {feature}: {value:.2f}')
            return cluster_profiles

        def business_insights_recommendations(self):  # Create visualization
            """Provide business insights and recommendations based on clustering"""
            print('Generating business insights and recommendations...')
            insights = []  # PCA visualization
            for cluster_id in range(self.optimal_k):
                cluster_data = self.df[self.df['cluster'] == cluster_id]
                if 'delivery_cost' in self.df.columns:
                    avg_cost = cluster_data['delivery_cost'].mean()
                    overall_avg = self.df['delivery_cost'].mean()
                    if avg_cost > overall_avg * 1.1:
                        insight = f'Cluster {cluster_id}: High-cost delivery group. Consider route optimization or partner renegotiation.'
                    elif avg_cost < overall_avg * 0.9:  # t-SNE visualization
                        insight = f'Cluster {cluster_id}: Low-cost efficient group. Replicate these delivery patterns.'
                    else:
                        insight = f'Cluster {cluster_id}: Average cost group. Maintain current operations.'
                    insights.append(insight)
            print('\nBusiness Insights:')
            for insight in insights:
                print(f'- {insight}')
            print('\nStrategic Recommendations:')
            print('1. Focus on high-cost clusters for cost optimization')
            print('2. Analyze successful patterns in low-cost clusters')
            print('3. Develop cluster-specific delivery strategies')
            print('4. Monitor cluster migration over time')
            print('5. Implement targeted improvements based on cluster characteristics')
            return insights

        def final_model_evaluation(self):
            """Final evaluation of clustering model"""  # Basic cluster statistics
            print('Final clustering model evaluation...')
            silhouette_avg = silhouette_score(self.scaled_features, self.cluster_labels)
            print(f'Final Silhouette Score: {silhouette_avg:.3f}')
            cluster_counts = self.df['cluster'].value_counts()
            balance_score = cluster_counts.std() / cluster_counts.mean()
            print(f'Cluster Balance Score (lower is better): {balance_score:.3f}')
            wcss = self.kmeans_model.inertia_
            print(f'Within-Cluster Sum of Squares: {wcss:.2f}')  # Feature importance per cluster
            evaluation_metrics = {'silhouette_score': silhouette_avg, 'cluster_balance': balance_score, 'wcss': wcss, 'n_clusters': self.optimal_k}
            return evaluation_metrics

        def export_results(self):
            """Export clustering results"""
            print('Exporting clustering results...')
            self.df.to_csv('clustered_delivery_data.csv', index=False)
            cluster_profiles = self.detailed_cluster_interpretation()
            model_info = {'optimal_k': self.optimal_k, 'cluster_centers': self.kmeans_model.cluster_centers_.tolist(), 'features_used': self.features_for_clustering.select_dtypes(include=[np.number]).columns.tolist()}
            print('Results exported successfully!')
            return (self.df, cluster_profiles, model_info)
      # Calculate mean values for each feature per cluster
        def create_visualization_dashboard(self):
            """Create comprehensive visualization dashboard"""
            print('Creating visualization dashboard...')  # Normalize the means for better comparison
            _fig, _axes = plt.subplots(2, 3, figsize=(18, 12))
            _fig.suptitle('Delivery Logistics Clustering Dashboard', fontsize=16, fontweight='bold')
            cluster_counts = self.df['cluster'].value_counts().sort_index()  # Plot feature importance heatmap
            _axes[0, 0].bar(cluster_counts.index, cluster_counts.values, color='skyblue', alpha=0.7)
            _axes[0, 0].set_title('Cluster Distribution')
            _axes[0, 0].set_xlabel('Cluster')
            _axes[0, 0].set_ylabel('Number of Deliveries')
            if 'delivery_cost' in self.df.columns:
                sns.boxplot(data=self.df, x='cluster', y='delivery_cost', ax=_axes[0, 1])
                _axes[0, 1].set_title('Delivery Cost Distribution by Cluster')
            numerical_features = self.df.select_dtypes(include=[np.number]).columns
            numerical_features = [_col for _col in numerical_features if _col != 'cluster' and _col != self.target_col]
            cluster_means = self.df.groupby('cluster')[numerical_features].mean()
            normalized_means = (cluster_means - cluster_means.mean()) / cluster_means.std()
            sns.heatmap(normalized_means.T.iloc[:10], ax=_axes[0, 2], cmap='RdYlBu', center=0, annot=True, fmt='.2f')
            _axes[0, 2].set_title('Top 10 Features per Cluster (Normalized)')
            pca = PCA(n_components=2, random_state=42)
            features_pca = pca.fit_transform(self.scaled_features)
            scatter = _axes[1, 0].scatter(features_pca[:, 0], features_pca[:, 1], c=self.cluster_labels, cmap='viridis', alpha=0.6)
            _axes[1, 0].set_title(f'PCA Visualization\n(Variance: {pca.explained_variance_ratio_.sum():.2%})')
            _axes[1, 0].set_xlabel('PC1')
            _axes[1, 0].set_ylabel('PC2')
            plt.colorbar(scatter, ax=_axes[1, 0])
            if len(normalized_means.columns) >= 3:
                top_features = normalized_means.T.abs().mean(axis=1).nlargest(5).index
                cluster_data = normalized_means[top_features].T
                cluster_data.plot(kind='bar', ax=_axes[1, 1])
                _axes[1, 1].set_title('Top Features Across Clusters')
                _axes[1, 1].legend(title='Cluster')  # Add key statistics for important features
                _axes[1, 1].tick_params(axis='x', rotation=45)
            _axes[1, 2].axis('off')
            if 'delivery_cost' in self.df.columns:
                insights_text = 'Business Insights:\n\n'  # Identify distinguishing features
                for cluster_id in range(self.optimal_k):
                    cluster_data = self.df[self.df['cluster'] == cluster_id]
                    avg_cost = cluster_data['delivery_cost'].mean()
                    overall_avg = self.df['delivery_cost'].mean()
                    if avg_cost > overall_avg * 1.1:
                        insight = f'Cluster {cluster_id}: High Cost\n'
                    elif avg_cost < overall_avg * 0.9:  # Find features that are significantly different from overall mean
                        insight = f'Cluster {cluster_id}: Low Cost\n'
                    else:
                        insight = f'Cluster {cluster_id}: Avg Cost\n'
                    insights_text += insight
                _axes[1, 2].text(0.1, 0.9, insights_text, transform=_axes[1, 2].transAxes, fontsize=12, verticalalignment='top')
            plt.tight_layout()  # Print cluster profiles
            plt.show()

        def execute_complete_clustering(self):
            """Execute complete clustering pipeline"""
            print('STARTING COMPREHENSIVE CLUSTERING ANALYSIS')
            print('=' * 60)
            self.prepare_clustering_data()
            self.determine_optimal_clusters()
            self.apply_kmeans()
            self.dimensionality_reduction_visualization()
            self.cluster_profiling_analysis()
            self.detailed_cluster_interpretation()
            self.business_insights_recommendations()
            self.final_model_evaluation()
            self.create_visualization_dashboard()
            self.export_results()
            print('\n' + '=' * 60)
            print('CLUSTERING ANALYSIS COMPLETED SUCCESSFULLY!')  # Analyze each cluster for business implications
            print('=' * 60)
            return self.df
    clustering_analysis = ComprehensiveClustering(df_processed)
    # Execute comprehensive clustering
    df_clustered = clustering_analysis.execute_complete_clustering()  # General recommendations  # Silhouette score  # Cluster balance  # Within-cluster sum of squares  # Save clustered data  # Save cluster profiles  # Save model information  # 1. Cluster distribution  # 2. Delivery cost by cluster (if available)  # 3. Feature importance heatmap  # 4. PCA visualization  # 5. Cluster characteristics radar chart (simplified)  # Simple bar chart instead of radar for simplicity  # 6. Business insights summary  # Data preparation  # Determine optimal clusters  # Apply K-means  # Dimensionality reduction  # Cluster analysis  # Business insights  # Final evaluation  # Visualization dashboard  # Export results
    return (clustering_analysis,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Modeling
    """)
    return


@app.cell
def _(
    ColumnTransformer,
    GridSearchCV,
    LGBMRegressor,
    OneHotEncoder,
    Pipeline,
    RandomForestRegressor,
    StandardScaler,
    XGBRegressor,
    df_processed,
    joblib,
    mean_absolute_error,
    mean_squared_error,
    np,
    pd,
    permutation_importance,
    plt,
    r2_score,
    shap,
    sns,
    train_test_split,
):
    class AdvancedPredictiveModeling:

        def __init__(self, df, target_col='delivery_cost'):
            self.df = df.copy()
            self.target_col = target_col
            self.models = {}
            self.results = {}
            self.best_model = None
            self.feature_names = None

        def prepare_modeling_data(self, test_size=0.2, random_state=42):
            """Prepare data for modeling"""
            print('Preparing data for predictive modeling...')
            if self.target_col not in self.df.columns:  # Separate features and target
                raise ValueError(f"Target column '{self.target_col}' not found in dataset")
            X = self.df.drop(columns=[self.target_col])
            y = self.df[self.target_col]
            if 'cluster' in X.columns:
                X = X.drop(columns=['cluster'])
            numerical_features = X.select_dtypes(include=[np.number]).columns.tolist()
            categorical_features = X.select_dtypes(include=['object']).columns.tolist()  # Remove cluster column if it exists (to avoid data leakage)
            self.feature_names = numerical_features + categorical_features
            print(f'Numerical features: {len(numerical_features)}')
            print(f'Categorical features: {len(categorical_features)}')
            print(f'Total features: {len(self.feature_names)}')  # Identify numerical and categorical features
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
            self.X_train, self.X_test, self.y_train, self.y_test = (X_train, X_test, y_train, y_test)
            self.numerical_features = numerical_features
            self.categorical_features = categorical_features
            return (X_train, X_test, y_train, y_test)

        def create_preprocessor(self):
            """Create preprocessing pipeline"""
            print('Creating preprocessing pipeline...')
            numerical_transformer = Pipeline(steps=[('scaler', StandardScaler())])  # Split the data
            categorical_transformer = Pipeline(steps=[('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))])
            preprocessor = ColumnTransformer(transformers=[('num', numerical_transformer, self.numerical_features), ('cat', categorical_transformer, self.categorical_features)])
            return preprocessor

        def initialize_models(self):
            """Initialize multiple models with their parameter grids"""
            print('Initializing models...')
            models = {'RandomForest': {'model': RandomForestRegressor(random_state=42), 'params': {'model__n_estimators': [100, 200], 'model__max_depth': [10, 20, None], 'model__min_samples_split': [2, 5], 'model__min_samples_leaf': [1, 2]}}, 'XGBoost': {'model': XGBRegressor(random_state=42, verbosity=0), 'params': {'model__n_estimators': [100, 200], 'model__max_depth': [3, 6, 9], 'model__learning_rate': [0.01, 0.1, 0.2], 'model__subsample': [0.8, 1.0]}}, 'LightGBM': {'model': LGBMRegressor(random_state=42, verbose=-1), 'params': {'model__n_estimators': [100, 200], 'model__max_depth': [5, 10, -1], 'model__learning_rate': [0.01, 0.1, 0.2], 'model__num_leaves': [31, 50, 100]}}}
            return models

        def train_models(self, cv=5, scoring='neg_mean_squared_error'):
            """Train all models using pipeline and hyperparameter tuning"""
            print('Training models with hyperparameter tuning...')
            preprocessor = self.create_preprocessor()
            models_config = self.initialize_models()  # Preprocessing for numerical features
            best_score = float('-inf')
            self.best_model_name = None
            for model_name, config in models_config.items():
                print(f'\nTraining {model_name}...')
                pipeline = Pipeline(steps=[('preprocessor', preprocessor), ('model', config['model'])])  # Preprocessing for categorical features
                grid_search = GridSearchCV(pipeline, config['params'], cv=cv, scoring=scoring, n_jobs=-1, verbose=1)
                grid_search.fit(self.X_train, self.y_train)
                self.models[model_name] = {'grid_search': grid_search, 'best_estimator': grid_search.best_estimator_}
                y_pred = grid_search.best_estimator_.predict(self.X_test)
                mae = mean_absolute_error(self.y_test, y_pred)  # Combine preprocessing steps
                mse = mean_squared_error(self.y_test, y_pred)
                rmse = np.sqrt(mse)
                r2 = r2_score(self.y_test, y_pred)
                self.results[model_name] = {'best_params': grid_search.best_params_, 'best_score': grid_search.best_score_, 'mae': mae, 'mse': mse, 'rmse': rmse, 'r2': r2, 'predictions': y_pred}
                print(f'{model_name} - Best CV Score: {-grid_search.best_score_:.4f}')
                print(f'{model_name} - Test R²: {r2:.4f}')
                print(f'{model_name} - Test RMSE: {rmse:.4f}')
                if grid_search.best_score_ > best_score:
                    best_score = grid_search.best_score_
                    self.best_model = grid_search.best_estimator_
                    self.best_model_name = model_name
            print(f'\nBest model: {self.best_model_name}')
            return (self.models, self.results)

        def evaluate_models(self):
            """Comprehensive model evaluation"""
            print('\n' + '=' * 50)
            print('COMPREHENSIVE MODEL EVALUATION')
            print('=' * 50)
            comparison_data = []
            for model_name, result in self.results.items():
                comparison_data.append({'Model': model_name, 'R²': result['r2'], 'RMSE': result['rmse'], 'MAE': result['mae'], 'MSE': result['mse']})
            comparison_df = pd.DataFrame(comparison_data)
            comparison_df = comparison_df.sort_values('R²', ascending=False)
            print('Model Performance Comparison:')
            print(comparison_df.to_string(index=False))
            _fig, _axes = plt.subplots(2, 2, figsize=(15, 12))
            _axes[0, 0].barh(comparison_df['Model'], comparison_df['R²'], color='skyblue')
            _axes[0, 0].set_title('Model Comparison - R² Score')
            _axes[0, 0].set_xlabel('R² Score')
            _axes[0, 1].barh(comparison_df['Model'], comparison_df['RMSE'], color='lightcoral')
            _axes[0, 1].set_title('Model Comparison - RMSE')
            _axes[0, 1].set_xlabel('RMSE')
            best_model_results = self.results[self.best_model_name]
            _axes[1, 0].scatter(self.y_test, best_model_results['predictions'], alpha=0.6)
            _axes[1, 0].plot([self.y_test.min(), self.y_test.max()], [self.y_test.min(), self.y_test.max()], 'r--', lw=2)
            _axes[1, 0].set_xlabel('Actual')
            _axes[1, 0].set_ylabel('Predicted')
            _axes[1, 0].set_title(f'Actual vs Predicted - {self.best_model_name}')
            residuals = self.y_test - best_model_results['predictions']
            _axes[1, 1].scatter(best_model_results['predictions'], residuals, alpha=0.6)
            _axes[1, 1].axhline(y=0, color='r', linestyle='--')
            _axes[1, 1].set_xlabel('Predicted')
            _axes[1, 1].set_ylabel('Residuals')
            _axes[1, 1].set_title(f'Residual Plot - {self.best_model_name}')
            plt.tight_layout()
            plt.show()
            return comparison_df

        def feature_importance_analysis(self):
            """Comprehensive feature importance analysis"""
            print('\n' + '=' * 50)
            print('FEATURE IMPORTANCE ANALYSIS')
            print('=' * 50)
            if self.best_model is None:
                print('No best model found. Please train models first.')
                return None
            best_estimator = self.best_model
            preprocessor = best_estimator.named_steps['preprocessor']  # Create pipeline
            model = best_estimator.named_steps['model']
            feature_names = []
            feature_names.extend(self.numerical_features)
            if len(self.categorical_features) > 0:
                categorical_transformer = preprocessor.named_transformers_['cat']
                if hasattr(categorical_transformer, 'named_steps'):  # Perform grid search
                    onehot = categorical_transformer.named_steps['onehot']
                    if hasattr(onehot, 'get_feature_names_out'):
                        cat_features = onehot.get_feature_names_out(self.categorical_features)
                        feature_names.extend(cat_features)
                else:
                    onehot = categorical_transformer
                    if hasattr(onehot, 'get_feature_names_out'):  # Increased verbosity to see progress
                        cat_features = onehot.get_feature_names_out(self.categorical_features)
                        feature_names.extend(cat_features)
            if hasattr(model, 'feature_importances_'):  # Fit the model
                importances = model.feature_importances_
                feature_imp_df = pd.DataFrame({'feature': feature_names[:len(importances)], 'importance': importances}).sort_values('importance', ascending=False)
                print('Top 10 Most Important Features:')  # Store the model and results
                print(feature_imp_df.head(10).to_string(index=False))
                plt.figure(figsize=(12, 8))
                top_features = feature_imp_df.head(15)
                sns.barplot(data=top_features, y='feature', x='importance', palette='viridis')
                plt.title(f'Feature Importance - {self.best_model_name}')
                plt.xlabel('Importance Score')  # Make predictions using the best estimator
                plt.tight_layout()
                plt.show()
                return feature_imp_df  # Calculate metrics
            else:
                print("Model doesn't have built-in feature importance")
                return self.calculate_permutation_importance()

        def calculate_permutation_importance(self):
            """Calculate permutation importance as fallback"""
            print('Calculating permutation importance...')
            try:
                preprocessor = self.best_model.named_steps['preprocessor']
                X_test_processed = preprocessor.transform(self.X_test)
                result = permutation_importance(self.best_model.named_steps['model'], X_test_processed, self.y_test, n_repeats=10, random_state=42)
                feature_names = preprocessor.get_feature_names_out()
                perm_imp_df = pd.DataFrame({'feature': feature_names, 'importance': result.importances_mean}).sort_values('importance', ascending=False)
                print('Top 10 Features by Permutation Importance:')
                print(perm_imp_df.head(10).to_string(index=False))
                plt.figure(figsize=(12, 8))
                top_features = perm_imp_df.head(15)
                sns.barplot(data=top_features, y='feature', x='importance', palette='coolwarm')
                plt.title(f'Permutation Feature Importance - {self.best_model_name}')
                plt.xlabel('Importance Score')  # Update best model
                plt.tight_layout()
                plt.show()
                return perm_imp_df
            except Exception as e:
                print(f'Permutation importance calculation failed: {e}')
                return None

        def shap_analysis(self):
            """SHAP analysis for model interpretation"""
            print('\n' + '=' * 50)
            print('SHAP ANALYSIS')
            print('=' * 50)
            if self.best_model is None:
                print('No best model found. Please train models first.')
                return (None, None)
            try:  # Create comparison table
                preprocessor = self.best_model.named_steps['preprocessor']
                X_train_processed = preprocessor.transform(self.X_train)
                X_test_processed = preprocessor.transform(self.X_test)
                feature_names = preprocessor.get_feature_names_out()
                model = self.best_model.named_steps['model']
                if self.best_model_name in ['RandomForest', 'XGBoost', 'LightGBM']:
                    explainer = shap.TreeExplainer(model)
                else:
                    explainer = shap.Explainer(model, X_train_processed, feature_names=feature_names)
                shap_values = explainer(X_test_processed)
                plt.figure(figsize=(12, 8))
                shap.summary_plot(shap_values, X_test_processed, feature_names=feature_names, show=False)
                plt.title(f'SHAP Summary Plot - {self.best_model_name}', fontsize=14, fontweight='bold')
                plt.tight_layout()
                plt.show()
                plt.figure(figsize=(12, 8))
                shap.plots.bar(shap_values, max_display=15, show=False)  # Visualization
                plt.title(f'SHAP Feature Importance - {self.best_model_name}', fontsize=14, fontweight='bold')
                plt.tight_layout()
                plt.show()  # 1. R² comparison
                print('\nGenerating individual prediction explanations...')
                for _i in range(min(3, len(self.X_test))):
                    plt.figure(figsize=(12, 6))
                    shap.plots.waterfall(shap_values[_i], max_display=10, show=False)
                    plt.title(f'SHAP Explanation for Sample {_i + 1} - {self.best_model_name}', fontsize=12, fontweight='bold')  # 2. RMSE comparison
                    plt.tight_layout()
                    plt.show()
                    actual = self.y_test.iloc[_i] if hasattr(self.y_test, 'iloc') else self.y_test[_i]
                    predicted = self.results[self.best_model_name]['predictions'][_i]
                    print(f'Sample {_i + 1}: Actual = {actual:.2f}, Predicted = {predicted:.2f}')  # 3. Actual vs Predicted for best model
                feature_imp_df = self.feature_importance_analysis()
                if feature_imp_df is not None:
                    top_features = feature_imp_df.head(3)['feature'].tolist()
                    print(f'\nGenerating SHAP dependence plots for top features: {top_features}')
                    for _i, feature in enumerate(top_features):
                        try:
                            feature_idx = list(feature_names).index(feature)
                            plt.figure(figsize=(10, 6))
                            shap.dependence_plot(feature_idx, shap_values.values, X_test_processed, feature_names=feature_names, show=False)  # 4. Residual plot for best model
                            plt.title(f'SHAP Dependence Plot for {feature}', fontsize=12, fontweight='bold')
                            plt.tight_layout()
                            plt.show()
                        except (ValueError, IndexError) as e:
                            print(f'Could not create dependence plot for {feature}: {e}')
                            continue
                return (explainer, shap_values)
            except Exception as e:
                print(f'SHAP analysis failed: {e}')
                print('This might be due to model type or data size limitations.')
                import traceback
                traceback.print_exc()
                return (None, None)

        def business_recommendations(self, feature_imp_df):
            """Generate business recommendations based on model insights"""
            print('\n' + '=' * 50)
            print('BUSINESS RECOMMENDATIONS')
            print('=' * 50)
            best_result = self.results[self.best_model_name]
            print(f'MODEL PERFORMANCE SUMMARY:')
            print(f'- Best Model: {self.best_model_name}')
            print(f"- R² Score: {best_result['r2']:.4f}")  # Get the best estimator (which is a Pipeline)
            print(f"- RMSE: {best_result['rmse']:.4f}")
            print(f"- MAE: {best_result['mae']:.4f}")
            if feature_imp_df is not None and len(feature_imp_df) > 0:  # Get feature names after preprocessing
                top_features = feature_imp_df.head(5)['feature'].tolist()
                print(f'\nKEY COST DRIVERS IDENTIFIED:')
                for _i, feature in enumerate(top_features, 1):
                    importance_score = feature_imp_df[feature_imp_df['feature'] == feature]['importance'].values[0]  # Get feature names
                    print(f'{_i}. {feature} (importance: {importance_score:.4f})')
            print(f'\nSTRATEGIC RECOMMENDATIONS:')
            print('1. COST OPTIMIZATION:')  # Numerical features
            print('   - Focus on the top cost drivers identified above')
            print('   - Implement monitoring systems for these key variables')
            print('   - Develop targeted cost reduction strategies')  # Categorical features (after one-hot encoding)
            print('2. OPERATIONAL EFFICIENCY:')
            print('   - Use the predictive model for route planning and resource allocation')
            print('   - Optimize delivery schedules based on cost patterns')
            print('   - Implement performance benchmarking across different segments')
            print('3. DATA-DRIVEN DECISION MAKING:')
            print('   - Integrate the model into operational systems')
            print('   - Establish regular model retraining cycles')
            print('   - Create dashboards for real-time cost monitoring')
            print('4. PARTNER MANAGEMENT:')  # Direct OneHotEncoder case
            print('   - Use insights for partner performance evaluation')
            print('   - Develop incentive structures based on cost efficiency')
            print('   - Implement collaborative planning with top partners')
            print(f'\nMODEL DEPLOYMENT STRATEGY:')
            if best_result['r2'] > 0.8:
                print('✅ EXCELLENT - Model is ready for production deployment')  # Model-based feature importance
                print('   - Can be used for automated decision making')
                print('   - Suitable for real-time cost prediction')
            elif best_result['r2'] > 0.6:
                print('✅ GOOD - Model is suitable for decision support')  # Create feature importance dataframe
                print('   - Use for planning and strategic decisions')
                print('   - Combine with business expertise for best results')
            elif best_result['r2'] > 0.4:
                print('⚠️ MODERATE - Model needs improvement')
                print('   - Use for directional insights')
                print('   - Combine with other business intelligence')
            else:
                print('❌ NEEDS IMPROVEMENT - Review feature engineering and data quality')
                print('   - Focus on data collection improvement')  # Plot feature importance
                print('   - Consider additional feature engineering')

        def save_final_model(self):
            """Save the final model and results"""
            print('\nSaving final model and results...')
            try:
                joblib.dump(self.best_model, 'best_delivery_cost_model.pkl')
                results_to_save = {'best_model_name': self.best_model_name, 'model_results': self.results, 'feature_names': self.feature_names, 'best_params': self.results[self.best_model_name]['best_params']}
                joblib.dump(results_to_save, 'model_results.pkl')
                predictions_df = self.X_test.copy()
                predictions_df['actual_delivery_cost'] = self.y_test.values
                predictions_df['predicted_delivery_cost'] = self.results[self.best_model_name]['predictions']
                predictions_df['residual'] = predictions_df['actual_delivery_cost'] - predictions_df['predicted_delivery_cost']  # Use permutation importance as fallback
                predictions_df.to_csv('model_predictions.csv', index=False)
                feature_imp_df = self.feature_importance_analysis()
                if feature_imp_df is not None:
                    feature_imp_df.to_csv('feature_importance.csv', index=False)
                print('✅ Model and results saved successfully!')
                print('   - best_delivery_cost_model.pkl: Trained model')
                print('   - model_results.pkl: Training results and parameters')
                print('   - model_predictions.csv: Prediction results')  # Use the best model to calculate permutation importance
                print('   - feature_importance.csv: Feature importance scores')
            except Exception as e:
                print(f'❌ Error saving model: {e}')
      # Calculate permutation importance
        def execute_complete_modeling(self):
            """Execute complete predictive modeling pipeline"""
            print('STARTING ADVANCED PREDICTIVE MODELING')
            print('=' * 60)
            try:
                self.prepare_modeling_data()
                self.train_models()
                comparison_df = self.evaluate_models()
                feature_imp_df = self.feature_importance_analysis()  # Get feature names
                explainer, shap_values = self.shap_analysis()
                self.business_recommendations(feature_imp_df)
                self.save_final_model()  # Create importance dataframe
                print('\n' + '=' * 60)
                print('✅ PREDICTIVE MODELING COMPLETED SUCCESSFULLY!')
                print('=' * 60)
                return {'best_model': self.best_model, 'best_model_name': self.best_model_name, 'results': self.results, 'feature_importance': feature_imp_df, 'model_comparison': comparison_df, 'shap_explainer': explainer, 'shap_values': shap_values}
            except Exception as e:
                print(f'❌ Error in modeling pipeline: {e}')
                import traceback
                traceback.print_exc()
                return None  # Plot
    print('Initializing predictive modeling...')
    modeling_pipeline = AdvancedPredictiveModeling(df_processed)
    print('\nStarting model training...')
    modeling_results = modeling_pipeline.execute_complete_modeling()
    if modeling_results is not None:
        print('\n🎯 MODELING RESULTS SUMMARY:')
        print(f"Best Model: {modeling_results['best_model_name']}")
        print(f"Best R² Score: {modeling_results['results'][modeling_results['best_model_name']]['r2']:.4f}")
    else:
    # Execute predictive modeling with error handling
        print('\n❌ Modeling failed. Please check the error messages above.')  # Get preprocessed data using the best model's preprocessor  # Get feature names  # Get the actual model from the pipeline  # Create SHAP explainer based on model type  # Calculate SHAP values for test set  # SHAP summary plot  # SHAP bar plot (mean absolute SHAP values)  # Individual prediction explanations (first 3 samples)  # Print actual vs predicted for context  # SHAP dependence plots for top 3 features  # Find the index of the feature  # Model deployment guidance  # Save the best model  # Save model results  # Save predictions with actual data  # Save feature importance if available  # Data preparation  # Model training  # Model evaluation  # Feature importance  # SHAP analysis  # Business recommendations  # Save final model
    return (modeling_pipeline,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Conclusion
    """)
    return


@app.cell
def _(clustering_analysis, modeling_pipeline):
    # FINAL COMPREHENSIVE CONCLUSION
    """
    DELIVERY LOGISTICS OPTIMIZATION: COMPREHENSIVE ANALYSIS CONCLUSION
    ==================================================================

    EXECUTIVE SUMMARY:
    This comprehensive data science project successfully analyzed the Indian multi-partner 
    delivery logistics dataset to uncover valuable insights, segment operations, and build 
    predictive models for cost optimization.

    KEY ACHIEVEMENTS:

    1. DATA UNDERSTANDING & PREPROCESSING:
       - Successfully loaded and explored the delivery logistics dataset
       - Handled missing values and performed comprehensive feature engineering
       - Created derived features to enhance model performance

    2. CLUSTERING ANALYSIS:
       - Identified optimal number of clusters using advanced techniques
       - Segmented delivery operations into distinct groups
       - Provided detailed cluster profiling and interpretation
       - Delivered actionable business insights for each segment

    3. PREDICTIVE MODELING:
       - Built multiple advanced models (Random Forest, XGBoost, LightGBM)
       - Implemented robust pipeline with hyperparameter tuning
       - Achieved strong predictive performance for delivery costs
       - Conducted comprehensive model interpretation using SHAP

    BUSINESS IMPACT:

    COST OPTIMIZATION:
    - Identified key cost drivers in delivery operations
    - Provided cluster-specific optimization strategies
    - Enabled accurate cost prediction for better planning

    OPERATIONAL EFFICIENCY:
    - Revealed patterns in delivery performance
    - Highlighted opportunities for route optimization
    - Supported data-driven decision making

    STRATEGIC INSIGHTS:
    - Partner performance segmentation
    - Resource allocation optimization
    - Service level improvements

    TECHNICAL EXCELLENCE:

    METHODOLOGICAL RIGOR:
    - Comprehensive EDA with advanced visualizations
    - Robust feature engineering pipeline
    - Systematic clustering approach
    - Ensemble modeling with cross-validation

    MODEL PERFORMANCE:
    - Multiple algorithms compared and optimized
    - Strong predictive accuracy achieved
    - Comprehensive model interpretation
    - Production-ready model deployment

    LIMITATIONS & FUTURE WORK:

    DATA LIMITATIONS:
    - Potential missing contextual information
    - Temporal aspects could be further explored
    - External factors (weather, traffic) not included

    ENHANCEMENT OPPORTUNITIES:
    - Real-time prediction capabilities
    - Integration with operational systems
    - Advanced time series analysis
    - Reinforcement learning for dynamic optimization

    FINAL RECOMMENDATIONS:

    IMMEDIATE ACTIONS:
    1. Implement cluster-specific delivery strategies
    2. Optimize routes based on cost drivers identified
    3. Use predictive model for cost estimation and budgeting
    4. Monitor high-cost clusters for continuous improvement

    STRATEGIC INITIATIVES:
    1. Develop partner performance scoring system
    2. Implement dynamic pricing models
    3. Create automated anomaly detection
    4. Establish data-driven KPI monitoring

    CONCLUSION:
    This project demonstrates the significant value of data science in logistics optimization. 
    The insights generated provide a solid foundation for cost reduction, operational 
    efficiency improvements, and strategic decision-making in the competitive Indian 
    logistics market.

    The comprehensive approach—from data understanding through modeling to business 
    recommendations—showcases the full spectrum of data science capabilities and their 
    practical application in solving real-world business problems.
    """

    # Export final report
    def generate_final_report():
        """Generate and save final project report"""

        report_content = """
        DELIVERY LOGISTICS OPTIMIZATION - FINAL PROJECT REPORT
        ======================================================

        PROJECT OVERVIEW:
        This project analyzed delivery logistics data to optimize operations and reduce costs.

        KEY FINDINGS:

        1. DATA CHARACTERISTICS:
           - Dataset successfully processed and engineered
           - Multiple feature types identified and handled
           - High-quality data suitable for advanced analysis

        2. CLUSTERING RESULTS:
           - Optimal number of clusters identified: {optimal_k}
           - Clear segmentation of delivery patterns achieved
           - Business-interpretable clusters created

        3. PREDICTIVE MODELING:
           - Best model: {best_model}
           - Model performance: R² = {r2_score:.4f}
           - Strong predictive capability demonstrated

        4. BUSINESS INSIGHTS:
           - Key cost drivers identified and analyzed
           - Actionable recommendations provided
           - Optimization opportunities highlighted

        TECHNICAL DELIVERABLES:
        - Comprehensive EDA with visualizations
        - Advanced feature engineering pipeline
        - Clustering analysis with profiling
        - Ensemble predictive modeling
        - Model interpretation using SHAP
        - Production-ready model deployment

        BUSINESS VALUE:
        - Cost reduction opportunities identified
        - Operational efficiency improvements
        - Data-driven decision support
        - Strategic planning enhancement

        NEXT STEPS:
        - Implement recommended strategies
        - Deploy predictive model in production
        - Establish monitoring system
        - Plan continuous improvement cycle
        """.format(
            optimal_k=clustering_analysis.optimal_k,
            best_model=modeling_pipeline.best_model_name,
            r2_score=modeling_pipeline.results[modeling_pipeline.best_model_name]['r2']
        )

        with open('delivery_logistics_final_report.txt', 'w') as f:
            f.write(report_content)

        print("Final report generated successfully!")

    # Generate final report
    generate_final_report()

    print("\n" + "="*80)
    print("PROJECT COMPLETED SUCCESSFULLY!")
    print("All analyses, models, and reports have been generated.")
    print("Check the generated files for detailed results.")
    print("="*80)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Reference

    1. https://xgboost.readthedocs.io/en/release_3.0.0/tutorials/index.html

    2. https://xbe.at/

    3. https://scikit-learn.org/stable/index.html
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Contact

    [Linkedin](http://linkedin.com/in/muh-amri-sidiq)
    """)
    return


@app.cell
def _():
    import marimo as mo
    return (mo,)


if __name__ == "__main__":
    app.run()
