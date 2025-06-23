"""
Lab1: PREDICTING PRODUCT SALES

This script implements the following models:
2. Logistic Regression


Dataset:
1. Enron Email Dataset: https://www.kaggle.com/datasets/thedevastator/unlock-profits-with-e-commerce-sales-data/data
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor, ExtraTreesClassifier
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from scipy import stats
import numpy as np
import holidays
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV

def read_data(filepath: str) -> pd.DataFrame:
    df = pd.read_csv(filepath, parse_dates=["Date"], date_format='%m-%d-%y', low_memory=False)

    # Trim khoảng trắng cho tên các cột
    df.columns = df.columns.str.strip()
    
    # Loại bỏ các cột không có giá trị trong việc dự đoán doanh thu
    columns_to_remove = [
        'index', 'Order ID', 'Unnamed: 22'
    ]
    
    # Chỉ loại bỏ các cột thực sự tồn tại
    existing_columns_to_remove = [col for col in columns_to_remove if col in df.columns]
    if existing_columns_to_remove:
        df = df.drop(columns=existing_columns_to_remove)

    # Ép kiểu cơ bản
    df["Amount"] = pd.to_numeric(df["Amount"], errors="coerce")
    df["Qty"] = pd.to_numeric(df["Qty"], errors="coerce").fillna(0).astype(int)
    
    # Loại bỏ các dòng có Amount = 0
    df = df[df["Amount"] != 0]

    # Điền giá trị NaN bằng giá trị trung bình cho các cột số
    numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns
    df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].mean())
    
    # Điền giá trị NaN bằng giá trị phổ biến nhất cho các cột phân loại
    categorical_cols = df.select_dtypes(include=['object']).columns
    df[categorical_cols] = df[categorical_cols].fillna(df[categorical_cols].mode().iloc[0])

    return df

def summarize_data(df: pd.DataFrame):
    print(set(df.dtypes.index) - set(df.columns))
    print(set(df.columns) - set(df.dtypes.index))
    print("Tổng quan dữ liệu:")
    print(f"- Số dòng: {df.shape[0]}")
    print(f"- Số cột: {df.shape[1]}")
    print()

    print("Các cột và kiểu dữ liệu:")
    print(df.dtypes)
    print()

    if 'Amount' in df.columns:
        print("Thống kê chi tiết cột Amount:")
        print(f"- Giá trị nhỏ nhất (min): {df['Amount'].min():.2f}")
        print(f"- Giá trị lớn nhất (max): {df['Amount'].max():.2f}")
        print(f"- Giá trị trung bình (mean): {df['Amount'].mean():.2f}")

def prepare_features(df: pd.DataFrame) -> pd.DataFrame:
    df_engineered = df.copy()

    # Nhóm đặc trưng thời gian
    df_engineered["day"] = df_engineered["Date"].dt.day
    df_engineered["year"] = df_engineered["Date"].dt.year
    df_engineered["month"] = df_engineered["Date"].dt.month
    df_engineered["weekday"] = df_engineered["Date"].dt.dayofweek
    df_engineered["is_weekend"] = df_engineered["weekday"].isin([5, 6]).astype(int)

    us_holidays = holidays.US()
    df_engineered["is_holiday"] = df_engineered["Date"].apply(lambda x: x in us_holidays).astype(int)
    df_engineered["promotion_applied"] = df_engineered["promotion-ids"].notnull().astype(int)
    df_engineered["b2b_flag"] = df['B2B'].astype(str).str.upper().map({"TRUE": 1, "FALSE": 0}).fillna(0).astype(int)
    sku_counts = df_engineered['SKU'].value_counts()
    df_engineered['sku_popularity'] = df_engineered['SKU'].map(sku_counts)
    category_counts = df_engineered['Category'].value_counts()
    df_engineered['category_popularity'] = df_engineered['Category'].map(category_counts)
    df_engineered['order_size_bucket'] = pd.cut(
        df['Qty'],
        bins=[-1, 1, 5, float('inf')],
        labels=['small', 'medium', 'large']
    ).astype(str)
    
    features_to_remove = [
        'Date', 'promotion-ids', 'B2B', 'SKU', 'Category'
    ]

    existing_features_to_remove = [col for col in features_to_remove if col in df_engineered.columns]
    if existing_features_to_remove:
        df_engineered = df_engineered.drop(columns=existing_features_to_remove)

    print(f"Created {len([col for col in df_engineered.columns if col not in df.columns])} new engineered features.")
    print(f"Removed {len(features_to_remove)} original features.")
    print(f"Total features after engineering: {len(df_engineered.columns)}")

    return df_engineered


def handle_outliers(df, z_threshold=3.45):
    df_cleaned = df.copy()
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    
    # Don't handle outliers in target variable if it exists
    if 'Amount' in numeric_cols:
        numeric_cols = numeric_cols.drop('Amount')
    
    # Handle outliers in each numerical column
    for col in numeric_cols:
        # Skip if standard deviation is too small
        if df[col].std() < 1e-10:
            continue
        # Calculate z-scores with error handling
        try:
            z_scores = np.abs(stats.zscore(df[col]))
            df_cleaned[col] = df[col].mask(z_scores > z_threshold, df[col].mean())
        except:
            # If z-score calculation fails, skip this column
            continue
    
    return df_cleaned

def encode_categorical(df):
    df_encoded = df.copy()
    categorical_cols = df.select_dtypes(include=['object']).columns
    
    # Encode each categorical column
    for col in categorical_cols:
        le = LabelEncoder()
        df_encoded[col] = le.fit_transform(df[col])
    
    return df_encoded


def encode_categorical(df):
    """
    Encode categorical variables using LabelEncoder
    
    Args:
        df (pandas.DataFrame): Input dataframe
        
    Returns:
        pandas.DataFrame: DataFrame with encoded categorical variables
    """
    df_encoded = df.copy()
    categorical_cols = df.select_dtypes(include=['object']).columns
    
    # Encode each categorical column
    for col in categorical_cols:
        le = LabelEncoder()
        df_encoded[col] = le.fit_transform(df[col])
    
    return df_encoded

def normalize_data(df):
    """
    Normalize numerical data using StandardScaler
    
    Args:
        df (pandas.DataFrame): Input dataframe
        
    Returns:
        pandas.DataFrame: DataFrame with normalized numerical variables
    """
    df_normalized = df.copy()
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    
    # Don't normalize target variable if it exists
    if 'Amount' in numeric_cols:
        numeric_cols = numeric_cols.drop('Amount')
    
    # Normalize each numerical column
    scaler = MinMaxScaler()
    df_normalized[numeric_cols] = scaler.fit_transform(df[numeric_cols])
    
    return df_normalized

def feature_importance(df: pd.DataFrame, importance_threshold: float = 0.95):
    if 'Amount' not in df.columns:
        raise ValueError("Input DataFrame must contain 'Amount' column.")

    # --- 1. Train Model to Get Importance Scores ---
    X = df.drop('Amount', axis=1)
    y = df['Amount']

    # Using RandomForestRegressor for proper regression feature importance
    from sklearn.ensemble import RandomForestRegressor
    model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    model.fit(X, y)

    # --- 2. Create Importance DataFrame ---
    importance_df = pd.DataFrame({
        'Feature': X.columns,
        'Importance': model.feature_importances_
    }).sort_values('Importance', ascending=False)
    
    print("Feature Importance Ranking:")
    print(importance_df.to_string(index=False))
    print()

    # Calculate cumulative importance
    importance_df['Cumulative_Importance'] = importance_df['Importance'].cumsum()

    # --- 3. Select Features Based on Threshold ---
    # Find features that contribute to the threshold
    selected_features = importance_df[importance_df['Cumulative_Importance'] <= importance_threshold]
    
    # If no features meet the threshold, select at least the top feature
    if selected_features.empty:
        selected_features = importance_df.head(1)
    # If the top feature already exceeds threshold, still include it
    elif importance_df.iloc[0]['Cumulative_Importance'] > importance_threshold:
        selected_features = importance_df.head(1)
        
    # --- 4. Print Report and Prepare Final DataFrame ---
    print("="*60)
    print("         FEATURE IMPORTANCE & SELECTION REPORT")
    print("="*60)
    print(f"Model used for selection: RandomForestRegressor")
    print(f"Selection threshold (cumulative importance): {importance_threshold}")
    print("-"*60)
    print(f"Total features analyzed: {len(X.columns)}")
    print(f"Features selected: {len(selected_features)}")
    print(f"Features removed: {len(X.columns) - len(selected_features)}")
    print(f"Cumulative importance of selected features: {selected_features['Cumulative_Importance'].iloc[-1]:.4f}")
    print("-"*60)
    
    print("Selected Features:")
    for _, row in selected_features.iterrows():
        print(f"  {row['Feature']}: {row['Importance']:.4f} (cumulative: {row['Cumulative_Importance']:.4f})")
    
    print("\nRemoved Features:")
    removed_features = importance_df[~importance_df['Feature'].isin(selected_features['Feature'])]
    for _, row in removed_features.iterrows():
        print(f"  {row['Feature']}: {row['Importance']:.4f}")

    # Get the list of feature names to keep and add the target variable back
    features_to_keep = selected_features['Feature'].tolist()
    features_to_keep.append('Amount')
    
    # Return the original DataFrame filtered to the selected features
    df_filtered = df[features_to_keep]
    print(f"\nReturning DataFrame with {len(df_filtered.columns)} columns.")
    print("="*60 + "\n")

    return df_filtered


def grid_search_gradient_boosting(df: pd.DataFrame):
    """
    Perform Grid Search to find best hyperparameters for Gradient Boosting
    Uses only 80% of data for grid search
    
    Args:
        df (pd.DataFrame): Input dataframe with features and target
        
    Returns:
        dict: Best parameters found
    """
    from sklearn.model_selection import GridSearchCV
    from sklearn.ensemble import GradientBoostingRegressor
    
    print("=== GRID SEARCH FOR GRADIENT BOOSTING (80% DATA) ===")
    
    # Prepare data
    X = df.drop('Amount', axis=1)
    y = df['Amount']
    # Split data: 80% for grid search, 20% for final testing
    X_grid, X_test, y_grid, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
   
    # Define parameter grid
    param_grid = {
        'n_estimators': [100, 200, 300],
        'learning_rate': [0.01, 0.05, 0.1, 0.2],
        'max_depth': [3, 4, 5, 6],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'subsample': [0.8, 0.9, 1.0]
    }
    print("Parameter grid:")
    for param, values in param_grid.items():
        print(f"  {param}: {values}")
    print()
    
    # Initialize base model
    base_model = GradientBoostingRegressor(random_state=42)
    
    # Initialize GridSearchCV
    grid_search = GridSearchCV(
        estimator=base_model,
        param_grid=param_grid,
        cv=5,  # 5-fold cross-validation
        scoring='neg_root_mean_squared_error',  # Use RMSE for regression
        n_jobs=-1,  # Use all CPU cores
        verbose=1
    )
    
    # Fit the grid search on 80% of data
    grid_search.fit(X_grid, y_grid)
    
    # Get best parameters and score
    best_params = grid_search.best_params_
    best_score = -grid_search.best_score_  # Convert back to positive MSE
    best_rmse = np.sqrt(best_score)
    
    print("=== GRID SEARCH RESULTS (on 80% data) ===")
    print(f"Best parameters: {best_params}")
    print(f"Best MSE: {best_score:.4f}")
    print(f"Best RMSE: {best_rmse:.4f}")
    print()
    
    return best_params

def grid_search_random_forest(df: pd.DataFrame):
    from sklearn.model_selection import GridSearchCV
    from sklearn.ensemble import RandomForestRegressor
    
    print("=== GRID SEARCH FOR RANDOM FOREST (80% DATA) ===")
    
    # Prepare data
    X = df.drop('Amount', axis=1)
    y = df['Amount']
    # Split data: 80% for grid search, 20% for final testing
    X_grid, X_test, y_grid, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
   
    # Define parameter grid for Random Forest
    param_grid = {
        'n_estimators': [100, 200, 300],
        'max_depth': [5, 10, 15, None],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4]
    }
    
    print("Parameter grid:")
    for param, values in param_grid.items():
        print(f"  {param}: {values}")
    print()
    
    # Initialize base model
    base_model = RandomForestRegressor(random_state=42, n_jobs=-1)
    
    # Initialize GridSearchCV
    grid_search = GridSearchCV(
        estimator=base_model,
        param_grid=param_grid,
        cv=5,  # 5-fold cross-validation
        scoring='neg_root_mean_squared_error',  # Use MSE for regression
        n_jobs=-1,  # Use all CPU cores
        verbose=3
    )
    
    # Fit the grid search on 80% of data
    grid_search.fit(X_grid, y_grid)
    
    # Get best parameters and score
    best_params = grid_search.best_params_
    best_score = -grid_search.best_score_  # Convert back to positive MSE
    best_rmse = np.sqrt(best_score)
    
    print("=== GRID SEARCH RESULTS (on 80% data) ===")
    print(f"Best parameters: {best_params}")
    print(f"Best MSE: {best_score:.4f}")
    print(f"Best RMSE: {best_rmse:.4f}")
    print()
    
    return best_params


def predict_with_linear_regression(df: pd.DataFrame):
    """
    Improved Linear Regression with diagnostics and preprocessing
    """
    X = df.drop('Amount', axis=1)
    y = df['Amount']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = LinearRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    # Calculate metrics
    metrics = {
        "MAE": mean_absolute_error(y_test, y_pred),
        "MSE": mean_squared_error(y_test, y_pred),
        "RMSE": np.sqrt(mean_squared_error(y_test, y_pred)),
        "R2 Score": r2_score(y_test, y_pred)
    }
    
    print("=== LINEAR REGRESSION RESULTS ===")
    print("Test Metrics:")
    print(metrics)

    return model, metrics, X_train, X_test, y_train, y_test
    
def predict_with_gradient_boosting(df: pd.DataFrame):
    X = df.drop('Amount', axis=1)
    y = df['Amount']
    # Chia dữ liệu train/test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    # Khởi tạo mô hình Gradient Boosting
    model = GradientBoostingRegressor(
        n_estimators=300,
        learning_rate=0.2,
        max_depth=6,
        min_samples_split=10,
        min_samples_leaf=2,
        subsample=1.0,
        random_state=42
    )
    # Huấn luyện mô hình
    model.fit(X_train, y_train)
    # Dự đoán
    y_pred = model.predict(X_test)
    # Tính các chỉ số đánh giá
    metrics = {
        "MAE": mean_absolute_error(y_test, y_pred),
        "MSE": mean_squared_error(y_test, y_pred),
        "RMSE": np.sqrt(mean_squared_error(y_test, y_pred)),
        "R2 Score": r2_score(y_test, y_pred)
    }
    print("Gradient Boosting Metrics:")
    print(metrics)
    print()

    return model, metrics, X_train, X_test, y_train, y_test

def predict_with_random_forest(df: pd.DataFrame):
    X = df.drop('Amount', axis=1)
    y = df['Amount']
    # Chia dữ liệu train/test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Khởi tạo và huấn luyện mô hình Random Forest
    model = RandomForestRegressor(
        n_estimators=300,  # Số lượng cây
        max_depth=None,      # Độ sâu tối đa của mỗi cây
        min_samples_split=2,  # Số lượng mẫu tối thiểu để chia node
        min_samples_leaf=1,   # Số lượng mẫu tối thiểu ở lá
        random_state=42
    )
    
    # Huấn luyện mô hình
    model.fit(X_train, y_train)
    
    # Dự đoán
    y_pred = model.predict(X_test)
    
    # Tính các chỉ số đánh giá
    metrics = {
        "MAE": mean_absolute_error(y_test, y_pred),
        "MSE": mean_squared_error(y_test, y_pred),
        "RMSE": np.sqrt(mean_squared_error(y_test, y_pred)),
        "R2 Score": r2_score(y_test, y_pred)
    }
    
    # In kết quả
    print("Random Forest Metrics:")
    print(metrics)
    print()
    
    return model, metrics, X_train, X_test, y_train, y_test

def main():
    df = read_data("datasets/amazon_sale_report.csv")
    df = prepare_features(df)          # 2. Tạo đặc trưng thủ công (dùng giá trị gốc chưa bị encode)
    df = handle_outliers(df)            # 3. Xử lý ngoại lệ (trên các cột số gốc hoặc vừa tạo)
    df = encode_categorical(df)         # 4. Mã hóa biến phân loại (LabelEncoder)
    df = normalize_data(df)             # 5. Chuẩn hóa dữ liệu số
    df = feature_importance(df, importance_threshold=0.99)

    #Grid Search
    best_params_gradient_boosting = grid_search_gradient_boosting(df)
    best_params_random_forest = grid_search_random_forest(df)
    
    predict_with_linear_regression(df)
    predict_with_gradient_boosting(df)
    predict_with_random_forest(df)

if __name__ == "__main__":
    main()