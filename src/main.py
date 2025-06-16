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
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import numpy as np
import holidays
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_score

def read_data(filepath: str) -> pd.DataFrame:
    df = pd.read_csv(filepath, parse_dates=["Date"], date_format='%m-%d-%y', low_memory=False)

    # Trim khoảng trắng cho tên các cột
    df.columns = df.columns.str.strip()

    # Ép kiểu cơ bản
    df["Amount"] = pd.to_numeric(df["Amount"], errors="coerce")
    df["Qty"] = pd.to_numeric(df["Qty"], errors="coerce").fillna(0).astype(int)

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

    print("Số lượng giá trị thiếu:")
    print(df.isnull().sum()[df.isnull().sum() > 0])
    print()

    print("Các cột phân loại (categorical) phổ biến:")
    categorical_cols = df.select_dtypes(include='object').columns
    for col in categorical_cols:
        print(f"- {col}: {df[col].nunique()} giá trị (top: {df[col].value_counts().idxmax()})")

    print()

    print("Các cột số (numeric) cơ bản:")
    numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns
    print(df[numeric_cols].describe().T)

def prepare_features(df: pd.DataFrame, drop_first: bool = True) -> pd.DataFrame:
    df = df.copy()

    # Nhóm đặc trưng thời gian
    df["day"] = df["Date"].dt.day
    df["year"] = df["Date"].dt.year
    df["month"] = df["Date"].dt.month
    df["weekday"] = df["Date"].dt.dayofweek
    df["week"] = df["Date"].dt.isocalendar().week
    df["is_weekend"] = df["weekday"].isin([5, 6]).astype(int)  # 5=Saturday, 6=Sunday
    season_map = {
        12: 'Winter', 1: 'Winter', 2: 'Winter',  # Winter: Dec, Jan, Feb
        3: 'Spring', 4: 'Spring', 5: 'Spring',    # Spring: Mar, Apr, May
        6: 'Summer', 7: 'Summer', 8: 'Summer',    # Summer: Jun, Jul, Aug
        9: 'Fall', 10: 'Fall', 11: 'Fall'         # Fall: Sep, Oct, Nov
    }
    df["season"] = df["month"].map(season_map)
    us_holidays = holidays.US()
    df["is_holiday"] = df["Date"].apply(lambda x: x in us_holidays).astype(int)
    df["holiday_name"] = df["Date"].apply(lambda x: us_holidays.get(x, "Not Holiday"))
    
    # Nhóm đặc trưng khuyến mãi
    df["promotion_applied"] = df["promotion-ids"].notnull().astype(int)
    df["qty_x_promo"] = df["Qty"] * df["promotion_applied"]
    
    # Nhóm đặc trưng giá cả
    df["price_per_unit"] = df["Amount"] / df["Qty"]  # Giá trung bình trên mỗi đơn vị
    df["price_per_unit"] = df["price_per_unit"].replace([np.inf, -np.inf], np.nan)  # Xử lý chia cho 0
    df["price_per_unit"] = df["price_per_unit"].fillna(df["price_per_unit"].mean())  # Điền giá trị trung bình
    df["price_promo"] = df["price_per_unit"] * df["promotion_applied"]
    df["price_season"] = df["price_per_unit"] * df["month"].map(lambda x: 1 if x in [12, 1, 2] else  # Winter
                                                                    2 if x in [3, 4, 5] else  # Spring
                                                                    3 if x in [6, 7, 8] else  # Summer
                                                                    4)  # Fall
    # Nhóm đặc trưng giá tương tác
    df["season_holiday"] = df["season"] + "_" + df["holiday_name"]
    df["promo_holiday"] = df["promotion_applied"].astype(str) + "_" + df["holiday_name"]
    
    # Nhóm đặc trưng theo sản phẩm
    sku_avg_sales = df.groupby('SKU')['Amount'].transform('mean')
    df['sku_avg_sales'] = sku_avg_sales.fillna(df['Amount'].mean())  # Điền giá trị trung bình chung nếu không có dữ liệu SKU
    category_avg_sales = df.groupby('Category')['Amount'].transform('mean')
    df['category_avg_sales'] = category_avg_sales.fillna(df['Amount'].mean())  # Điền giá trị trung bình chung nếu không có dữ liệu Category
    
    # Tính tỷ lệ doanh thu của SKU so với trung bình Category
    df['sku_to_category_ratio'] = df['sku_avg_sales'] / df['category_avg_sales']
    df['sku_to_category_ratio'] = df['sku_to_category_ratio'].replace([np.inf, -np.inf], 1.0)  # Xử lý chia cho 0
    df['sku_to_category_ratio'] = df['sku_to_category_ratio'].fillna(1.0)  # Điền 1.0 nếu không có dữ liệu
    
    # Nhóm đặc trưng theo xu hướng ngắn hạn
    df['sku_rolling_avg_30d'] = df.groupby('SKU')['Amount'].transform(
        lambda x: x.rolling(window=30, min_periods=1).mean()
    )
    df['sku_rolling_avg_30d'] = df['sku_rolling_avg_30d'].fillna(df['sku_avg_sales'])  # Điền bằng trung bình SKU nếu không có dữ liệu 30 ngày
    
    # Tính doanh thu trung bình theo Category trong 30 ngày gần nhất
    df['category_rolling_avg_30d'] = df.groupby('Category')['Amount'].transform(
        lambda x: x.rolling(window=30, min_periods=1).mean()
    )
    df['category_rolling_avg_30d'] = df['category_rolling_avg_30d'].fillna(df['category_avg_sales'])  # Điền bằng trung bình Category nếu không có dữ liệu 30 ngày
    
    # Tính tỷ lệ doanh thu hiện tại so với trung bình 30 ngày của SKU
    df['sku_sales_trend'] = df['Amount'] / df['sku_rolling_avg_30d']
    df['sku_sales_trend'] = df['sku_sales_trend'].replace([np.inf, -np.inf], 1.0)  # Xử lý chia cho 0
    df['sku_sales_trend'] = df['sku_sales_trend'].fillna(1.0)  # Điền 1.0 nếu không có dữ liệu
    
    # Tính tỷ lệ doanh thu hiện tại so với trung bình 30 ngày của Category
    df['category_sales_trend'] = df['Amount'] / df['category_rolling_avg_30d']
    df['category_sales_trend'] = df['category_sales_trend'].replace([np.inf, -np.inf], 1.0)  # Xử lý chia cho 0
    df['category_sales_trend'] = df['category_sales_trend'].fillna(1.0)  # Điền 1.0 nếu không có dữ liệu

    df_model = df.copy()
    # One-hot encode các cột phân loại và xử lý NaN
    df_model = pd.get_dummies(df_model, 
                            columns=["Category", "Sales Channel", "fulfilled-by", "ship-country", 
                                   "season", "holiday_name", "season_holiday", "promo_holiday"], 
                            drop_first=drop_first,
                            dummy_na=True)  # Thêm cột cho giá trị NaN
    
    return df_model

def forecast_sales_with_linear_regression(df: pd.DataFrame, print_top_features: bool = False):
    #DỰ BÁO DOANH THU THEO THỜI GIAN
    df = df.dropna(subset=["Amount"])
    
    # Lấy tất cả các cột sau khi one-hot encoding
    time_features = ["day", "year", "month", "weekday", "week", "is_weekend", "is_holiday"]
    product_features = ["Qty", "promotion_applied", "qty_x_promo", "price_per_unit", "price_promo"]
    category_features = [col for col in df.columns if col.startswith("Category_")]
    sales_channel_features = [col for col in df.columns if col.startswith("Sales Channel_")]
    season_features = [col for col in df.columns if col.startswith("season_")]
    holiday_features = [col for col in df.columns if col.startswith("holiday_name_")]
    season_holiday_features = [col for col in df.columns if col.startswith("season_holiday_")]
    promo_holiday_features = [col for col in df.columns if col.startswith("promo_holiday_")]
    price_season_features = ["price_season"]
    
    # Thêm features mới về doanh thu
    sales_features = [
        "sku_avg_sales", "category_avg_sales", "sku_to_category_ratio",
        "sku_rolling_avg_30d", "category_rolling_avg_30d",
        "sku_sales_trend", "category_sales_trend"
    ]
    
    features = (time_features + product_features + category_features + 
               sales_channel_features + season_features + holiday_features + 
               season_holiday_features + promo_holiday_features + price_season_features +
               sales_features)
    
    X = df[features]
    y = df["Amount"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # In kích thước các tập dữ liệu
    print("\nKích thước các tập dữ liệu sau khi tách:")
    print(f"X_train shape: {X_train.shape}")
    print(f"X_test shape: {X_test.shape}")
    print(f"y_train shape: {y_train.shape}")
    print(f"y_test shape: {y_test.shape}")
    
    # In một số mẫu đầu tiên của tập train
    print("\nMẫu đầu tiên của tập train:")
    print("X_train:")
    print(X_train.head())
    print("\ny_train:")
    print(y_train.head())
    
    model = LinearRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    metrics = {
        "MAE": mean_absolute_error(y_test, y_pred),
        "MSE": mean_squared_error(y_test, y_pred),
        "RMSE": np.sqrt(mean_squared_error(y_test, y_pred)),
        "R2 Score": r2_score(y_test, y_pred)
    }
    print("Linear Regression Metrics:")
    print(metrics)
    print()

    if print_top_features:
        # In tầm quan trọng của các features dựa trên coefficients
        feature_importance = pd.DataFrame({
            'feature': features,
            'coefficient': np.abs(model.coef_)  # Lấy giá trị tuyệt đối của coefficients
        }).sort_values('coefficient', ascending=False)
        print("Top 5 Important Features (based on coefficients):")
        print(feature_importance.head(5))
        print()
    
    return model, metrics, X_train, X_test, y_train, y_test
    
def forecast_sales_with_gradient_boosting(df, print_top_features: bool = False):
    #DỰ BÁO DOANH THU THEO THỜI GIAN
    # Loại bỏ bản ghi không có Amount
    df = df.dropna(subset=["Amount"])
    
    # Lựa chọn các feature
    time_features = ["day", "year", "month", "weekday", "week", "is_weekend", "is_holiday"]
    product_features = ["Qty", "promotion_applied", "qty_x_promo", "price_per_unit", "price_promo"]
    category_features = [col for col in df.columns if col.startswith("Category_")]
    sales_channel_features = [col for col in df.columns if col.startswith("Sales Channel_")]
    season_features = [col for col in df.columns if col.startswith("season_")]
    holiday_features = [col for col in df.columns if col.startswith("holiday_name_")]
    season_holiday_features = [col for col in df.columns if col.startswith("season_holiday_")]
    promo_holiday_features = [col for col in df.columns if col.startswith("promo_holiday_")]
    price_season_features = ["price_season"]
    
    # Thêm features mới về doanh thu
    sales_features = [
        "sku_avg_sales", "category_avg_sales", "sku_to_category_ratio",
        "sku_rolling_avg_30d", "category_rolling_avg_30d",
        "sku_sales_trend", "category_sales_trend"
    ]
    
    features = (time_features + product_features + category_features + 
               sales_channel_features + season_features + holiday_features + 
               season_holiday_features + promo_holiday_features + price_season_features +
               sales_features)
    
    X = df[features]
    y = df["Amount"]
    # Chia dữ liệu train/test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    # Khởi tạo mô hình Gradient Boosting
    model = GradientBoostingRegressor(
        n_estimators=100,
        learning_rate=0.1,
        max_depth=3,
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

    if print_top_features:
        # In tầm quan trọng của các features
        feature_importance = pd.DataFrame({
            'feature': features,
            'importance': model.feature_importances_
        }).sort_values('importance', ascending=False)
        print("Top 5 Important Features:")
        print(feature_importance.head(5))
        print()

    return model, metrics, X_train, X_test, y_train, y_test

def forecast_sales_with_random_forest(df: pd.DataFrame, print_top_features: bool = False):
    #DỰ BÁO DOANH THU THEO THỜI GIAN
    # Loại bỏ bản ghi không có Amount
    df = df.dropna(subset=["Amount"])
    
    # Lấy tất cả các cột sau khi one-hot encoding
    time_features = ["day", "year", "month", "weekday", "week", "is_weekend", "is_holiday"]
    product_features = ["Qty", "promotion_applied", "qty_x_promo", "price_per_unit", "price_promo"]
    category_features = [col for col in df.columns if col.startswith("Category_")]
    sales_channel_features = [col for col in df.columns if col.startswith("Sales Channel_")]
    season_features = [col for col in df.columns if col.startswith("season_")]
    holiday_features = [col for col in df.columns if col.startswith("holiday_name_")]
    season_holiday_features = [col for col in df.columns if col.startswith("season_holiday_")]
    promo_holiday_features = [col for col in df.columns if col.startswith("promo_holiday_")]
    price_season_features = ["price_season"]
    
    # Thêm features mới về doanh thu
    sales_features = [
        "sku_avg_sales", "category_avg_sales", "sku_to_category_ratio",
        "sku_rolling_avg_30d", "category_rolling_avg_30d",
        "sku_sales_trend", "category_sales_trend"
    ]
    
    features = (time_features + product_features + category_features + 
               sales_channel_features + season_features + holiday_features + 
               season_holiday_features + promo_holiday_features + price_season_features +
               sales_features)
    
    X = df[features]
    y = df["Amount"]
    
    # Chia dữ liệu train/test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Khởi tạo và huấn luyện mô hình Random Forest
    model = RandomForestRegressor(
        n_estimators=100,  # Số lượng cây
        max_depth=10,      # Độ sâu tối đa của mỗi cây
        min_samples_split=5,  # Số lượng mẫu tối thiểu để chia node
        min_samples_leaf=2,   # Số lượng mẫu tối thiểu ở lá
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
    
    if print_top_features:
        # In tầm quan trọng của các features
        feature_importance = pd.DataFrame({
            'feature': features,
            'importance': model.feature_importances_
        }).sort_values('importance', ascending=False)
        print("Top 5 Important Features:")
        print(feature_importance.head(5))
        print()
    
    return model, metrics, X_train, X_test, y_train, y_test

def check_overfitting(model, X_train, y_train, X_test, y_test, cv=5):
    """
    Đánh giá overfitting của mô hình:
        - In R², MAE, RMSE trên cả train và test
        - In Cross-validation R²
        - Hiển thị biểu đồ so sánh y_test vs y_pred
    """
    # Train metrics
    y_train_pred = model.predict(X_train)
    r2_train = r2_score(y_train, y_train_pred)
    mae_train = mean_absolute_error(y_train, y_train_pred)
    rmse_train = np.sqrt(mean_squared_error(y_train, y_train_pred))

    # Test metrics
    y_test_pred = model.predict(X_test)
    r2_test = r2_score(y_test, y_test_pred)
    mae_test = mean_absolute_error(y_test, y_test_pred)
    rmse_test = np.sqrt(mean_squared_error(y_test, y_test_pred))

    # Cross-validation R2
    cv_scores = cross_val_score(model, X_train, y_train, cv=cv, scoring='r2')
    cv_r2_mean = np.mean(cv_scores)
    cv_r2_std = np.std(cv_scores)

    print(f"Train:    R2={r2_train:.4f}, MAE={mae_train:.4f}, RMSE={rmse_train:.4f}")
    print(f"Test:     R2={r2_test:.4f}, MAE={mae_test:.4f}, RMSE={rmse_test:.4f}")
    print(f"CV R2:    Mean={cv_r2_mean:.4f}, Std={cv_r2_std:.4f}")

    # Plot y_test vs y_pred
    plt.figure(figsize=(6,6))
    plt.scatter(y_test, y_test_pred, alpha=0.5)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
    plt.xlabel('Actual y_test')
    plt.ylabel('Predicted y_test')
    plt.title('y_test vs y_pred')
    plt.grid(True)
    plt.show()

def main():
    df = read_data("datasets/amazon_sale_report.csv")
    summarize_data(df)
    df_linear_regression = prepare_features(df, drop_first=True)
    df_gradient_boosting = prepare_features(df, drop_first=False)
    df_random_forest = prepare_features(df, drop_first=False)
    # Dự đoán doanh thu theo thời gian
    # model_lr, metrics_lr, X_train_lr, X_test_lr, y_train_lr, y_test_lr = forecast_sales_with_linear_regression(df_linear_regression, print_top_features=False)
    # check_overfitting(model_lr, X_train_lr, y_train_lr, X_test_lr, y_test_lr)
    # model_gb, metrics_gb, X_train_gb, X_test_gb, y_train_gb, y_test_gb = forecast_sales_with_gradient_boosting(df_gradient_boosting, print_top_features=False)
    # check_overfitting(model_gb, X_train_gb, y_train_gb, X_test_gb, y_test_gb)
    model_rf, metrics_rf, X_train_rf, X_test_rf, y_train_rf, y_test_rf = forecast_sales_with_random_forest(df_random_forest, print_top_features=False)
    check_overfitting(model_rf, X_train_rf, y_train_rf, X_test_rf, y_test_rf)

if __name__ == "__main__":
    main()