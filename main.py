import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import seaborn as sns
import matplotlib.pyplot as plt

def load_and_clean_data():
    """Tải và làm sạch dữ liệu"""
    # Tải dữ liệu
    df = pd.read_csv('hotel_bookings.csv')
    
    # Xử lý missing values
    df['children'].fillna(0, inplace=True)
    df['country'].fillna('Unknown', inplace=True)
    df['agent'].fillna(0, inplace=True)
    df['company'].fillna(0, inplace=True)
    
    # Tạo biến mục tiêu: tổng số người (adults + children + babies)
    df['total_guests'] = df['adults'] + df['children'] + df['babies']
    
    # Chọn các features quan trọng
    selected_features = [
        'hotel', 'arrival_date_month', 'arrival_date_week_number',
        'arrival_date_day_of_month', 'stays_in_weekend_nights',
        'stays_in_week_nights', 'adults', 'children', 'babies',
        'meal', 'market_segment', 'distribution_channel',
        'is_repeated_guest', 'previous_cancellations',
        'previous_bookings_not_canceled', 'reserved_room_type',
        'assigned_room_type', 'booking_changes', 'deposit_type',
        'days_in_waiting_list', 'customer_type', 'adr',
        'required_car_parking_spaces', 'total_of_special_requests',
        'total_guests'
    ]
    
    return df[selected_features]

def prepare_data(df):
    """Tiền xử lý dữ liệu"""
    # Tạo copy của DataFrame
    df_processed = df.copy()
    
    # Mã hóa biến phân loại
    le = LabelEncoder()
    categorical_columns = [
        'hotel', 'arrival_date_month', 'meal', 'market_segment',
        'distribution_channel', 'reserved_room_type', 'assigned_room_type',
        'deposit_type', 'customer_type'
    ]
    
    for col in categorical_columns:
        df_processed[col] = le.fit_transform(df_processed[col])
    
    # Chuẩn hóa các biến số
    scaler = StandardScaler()
    numerical_columns = [
        'arrival_date_week_number', 'arrival_date_day_of_month',
        'stays_in_weekend_nights', 'stays_in_week_nights',
        'previous_cancellations', 'previous_bookings_not_canceled',
        'booking_changes', 'days_in_waiting_list', 'adr',
        'required_car_parking_spaces', 'total_of_special_requests'
    ]
    
    df_processed[numerical_columns] = scaler.fit_transform(df_processed[numerical_columns])
    
    return df_processed

def evaluate_model(y_true, y_pred, model_name):
    """Đánh giá mô hình"""
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_true, y_pred)
    
    print(f"\nKết quả đánh giá mô hình {model_name}:")
    print(f"MAE: {mae:.2f}")
    print(f"MSE: {mse:.2f}")
    print(f"RMSE: {rmse:.2f}")
    print(f"R-squared: {r2:.4f}")
    
    return {'mae': mae, 'mse': mse, 'rmse': rmse, 'r2': r2}

def plot_feature_importance(importance_df):
    """Vẽ biểu đồ tầm quan trọng của features"""
    plt.figure(figsize=(12, 6))
    sns.barplot(x='importance', y='feature', data=importance_df)
    plt.title('Feature Importance (Random Forest)')
    plt.xlabel('Importance Score')
    plt.tight_layout()
    plt.show()

def main():
    # Tải và làm sạch dữ liệu
    print("Đang tải dữ liệu...")
    df = load_and_clean_data()
    print(f"Kích thước dữ liệu: {df.shape}")
    
    # Tiền xử lý dữ liệu
    print("\nĐang tiền xử lý dữ liệu...")
    processed_df = prepare_data(df)
    
    # Chia features và target
    X = processed_df.drop('total_guests', axis=1)
    y = processed_df['total_guests']
    
    # Chia tập train/test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    print(f"\nKích thước tập train: {X_train.shape}")
    print(f"Kích thước tập test: {X_test.shape}")
    
    # Linear Regression
    print("\nĐang huấn luyện Linear Regression...")
    lr_model = LinearRegression()
    lr_model.fit(X_train, y_train)
    lr_pred = lr_model.predict(X_test)
    lr_metrics = evaluate_model(y_test, lr_pred, "Linear Regression")
    
    # Random Forest
    print("\nĐang huấn luyện Random Forest...")
    rf_model = RandomForestRegressor(
        n_estimators=100,
        random_state=42,
        n_jobs=-1  # Sử dụng tất cả CPU cores
    )
    rf_model.fit(X_train, y_train)
    rf_pred = rf_model.predict(X_test)
    rf_metrics = evaluate_model(y_test, rf_pred, "Random Forest")
    
    # Phân tích feature importance
    feature_importance = pd.DataFrame({
        'feature': X.columns,
        'importance': rf_model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    print("\nTop 10 đặc trưng quan trọng nhất:")
    print(feature_importance.head(10))
    
    # Vẽ biểu đồ feature importance
    plot_feature_importance(feature_importance)
    
    # Kết luận
    print("\nKết luận:")
    if rf_metrics['r2'] > lr_metrics['r2']:
        print("Random Forest cho kết quả tốt hơn với R-squared cao hơn")
        best_model = "Random Forest"
        best_metrics = rf_metrics
    else:
        print("Linear Regression cho kết quả tốt hơn với R-squared cao hơn")
        best_model = "Linear Regression"
        best_metrics = lr_metrics
    
    print(f"\nMô hình {best_model} tốt nhất với các chỉ số:")
    print(f"- MAE: {best_metrics['mae']:.2f}")
    print(f"- RMSE: {best_metrics['rmse']:.2f}")
    print(f"- R-squared: {best_metrics['r2']:.4f}")

if __name__ == "__main__":
    main()