import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error, mean_absolute_error
import numpy as np
import matplotlib
from sklearn.preprocessing import OneHotEncoder, RobustScaler
from sklearn.neural_network import MLPRegressor

matplotlib.use('Agg')  # Use non-GUI backend for compatibility

def perform_analysis():
    print("Starting perform_analysis for all months of 2024...")

    # Load data
    excel_file_path = "RC Projection data since 2022 28AUG24.xlsx"
    sheets_to_load = ['2022', '2023', '2024']
    dfs = {sheet: pd.read_excel(excel_file_path, sheet_name=sheet) for sheet in sheets_to_load}

    # Clean and preprocess data
    for year in ['2022', '2023', '2024']:
        dfs[year]['TRANSACTION_AMOUNT'] = pd.to_numeric(dfs[year]['TRANSACTION_AMOUNT'], errors='coerce')
        dfs[year].dropna(subset=['TRANSACTION_AMOUNT'], inplace=True)
        dfs[year]['MONTH'] = pd.to_datetime(dfs[year]['MONTH']).dt.month

    # Group data for Linear Regression
    grouped_2022 = dfs['2022'].groupby(['YEAR', 'MONTH', 'FIN_SOURCE_TYPE_DESC'])['TRANSACTION_AMOUNT'].sum().reset_index()
    grouped_2023 = dfs['2023'].groupby(['YEAR', 'MONTH', 'FIN_SOURCE_TYPE_DESC'])['TRANSACTION_AMOUNT'].sum().reset_index()
    grouped_2024 = dfs['2024'].groupby(['YEAR', 'MONTH', 'FIN_SOURCE_TYPE_DESC'])['TRANSACTION_AMOUNT'].sum().reset_index()

    # Train on 2022-2023, predict for 2024 (Linear Regression)
    train_data = pd.concat([grouped_2022, grouped_2023])
    test_data = grouped_2024
    financial_sources = train_data['FIN_SOURCE_TYPE_DESC'].unique()
    predictions = []

    for source in financial_sources:
        train_source = train_data[train_data['FIN_SOURCE_TYPE_DESC'] == source]
        test_source = test_data[test_data['FIN_SOURCE_TYPE_DESC'] == source]

        if train_source.empty or test_source.empty:
            print(f"Skipping {source} due to lack of data.")
            continue

        X_train = train_source[['YEAR', 'MONTH']].astype(int)
        y_train = train_source['TRANSACTION_AMOUNT']
        X_test = test_source[['YEAR', 'MONTH']].astype(int)

        model = LinearRegression()
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        test_source = test_source.copy()
        test_source['PREDICTED_AMOUNT'] = y_pred
        predictions.append(test_source)

    predictions_df = pd.concat(predictions)

    # Linear Regression - Plot 1
    plt.figure(figsize=(15, 8))
    sns.lineplot(data=predictions_df, x='MONTH', y='TRANSACTION_AMOUNT', hue='FIN_SOURCE_TYPE_DESC', linestyle='-')
    sns.lineplot(data=predictions_df, x='MONTH', y='PREDICTED_AMOUNT', hue='FIN_SOURCE_TYPE_DESC', linestyle='--', alpha=0.6)
    plt.title('Actual vs Predicted Monthly Transaction Amount by Financial Source Type (2024)')
    plt.xlabel('Month')
    plt.ylabel('Transaction Amount')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    image_path_1 = "static/predicted_image_sources_lr_2024.png"
    plt.savefig(image_path_1)
    plt.close()

    # Linear Regression - Plot 2
    monthly_actual_total = predictions_df.groupby('MONTH')['TRANSACTION_AMOUNT'].sum()
    monthly_predicted_total = predictions_df.groupby('MONTH')['PREDICTED_AMOUNT'].sum()

    plt.figure(figsize=(10, 6))
    plt.plot(monthly_actual_total.index, monthly_actual_total, label='Actual Total', color='blue', marker='o')
    plt.plot(monthly_predicted_total.index, monthly_predicted_total, label='Predicted Total', color='red', linestyle='--', marker='x')
    plt.title('Aggregate Monthly Amounts (Linear Regression - 2024)')
    plt.xlabel('Month')
    plt.ylabel('Transaction Amount')
    plt.legend(loc='upper left')
    plt.tight_layout()
    image_path_2 = "static/predicted_image_aggregate_lr_2024.png"
    plt.savefig(image_path_2)
    plt.close()

    # --- ARIMA Part ---
    train_data_agg = train_data.groupby(['YEAR', 'MONTH'])['TRANSACTION_AMOUNT'].sum().reset_index()
    test_data_agg = test_data.groupby(['YEAR', 'MONTH'])['TRANSACTION_AMOUNT'].sum().reset_index()

    # Convert to datetime format and set as index
    train_data_agg['YEAR_MONTH'] = pd.to_datetime(train_data_agg['YEAR'].astype(str) + '-' + train_data_agg['MONTH'].astype(str).str.zfill(2), format='%Y-%m', errors='coerce')
    test_data_agg['YEAR_MONTH'] = pd.to_datetime(test_data_agg['YEAR'].astype(str) + '-' + test_data_agg['MONTH'].astype(str).str.zfill(2), format='%Y-%m', errors='coerce')
    train_data_agg.set_index('YEAR_MONTH', inplace=True)
    test_data_agg.set_index('YEAR_MONTH', inplace=True)

    # ARIMA Model Training and Forecast
    model_arima = ARIMA(train_data_agg['TRANSACTION_AMOUNT'], order=(1, 1, 1))
    model_arima_fit = model_arima.fit()
    forecast_arima = model_arima_fit.forecast(steps=len(test_data_agg))

    comparison_df_arima = test_data_agg.copy()
    comparison_df_arima['PREDICTED_AMOUNT'] = forecast_arima.values

    # ARIMA - Plot 1: ACF/PACF for Model Check
    from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
    series_diff = train_data_agg['TRANSACTION_AMOUNT'].diff().dropna()

    plt.figure(figsize=(12, 6))
    plt.subplot(121)
    plot_acf(series_diff, ax=plt.gca())
    plt.title("ARIMA Model - ACF Plot")

    plt.subplot(122)
    plot_pacf(series_diff, ax=plt.gca())
    plt.title("ARIMA Model - PACF Plot")
    plt.tight_layout()
    image_path_3 = "static/arima_acf_pacf_2024.png"
    plt.savefig(image_path_3)
    plt.close()

    # ARIMA - Plot 2: Aggregate Monthly Predictions
    plt.figure(figsize=(10, 6))
    plt.plot(comparison_df_arima.index, comparison_df_arima['TRANSACTION_AMOUNT'], label='Actual Total', color='blue', marker='o')
    plt.plot(comparison_df_arima.index, comparison_df_arima['PREDICTED_AMOUNT'], label='Predicted Total (ARIMA)', color='red', linestyle='--', marker='x')
    plt.title('Aggregate Monthly Amounts (ARIMA - 2024)')
    plt.xlabel('Month')
    plt.ylabel('Transaction Amount')
    plt.xticks(rotation=45)
    plt.legend(loc='upper left')
    plt.tight_layout()
    image_path_4 = "static/predicted_image_aggregate_arima_2024.png"
    plt.savefig(image_path_4)
    plt.close()

    # --- Neural Network Part ---
    print("Starting Neural Network analysis...")

    # Encode categorical feature
    encoder = OneHotEncoder(sparse_output=False)
    encoded_features = encoder.fit_transform(train_data[['FIN_SOURCE_TYPE_DESC']])

    nn_train_features = pd.concat([
        train_data[['MONTH']].reset_index(drop=True),
        pd.DataFrame(encoded_features, columns=encoder.get_feature_names_out(['FIN_SOURCE_TYPE_DESC']))
    ], axis=1)

    encoded_test_features = encoder.transform(test_data[['FIN_SOURCE_TYPE_DESC']])
    nn_test_features = pd.concat([
        test_data[['MONTH']].reset_index(drop=True),
        pd.DataFrame(encoded_test_features, columns=encoder.get_feature_names_out(['FIN_SOURCE_TYPE_DESC']))
    ], axis=1)

    nn_y_train = train_data['TRANSACTION_AMOUNT'].values
    nn_y_test = test_data['TRANSACTION_AMOUNT'].values

    # Scale features
    scaler = RobustScaler()
    X_train_scaled = scaler.fit_transform(nn_train_features)
    X_test_scaled = scaler.transform(nn_test_features)

    # Neural Network model
    mlp = MLPRegressor(hidden_layer_sizes=(100, 100, 50), activation='relu', solver='adam', max_iter=2000, random_state=42, learning_rate_init=0.01)
    mlp.fit(X_train_scaled, nn_y_train)

    nn_y_pred = mlp.predict(X_test_scaled)

    # Plot NN Predictions
    actual_total_2024 = test_data.copy()
    actual_total_2024['TRANSACTION_AMOUNT'] = nn_y_test
    actual_monthly_sum = actual_total_2024.groupby('MONTH')['TRANSACTION_AMOUNT'].sum().reset_index()

    predicted_total_2024 = test_data.copy()
    predicted_total_2024['TRANSACTION_AMOUNT'] = nn_y_pred
    predicted_monthly_sum = predicted_total_2024.groupby('MONTH')['TRANSACTION_AMOUNT'].sum().reset_index()

    plt.figure(figsize=(10, 6))
    plt.plot(actual_monthly_sum['MONTH'], actual_monthly_sum['TRANSACTION_AMOUNT'], label='Actual Total Transaction Amount', color='blue', marker='o')
    plt.plot(predicted_monthly_sum['MONTH'], predicted_monthly_sum['TRANSACTION_AMOUNT'], label='Predicted Total Transaction Amount', color='red', linestyle='--', marker='x')
    plt.title('Actual vs Predicted Monthly Transaction Amounts (Neural Network - 2024)')
    plt.xlabel('Month')
    plt.ylabel('Transaction Amount')
    plt.xticks(ticks=actual_monthly_sum['MONTH'], labels=actual_monthly_sum['MONTH'])
    plt.legend()
    plt.tight_layout()
    image_path_nn = "static/predicted_image_aggregate_nn_2024.png"
    plt.savefig(image_path_nn)
    plt.close()

    print("Neural Network analysis complete. Saved plot to", image_path_nn)

    print("Analysis complete.")
    return image_path_1, image_path_2, image_path_3, image_path_4, image_path_nn

# If testing directly
if __name__ == "__main__":
    image_path_1, image_path_2, image_path_3, image_path_4, image_path_nn = perform_analysis()
    if all([image_path_1, image_path_2, image_path_3, image_path_4, image_path_nn]):
        print(f"Analysis completed and images saved at '{image_path_1}', '{image_path_2}', '{image_path_3}', '{image_path_4}', and '{image_path_nn}'.")
    else:
        print("No output generated.")