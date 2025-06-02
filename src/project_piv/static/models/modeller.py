# modeller.py
import pandas as pd
import numpy as np
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))

# Set matplotlib backend to 'Agg' before importing pyplot
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import pickle
from sklearn.model_selection import train_test_split, TimeSeriesSplit, GridSearchCV
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.pipeline import Pipeline
from xgboost import XGBRegressor
from matplotlib.backends.backend_pdf import PdfPages
from project_piv.logger import Logger

class Modeller:
    """
    Class for building, training and evaluating predictive models for financial data.
    """

    def __init__(self, logger, data_dir=None, indicator_symbol="ES=F", prediction_horizon=5):
        """
        Initialize the Modeller class.

        Args:
            logger: Logger instance for logging
            data_dir: Directory where data is stored
            indicator_symbol: Symbol of the financial indicator
            prediction_horizon: Number of days to predict ahead
        """
        self.logger = logger
        self.class_name = self.__class__.__name__
        self.indicator_symbol = indicator_symbol
        self.prediction_horizon = prediction_horizon

        # Set data directory
        if data_dir is None:
            self.data_dir = os.path.join(
                os.path.dirname(os.path.abspath(__file__)),
                '..',  # Go up one level to reach static
                'data'
            )
        else:
            self.data_dir = data_dir

        # Define file paths
        self.data_path = os.path.join(
            self.data_dir,
            f'enriched_{indicator_symbol.replace("=F", "")}.csv'
        )

        # Create models directory if it doesn't exist
        self.models_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'models')
        os.makedirs(self.models_dir, exist_ok=True)

        self.model_path = os.path.join(self.models_dir, 'model.pkl')
        self.scaler_path = os.path.join(self.models_dir, 'scaler.pkl')
        self.metrics_path = os.path.join(self.models_dir, 'model_metrics.csv')
        self.feature_importance_path = os.path.join(
            self.models_dir, 'feature_importance.csv')
        self.model_report_path = os.path.join(
            self.models_dir, 'model_report.pdf')

        # Define evaluation metrics
        self.metrics = {
            'RMSE': lambda y_true, y_pred: np.sqrt(mean_squared_error(y_true, y_pred)),
            'MAE': mean_absolute_error,
            'R2': r2_score
        }

        self.logger.info(
            self.class_name,
            "__init__",
            f"Modeller initialized for {indicator_symbol} with prediction horizon of {prediction_horizon} days."
        )

    def load_data(self):
        """
        Load enriched data from CSV file.

        Returns:
            DataFrame with enriched data
        """
        function_name = "load_data"

        try:
            if not os.path.exists(self.data_path):
                self.logger.error(
                    self.class_name,
                    function_name,
                    f"Enriched data file not found: {self.data_path}"
                )
                return pd.DataFrame()

            df = pd.read_csv(self.data_path)

            # Convert date column to datetime
            if 'date' in df.columns:
                df['date'] = pd.to_datetime(df['date'])

            # Sort by date
            df = df.sort_values('date')

            self.logger.info(
                self.class_name,
                function_name,
                f"Loaded {len(df)} rows from {self.data_path}"
            )

            return df

        except Exception as e:
            self.logger.error(
                self.class_name,
                function_name,
                f"Error loading data: {e}"
            )
            return pd.DataFrame()

    def prepare_features_target(self, df):
        """
        Prepare features and target variables for model training.

        Args:
            df: DataFrame with enriched data

        Returns:
            Tuple of (X, y, feature_names)
        """
        function_name = "prepare_features_target"

        if df.empty:
            self.logger.error(
                self.class_name,
                function_name,
                "DataFrame is empty, cannot prepare features and target."
            )
            return None, None, None

        try:
            # Create target variable: future price after prediction_horizon days
            df['target'] = df['close'].shift(-self.prediction_horizon)

            # Drop rows with NaN in target
            df = df.dropna(subset=['target'])

            # Exclude columns that shouldn't be used as features
            exclude_cols = ['date', 'target']

            # Identify columns with too many NaN values (more than 10%)
            na_threshold = len(df) * 0.1
            cols_with_na = df.columns[df.isna().sum() > na_threshold].tolist()
            exclude_cols.extend(cols_with_na)

            # Select feature columns
            feature_cols = [
                col for col in df.columns if col not in exclude_cols]

            # Drop rows with NaN in features
            df = df.dropna(subset=feature_cols)

            # Handle infinite values and large numbers
            for col in feature_cols:
                # Replace infinite values with NaN
                df[col] = df[col].replace([np.inf, -np.inf], np.nan)
                
                # Calculate the 1st and 99th percentiles
                q1 = df[col].quantile(0.01)
                q99 = df[col].quantile(0.99)
                
                # Clip values outside these percentiles
                df[col] = df[col].clip(lower=q1, upper=q99)

            # Drop any remaining NaN values
            df = df.dropna()

            # Extract features and target
            X = df[feature_cols]
            y = df['target']

            self.logger.info(
                self.class_name,
                function_name,
                f"Prepared features and target. X shape: {X.shape}, y shape: {y.shape}"
            )

            return X, y, feature_cols

        except Exception as e:
            self.logger.error(
                self.class_name,
                function_name,
                f"Error preparing features and target: {e}"
            )
            return None, None, None

    def train_test_split_time_series(self, X, y, test_size=0.2):
        """
        Split data into training and testing sets, respecting time order.

        Args:
            X: Feature DataFrame
            y: Target Series
            test_size: Proportion of data to use for testing

        Returns:
            Tuple of (X_train, X_test, y_train, y_test)
        """
        function_name = "train_test_split_time_series"

        try:
            # Calculate split index
            split_idx = int(len(X) * (1 - test_size))

            # Split data
            X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
            y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

            self.logger.info(
                self.class_name,
                function_name,
                f"Split data into training ({len(X_train)} samples) and testing ({len(X_test)} samples)"
            )

            return X_train, X_test, y_train, y_test

        except Exception as e:
            self.logger.error(
                self.class_name,
                function_name,
                f"Error splitting data: {e}"
            )
            return None, None, None, None

    def train_models(self, X_train, y_train, X_test, y_test):
        """
        Train multiple models and select the best one.

        Args:
            X_train: Training features
            y_train: Training target
            X_test: Testing features
            y_test: Testing target

        Returns:
            Tuple of (best_model, scaler, metrics, model_name)
        """
        function_name = "train_models"

        try:
            # Initialize scaler
            scaler = StandardScaler()

            # Scale features
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)

            # Define models to try
            models = {
                'LinearRegression': LinearRegression(),
                'Ridge': Ridge(alpha=1.0),
                'Lasso': Lasso(alpha=0.1),
                'RandomForest': RandomForestRegressor(n_estimators=100, random_state=42),
                'GradientBoosting': GradientBoostingRegressor(n_estimators=100, random_state=42),
                'SVR': SVR(kernel='rbf', C=1.0, epsilon=0.1),
                'XGBoost': XGBRegressor(n_estimators=100, learning_rate=0.1, random_state=42)
            }

            # Train and evaluate each model
            results = {}

            for name, model in models.items():
                try:
                    # Train model
                    model.fit(X_train_scaled, y_train)

                    # Make predictions
                    y_pred = model.predict(X_test_scaled)

                    # Calculate metrics
                    model_metrics = {}
                    for metric_name, metric_func in self.metrics.items():
                        model_metrics[metric_name] = metric_func(
                            y_test, y_pred)

                    results[name] = {
                        'model': model,
                        'metrics': model_metrics
                    }

                    self.logger.info(
                        self.class_name,
                        function_name,
                        f"Model {name} - RMSE: {model_metrics['RMSE']:.4f}, MAE: {model_metrics['MAE']:.4f}, R2: {model_metrics['R2']:.4f}"
                    )

                except Exception as e:
                    self.logger.error(
                        self.class_name,
                        function_name,
                        f"Error training {name} model: {e}"
                    )

            # Select best model based on RMSE
            best_model_name = min(
                results, key=lambda k: results[k]['metrics']['RMSE'])
            best_result = results[best_model_name]

            self.logger.info(
                self.class_name,
                function_name,
                f"Best model: {best_model_name} with RMSE: {best_result['metrics']['RMSE']:.4f}"
            )

            return best_result['model'], scaler, best_result['metrics'], best_model_name

        except Exception as e:
            self.logger.error(
                self.class_name,
                function_name,
                f"Error training models: {e}"
            )
            return None, None, None, None

    def save_model_artifacts(self, model, scaler, metrics, model_name, feature_names):
        """
        Save model, scaler, metrics and feature importance.

        Args:
            model: Trained model
            scaler: Fitted scaler
            metrics: Model evaluation metrics
            model_name: Name of the model
            feature_names: List of feature names

        Returns:
            True if saved successfully, False otherwise
        """
        function_name = "save_model_artifacts"

        try:
            # Save model
            with open(self.model_path, 'wb') as f:
                pickle.dump(model, f)

            # Save scaler
            with open(self.scaler_path, 'wb') as f:
                pickle.dump(scaler, f)

            # Save metrics
            metrics_df = pd.DataFrame([metrics])
            metrics_df['model'] = model_name
            metrics_df['timestamp'] = datetime.now().strftime(
                '%Y-%m-%d %H:%M:%S')
            metrics_df.to_csv(self.metrics_path, index=False)

            # Save feature importance if available
            if hasattr(model, 'feature_importances_'):
                feature_importance = pd.DataFrame({
                    'feature': feature_names,
                    'importance': model.feature_importances_
                })
                feature_importance = feature_importance.sort_values(
                    'importance', ascending=False)
                feature_importance.to_csv(
                    self.feature_importance_path, index=False)

            self.logger.info(
                self.class_name,
                function_name,
                f"Saved model artifacts to {self.models_dir}"
            )

            return True

        except Exception as e:
            self.logger.error(
                self.class_name,
                function_name,
                f"Error saving model artifacts: {e}"
            )
            return False

    def generate_model_report(self, X_train, y_train, X_test, y_test, model, scaler, metrics, model_name, feature_names):
        """
        Generate a PDF report with model evaluation and visualizations.

        Args:
            X_train: Training features
            y_train: Training target
            X_test: Testing features
            y_test: Testing target
            model: Trained model
            scaler: Fitted scaler
            metrics: Model evaluation metrics
            model_name: Name of the model
            feature_names: List of feature names

        Returns:
            True if report was generated successfully, False otherwise
        """
        function_name = "generate_model_report"

        try:
            # Set Seaborn style
            sns.set(style="whitegrid")

            # Scale data
            X_test_scaled = scaler.transform(X_test)

            # Make predictions
            y_pred = model.predict(X_test_scaled)

            # Create PDF
            with PdfPages(self.model_report_path) as pdf:
                # Title page
                plt.figure(figsize=(11, 8.5))
                plt.axis('off')
                plt.text(0.5, 0.5, f"E-Mini S&P 500 ({self.indicator_symbol}) Model Report",
                         ha='center', va='center', fontsize=24, fontweight='bold')
                plt.text(0.5, 0.45, f"Model: {model_name}",
                         ha='center', va='center', fontsize=18)
                plt.text(0.5, 0.4, f"Generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
                         ha='center', va='center', fontsize=14)
                pdf.savefig()
                plt.close()

                # 1. Actual vs Predicted
                plt.figure(figsize=(11, 8.5))
                plt.scatter(y_test, y_pred, alpha=0.5)
                plt.plot([y_test.min(), y_test.max()], [
                         y_test.min(), y_test.max()], 'r--')
                plt.xlabel('Actual')
                plt.ylabel('Predicted')
                plt.title('Actual vs Predicted Values')
                plt.grid(True)
                pdf.savefig()
                plt.close()

                # 2. Prediction Error
                plt.figure(figsize=(11, 8.5))
                error = y_test - y_pred
                plt.hist(error, bins=50, alpha=0.75)
                plt.xlabel('Prediction Error')
                plt.ylabel('Frequency')
                plt.title('Histogram of Prediction Errors')
                plt.grid(True)
                pdf.savefig()
                plt.close()

                # 3. Error vs Predicted
                plt.figure(figsize=(11, 8.5))
                plt.scatter(y_pred, error, alpha=0.5)
                plt.axhline(y=0, color='r', linestyle='--')
                plt.xlabel('Predicted Value')
                plt.ylabel('Error')
                plt.title('Error vs Predicted Value')
                plt.grid(True)
                pdf.savefig()
                plt.close()

                # 4. Feature Importance (if available)
                if hasattr(model, 'feature_importances_'):
                    plt.figure(figsize=(11, 8.5))

                    # Get feature importance
                    importance = model.feature_importances_
                    indices = np.argsort(importance)[-20:]  # Top 20 features

                    plt.barh(range(len(indices)), importance[indices])
                    plt.yticks(range(len(indices)), [
                               feature_names[i] for i in indices])
                    plt.xlabel('Importance')
                    plt.title('Top 20 Feature Importance')
                    plt.tight_layout()
                    pdf.savefig()
                    plt.close()

                # 5. Time Series Plot of Actual vs Predicted
                plt.figure(figsize=(11, 8.5))

                # Create a DataFrame with actual and predicted values
                results_df = pd.DataFrame({
                    'Actual': y_test.values,
                    'Predicted': y_pred
                })

                # Plot
                plt.plot(results_df.index,
                         results_df['Actual'], label='Actual')
                plt.plot(results_df.index,
                         results_df['Predicted'], label='Predicted')
                plt.xlabel('Time')
                plt.ylabel('Price')
                plt.title('Actual vs Predicted Values Over Time')
                plt.legend()
                plt.grid(True)
                pdf.savefig()
                plt.close()

                # 6. Model Metrics
                plt.figure(figsize=(11, 8.5))
                plt.axis('off')

                plt.text(0.5, 0.95, "Model Evaluation Metrics",
                         ha='center', fontsize=16, fontweight='bold')

                y_pos = 0.85
                for key, value in metrics.items():
                    plt.text(0.3, y_pos, key, ha='right', fontsize=12)
                    plt.text(
                        0.35, y_pos, f": {value:.4f}", ha='left', fontsize=12)
                    y_pos -= 0.05

                # Add explanation of metrics
                plt.text(0.5, 0.65, "Metrics Explanation:",
                         ha='center', fontsize=14, fontweight='bold')

                explanations = {
                    'RMSE': "Root Mean Squared Error - Measures the average magnitude of errors in predictions. Lower is better.",
                    'MAE': "Mean Absolute Error - Average absolute difference between predicted and actual values. Lower is better.",
                    'R2': "R-squared - Proportion of variance in the dependent variable predictable from the independent variables. Higher is better (max 1.0)."
                }

                y_pos = 0.6
                for key, value in explanations.items():
                    plt.text(0.5, y_pos, f"{key}: {value}",
                             ha='left', fontsize=10)
                    y_pos -= 0.05

                # Add justification for chosen metric
                plt.text(0.5, 0.4, "Metric Selection Justification:",
                         ha='center', fontsize=14, fontweight='bold')
                plt.text(
                    0.5, 0.35, "RMSE was chosen as the primary metric for model selection because:", ha='left', fontsize=10)
                plt.text(
                    0.5, 0.3, "1. It penalizes large errors more heavily than MAE, which is important in financial forecasting", ha='left', fontsize=10)
                plt.text(
                    0.5, 0.25, "2. It's in the same units as the target variable, making it interpretable", ha='left', fontsize=10)
                plt.text(
                    0.5, 0.2, "3. It's widely used in time series forecasting and financial applications", ha='left', fontsize=10)

                pdf.savefig()
                plt.close()

            self.logger.info(
                self.class_name,
                function_name,
                f"Model report generated successfully: {self.model_report_path}"
            )

            return True

        except Exception as e:
            self.logger.error(
                self.class_name,
                function_name,
                f"Error generating model report: {e}"
            )
            return False

    def entrenar(self):
        """
        Train and save a predictive model.

        Returns:
            True if training was successful, False otherwise
        """
        function_name = "entrenar"

        self.logger.info(
            self.class_name,
            function_name,
            "Starting model training process."
        )

        try:
            # 1. Load data
            df = self.load_data()

            if df.empty:
                self.logger.error(
                    self.class_name,
                    function_name,
                    "Failed to load data. Training process aborted."
                )
                return False

            # 2. Prepare features and target
            X, y, feature_names = self.prepare_features_target(df)

            if X is None or y is None:
                self.logger.error(
                    self.class_name,
                    function_name,
                    "Failed to prepare features and target. Training process aborted."
                )
                return False

            # 3. Split data
            X_train, X_test, y_train, y_test = self.train_test_split_time_series(
                X, y)

            if X_train is None:
                self.logger.error(
                    self.class_name,
                    function_name,
                    "Failed to split data. Training process aborted."
                )
                return False

            # 4. Train models
            model, scaler, metrics, model_name = self.train_models(
                X_train, y_train, X_test, y_test)

            if model is None:
                self.logger.error(
                    self.class_name,
                    function_name,
                    "Failed to train models. Training process aborted."
                )
                return False

            # 5. Save model artifacts
            save_success = self.save_model_artifacts(
                model, scaler, metrics, model_name, feature_names)

            if not save_success:
                self.logger.error(
                    self.class_name,
                    function_name,
                    "Failed to save model artifacts."
                )

            # 6. Generate model report
            report_success = self.generate_model_report(
                X_train, y_train, X_test, y_test,
                model, scaler, metrics, model_name, feature_names
            )

            if not report_success:
                self.logger.error(
                    self.class_name,
                    function_name,
                    "Failed to generate model report."
                )

            self.logger.info(
                self.class_name,
                function_name,
                f"Model training process completed successfully. Best model: {model_name}"
            )

            return True

        except Exception as e:
            self.logger.error(
                self.class_name,
                function_name,
                f"Error in training process: {e}"
            )
            return False

    def load_model(self):
        """
        Load trained model and scaler.

        Returns:
            Tuple of (model, scaler)
        """
        function_name = "load_model"

        try:
            # Check if model and scaler files exist
            if not os.path.exists(self.model_path) or not os.path.exists(self.scaler_path):
                self.logger.error(
                    self.class_name,
                    function_name,
                    f"Model file {self.model_path} or scaler file {self.scaler_path} not found."
                )
                return None, None

            # Load model
            with open(self.model_path, 'rb') as f:
                model = pickle.load(f)

            # Load scaler
            with open(self.scaler_path, 'rb') as f:
                scaler = pickle.load(f)

            self.logger.info(
                self.class_name,
                function_name,
                f"Loaded model from {self.model_path} and scaler from {self.scaler_path}"
            )

            return model, scaler

        except Exception as e:
            self.logger.error(
                self.class_name,
                function_name,
                f"Error loading model: {e}"
            )
            return None, None

    def predecir(self):
        """
        Make predictions using the trained model.

        Returns:
            DataFrame with predictions
        """
        function_name = "predecir"

        self.logger.info(
            self.class_name,
            function_name,
            "Starting prediction process."
        )

        try:
            # 1. Load model and scaler
            model, scaler = self.load_model()

            if model is None or scaler is None:
                self.logger.error(
                    self.class_name,
                    function_name,
                    "Failed to load model or scaler. Prediction process aborted."
                )
                return pd.DataFrame()

            # 2. Load data
            df = self.load_data()

            if df.empty:
                self.logger.error(
                    self.class_name,
                    function_name,
                    "Failed to load data. Prediction process aborted."
                )
                return pd.DataFrame()

            # 3. Prepare features
            X, _, feature_names = self.prepare_features_target(df)

            if X is None:
                self.logger.error(
                    self.class_name,
                    function_name,
                    "Failed to prepare features. Prediction process aborted."
                )
                return pd.DataFrame()

            # 4. Get the most recent data points
            latest_data = X.iloc[-self.prediction_horizon:]

            # 5. Scale features
            latest_data_scaled = scaler.transform(latest_data)

            # 6. Make predictions
            predictions = model.predict(latest_data_scaled)

            # 7. Create prediction DataFrame
            dates = pd.date_range(
                start=df['date'].iloc[-1] + pd.Timedelta(days=1),
                periods=self.prediction_horizon
            )

            pred_df = pd.DataFrame({
                'date': dates,
                'predicted_close': predictions
            })

            self.logger.info(
                self.class_name,
                function_name,
                f"Made {len(predictions)} predictions for future dates."
            )

            return pred_df

        except Exception as e:
            self.logger.error(
                self.class_name,
                function_name,
                f"Error in prediction process: {e}"
            )
            return pd.DataFrame()


# Example usage
if __name__ == "__main__":
    from logger import Logger

    logger = Logger(logger_name="ES_MINI_Modeller_Test")
    modeller = Modeller(logger)

    # Train model
    training_success = modeller.entrenar()

    if training_success:
        print("Model training successful.")

        # Make predictions
        predictions = modeller.predecir()

        if not predictions.empty:
            print("Predictions:")
            print(predictions)
        else:
            print("Failed to make predictions.")
    else:
        print("Model training failed.")
