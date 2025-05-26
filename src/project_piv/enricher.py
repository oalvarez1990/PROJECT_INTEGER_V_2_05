# enricher.py
import pandas as pd
import numpy as np
import os
import sqlite3
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.backends.backend_pdf import PdfPages

class Enricher:
    """
    Class for enriching financial data with technical indicators and KPIs.
    """

    def __init__(self, logger, data_dir=None, indicator_symbol="ES=F"):
        """
        Initialize the Enricher class.

        Args:
            logger: Logger instance for logging
            data_dir: Directory where data is stored
            indicator_symbol: Symbol of the financial indicator
        """
        self.logger = logger
        self.class_name = self.__class__.__name__
        self.indicator_symbol = indicator_symbol

        # Set data directory
        if data_dir is None:
            self.data_dir = os.path.join(
                os.path.dirname(os.path.abspath(__file__)),
                'static',
                'data'
            )
        else:
            self.data_dir = data_dir

        # Create utils directory if it doesn't exist
        self.utils_dir = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            'static',
            'utils'
        )
        os.makedirs(self.utils_dir, exist_ok=True)

        # Define file paths
        self.base_csv_path = os.path.join(
            self.data_dir,
            f'historical_{indicator_symbol.replace("=F", "")}.csv'
        )
        self.enriched_csv_path = os.path.join(
            self.data_dir,
            f'enriched_{indicator_symbol.replace("=F", "")}.csv'
        )
        self.db_path = os.path.join(
            self.data_dir,
            f'enriched_{indicator_symbol.replace("=F", "")}.db'
        )
        self.kpi_pdf_path = os.path.join(
            self.utils_dir,
            f'kpi_report_{indicator_symbol.replace("=F", "")}.pdf'
        )

        self.logger.info(
            self.class_name,
            "__init__",
            f"Enricher initialized for {indicator_symbol}. Data will be processed from {self.base_csv_path}"
        )

    def load_data(self):
        """
        Load historical data from CSV file.

        Returns:
            DataFrame with historical data
        """
        function_name = "load_data"

        try:
            if not os.path.exists(self.base_csv_path):
                self.logger.error(
                    self.class_name,
                    function_name,
                    f"Data file not found: {self.base_csv_path}"
                )
                return pd.DataFrame()

            df = pd.read_csv(self.base_csv_path)

            # Convert date column to datetime
            if 'date' in df.columns:
                df['date'] = pd.to_datetime(df['date'])

            # Sort by date
            df = df.sort_values('date')

            self.logger.info(
                self.class_name,
                function_name,
                f"Loaded {len(df)} rows from {self.base_csv_path}"
            )

            return df

        except Exception as e:
            self.logger.error(
                self.class_name,
                function_name,
                f"Error loading data: {e}"
            )
            return pd.DataFrame()

    def clean_data(self, df):
        """
        Clean and preprocess the data.

        Args:
            df: DataFrame with historical data

        Returns:
            Cleaned DataFrame
        """
        function_name = "clean_data"

        if df.empty:
            self.logger.warning(
                self.class_name,
                function_name,
                "DataFrame is empty, skipping data cleaning."
            )
            return df

        try:
            # Make a copy to avoid modifying the original
            df_clean = df.copy()

            # Check for missing values
            missing_values = df_clean.isnull().sum()

            if missing_values.sum() > 0:
                self.logger.warning(
                    self.class_name,
                    function_name,
                    f"Found {missing_values.sum()} missing values: {missing_values[missing_values > 0].to_dict()}"
                )

                # Fill missing values in OHLC with forward fill
                for col in ['open', 'high', 'low', 'close']:
                    if col in df_clean.columns:
                        df_clean[col] = df_clean[col].fillna(method='ffill')

                # Fill missing volume with 0
                if 'volume' in df_clean.columns:
                    df_clean['volume'] = df_clean['volume'].fillna(0)

            # Check for duplicated dates
            duplicates = df_clean.duplicated(subset=['date'], keep='first')
            if duplicates.sum() > 0:
                self.logger.warning(
                    self.class_name,
                    function_name,
                    f"Found {duplicates.sum()} duplicate dates. Keeping first occurrence."
                )
                df_clean = df_clean.drop_duplicates(subset=['date'], keep='first')

            # Check for outliers in price data (using IQR method)
            for col in ['open', 'high', 'low', 'close']:
                if col in df_clean.columns:
                    Q1 = df_clean[col].quantile(0.25)
                    Q3 = df_clean[col].quantile(0.75)
                    IQR = Q3 - Q1
                    lower_bound = Q1 - 3 * IQR
                    upper_bound = Q3 + 3 * IQR

                    outliers = df_clean[(df_clean[col] < lower_bound) | (df_clean[col] > upper_bound)]

                    if len(outliers) > 0:
                        self.logger.warning(
                            self.class_name,
                            function_name,
                            f"Found {len(outliers)} outliers in {col} column."
                        )

                        # Instead of removing outliers, we'll cap them
                        df_clean[col] = df_clean[col].clip(lower=lower_bound, upper=upper_bound)

            # Ensure data is sorted by date
            df_clean = df_clean.sort_values('date')

            self.logger.info(
                self.class_name,
                function_name,
                f"Data cleaning completed. Rows after cleaning: {len(df_clean)}"
            )

            return df_clean

        except Exception as e:
            self.logger.error(
                self.class_name,
                function_name,
                f"Error cleaning data: {e}"
            )
            return df

    def calculate_kpis(self, df):
        """
        Calculate Key Performance Indicators (KPIs) for the financial data.

        Args:
            df: DataFrame with historical data

        Returns:
            DataFrame with added KPIs
        """
        function_name = "calculate_kpis"

        if df.empty:
            self.logger.warning(
                self.class_name,
                function_name,
                "DataFrame is empty, skipping KPI calculation."
            )
            return df

        try:
            # Make a copy to avoid modifying the original
            df_kpi = df.copy()

            # 1. Daily Returns
            df_kpi['daily_return'] = df_kpi['close'].pct_change() * 100

            # 2. Moving Averages (5, 10, 20, 50, 200 days)
            for window in [5, 10, 20, 50, 200]:
                df_kpi[f'sma_{window}'] = df_kpi['close'].rolling(window=window).mean()

            # 3. Exponential Moving Averages (5, 10, 20, 50 days)
            for window in [5, 10, 20, 50]:
                df_kpi[f'ema_{window}'] = df_kpi['close'].ewm(span=window, adjust=False).mean()

            # 4. Volatility (5, 10, 20, 50 days)
            for window in [5, 10, 20, 50]:
                df_kpi[f'volatility_{window}'] = df_kpi['daily_return'].rolling(window=window).std() * np.sqrt(window)

            # 5. Rate of Change (1, 5, 10, 20 days)
            for window in [1, 5, 10, 20]:
                df_kpi[f'roc_{window}'] = df_kpi['close'].pct_change(periods=window) * 100

            # 6. Relative Strength Index (RSI) - 14 days
            delta = df_kpi['close'].diff()
            gain = delta.where(delta > 0, 0)
            loss = -delta.where(delta < 0, 0)

            avg_gain = gain.rolling(window=14).mean()
            avg_loss = loss.rolling(window=14).mean()

            rs = avg_gain / avg_loss
            df_kpi['rsi_14'] = 100 - (100 / (1 + rs))

            # 7. Bollinger Bands (20 days, 2 standard deviations)
            df_kpi['bollinger_mid'] = df_kpi['close'].rolling(window=20).mean()
            df_kpi['bollinger_std'] = df_kpi['close'].rolling(window=20).std()
            df_kpi['bollinger_upper'] = df_kpi['bollinger_mid'] + 2 * df_kpi['bollinger_std']
            df_kpi['bollinger_lower'] = df_kpi['bollinger_mid'] - 2 * df_kpi['bollinger_std']

            # 8. MACD (Moving Average Convergence Divergence)
            df_kpi['ema_12'] = df_kpi['close'].ewm(span=12, adjust=False).mean()
            df_kpi['ema_26'] = df_kpi['close'].ewm(span=26, adjust=False).mean()
            df_kpi['macd'] = df_kpi['ema_12'] - df_kpi['ema_26']
            df_kpi['macd_signal'] = df_kpi['macd'].ewm(span=9, adjust=False).mean()
            df_kpi['macd_histogram'] = df_kpi['macd'] - df_kpi['macd_signal']

            # 9. Average True Range (ATR) - 14 days
            high_low = df_kpi['high'] - df_kpi['low']
            high_close = (df_kpi['high'] - df_kpi['close'].shift()).abs()
            low_close = (df_kpi['low'] - df_kpi['close'].shift()).abs()

            true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
            df_kpi['atr_14'] = true_range.rolling(window=14).mean()

            # 10. Cumulative Return
            df_kpi['cum_return'] = (1 + df_kpi['daily_return'] / 100).cumprod() - 1

            # 11. Drawdown
            rolling_max = df_kpi['close'].cummax()
            df_kpi['drawdown'] = (df_kpi['close'] / rolling_max - 1) * 100

            # 12. Volume Moving Average (5, 20 days)
            for window in [5, 20]:
                df_kpi[f'volume_sma_{window}'] = df_kpi['volume'].rolling(window=window).mean()

            # 13. Price to Volume Ratio
            df_kpi['price_volume_ratio'] = df_kpi['close'] / df_kpi['volume']

            # 14. Temporal features
            df_kpi['day_of_week'] = df_kpi['date'].dt.dayofweek
            df_kpi['month'] = df_kpi['date'].dt.month
            df_kpi['quarter'] = df_kpi['date'].dt.quarter
            df_kpi['year'] = df_kpi['date'].dt.year
            df_kpi['is_month_end'] = df_kpi['date'].dt.is_month_end.astype(int)
            df_kpi['is_quarter_end'] = df_kpi['date'].dt.is_quarter_end.astype(int)

            self.logger.info(
                self.class_name,
                function_name,
                f"KPI calculation completed. Added {len(df_kpi.columns) - len(df.columns)} new columns."
            )

            return df_kpi

        except Exception as e:
            self.logger.error(
                self.class_name,
                function_name,
                f"Error calculating KPIs: {e}"
            )
            return df

    def generate_kpi_report(self, df):
        """
        Generate a PDF report with KPI visualizations.

        Args:
            df: DataFrame with KPIs

        Returns:
            True if report was generated successfully, False otherwise
        """
        function_name = "generate_kpi_report"

        if df.empty:
            self.logger.warning(
                self.class_name,
                function_name,
                "DataFrame is empty, skipping KPI report generation."
            )
            return False

        try:
            # Set Seaborn style
            sns.set(style="whitegrid")

            # Create PDF
            with PdfPages(self.kpi_pdf_path) as pdf:
                # Title page
                plt.figure(figsize=(11, 8.5))
                plt.axis('off')
                plt.text(0.5, 0.5, f"E-Mini S&P 500 ({self.indicator_symbol}) KPI Report",
                         ha='center', va='center', fontsize=24, fontweight='bold')
                plt.text(0.5, 0.45, f"Generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
                         ha='center', va='center', fontsize=14)
                pdf.savefig()
                plt.close()

                # 1. Price and Moving Averages
                plt.figure(figsize=(11, 8.5))
                plt.subplot(2, 1, 1)
                plt.plot(df['date'], df['close'], label='Close Price')
                plt.plot(df['date'], df['sma_50'], label='SMA 50')
                plt.plot(df['date'], df['sma_200'], label='SMA 200')
                plt.title('Price and Moving Averages')
                plt.legend()
                plt.grid(True)

                # 2. Daily Returns
                plt.subplot(2, 1, 2)
                plt.plot(df['date'], df['daily_return'])
                plt.title('Daily Returns (%)')
                plt.grid(True)
                pdf.savefig()
                plt.close()

                # 3. Volatility
                plt.figure(figsize=(11, 8.5))
                plt.subplot(2, 1, 1)
                plt.plot(df['date'], df['volatility_20'], label='20-day Volatility')
                plt.plot(df['date'], df['volatility_50'], label='50-day Volatility')
                plt.title('Volatility')
                plt.legend()
                plt.grid(True)

                # 4. Drawdown
                plt.subplot(2, 1, 2)
                plt.plot(df['date'], df['drawdown'])
                plt.title('Drawdown (%)')
                plt.grid(True)
                pdf.savefig()
                plt.close()

                # 5. RSI and MACD
                plt.figure(figsize=(11, 8.5))
                plt.subplot(2, 1, 1)
                plt.plot(df['date'], df['rsi_14'])
                plt.axhline(y=70, color='r', linestyle='--')
                plt.axhline(y=30, color='g', linestyle='--')
                plt.title('Relative Strength Index (RSI)')
                plt.grid(True)

                plt.subplot(2, 1, 2)
                plt.plot(df['date'], df['macd'], label='MACD')
                plt.plot(df['date'], df['macd_signal'], label='Signal Line')
                plt.bar(df['date'], df['macd_histogram'], alpha=0.3, label='Histogram')
                plt.title('MACD')
                plt.legend()
                plt.grid(True)
                pdf.savefig()
                plt.close()

                # 6. Bollinger Bands
                plt.figure(figsize=(11, 8.5))
                plt.plot(df['date'], df['close'], label='Close Price')
                plt.plot(df['date'], df['bollinger_mid'], label='20-day SMA')
                plt.plot(df['date'], df['bollinger_upper'], label='Upper Band')
                plt.plot(df['date'], df['bollinger_lower'], label='Lower Band')
                plt.title('Bollinger Bands')
                plt.legend()
                plt.grid(True)
                pdf.savefig()
                plt.close()

                # 7. Volume Analysis
                plt.figure(figsize=(11, 8.5))
                plt.subplot(2, 1, 1)
                plt.bar(df['date'], df['volume'], alpha=0.5)
                plt.plot(df['date'], df['volume_sma_20'], color='r', label='20-day SMA')
                plt.title('Volume')
                plt.legend()
                plt.grid(True)

                plt.subplot(2, 1, 2)
                plt.plot(df['date'], df['price_volume_ratio'])
                plt.title('Price to Volume Ratio')
                plt.grid(True)
                pdf.savefig()
                plt.close()

                # 8. Cumulative Return
                plt.figure(figsize=(11, 8.5))
                plt.plot(df['date'], df['cum_return'] * 100)
                plt.title('Cumulative Return (%)')
                plt.grid(True)
                pdf.savefig()
                plt.close()

                # 9. Rate of Change
                plt.figure(figsize=(11, 8.5))
                plt.plot(df['date'], df['roc_5'], label='5-day ROC')
                plt.plot(df['date'], df['roc_20'], label='20-day ROC')
                plt.title('Rate of Change (%)')
                plt.legend()
                plt.grid(True)
                pdf.savefig()
                plt.close()

                # 10. Summary Statistics
                plt.figure(figsize=(11, 8.5))
                plt.axis('off')

                # Calculate summary statistics
                recent_df = df.iloc[-252:]  # Last year of data (assuming 252 trading days)

                stats = {
                    'Current Price': df['close'].iloc[-1],
                    'Daily Return (%)': df['daily_return'].iloc[-1],
                    'Cumulative Return (%)': df['cum_return'].iloc[-1] * 100,
                    'Volatility (20-day)': df['volatility_20'].iloc[-1],
                    'RSI (14-day)': df['rsi_14'].iloc[-1],
                    'Average Daily Return (%)': recent_df['daily_return'].mean(),
                    'Std Dev of Daily Return (%)': recent_df['daily_return'].std(),
                    'Max Drawdown (%)': recent_df['drawdown'].min(),
                    'Sharpe Ratio (Rf=0)': recent_df['daily_return'].mean() / recent_df['daily_return'].std() * np.sqrt(252),
                    'Positive Days (%)': (recent_df['daily_return'] > 0).mean() * 100,
                    'Negative Days (%)': (recent_df['daily_return'] < 0).mean() * 100
                }

                # Create a table-like display
                plt.text(0.5, 0.95, "Summary Statistics (Last Year)", ha='center', fontsize=16, fontweight='bold')

                y_pos = 0.85
                for key, value in stats.items():
                    plt.text(0.3, y_pos, key, ha='right', fontsize=12)
                    plt.text(0.35, y_pos, f": {value:.4f}", ha='left', fontsize=12)
                    y_pos -= 0.05

                pdf.savefig()
                plt.close()

            self.logger.info(
                self.class_name,
                function_name,
                f"KPI report generated successfully: {self.kpi_pdf_path}"
            )

            return True

        except Exception as e:
            self.logger.error(
                self.class_name,
                function_name,
                f"Error generating KPI report: {e}"
            )
            return False

    def save_to_csv(self, df):
        """
        Save enriched data to CSV file.

        Args:
            df: DataFrame with enriched data

        Returns:
            True if saved successfully, False otherwise
        """
        function_name = "save_to_csv"

        if df.empty:
            self.logger.warning(
                self.class_name,
                function_name,
                "DataFrame is empty, nothing to save."
            )
            return False

        try:
            # Save to CSV
            df.to_csv(self.enriched_csv_path, index=False)

            self.logger.info(
                self.class_name,
                function_name,
                f"Saved {len(df)} rows to {self.enriched_csv_path}"
            )

            return True

        except Exception as e:
            self.logger.error(
                self.class_name,
                function_name,
                f"Error saving to CSV: {e}"
            )
            return False

    def save_to_sqlite(self, df):
        """
        Save enriched data to SQLite database.

        Args:
            df: DataFrame with enriched data

        Returns:
            True if saved successfully, False otherwise
        """
        function_name = "save_to_sqlite"

        if df.empty:
            self.logger.warning(
                self.class_name,
                function_name,
                "DataFrame is empty, nothing to save."
            )
            return False

        try:
            # Create connection to SQLite database
            conn = sqlite3.connect(self.db_path)

            # Save to SQLite
            df.to_sql('enriched_data', conn, if_exists='replace', index=False)

            # Close connection
            conn.close()

            self.logger.info(
                self.class_name,
                function_name,
                f"Saved {len(df)} rows to SQLite database: {self.db_path}"
            )

            return True

        except Exception as e:
            self.logger.error(
                self.class_name,
                function_name,
                f"Error saving to SQLite: {e}"
            )
            return False

    def run(self):
        """
        Run the complete enrichment process.

        Returns:
            DataFrame with enriched data
        """
        function_name = "run"

        self.logger.info(
            self.class_name,
            function_name,
            "Starting enrichment process."
        )

        # 1. Load data
        df = self.load_data()

        if df.empty:
            self.logger.error(
                self.class_name,
                function_name,
                "Failed to load data. Enrichment process aborted."
            )
            return df

        # 2. Clean data
        df_clean = self.clean_data(df)

        # 3. Calculate KPIs
        df_enriched = self.calculate_kpis(df_clean)

        # 4. Generate KPI report
        report_success = self.generate_kpi_report(df_enriched)

        # 5. Save enriched data
        csv_success = self.save_to_csv(df_enriched)
        sqlite_success = self.save_to_sqlite(df_enriched)

        self.logger.info(
            self.class_name,
            function_name,
            f"Enrichment process completed. Report: {'Success' if report_success else 'Failed'}, "
            f"CSV: {'Success' if csv_success else 'Failed'}, "
            f"SQLite: {'Success' if sqlite_success else 'Failed'}"
        )

        return df_enriched

# Example usage
if __name__ == "__main__":
    from logger import Logger

    logger = Logger(logger_name="ES_MINI_Enricher_Test")
    enricher = Enricher(logger)
    enriched_data = enricher.run()

    if not enriched_data.empty:
        print(f"Enrichment successful. Shape: {enriched_data.shape}")
        print("Columns:", enriched_data.columns.tolist())
    else:
        print("Enrichment failed.")