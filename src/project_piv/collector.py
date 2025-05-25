import pandas as pd
import os
import sqlite3
import yfinance as yf
from datetime import datetime
from logger import Logger

class Collector:
    def __init__(self, logger: Logger, indicator_symbol: str = "ES=F", data_dir: str = None):
        """
        Initialize the collector for downloading financial data.

        Args:
            logger: Instance of the Logger class
            indicator_symbol: Symbol of the indicator on Yahoo Finance
            data_dir: Directory where data files will be stored
        """
        # Set up class attributes
        self.logger = logger
        self.class_name = self.__class__.__name__
        self.indicator_symbol = indicator_symbol

        # Set up data directory
        if data_dir is None:
            # Default to static/data in the same directory as this file
            self.data_dir = os.path.join(
                os.path.dirname(os.path.abspath(__file__)),
                'static',
                'data'
            )
        else:
            self.data_dir = data_dir

        # Ensure data directory exists
        os.makedirs(self.data_dir, exist_ok=True)

        # Define file paths
        self.csv_path = os.path.join(
            self.data_dir,
            f'historical_{indicator_symbol.replace("=F", "")}.csv'
        )
        self.db_path = os.path.join(
            self.data_dir,
            f'historical_{indicator_symbol.replace("=F", "")}.db'
        )
        self.db_table_name = 'historical_data'

        self.logger.info(
            self.class_name,
            "__init__",
            f"Collector initialized for {indicator_symbol}. Data will be saved to {self.data_dir}"
        )

    def fetch_data(self) -> pd.DataFrame:
        """
        Fetch historical data from Yahoo Finance using yfinance.

        Returns:
            DataFrame containing the historical data
        """
        function_name = "fetch_data"
        self.logger.info(self.class_name, function_name, f"Attempting to fetch data for: {self.indicator_symbol}")

        try:
            # Use yfinance to download data
            # Start date from Dec 1, 2003
            start_date = "2003-12-01"
            end_date = datetime.now().strftime("%Y-%m-%d")

            self.logger.info(
                self.class_name,
                function_name,
                f"Downloading data from {start_date} to {end_date}"
            )

            # Download the data
            ticker = yf.Ticker(self.indicator_symbol)
            df = ticker.history(start=start_date, end=end_date, interval="1d")

            # Reset index to make Date a column
            df = df.reset_index()

            # Rename columns to match our expected format
            df = df.rename(columns={
                'Date': 'date',
                'Open': 'open',
                'High': 'high',
                'Low': 'low',
                'Close': 'close',
                'Volume': 'volume',
                'Dividends': 'dividends',
                'Stock Splits': 'stock_splits'
            })

            # Drop unnecessary columns
            if 'dividends' in df.columns:
                df = df.drop('dividends', axis=1)
            if 'stock_splits' in df.columns:
                df = df.drop('stock_splits', axis=1)

            # Rename 'Adj Close' to 'adj_close' if it exists
            if 'Adj Close' in df.columns:
                df = df.rename(columns={'Adj Close': 'adj_close'})

            self.logger.info(
                self.class_name,
                function_name,
                f"Successfully downloaded {len(df)} rows of data."
            )

            return df

        except Exception as e:
            self.logger.error(
                self.class_name,
                function_name,
                f"Error fetching data: {e}"
            )
            return pd.DataFrame()

    def clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Clean and standardize the data.

        Args:
            df: DataFrame to clean

        Returns:
            Cleaned DataFrame
        """
        function_name = "clean_data"

        if df.empty:
            self.logger.info(
                self.class_name,
                function_name,
                "DataFrame is empty, skipping cleaning."
            )
            return df

        self.logger.info(
            self.class_name,
            function_name,
            f"Starting data cleaning. Initial shape: {df.shape}"
        )

        try:
            # Ensure date column is properly formatted
            if 'date' in df.columns:
                df['date'] = pd.to_datetime(df['date']).dt.strftime('%Y-%m-%d')

            # Convert numeric columns to float
            numeric_cols = ['open', 'high', 'low', 'close', 'adj_close', 'volume']
            for col in numeric_cols:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors='coerce')

            # Drop rows with missing essential data
            essential_cols = ['date', 'open', 'high', 'low', 'close']
            essential_cols = [col for col in essential_cols if col in df.columns]
            df = df.dropna(subset=essential_cols)

            # Sort by date
            if 'date' in df.columns:
                df = df.sort_values('date').reset_index(drop=True)

            self.logger.info(
                self.class_name,
                function_name,
                f"Data cleaning finished. Final shape: {df.shape}"
            )

            return df

        except Exception as e:
            self.logger.error(
                self.class_name,
                function_name,
                f"Error during data cleaning: {e}"
            )
            return pd.DataFrame()

    def save_to_csv(self, df: pd.DataFrame) -> bool:
        """
        Save data to CSV file, appending new data to existing file.

        Args:
            df: DataFrame to save

        Returns:
            True if successful, False otherwise
        """
        function_name = "save_to_csv"

        if df.empty:
            self.logger.info(
                self.class_name,
                function_name,
                "DataFrame is empty, nothing to save to CSV."
            )
            return False

        try:
            # Check if file exists and has content
            file_exists = os.path.exists(self.csv_path) and os.path.getsize(self.csv_path) > 0

            if file_exists:
                # Read existing data
                existing_df = pd.read_csv(self.csv_path)

                if 'date' in existing_df.columns and 'date' in df.columns:
                    # Convert dates to same format for comparison
                    existing_df['date'] = pd.to_datetime(existing_df['date']).dt.strftime('%Y-%m-%d')
                    df['date'] = pd.to_datetime(df['date']).dt.strftime('%Y-%m-%d')

                    # Find new data (not in existing file)
                    existing_dates = set(existing_df['date'])
                    df_to_append = df[~df['date'].isin(existing_dates)]

                    if not df_to_append.empty:
                        # Append only new data
                        df_to_append.to_csv(self.csv_path, mode='a', header=False, index=False)
                        self.logger.info(
                            self.class_name,
                            function_name,
                            f"Appended {len(df_to_append)} new rows to {self.csv_path}"
                        )
                    else:
                        self.logger.info(
                            self.class_name,
                            function_name,
                            "No new data to append to CSV."
                        )
                else:
                    # If date column missing, just append all
                    df.to_csv(self.csv_path, mode='a', header=False, index=False)
                    self.logger.info(
                        self.class_name,
                        function_name,
                        f"Appended {len(df)} rows to {self.csv_path} (date column not found for comparison)"
                    )
            else:
                # Create new file with header
                df.to_csv(self.csv_path, index=False)
                self.logger.info(
                    self.class_name,
                    function_name,
                    f"Created new CSV file with {len(df)} rows at {self.csv_path}"
                )

            return True

        except Exception as e:
            self.logger.error(
                self.class_name,
                function_name,
                f"Error saving to CSV: {e}"
            )
            return False

    def save_to_sqlite(self, df: pd.DataFrame) -> bool:
        """
        Save data to SQLite database, updating with new data.

        Args:
            df: DataFrame to save

        Returns:
            True if successful, False otherwise
        """
        function_name = "save_to_sqlite"

        if df.empty:
            self.logger.info(
                self.class_name,
                function_name,
                "DataFrame is empty, nothing to save to SQLite."
            )
            return False

        try:
            # Connect to SQLite database
            conn = sqlite3.connect(self.db_path)

            # Create table if it doesn't exist
            conn.execute(f'''
                CREATE TABLE IF NOT EXISTS {self.db_table_name} (
                    date TEXT PRIMARY KEY,
                    open REAL,
                    high REAL,
                    low REAL,
                    close REAL,
                    adj_close REAL,
                    volume REAL
                )
            ''')

            # Get existing dates to avoid duplicates
            existing_dates = pd.read_sql(f"SELECT date FROM {self.db_table_name}", conn)

            if not existing_dates.empty:
                # Find new data (not in existing database)
                existing_date_set = set(existing_dates['date'])
                df_to_insert = df[~df['date'].isin(existing_date_set)]
            else:
                df_to_insert = df

            if not df_to_insert.empty:
                # Insert new data
                df_to_insert.to_sql(
                    self.db_table_name,
                    conn,
                    if_exists='append',
                    index=False
                )

                self.logger.info(
                    self.class_name,
                    function_name,
                    f"Added {len(df_to_insert)} new rows to SQLite database {self.db_path}"
                )
            else:
                self.logger.info(
                    self.class_name,
                    function_name,
                    "No new data to add to SQLite database."
                )

            conn.close()
            return True

        except sqlite3.Error as e:
            self.logger.error(
                self.class_name,
                function_name,
                f"SQLite error: {e}"
            )
            return False

        except Exception as e:
            self.logger.error(
                self.class_name,
                function_name,
                f"Error saving to SQLite: {e}"
            )
            return False

    def run(self) -> bool:
        """
        Execute the full data collection process.

        Returns:
            True if successful, False otherwise
        """
        function_name = "run"
        self.logger.info(self.class_name, function_name, "Collection process started.")

        # Step 1: Fetch data
        raw_df = self.fetch_data()

        if raw_df.empty:
            self.logger.warning(
                self.class_name,
                function_name,
                "Fetched data was empty, no further processing."
            )
            return False

        # Step 2: Clean data
        cleaned_df = self.clean_data(raw_df)

        if cleaned_df.empty:
            self.logger.warning(
                self.class_name,
                function_name,
                "Data became empty after cleaning, nothing was saved."
            )
            return False

        # Step 3: Save data to CSV and SQLite
        csv_success = self.save_to_csv(cleaned_df)
        sqlite_success = self.save_to_sqlite(cleaned_df)

        self.logger.info(
            self.class_name,
            function_name,
            f"Collection process finished. CSV: {'Success' if csv_success else 'Failed'}, "
            f"SQLite: {'Success' if sqlite_success else 'Failed'}"
        )

        return csv_success and sqlite_success