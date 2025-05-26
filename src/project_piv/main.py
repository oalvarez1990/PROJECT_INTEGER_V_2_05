# main.py
import os
import pandas as pd
from logger import Logger
from collector import Collector
from datetime import datetime


def main():
    """
    Main function to run the data collection and enrichment process.
    """
    # Initialize the Logger
    logger_instance = Logger(
        logger_name="ES_MINI_Main_App",
        log_file_prefix="app_es_mini_data"
    )

    class_name_main = "MainApp"
    function_name_main = "main"

    # Define data directory
    data_dir = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        'static',
        'data'
    )
    os.makedirs(data_dir, exist_ok=True)

    logger_instance.info(
        class_name_main,
        function_name_main,
        f"Application started. Data directory: {data_dir}"
    )

    try:
        # Initialize and run the Collector
        collector_instance = Collector(
            logger=logger_instance,
            indicator_symbol="ES=F",
            data_dir=data_dir
        )

        logger_instance.info(
            class_name_main,
            function_name_main,
            f"Collector initialized for ES=F. Data will be saved in {data_dir}"
        )

        # Run the collection process
        collection_success = collector_instance.run()

        if not collection_success:
            logger_instance.warning(
                class_name_main,
                function_name_main,
                "Data collection process completed with warnings or errors."
            )

        # Check if data files exist
        csv_path = os.path.join(data_dir, 'historical_ES.csv')

        if not os.path.exists(csv_path):
            logger_instance.error(
                class_name_main,
                function_name_main,
                f"Data file {csv_path} not found. Cannot proceed with analysis."
            )
            print(f"Error: Data file {csv_path} not found.")
            return

        # Read the collected data for verification
        try:
            df = pd.read_csv(csv_path)
            print(f"\nSuccessfully collected data for E-Mini S&P 500 (ES=F)")
            print(f"Total records: {len(df)}")
            print(f"Date range: {df['date'].min()} to {df['date'].max()}")
            print(f"Data saved to:")
            print(f"  - CSV: {csv_path}")
            print(f"  - SQLite: {os.path.join(data_dir, 'historical_ES.db')}")
            print("\nSample data (latest 5 records):")
            print(df.tail(5))

        except Exception as e:
            logger_instance.error(
                class_name_main,
                function_name_main,
                f"Error reading collected data: {e}"
            )
            print(f"Error reading collected data: {e}")

        logger_instance.info(
            class_name_main,
            function_name_main,
            "Application process completed."
        )

    except Exception as e:
        logger_instance.error(
            class_name_main,
            function_name_main,
            f"An unhandled error occurred: {e}"
        )
        print(
            f"An unhandled error occurred: {e}. Check the logs for more details.")


if __name__ == "__main__":
    main()