import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))

from project_piv.logger import Logger
from modeller import Modeller

def main():
    # Initialize logger
    logger = Logger(logger_name="ES_MINI_Modeller")
    
    # Initialize modeller
    modeller = Modeller(logger, indicator_symbol="ES=F", prediction_horizon=5)
    
    # Train the model
    print("Starting model training...")
    modeller.entrenar()
    print("Model training completed!")

if __name__ == "__main__":
    main() 