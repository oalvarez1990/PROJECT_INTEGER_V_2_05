# dashboard.py
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import os
import sqlite3
from datetime import datetime, timedelta
import sys

# Add current directory to path to import local modules
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from logger import Logger
from project_piv.modeller import Modeller

# Page configuration
st.set_page_config(
    page_title="E-Mini S&P 500 Dashboard",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        color: #1E88E5;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.5rem;
        font-weight: 600;
        color: #424242;
        margin-bottom: 1rem;
    }
    .kpi-card {
        background-color: #f5f5f5;
        border-radius: 5px;
        padding: 1rem;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .kpi-value {
        font-size: 2rem;
        font-weight: 700;
        color: #1E88E5;
    }
    .kpi-title {
        font-size: 1rem;
        color: #616161;
    }
    .positive {
        color: #4CAF50;
    }
    .negative {
        color: #F44336;
    }
    .neutral {
        color: #9E9E9E;
    }
</style>
""", unsafe_allow_html=True)

class Dashboard:
    """
    Class for creating an interactive dashboard for financial data visualization.
    """
    
    def __init__(self, data_dir=None):
        """
        Initialize the Dashboard class.
        
        Args:
            data_dir: Directory where data is stored
        """
        # Set data directory
        if data_dir is None:
            self.data_dir = os.path.join(
                os.path.dirname(os.path.abspath(__file__)),
                'static',
                'data'
            )
        else:
            self.data_dir = data_dir
        
        # Define file paths
        self.enriched_csv_path = os.path.join(
            self.data_dir,
            'enriched_ES.csv'
        )
        
        # Initialize logger
        self.logger = Logger(logger_name="ES_MINI_Dashboard")
        
        # Initialize modeller
        self.modeller = Modeller(self.logger)
        
        # Load data
        self.df = self.load_data()
        
        # Get predictions
        self.predictions = self.modeller.predecir()
    
    def load_data(self):
        """
        Load enriched data from CSV file.
        
        Returns:
            DataFrame with enriched data
        """
        try:
            if os.path.exists(self.enriched_csv_path):
                df = pd.read_csv(self.enriched_csv_path)
                
                # Convert date column to datetime
                if 'date' in df.columns:
                    df['date'] = pd.to_datetime(df['date'])
                
                return df
            else:
                st.error(f"Data file not found: {self.enriched_csv_path}")
                return pd.DataFrame()
        
        except Exception as e:
            st.error(f"Error loading data: {e}")
            return pd.DataFrame()
    
    def run(self):
        """
        Run the dashboard.
        """
        # Check if data is available
        if self.df.empty:
            st.error("No data available. Please make sure the data files exist.")
            return
        
        # Main title
        st.markdown('<div class="main-header">E-Mini S&P 500 Dashboard</div>', unsafe_allow_html=True)
        
        # General information
        st.markdown("This dashboard shows information and analysis of the E-Mini S&P 500, a futures contract based on the S&P 500 index.")
        
        # Last update date
        last_date = self.df['date'].max()
        st.markdown(f"**Last update:** {last_date.strftime('%B %d, %Y')}")
        
        # Create tabs
        tab1, tab2, tab3, tab4 = st.tabs(["Summary", "Technical Analysis", "Correlations", "Predictions"])
        
        with tab1:
            self.show_summary_tab()
        
        with tab2:
            self.show_technical_analysis_tab()
        
        with tab3:
            self.show_correlations_tab()
        
        with tab4:
            self.show_predictions_tab()
    
    def show_summary_tab(self):
        """
        Show summary tab with key performance indicators (KPIs).
        """
        st.markdown('<div class="sub-header">Summary</div>', unsafe_allow_html=True)
        
        # Key performance indicators
        st.markdown('<div class="kpi-card">', unsafe_allow_html=True)
        st.markdown('<div class="kpi-title">Last Close Price</div>', unsafe_allow_html=True)
        last_close = self.df['close'].iloc[-1]
        st.markdown(f'<div class="kpi-value">{last_close:.2f}</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
        st.markdown('<div class="kpi-card">', unsafe_allow_html=True)
        st.markdown('<div class="kpi-title">Last Open Price</div>', unsafe_allow_html=True)
        last_open = self.df['open'].iloc[-1]
        st.markdown(f'<div class="kpi-value">{last_open:.2f}</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
        st.markdown('<div class="kpi-card">', unsafe_allow_html=True)
        st.markdown('<div class="kpi-title">Last High Price</div>', unsafe_allow_html=True)
        last_high = self.df['high'].iloc[-1]                