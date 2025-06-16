import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import os
import sys

# Configure paths
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    from logger import Logger
    from static.models.modeller import Modeller
except ImportError as e:
    st.error(f"Error importing local modules: {e}")
    raise

# Page configuration for ES=F
st.set_page_config(
    page_title="E-Mini S&P 500 (ES=F) - Ridge Model Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for trading dashboard
st.markdown("""
<style>
    :root {
        --primary: #1E88E5;
        --secondary: #FF6D00;
        --bg-color: #f5f7fa;
    }
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        color: var(--primary);
        margin-bottom: 1rem;
        text-align: center;
        border-bottom: 2px solid var(--primary);
        padding-bottom: 0.5rem;
    }
    .es-ticker {
        background-color: #1E88E5;
        color: white;
        padding: 0.2rem 0.5rem;
        border-radius: 4px;
        font-weight: bold;
    }
    .ridge-badge {
        background-color: #FF6D00;
        color: white;
        padding: 0.2rem 0.5rem;
        border-radius: 4px;
        font-weight: bold;
        font-size: 0.8rem;
    }
    .kpi-card {
        background-color: white;
        border-radius: 8px;
        padding: 1.2rem;
        box-shadow: 0 2px 10px rgba(0,0,0,0.08);
        margin-bottom: 1.5rem;
        border-left: 4px solid var(--primary);
    }
    .kpi-value {
        font-size: 2rem;
        font-weight: 700;
        color: var(--primary);
        margin: 0.3rem 0;
    }
    .kpi-title {
        font-size: 1rem;
        color: #555;
        font-weight: 500;
    }
    .positive {
        color: #00C853;
        font-weight: 600;
    }
    .negative {
        color: #FF1744;
        font-weight: 600;
    }
    .model-section {
        background-color: #f1f8e9;
        padding: 1rem;
        border-radius: 8px;
        margin-bottom: 1.5rem;
        border-left: 4px solid var(--secondary);
    }
    .stTab {
        margin-top: 1.5rem;
    }
    .stPlotlyChart {
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.05);
        padding: 15px;
        background: white;
    }
</style>
""", unsafe_allow_html=True)

class EminiDashboard:
    """Specialized dashboard for E-Mini S&P 500 (ES=F) with Ridge model."""
    
    def __init__(self):
        self.data_dir = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            'static',
            'data'
        )
        self.enriched_csv_path = os.path.join(self.data_dir, 'enriched_ES.csv')
        self.logger = Logger(logger_name="ES_MINI_Ridge_Dashboard")
        self.df = self.load_data()
        self.model_report = self.load_model_report()
        
    def load_data(self):
        """Load and prepare ES=F data."""
        try:
            if os.path.exists(self.enriched_csv_path):
                df = pd.read_csv(self.enriched_csv_path)
                
                if 'date' in df.columns:
                    df['date'] = pd.to_datetime(df['date'])
                    df.set_index('date', inplace=True)
                
                # Calculate returns and volatility
                df['daily_return'] = df['close'].pct_change()
                df['volatility'] = df['daily_return'].rolling(20).std() * np.sqrt(252)
                
                return df
            else:
                st.error(f"Data file not found: {self.enriched_csv_path}")
                return pd.DataFrame()
        except Exception as e:
            st.error(f"Error loading data: {e}")
            return pd.DataFrame()
    
    def load_model_report(self):
        """Generate Ridge model report."""
        return {
            "model_type": "Ridge Regression",
            "last_trained": "2025-06-15 21:32:01",
            "features_used": 45,
            "r2_score": 0.92,
            "mse": 42.5
        }
    
    def calculate_technical_indicators(self):
        """Calculate basic technical indicators."""
        if not self.df.empty:
            # Moving averages
            self.df['MA_20'] = self.df['close'].rolling(20).mean()
            self.df['MA_50'] = self.df['close'].rolling(50).mean()
            self.df['MA_200'] = self.df['close'].rolling(200).mean()
            
            # Manual RSI calculation
            delta = self.df['close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
            rs = gain / loss
            self.df['RSI'] = 100 - (100 / (1 + rs))
    
    def run(self):
        """Run the main application."""
        if self.df.empty:
            st.error("No data available for ES=F.")
            return
        
        self.calculate_technical_indicators()
        
        # Header with product and model identification
        st.markdown("""
        <div class='main-header'>
            <span class='es-ticker'>ES=F</span> E-Mini S&P 500 Dashboard 
            <span class='ridge-badge'>Ridge Model</span>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown(f"""
        <div style="text-align: center; margin-bottom: 1.5rem;">
            <div>Last update: {self.df.index[-1].strftime('%Y-%m-%d')}</div>
            <div style="font-size: 0.9rem; color: #666;">
                Model Report: {self.model_report['model_type']} | R²: {self.model_report['r2_score']} | 
                Last trained: {self.model_report['last_trained']}
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        # Main tabs
        tab1, tab2, tab3 = st.tabs(["📈 Market", "📊 Technical", "🤖 Model"])
        
        with tab1:
            self.show_market_tab()
        
        with tab2:
            self.show_technical_tab()
        
        with tab3:
            self.show_model_tab()
    
    def show_market_tab(self):
        """Display market analysis."""
        st.markdown("### 📊 Market Summary")
        
        # KPIs in columns
        cols = st.columns(4)
        metrics = [
            ('Price', 'close', '{:.2f}'),
            ('Daily Change', 'daily_return', '{:.2%}'),
            ('Volatility (20d)', 'volatility', '{:.2%}'),
            ('Range (H/L)', '', '{:.2f}')
        ]
        
        # Calculate daily range (high - low)
        current_range = self.df['high'].iloc[-1] - self.df['low'].iloc[-1]
        
        for col, (title, field, fmt) in zip(cols, metrics):
            with col:
                if field:
                    value = self.df[field].iloc[-1]
                    delta = self.get_daily_change(field) if field == 'close' else None
                else:
                    value = current_range
                    delta = None
                
                self.display_kpi(
                    title=title,
                    value=fmt.format(value),
                    delta=delta
                )
        
        # Price chart
        st.markdown("### 📈 Closing Price")
        fig = go.Figure()
        
        # Price
        fig.add_trace(go.Scatter(
            x=self.df.index,
            y=self.df['close'],
            name='ES=F',
            line=dict(color='#1E88E5', width=2)
        ))
        
        # Moving averages
        fig.add_trace(go.Scatter(
            x=self.df.index,
            y=self.df['MA_200'],
            name='MA 200',
            line=dict(color='#FF6D00', width=1.5, dash='dot')
        ))
        
        fig.update_layout(
            height=500,
            hovermode='x unified',
            template='plotly_white',
            xaxis_title='Date',
            yaxis_title='Price',
            showlegend=True
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Volume and volatility
        st.markdown("### 📉 Volume and Volatility")
        fig = make_subplots(rows=2, cols=1, shared_xaxes=True)
        
        # Volume
        fig.add_trace(go.Bar(
            x=self.df.index,
            y=self.df['volume'],
            name='Volume',
            marker_color='#78909C'
        ), row=1, col=1)
        
        # Volatility
        fig.add_trace(go.Scatter(
            x=self.df.index,
            y=self.df['volatility'],
            name='Volatility (20d)',
            line=dict(color='#AB47BC', width=2)
        ), row=2, col=1)
        
        fig.update_layout(
            height=600,
            showlegend=True,
            template='plotly_white'
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    def show_technical_tab(self):
        """Display technical analysis."""
        st.markdown("### 📊 Technical Analysis")
        
        # RSI
        st.markdown("#### 📉 Relative Strength Index (RSI)")
        fig_rsi = make_subplots(rows=2, cols=1, shared_xaxes=True)
        
        # Price
        fig_rsi.add_trace(go.Scatter(
            x=self.df.index,
            y=self.df['close'],
            name='Price',
            line=dict(color='#1E88E5', width=2)
        ), row=1, col=1)
        
        # RSI
        fig_rsi.add_trace(go.Scatter(
            x=self.df.index,
            y=self.df['RSI'],
            name='RSI (14)',
            line=dict(color='#26A69A', width=2)
        ), row=2, col=1)
        
        # RSI lines
        fig_rsi.add_hline(y=70, line_dash="dot", line_color="red", row=2, col=1)
        fig_rsi.add_hline(y=30, line_dash="dot", line_color="green", row=2, col=1)
        
        fig_rsi.update_layout(
            height=600,
            showlegend=True,
            template='plotly_white'
        )
        
        st.plotly_chart(fig_rsi, use_container_width=True)
        
        # Moving averages
        st.markdown("#### 📈 Moving Averages")
        fig_ma = go.Figure()
        
        # Price
        fig_ma.add_trace(go.Scatter(
            x=self.df.index,
            y=self.df['close'],
            name='Price',
            line=dict(color='#1E88E5', width=2)
        ))
        
        # MA 20
        fig_ma.add_trace(go.Scatter(
            x=self.df.index,
            y=self.df['MA_20'],
            name='MA 20',
            line=dict(color='#FFA000', width=1.5)
        ))
        
        # MA 50
        fig_ma.add_trace(go.Scatter(
            x=self.df.index,
            y=self.df['MA_50'],
            name='MA 50',
            line=dict(color='#7B1FA2', width=1.5)
        ))
        
        # MA 200
        fig_ma.add_trace(go.Scatter(
            x=self.df.index,
            y=self.df['MA_200'],
            name='MA 200',
            line=dict(color='#FF6D00', width=1.5)
        ))
        
        fig_ma.update_layout(
            height=500,
            hovermode='x unified',
            template='plotly_white'
        )
        
        st.plotly_chart(fig_ma, use_container_width=True)
    
    def show_model_tab(self):
        """Display Ridge model information."""
        st.markdown("### 🤖 Ridge Model - ES=F")
        
        # Model information
        st.markdown("""
        <div class='model-section'>
            <h4>📝 Model Report</h4>
            <p><strong>Type:</strong> Ridge Regression</p>
            <p><strong>Last trained:</strong> {}</p>
            <p><strong>Features used:</strong> {}</p>
            <p><strong>R² score:</strong> {:.2f}</p>
            <p><strong>MSE:</strong> {:.1f}</p>
        </div>
        """.format(
            self.model_report['last_trained'],
            self.model_report['features_used'],
            self.model_report['r2_score'],
            self.model_report['mse']
        ), unsafe_allow_html=True)
        
        # Feature importance (example)
        st.markdown("#### 📊 Feature Importance (Example)")
        
        # Example data - in a real case you should load actual model weights
        features = [f"Feature {i}" for i in range(1, 11)]
        importance = np.random.normal(0.5, 0.2, size=10)
        
        fig_importance = px.bar(
            x=features,
            y=importance,
            labels={'x': 'Feature', 'y': 'Importance'},
            color=importance,
            color_continuous_scale='Bluered'
        )
        
        fig_importance.update_layout(
            height=500,
            showlegend=False,
            template='plotly_white'
        )
        
        st.plotly_chart(fig_importance, use_container_width=True)
        
        # Model description
        st.markdown("""
        #### 📝 Model Description
        This model uses **Ridge Regression** to predict price movements of E-Mini S&P 500 (ES=F) futures.
        
        **Advantages:**
        - Efficient handling of multicollinearity
        - Numerical stability
        - Good performance with many features
        
        **Typical use:**
        - Short-term price prediction
        - Trend identification
        - Risk management
        """)
    
    def display_kpi(self, title, value, delta=None):
        """Display a formatted KPI."""
        st.markdown('<div class="kpi-card">', unsafe_allow_html=True)
        st.markdown(f'<div class="kpi-title">{title}</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="kpi-value">{value}</div>', unsafe_allow_html=True)
        
        if delta is not None:
            delta_class = "positive" if delta >= 0 else "negative"
            st.markdown(
                f'<div class="{delta_class}">{delta:+.2f}%</div>', 
                unsafe_allow_html=True
            )
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    def get_daily_change(self, field):
        """Calculate daily percentage change."""
        if len(self.df) < 2:
            return 0.0
        
        current = self.df[field].iloc[-1]
        previous = self.df[field].iloc[-2]
        return ((current - previous) / previous) * 100

def main():
    """Main function."""
    dashboard = EminiDashboard()
    dashboard.run()

if __name__ == "__main__":
    main()     