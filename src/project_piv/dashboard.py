# dashboard.py
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import os
import sys
import webbrowser
import threading
import time

# Configuración de rutas para imports locales
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    from logger import Logger
    from static.models.modeller import Modeller
except ImportError as e:
    st.error(f"Error importing local modules: {e}")
    raise

# Configuración de página
st.set_page_config(
    page_title="E-Mini S&P 500 Dashboard",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS personalizado
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
        margin-bottom: 1rem;
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
    .stTab {
        margin-top: 1rem;
    }
</style>
""", unsafe_allow_html=True)

class Dashboard:
    """Clase principal del dashboard de análisis financiero."""
    
    def __init__(self, data_dir=None):
        """Inicializa el dashboard con configuración básica."""
        self.data_dir = data_dir or os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            'static',
            'data'
        )
        
        self.enriched_csv_path = os.path.join(self.data_dir, 'enriched_ES.csv')
        self.logger = Logger(logger_name="ES_MINI_Dashboard")
        self.modeller = Modeller(self.logger)
        self.df = self.load_data()
        
        try:
            self.predictions = self.modeller.predecir()
        except Exception as e:
            self.logger.error(f"Error getting predictions: {e}")
            self.predictions = None
    
    def load_data(self):
        """Carga los datos desde el archivo CSV."""
        try:
            if os.path.exists(self.enriched_csv_path):
                df = pd.read_csv(self.enriched_csv_path)
                
                if 'date' in df.columns:
                    df['date'] = pd.to_datetime(df['date'])
                
                return df
            else:
                st.error(f"Archivo de datos no encontrado: {self.enriched_csv_path}")
                return pd.DataFrame()
        
        except Exception as e:
            st.error(f"Error cargando datos: {e}")
            return pd.DataFrame()
    
    def run(self):
        """Ejecuta la aplicación principal del dashboard."""
        if self.df.empty:
            st.error("No hay datos disponibles. Verifica los archivos de datos.")
            return
        
        # Encabezado principal
        st.markdown('<div class="main-header">E-Mini S&P 500 Dashboard</div>', unsafe_allow_html=True)
        st.markdown("Dashboard interactivo para análisis del contrato de futuros E-Mini S&P 500.")
        
        # Última fecha de actualización
        last_date = self.df['date'].max()
        st.markdown(f"**Última actualización:** {last_date.strftime('%d/%m/%Y')}")
        
        # Pestañas principales
        tab1, tab2, tab3, tab4 = st.tabs([
            "Resumen", 
            "Análisis Técnico", 
            "Correlaciones", 
            "Predicciones"
        ])
        
        with tab1:
            self.show_summary_tab()
        
        with tab2:
            self.show_technical_tab()
        
        with tab3:
            self.show_correlations_tab()
        
        with tab4:
            self.show_predictions_tab()
    
    def show_summary_tab(self):
        """Muestra el tab de resumen con KPIs principales."""
        st.markdown('<div class="sub-header">Resumen del Mercado</div>', unsafe_allow_html=True)
        
        # KPIs en columnas
        cols = st.columns(3)
        metrics = [
            ('Último Cierre', 'close', '{:.2f}'),
            ('Apertura', 'open', '{:.2f}'),
            ('Máximo', 'high', '{:.2f}')
        ]
        
        for col, (title, field, fmt) in zip(cols, metrics):
            with col:
                self.display_kpi(
                    title=title,
                    value=fmt.format(self.df[field].iloc[-1]),
                    delta=self.get_daily_change(field)
                )
        
        # Gráfico de precios
        st.markdown('<div class="sub-header">Histórico de Precios</div>', unsafe_allow_html=True)
        fig = px.line(
            self.df, 
            x='date', 
            y='close', 
            title='Precio de Cierre',
            labels={'date': 'Fecha', 'close': 'Precio'}
        )
        st.plotly_chart(fig, use_container_width=True)
    
    def show_technical_tab(self):
        """Muestra indicadores técnicos."""
        st.markdown('<div class="sub-header">Indicadores Técnicos</div>', unsafe_allow_html=True)
        st.write("Aquí se mostrarán los indicadores técnicos calculados.")
    
    def show_correlations_tab(self):
        """Muestra análisis de correlaciones."""
        st.markdown('<div class="sub-header">Correlaciones de Mercado</div>', unsafe_allow_html=True)
        st.write("Análisis de correlaciones con otros activos e indicadores.")
    
    def show_predictions_tab(self):
        """Muestra las predicciones del modelo."""
        st.markdown('<div class="sub-header">Predicciones del Modelo</div>', unsafe_allow_html=True)
        
        if self.predictions is not None:
            st.write("Resultados de las predicciones:")
            st.dataframe(self.predictions)
        else:
            st.warning("No hay predicciones disponibles. Verifica el modelo.")
    
    def display_kpi(self, title, value, delta=None):
        """Muestra un KPI con formato."""
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
        """Calcula el cambio porcentual diario."""
        if len(self.df) < 2:
            return 0.0
        
        current = self.df[field].iloc[-1]
        previous = self.df[field].iloc[-2]
        return ((current - previous) / previous) * 100

def is_streamlit_running():
    """Determina si el script se ejecuta con streamlit run."""
    return 'streamlit' in sys.modules

def open_browser():
    """Abre el navegador automáticamente para python3."""
    time.sleep(3)
    webbrowser.open_new("http://localhost:8501")

def main():
    """Función principal de ejecución."""
    dashboard = Dashboard()
    dashboard.run()

if __name__ == "__main__":
    if not is_streamlit_running():
        threading.Thread(target=open_browser, daemon=True).start()
    
    main()           