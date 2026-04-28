import streamlit as st
import pandas as pd
import numpy as np
import scipy.stats as stats
import plotly.express as px
import plotly.graph_objects as go
from fpdf import FPDF
import tempfile

st.set_page_config(page_title="Weyland-Yutani Mining Ops", layout="wide")

@st.cache_data(ttl=600)
def load_data():
    url = "https://docs.google.com/spreadsheets/d/e/2PACX-1vRvGsL0oCRRFnFNdfXxs6sP9sBj9SL36Q7XNSTfFmSnp1Lim_-Em0q8WifeLfrLhM4krTpTOgwEt_k8/pub?gid=1613964847&single=true&output=csv"
    df = pd.read_csv(url)
    df['date'] = pd.to_datetime(df['date'])
    return df

df = load_data()

st.title("Weyland-Yutani Mining Ops Dashboard")
st.markdown("by Askhat Aubakirov. Data Engineering Itransition TASK5. April 2026.")

st.sidebar.title("Configuration")
selected_mine = st.sidebar.selectbox("Select Mine", ["Total Output"] + list(df['mine_name'].unique()))

if selected_mine == "Total Output":
    plot_df = df.groupby('date')['output_fin'].sum().reset_index()
else:
    plot_df = df[df['mine_name'] == selected_mine].copy()

#metrics
mean_val = plot_df['output_fin'].mean()
std_val = plot_df['output_fin'].std()
median_val = plot_df['output_fin'].median()
q1 = plot_df['output_fin'].quantile(0.25)
q3 = plot_df['output_fin'].quantile(0.75)
iqr_val = q3 - q1

st.title(f"Dashboard: {selected_mine}")
col1, col2, col3, col4 = st.columns(4)
col1.metric("Mean Daily Output", f"{mean_val:.2f}")
col2.metric("Standard Deviation", f"{std_val:.2f}")
col3.metric("Median", f"{median_val:.2f}")
col4.metric("IQR", f"{iqr_val:.2f}")

st.sidebar.subheader("Anomaly Detection Parameters")
z_thresh = st.sidebar.slider("Z-Score Threshold", 1.0, 4.0, 2.0)
ma_window = st.sidebar.slider("Moving Average Window (days)", 2, 14, 5)
ma_dist_pct = st.sidebar.slider("Max Distance from MA (%)", 5, 50, 20)
grubbs_alpha = st.sidebar.slider("Grubbs' Test Alpha", 0.01, 0.10, 0.05)

#1-IQR Rule
lower_bound = q1 - 1.5 * iqr_val
upper_bound = q3 + 1.5 * iqr_val
plot_df['anomaly_iqr'] = (plot_df['output_fin'] < lower_bound) | (plot_df['output_fin'] > upper_bound)

#2-Z-Score
plot_df['z_score'] = np.abs(stats.zscore(plot_df['output_fin']))
plot_df['anomaly_zscore'] = plot_df['z_score'] > z_thresh

#3-Distance from Moving Average
plot_df['moving_avg'] = plot_df['output_fin'].rolling(window=ma_window).mean()
pct_distance = np.abs(plot_df['output_fin'] - plot_df['moving_avg']) / plot_df['moving_avg'] * 100
plot_df['anomaly_ma'] = pct_distance > ma_dist_pct

#4-Grubbs' Test (Simplified Two-Sided)
def grubbs_test(data, alpha):
    n = len(data)
    t_crit = stats.t.ppf(1 - alpha / (2 * n), n - 2)
    g_crit = ((n - 1) / np.sqrt(n)) * np.sqrt((t_crit**2) / (n - 2 + t_crit**2))
    
    mean = np.mean(data)
    std = np.std(data)
    g_scores = np.abs(data - mean) / std
    return g_scores > g_crit

plot_df['anomaly_grubbs'] = grubbs_test(plot_df['output_fin'].values, grubbs_alpha)

chart_type = st.sidebar.selectbox("Chart Type", ["line", "bar"]) # Note: Stacked requires multi-mine data logic
anomaly_test = st.sidebar.selectbox("Highlight Anomalies From:", ["anomaly_iqr", "anomaly_zscore", "anomaly_ma", "anomaly_grubbs"])
poly_degree = st.sidebar.selectbox("Trendline Degree", [1, 2, 3, 4])

fig = go.Figure()
#line or bar for output:
if chart_type == "line":
    fig.add_trace(go.Scatter(x=plot_df['date'], y=plot_df['output_fin'], mode='lines', name='Output', line=dict(color='gray')))
elif chart_type == "bar":
    fig.add_trace(go.Bar(x=plot_df['date'], y=plot_df['output_fin'], name='Output', marker_color='gray'))

#show anomalies
anomalies = plot_df[plot_df[anomaly_test] == True]
fig.add_trace(go.Scatter(x=anomalies['date'], y=anomalies['output_fin'], mode='markers', name='Anomaly', marker=dict(color='red', size=10, symbol='x')))

# Add Polynomial Trendline
x_numeric = np.arange(len(plot_df)) # Convert dates to index for polyfit
coeffs = np.polyfit(x_numeric, plot_df['output_fin'], poly_degree)
poly_trend = np.polyval(coeffs, x_numeric)
fig.add_trace(go.Scatter(x=plot_df['date'], y=poly_trend, mode='lines', name=f'Trend (Degree {poly_degree})', line=dict(color='orange', dash='dash')))

st.plotly_chart(fig, width="stretch")

#pdf gen and download
st.subheader("Reporting")
if st.button("Generate Comprehensive PDF Report"):
    with st.spinner("Compiling Weyland-Yutani Operations Report..."):
        pdf = FPDF()
        
        mines = [m for m in df['mine_name'].unique() if pd.notna(m)]
        reporting_entities = ["Total Output"] + mines
        
        for entity in reporting_entities:
            if entity == "Total Output":
                entity_df = df.groupby('date')['output_fin'].sum().reset_index()
            else:
                entity_df = df[df['mine_name'] == entity].copy()
                
            mean_val = entity_df['output_fin'].mean()
            std_val = entity_df['output_fin'].std()
            median_val = entity_df['output_fin'].median()
            q1 = entity_df['output_fin'].quantile(0.25)
            q3 = entity_df['output_fin'].quantile(0.75)
            iqr_val = q3 - q1
            
            # IQR
            lower_bound = q1 - 1.5 * iqr_val
            upper_bound = q3 + 1.5 * iqr_val
            entity_df['anomaly_iqr'] = (entity_df['output_fin'] < lower_bound) | (entity_df['output_fin'] > upper_bound)
            # Z-Score
            entity_df['z_score'] = np.abs(stats.zscore(entity_df['output_fin']))
            entity_df['anomaly_zscore'] = entity_df['z_score'] > z_thresh
            # Moving Average
            entity_df['moving_avg'] = entity_df['output_fin'].rolling(window=ma_window).mean()
            pct_distance = np.abs(entity_df['output_fin'] - entity_df['moving_avg']) / entity_df['moving_avg'] * 100
            entity_df['anomaly_ma'] = pct_distance > ma_dist_pct
            # Grubbs' Test
            entity_df['anomaly_grubbs'] = grubbs_test(entity_df['output_fin'].values, grubbs_alpha)
            
            fig_pdf = go.Figure()
            if chart_type == "line":
                fig_pdf.add_trace(go.Scatter(x=entity_df['date'], y=entity_df['output_fin'], mode='lines', name='Output', line=dict(color='gray')))
            elif chart_type == "bar":
                fig_pdf.add_trace(go.Bar(x=entity_df['date'], y=entity_df['output_fin'], name='Output', marker_color='gray'))
            
            anomalies = entity_df[entity_df[anomaly_test] == True]
            fig_pdf.add_trace(go.Scatter(x=anomalies['date'], y=anomalies['output_fin'], mode='markers', name='Anomaly', marker=dict(color='red', size=10, symbol='x')))
            
            x_numeric = np.arange(len(entity_df))
            coeffs = np.polyfit(x_numeric, entity_df['output_fin'], poly_degree)
            poly_trend = np.polyval(coeffs, x_numeric)
            fig_pdf.add_trace(go.Scatter(x=entity_df['date'], y=poly_trend, mode='lines', name=f'Trend', line=dict(color='orange', dash='dash')))

            pdf.add_page()
            pdf.set_font("Arial", 'B', 16)
            pdf.cell(200, 10, txt=f"Operations Report: {entity}", ln=True, align='C')
            
            pdf.set_font("Arial", size=12)
            pdf.cell(200, 10, txt=f"Mean: {mean_val:.2f} | Std Dev: {std_val:.2f} | Median: {median_val:.2f} | IQR: {iqr_val:.2f}", ln=True)
            
            with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmpfile:
                fig_pdf.write_image(tmpfile.name)
                pdf.image(tmpfile.name, x=10, y=pdf.get_y(), w=190, h=100)
                
            pdf.set_y(pdf.get_y() + 105) 
            
            pdf.set_font("Arial", 'B', 12)
            pdf.cell(200, 10, txt=f"Detected Anomalies ({anomaly_test})", ln=True)
            pdf.set_font("Arial", size=10)
            
            if anomalies.empty:
                pdf.cell(200, 8, txt="No anomalies detected with current parameters.", ln=True)
            else:
                for idx, row in anomalies.iterrows():
                    date_str = row['date'].strftime('%Y-%m-%d')
                    val = row['output_fin']
                    pdf.cell(200, 8, txt=f"- Date: {date_str} | Value: {val:.2f}", ln=True)
        
        pdf_bytes = pdf.output(dest='S').encode('latin-1')
        st.download_button(label="Download Comprehensive Report", data=pdf_bytes, file_name="WY_Comprehensive_Report.pdf", mime="application/pdf")