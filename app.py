import streamlit as st
import pandas as pd
import numpy as np
import scipy.stats as stats
import plotly.express as px
import plotly.graph_objects as go
from fpdf import FPDF
import tempfile
import matplotlib.pyplot as plt
from datetime import datetime

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

# ==========================================
# 1. DEFINE THE CORPORATE PDF TEMPLATE
# ==========================================
class WY_Report_PDF(FPDF):
    def header(self):
        # Attempt to load the WY Logo
        try:
            # Place a file named 'wy_logo.png' in your project folder
            self.image("wy_logo.png", 10, 8, 30) 
        except:
            # Fallback if image is missing
            self.set_font('Arial', 'B', 12)
            self.set_text_color(200, 0, 0)
            self.cell(30, 10, '[WY-LOGO]', border=1, align='C')

        # Corporate Header Text
        self.set_font('Arial', 'B', 16)
        self.set_text_color(30, 30, 30)
        self.cell(0, 8, 'WEYLAND-YUTANI CORPORATION', ln=True, align='R')
        
        self.set_font('Arial', 'I', 10)
        self.set_text_color(100, 100, 100)
        self.cell(0, 5, 'Building Better Worlds', ln=True, align='R')
        self.cell(0, 5, 'Off-World Operations & Resource Management', ln=True, align='R')
        
        # Draw a horizontal line
        self.set_draw_color(200, 0, 0) # WY Corporate Red
        self.set_line_width(0.5)
        self.line(10, 30, 200, 30)
        self.ln(15)

    def footer(self):
        # Position at 25 mm from bottom
        self.set_y(-25)
        
        # Security Warning
        self.set_font('Arial', 'B', 8)
        self.set_text_color(200, 0, 0)
        self.cell(0, 4, 'RESTRICTED ACCESS - CLASSIFIED LEVEL 4', align='C', ln=True)
        
        # Signature and Title
        self.set_font('Arial', 'I', 8)
        self.set_text_color(100, 100, 100)
        self.cell(0, 4, 'Prepared & Authorized by: Askhat Aubakirov', align='C', ln=True)
        self.cell(0, 4, 'Principal Data Architect | Planetary Engineering Div.', align='C', ln=True)
        
        # Page Numbering
        self.set_font('Arial', '', 8)
        self.cell(0, 4, f'Page {self.page_no()}', align='C', ln=True)


# ==========================================
# 2. GENERATE AND DOWNLOAD LOGIC
# ==========================================
st.subheader("Automated Reporting Systems")

if st.button("Generate Classified PDF Report"):
    with st.spinner("Accessing Weyland-Yutani Mainframe & Compiling Report..."):
        
        # Initialize our custom WY PDF instead of the base FPDF
        pdf = WY_Report_PDF()
        
        # Get timestamp for the report
        report_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S UTC")
        
        mines = [m for m in df['mine_name'].unique() if pd.notna(m)]
        reporting_entities = ["Total Output"] + mines
        
        for entity in reporting_entities:
            # 1. Filter Data
            if entity == "Total Output":
                entity_df = df.groupby('date')['output_fin'].sum().reset_index()
            else:
                entity_df = df[df['mine_name'] == entity].copy()
                
            # 2. Calculate Metrics
            mean_val = entity_df['output_fin'].mean()
            std_val = entity_df['output_fin'].std()
            median_val = entity_df['output_fin'].median()
            q1 = entity_df['output_fin'].quantile(0.25)
            q3 = entity_df['output_fin'].quantile(0.75)
            iqr_val = q3 - q1
            
            # 3. Detect Anomalies
            lower_bound = q1 - 1.5 * iqr_val
            upper_bound = q3 + 1.5 * iqr_val
            entity_df['anomaly_iqr'] = (entity_df['output_fin'] < lower_bound) | (entity_df['output_fin'] > upper_bound)
            
            entity_df['z_score'] = np.abs(stats.zscore(entity_df['output_fin']))
            entity_df['anomaly_zscore'] = entity_df['z_score'] > z_thresh
            
            entity_df['moving_avg'] = entity_df['output_fin'].rolling(window=ma_window).mean()
            pct_distance = np.abs(entity_df['output_fin'] - entity_df['moving_avg']) / entity_df['moving_avg'] * 100
            entity_df['anomaly_ma'] = pct_distance > ma_dist_pct
            
            entity_df['anomaly_grubbs'] = grubbs_test(entity_df['output_fin'].values, grubbs_alpha)
            
            anomalies = entity_df[entity_df[anomaly_test] == True]
            
            x_numeric = np.arange(len(entity_df))
            coeffs = np.polyfit(x_numeric, entity_df['output_fin'], poly_degree)
            poly_trend = np.polyval(coeffs, x_numeric)

            # 4. Generate Professional Matplotlib Chart
            # Use a cleaner style for corporate reports
            plt.style.use('bmh') 
            plt.figure(figsize=(10, 4.5))
            
            if chart_type == "line":
                plt.plot(entity_df['date'], entity_df['output_fin'], color='#1f497d', linewidth=2, label='Daily Extraction')
            elif chart_type == "bar":
                plt.bar(entity_df['date'], entity_df['output_fin'], color='#1f497d', alpha=0.8, label='Daily Extraction')
                
            # Add anomalies and trendline
            plt.scatter(anomalies['date'], anomalies['output_fin'], color='#c00000', marker='x', s=120, linewidths=2, label='Critical Anomaly', zorder=5)
            plt.plot(entity_df['date'], poly_trend, color='#e36c0a', linestyle='--', linewidth=2, label=f'Trajectory (Deg {poly_degree})')
            
            plt.title(f"Sector Output Assessment: {entity}", fontweight='bold')
            plt.xlabel("Local Sol Date")
            plt.ylabel("Resource Tonnage")
            plt.grid(True, linestyle=':', alpha=0.6)
            plt.legend(loc='upper right')
            plt.tight_layout()

            # 5. Build the PDF Page
            pdf.add_page()
            
            # Entity Header Banner
            pdf.set_fill_color(31, 73, 125) # Dark Corporate Blue
            pdf.set_text_color(255, 255, 255)
            pdf.set_font("Arial", 'B', 14)
            pdf.cell(0, 10, txt=f"  GEOLOGICAL ASSET: {entity.upper()}", ln=True, fill=True)
            
            # Timestamp
            pdf.set_text_color(100, 100, 100)
            pdf.set_font("Arial", 'I', 9)
            pdf.cell(0, 6, txt=f"  System Timestamp: {report_time}", ln=True)
            pdf.ln(5)
            
            # Stats Table
            pdf.set_text_color(0, 0, 0)
            pdf.set_font("Arial", 'B', 10)
            pdf.cell(45, 8, txt="Mean Output:", border=1, fill=False)
            pdf.set_font("Arial", '', 10)
            pdf.cell(45, 8, txt=f"{mean_val:.2f} Units", border=1, fill=False)
            
            pdf.set_font("Arial", 'B', 10)
            pdf.cell(45, 8, txt="Median Output:", border=1, fill=False)
            pdf.set_font("Arial", '', 10)
            pdf.cell(45, 8, txt=f"{median_val:.2f} Units", border=1, fill=False)
            pdf.ln(8)
            
            pdf.set_font("Arial", 'B', 10)
            pdf.cell(45, 8, txt="Volatility (Std Dev):", border=1, fill=False)
            pdf.set_font("Arial", '', 10)
            pdf.cell(45, 8, txt=f"{std_val:.2f}", border=1, fill=False)
            
            pdf.set_font("Arial", 'B', 10)
            pdf.cell(45, 8, txt="Variance (IQR):", border=1, fill=False)
            pdf.set_font("Arial", '', 10)
            pdf.cell(45, 8, txt=f"{iqr_val:.2f}", border=1, fill=False)
            pdf.ln(12)
            
            # Save Matplotlib chart to temp image directly
            with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmpfile:
                plt.savefig(tmpfile.name, format="png", dpi=200, bbox_inches='tight')
                plt.close() 
                
                # Insert image into PDF
                pdf.image(tmpfile.name, x=10, y=pdf.get_y(), w=190)
                
            # Move cursor below the image
            pdf.set_y(pdf.get_y() + 90) 
            
            # 6. Add Anomaly Log Section
            pdf.set_fill_color(220, 220, 220)
            pdf.set_font("Arial", 'B', 11)
            pdf.cell(0, 8, txt=f"  INCIDENT LOG - TRIGGER: {anomaly_test.upper()}", ln=True, fill=True)
            pdf.set_font("Arial", size=9)
            
            if anomalies.empty:
                pdf.set_text_color(0, 128, 0) # Green
                pdf.cell(0, 8, txt="  > Status Nominal. No operational deviations detected.", ln=True)
            else:
                pdf.set_text_color(200, 0, 0) # Red
                pdf.cell(0, 8, txt=f"  > WARNING: {len(anomalies)} structural deviations detected.", ln=True)
                pdf.set_text_color(0, 0, 0)
                for idx, row in anomalies.iterrows():
                    date_str = row['date'].strftime('%Y-%m-%d')
                    val = row['output_fin']
                    pdf.cell(0, 6, txt=f"     - Sol Date: {date_str} | Extraction Level: {val:.2f} | Priority: High", ln=True)
            
            pdf.set_text_color(0, 0, 0) # Reset color for next page loop
        
        # Output Final PDF
        pdf_bytes = pdf.output(dest='S').encode('latin-1')
        st.download_button(label="Download Classified File", data=pdf_bytes, file_name="WY_Classified_Ops_Report.pdf", mime="application/pdf")