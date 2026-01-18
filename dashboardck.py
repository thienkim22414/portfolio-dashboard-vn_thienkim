import streamlit as st
import pandas as pd
import numpy as np
from scipy.optimize import minimize
import plotly.express as px
import plotly.graph_objects as go

# Load dữ liệu bluechip
@st.cache_data
def load_data():
    fund = pd.read_csv("FUNDAMENTAL_FOR_PORTFOLIO.csv")
    
    # Đọc file price như text để xử lý dấu ngoặc kép và tách cột
    with open("PRICE_FOR_PORTFOLIO.csv", "r", encoding="utf-8") as f:
        lines = f.readlines()
    
    # Bỏ dòng header bị bao ngoặc kép (dòng 0)
    data_lines = lines[1:]
    
    # Tạo list các row
    rows = []
    for line in data_lines:
        # Bỏ dấu ngoặc kép ở đầu và cuối nếu có, rồi tách bằng dấu phẩy
        line = line.strip()
        if line.startswith('"') and line.endswith('"'):
            line = line[1:-1]
        row = line.split(',')
        rows.append(row)
    
    # Tạo DataFrame
    price = pd.DataFrame(rows, columns=['Date', 'Symbol', 'Open', 'High', 'Low', 'Close', 'Volume'])
    
    # Chuyển kiểu dữ liệu
    price['Open'] = pd.to_numeric(price['Open'], errors='coerce')
    price['High'] = pd.to_numeric(price['High'], errors='coerce')
    price['Low'] = pd.to_numeric(price['Low'], errors='coerce')
    price['Close'] = pd.to_numeric(price['Close'], errors='coerce')
    price['Volume'] = pd.to_numeric(price['Volume'], errors='coerce')
    
    # Chuyển cột Date
    price['DATE'] = pd.to_datetime(price['Date'])
    
    return fund, price

fund_df, price_df = load_data()

# Xử lý dữ liệu giá
price_df = price_df.drop_duplicates(subset=['DATE', 'Symbol'])
price_df = price_df.sort_values(['DATE', 'Symbol'])
price_pivot = price_df.pivot(index='DATE', columns='Symbol', values='Close')
price_pivot = price_pivot.ffill().bfill()
log_return = np.log(price_pivot / price_pivot.shift(1)).dropna(how='all')

# Tính ROE, P/E
if 'ROE' not in fund_df.columns:
    fund_df['ROE'] = fund_df['Net Income after Minority Interest'] / fund_df['Shareholders Equity - Common, PoP Avg'] * 100

latest_close = price_df.sort_values(['Symbol', 'DATE'], ascending=[True, False]).drop_duplicates('Symbol')[['Symbol', 'Close']]
fund_df = fund_df.merge(latest_close, on='Symbol', how='left')

if 'P/E' not in fund_df.columns:
    fund_df['P/E'] = fund_df['Close'] / fund_df['EPS - Basic - excl Extraordinary Items, Common - Total']
    fund_df['P/E'] = fund_df['P/E'].replace([np.inf, -np.inf], np.nan).fillna(0)

# DANH SÁCH NGÀNH
# ===============================
aggressive = [
    'IT Services', 'Software', 'Technology Hardware, Storage & Peripherals',
    'Retail', 'Textiles, Apparel & Luxury Goods',
    'Oil, Gas & Consumable Fuels', 'Chemicals',
    'Real Estate Management & Development', 'Construction & Engineering',
    'Metals & Mining', 'Transportation'
]
conservative = [
    'Pharmaceuticals', 'Health Care Equipment & Services',
    'Utilities', 'Independent Power and Renewable Electricity Producers',
    'Food Products', 'Beverages', 'Household Products', 'Banks'
]
balanced = [
    'Industrial Conglomerates', 'Machinery',
    'Trading Companies & Distributors',
    'Commercial Services & Supplies',
    'Transportation Infrastructure'
]
# ===============================

def classify(row):
    industry = row['GICS Industry Name']
    
    # ===== 1️⃣ TĂNG TRƯỞNG (trước đây là Tích cực) =====
    if industry in aggressive:
        # Điều kiện bắt buộc
        if row['ROE'] > 15:
            # Chỉ cần 1 trong 2
            cond_growth = sum([
                row['Beta 5 Year'] >= 1.0,
                row['P/E'] >= 15
            ])
            if cond_growth >= 1:
                return "Tăng trưởng"
    
    # ===== 2️⃣ BẢO THỦ =====
    if industry in conservative:
        # Điều kiện BẮT BUỘC
        if (
            row['Company Market Capitalization'] >= 25_000_000_000_000 and
            row['Dividend Yield - Common - Net - Issue - %, TTM'] >= 1.5
        ):
            # Điều kiện LINH HOẠT (chỉ cần 1 trong 2)
            score_conservative = sum([
                row['Beta 5 Year'] <= 1.0,
                row['ROE'] >= 10
            ])
            if score_conservative >= 2:
                return "Bảo thủ"
    
    # ===== 3️⃣ CÂN BẰNG =====
    # Không áp đặt ràng buộc theo ngành (khác với hai phong cách còn lại).
    # Sự linh hoạt về mặt ngành nghề cho phép hệ thống đề xuất một tập hợp cổ phiếu đa dạng hơn,
    # phù hợp với nhà đầu tư muốn cân bằng giữa tăng trưởng và ổn định mà không bị giới hạn bởi ngành cụ thể.
    score_balanced = sum([
        row['ROE'] > 12,
        0.8 <= row['Beta 5 Year'] <= 1.2,
        row['Dividend Yield - Common - Net - Issue - %, TTM'] > 1.0,
        row['P/E'] > 12
    ])
    if score_balanced >= 3:
        return "Cân bằng"
    
    return "Khác"

# Áp dụng phân loại
required_cols = [
    'ROE', 'Beta 5 Year', 'P/E', 'GICS Industry Name',
    'Dividend Yield - Common - Net - Issue - %, TTM',
    'Company Market Capitalization'
]
fund_df = fund_df.dropna(subset=required_cols)
fund_df['Khau_Vi_Rui_Ro'] = fund_df.apply(classify, axis=1)

# Hàm tính hiệu quả
def expected_return(weights, log_returns):
    return np.sum(log_returns.mean() * weights) * 252 * 100

def standard_deviation(weights, cov_matrix):
    variance = weights.T @ cov_matrix @ weights
    return np.sqrt(variance * 252) * 100

def sharpe_ratio(weights, log_returns, cov_matrix, risk_free_rate):
    ret = expected_return(weights, log_returns) / 100
    vol = standard_deviation(weights, cov_matrix) / 100
    return (ret - risk_free_rate) / vol if vol > 0 else 0

def optimize_portfolio(log_returns_query, risk_free_rate):
    stock_list = log_returns_query.columns.tolist()
    correlation_matrix = log_returns_query.corr()
    std_devs = log_returns_query.std()
    covariance_matrix = correlation_matrix.to_numpy() * np.outer(std_devs, std_devs)
    constraints = {'type': 'eq', 'fun': lambda weights: np.sum(weights) - 1}
    bounds = [(0, 0.4) for _ in range(len(stock_list))]
    initial_weights = np.array([1/len(stock_list)] * len(stock_list))
    optimized_results = minimize(lambda w: -sharpe_ratio(w, log_returns_query, covariance_matrix, risk_free_rate),
                                 initial_weights, method='SLSQP', constraints=constraints, bounds=bounds)
    return optimized_results.x if optimized_results.success else initial_weights

# Load benchmark
@st.cache_data
def load_benchmark():
    files = {
        'VN-Index': "Dữ liệu Lịch sử VN Index.csv",
        'VN30': "Dữ liệu Lịch sử VN 30.csv",
        'VN100': "Dữ liệu Lịch sử VN100.csv"
    }
    
    benchmarks = {}
    for name, file in files.items():
        try:
            df = pd.read_csv(file, quoting=1, quotechar='"', doublequote=True)
            df['Ngày'] = pd.to_datetime(df['Ngày'], format='%d/%m/%Y')
            df = df.sort_values('Ngày')
            df['Close'] = pd.to_numeric(df['Lần cuối'].str.replace(',', ''))
            df['log_return'] = np.log(df['Close'] / df['Close'].shift(1))
            log_r = df['log_return'].dropna()
            ret = log_r.mean() * 252 * 100
            vol = log_r.std() * np.sqrt(252) * 100
            sharpe = (ret/100 - 0.0418) / (vol/100) if vol > 0 else 0
            benchmarks[name] = (round(ret, 1), round(vol, 1), round(sharpe, 2))
        except Exception as e:
            benchmarks[name] = (0, 0, 0)  # Nếu lỗi file, bỏ qua benchmark
    return benchmarks

benchmarks = load_benchmark()

# Risk-free rate
risk_free_rate = 0.0418
st.sidebar.success("Risk-free rate VN (TPCP 10 năm): 4.18% (dữ liệu 30/12/2025)")

# Giao diện
st.title("🎯 Bảng Điều Khiển Danh Mục Đầu Tư Tối Ưu (Bluechip VN)")

khau_vi = st.sidebar.selectbox("Phong cách đầu tư", ["Bảo thủ", "Cân bằng", "Tăng trưởng"])

filtered = fund_df[fund_df['Khau_Vi_Rui_Ro'] == khau_vi].set_index('Symbol')

if filtered.empty:
    st.info(f"Không có cổ phiếu nào thuộc phong cách '{khau_vi}'.")
else:
    st.markdown(f"<h2 style='color: #2ca02c;'>Cổ phiếu phù hợp cho {khau_vi}</h2>", unsafe_allow_html=True)
    st.dataframe(filtered[['Company Common Name', 'ROE', 'Beta 5 Year', 'Dividend Yield - Common - Net - Issue - %, TTM', 'P/E']])
    
    symbols = [s for s in filtered.index if s in log_return.columns]
    
    if len(symbols) < 3:
        st.info(f"Chỉ có {len(symbols)} cổ phiếu – chưa đủ để tối ưu danh mục đa dạng.")
    else:
        log_returns_query = log_return[symbols]
        optimal_weights = optimize_portfolio(log_returns_query, risk_free_rate)
        
        portfolio_df = pd.DataFrame({'Ticker': symbols, 'Weight': optimal_weights})
        portfolio_df = portfolio_df[portfolio_df['Weight'] > 0.0001].sort_values('Weight', ascending=False)
        
        merged = portfolio_df.merge(filtered[['Dividend Yield - Common - Net - Issue - %, TTM']], left_on='Ticker', right_index=True)
        dividend_yield_port = (merged['Weight'] * merged['Dividend Yield - Common - Net - Issue - %, TTM']).sum()
        
        optimal_return = expected_return(optimal_weights, log_returns_query)
        optimal_vol = standard_deviation(optimal_weights, log_returns_query.cov())
        optimal_sharpe = sharpe_ratio(optimal_weights, log_returns_query, log_returns_query.cov(), risk_free_rate)
        
        st.markdown("<h2 style='color: #1f77b4;'>Thông tin danh mục tối ưu</h2>", unsafe_allow_html=True)
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Lợi nhuận kỳ vọng", f"{optimal_return:.1f}%")
        col2.metric("Biến động", f"{optimal_vol:.1f}%")
        col3.metric("Sharpe Ratio", f"{optimal_sharpe:.2f}")
        col4.metric("Dividend Yield", f"{dividend_yield_port:.2f}%")
        
        fig = px.pie(portfolio_df, values='Weight', names='Ticker', title='Tỷ trọng danh mục tối ưu',
                     color_discrete_sequence=px.colors.sequential.Viridis)
        fig.update_traces(textinfo='percent+label', pull=[0.1 if w > 0.1 else 0 for w in portfolio_df['Weight']])
        st.plotly_chart(fig, use_container_width=True)
        
        st.markdown("<h2 style='color: #9467bd;'>Ma trận tương quan cổ phiếu trong danh mục</h2>", unsafe_allow_html=True)
        corr = log_returns_query.corr()
        fig_corr = go.Figure(data=go.Heatmap(
            z=corr.values, x=corr.columns, y=corr.index, colorscale='RdBu', zmin=-1, zmax=1,
            text=corr.values.round(2), texttemplate="%{text}", textfont={"size": 10},
            colorbar=dict(title="Correlation")
        ))
        fig_corr.update_layout(title="Ma trận tương quan log-return", height=600, xaxis_title="Cổ phiếu", yaxis_title="Cổ phiếu", xaxis=dict(tickangle=45))
        st.plotly_chart(fig_corr, use_container_width=True)
        
        # So sánh benchmark
        st.markdown("<h2 style='color: #ff7f0e;'>So sánh với Benchmark (2020-2025)</h2>", unsafe_allow_html=True)
        comparison_data = [
            {"Nhóm": khau_vi, "Return (%)": optimal_return, "Vol (%)": optimal_vol, "Sharpe": optimal_sharpe},
            {"Nhóm": "VN-Index", "Return (%)": benchmarks.get('VN-Index', (0,0,0))[0], "Vol (%)": benchmarks.get('VN-Index', (0,0,0))[1], "Sharpe": benchmarks.get('VN-Index', (0,0,0))[2]},
            {"Nhóm": "VN30", "Return (%)": benchmarks.get('VN30', (0,0,0))[0], "Vol (%)": benchmarks.get('VN30', (0,0,0))[1], "Sharpe": benchmarks.get('VN30', (0,0,0))[2]},
            {"Nhóm": "VN100", "Return (%)": benchmarks.get('VN100', (0,0,0))[0], "Vol (%)": benchmarks.get('VN100', (0,0,0))[1], "Sharpe": benchmarks.get('VN100', (0,0,0))[2]},
        ]
        df_comp = pd.DataFrame(comparison_data)
        st.dataframe(df_comp.style.highlight_max(subset=['Sharpe'], color='lightgreen'))
        
        fig = go.Figure()
        for row in comparison_data:
            color = 'red' if row['Nhóm'] == khau_vi else 'gray'
            size = 20 if row['Nhóm'] == khau_vi else 10
            fig.add_trace(go.Scatter(x=[row['Vol (%)']], y=[row['Return (%)']], mode='markers+text',
                                     marker=dict(size=size, color=color), text=row['Nhóm'], textposition="top center"))
        fig.update_layout(title="So sánh Return vs Risk", xaxis_title="Biến động (%)", yaxis_title="Lợi nhuận kỳ vọng (%)")
        st.plotly_chart(fig, use_container_width=True)

# Dark mode CSS
st.markdown("""
<style>
.stApp {background-color: #121212; color: #e0e0e0;}
h1, h2, h3 {color: #ffffff;}
.stDataFrame {background-color: #1e1e1e;}
.stMetric {background-color: #1e1e1e; padding: 15px; border-radius: 10px; box-shadow: 0 4px 12px rgba(0,0,0,0.5); color: #ffffff; border: 1px solid #333333;}
.stMetric > label {color: #aaaaaa;}
.stSidebar {background-color: #0e0e0e;}
</style>
""", unsafe_allow_html=True)
