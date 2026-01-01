import streamlit as st
import pandas as pd
import numpy as np
from scipy.optimize import minimize
import plotly.express as px
import plotly.graph_objects as go

@st.cache_data
def load_data():
    fund = pd.read_csv("FUNDAMENTAL_FOR_PORTFOLIO.csv")
    
    # Đọc CSV với xử lý dấu ngoặc kép bao header
    price = pd.read_csv(
        "PRICE_FOR_PORTFOLIO.csv",
        header=None,             # Không dùng header tự động
        skiprows=1,              # Bỏ dòng đầu (header bị bao ngoặc kép)
        quoting=1,               # QUOTE_MINIMAL
        quotechar='"',
        doublequote=True
    )
    
    # Gán header thủ công
    price.columns = ['Date', 'Symbol', 'Open', 'High', 'Low', 'Close', 'Volume']
    
    # Chuyển cột Date thành datetime
    price['DATE'] = pd.to_datetime(price['Date'])
    
    # In header để kiểm tra (xóa sau khi chạy ổn)
    st.write("Header price CSV sau khi sửa:", price.columns.tolist())
    
    return fund, price

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

# Danh sách ngành
aggressive = ['IT Services', 'Software', 'Technology Hardware, Storage & Peripherals', 'Retail', 'Textiles, Apparel & Luxury Goods',
              'Oil, Gas & Consumable Fuels', 'Chemicals', 'Real Estate Management & Development', 'Construction & Engineering',
              'Metals & Mining', 'Transportation']

conservative = ['Pharmaceuticals', 'Health Care Equipment & Services', 'Utilities', 'Independent Power and Renewable Electricity Producers',
                'Food Products', 'Beverages', 'Household Products', 'Banks']

# Phân loại khẩu vị rủi ro
def classify(row):
    score_aggressive = sum([
        row['ROE'] > 15,
        row['Beta 5 Year'] > 1.0,
        row['P/E'] > 20,
        row['GICS Industry Name'] in aggressive
    ])
    if score_aggressive >= 3:
        return "Tích cực"
    
    elif (row['Company Market Capitalization'] > 25_000_000_000_000 and
          row['Dividend Yield - Common - Net - Issue - %, TTM'] > 1.5 and
          row['Beta 5 Year'] < 1.2 and row['ROE'] > 10 and
          row['GICS Industry Name'] in conservative):
        return "Bảo thủ"
    
    else:
        score = sum([
            row['ROE'] > 12,
            0.8 <= row['Beta 5 Year'] <= 1.3,
            row['Dividend Yield - Common - Net - Issue - %, TTM'] > 1.0,
            row['P/E'] > 12
        ])
        return "Cân bằng" if score >= 2 else "Khác"

if 'Khau_Vi_Rui_Ro' not in fund_df.columns:
    fund_df = fund_df.dropna(subset=['ROE', 'Beta 5 Year', 'P/E', 'GICS Industry Name',
                                     'Dividend Yield - Common - Net - Issue - %, TTM'])
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

# Risk-free rate
risk_free_rate = 0.0418
st.sidebar.success("Risk-free rate VN (TPCP 10 năm): 4.18% (dữ liệu 30/12/2025)")

# Giao diện
st.title("🎯 Bảng Điều Khiển Danh Mục Đầu Tư Tối Ưu (Bluechip VN)")

st.sidebar.info("Dữ liệu: 41 cổ phiếu bluechip lớn nhất HOSE")

khau_vi = st.sidebar.selectbox("Chọn khẩu vị rủi ro", ["Bảo thủ", "Cân bằng", "Tích cực"])

filtered = fund_df[fund_df['Khau_Vi_Rui_Ro'] == khau_vi].set_index('Symbol')

if filtered.empty:
    st.info(f"Không có cổ phiếu nào thuộc khẩu vị '{khau_vi}'.")
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

        st.markdown("<h2 style='color: #d62728;'>Lợi nhuận tích lũy danh mục</h2>", unsafe_allow_html=True)
        portfolio_cumulative = (log_returns_query + 1).cumprod().dot(optimal_weights)
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=portfolio_cumulative.index, y=portfolio_cumulative, name="Danh mục tối ưu", line=dict(color="#1f77b4", width=2.5)))
        fig.update_layout(title="Lợi nhuận tích lũy danh mục tối ưu", xaxis_title="Ngày", yaxis_title="Lợi nhuận tích lũy", height=500)
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
