# streamlit_etf_heatmap.py
import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.express as px
from typing import List

# If you already had a CSS injector, keep it, otherwise just comment this
st.set_page_config(page_title="ETF Screener", layout="wide")
def inject_custom_styles():
    st.markdown(
        """
        <style>
        .stPlotlyChart { border-radius: 12px; }
        </style>
        """,
        unsafe_allow_html=True
    )

inject_custom_styles()

# --- Load ETF universe (adjust path for your system) ---
ETF_universe = pd.read_csv(r'C:\Users\Armstrong Admin\Downloads\VESTED-PROJECT\US ETF UNIVERSE.csv')
st.session_state.etf_universe = ETF_universe

# --- Filters for thematic groups ---
st.session_state.healthcare_etfs = ETF_universe[
    (ETF_universe['ETF Name'].str.contains('Health', case=False, na=False)) &
    (ETF_universe['Inverse'] == "No") &
    (ETF_universe['Leveraged'] == "False")]

st.session_state.semiconductor_etfs = ETF_universe[
    (ETF_universe['ETF Name'].str.contains('Semiconductor', case=False, na=False)) &
    (ETF_universe['Inverse'] == "No") &
    (ETF_universe['Leveraged'] == "False")]

st.session_state.artificial_intelligence_etfs = ETF_universe[
    (ETF_universe['ETF Name'].str.contains('Artificial Intelligence', case=False, na=False)) &
    (ETF_universe['Inverse'] == "No") &
    (ETF_universe['Leveraged'] == "False")]

st.session_state.video_game_etfs = ETF_universe[
    (ETF_universe['ETF Name'].str.contains('Video Game', case=False, na=False)) &
    (ETF_universe['Inverse'] == "No") &
    (ETF_universe['Leveraged'] == "False")]

st.session_state.defense_etfs = ETF_universe[
    (ETF_universe['ETF Name'].str.contains('Aerospace & Defense', case=False, na=False)) &
    (ETF_universe['Inverse'] == "No") &
    (ETF_universe['Leveraged'] == "False")]

st.session_state.luxury_etfs = ETF_universe[
    (ETF_universe['ETF Name'].str.contains('Luxury', case=False, na=False)) &
    (ETF_universe['Inverse'] == "No") &
    (ETF_universe['Leveraged'] == "False")]

st.session_state.cloud_etfs = ETF_universe[
    (ETF_universe['ETF Name'].str.contains('Cloud', case=False, na=False)) &
    (ETF_universe['Inverse'] == "No") &
    (ETF_universe['Leveraged'] == "False")]

st.session_state.nuclear_etfs = ETF_universe[
    (ETF_universe['ETF Name'].str.contains('Nuclear', case=False, na=False)) &
    (ETF_universe['Inverse'] == "No") &
    (ETF_universe['Leveraged'] == "False")]

# --- Helper: detect the ticker column in your CSV ---
def detect_ticker_column(df: pd.DataFrame) -> str:
    candidates = ['Ticker', 'Symbol', 'ETF Ticker', 'Ticker Symbol', 'ticker', 'symbol']
    for c in candidates:
        if c in df.columns:
            return c
    if 'ETF Name' in df.columns:
        sample = df['ETF Name'].astype(str).iloc[:20].tolist()
        import re
        for s in sample:
            m = re.search(r'\(([A-Z0-9\.\-]{1,6})\)', s)
            if m:
                return 'ETF Name'
    return None

ticker_col = detect_ticker_column(ETF_universe)
if ticker_col is None:
    st.error("No ticker column detected. Make sure your CSV has 'Ticker' or 'Symbol', or tickers in parentheses inside 'ETF Name'.")
    st.stop()

# --- Cached fetch function ---
@st.cache_data(ttl=3600)
def get_etf_history_close(ticker: str, period="1y") -> pd.Series:
    try:
        ticker = str(ticker).strip()
        if ticker_col == 'ETF Name':
            import re
            m = re.search(r'\(([A-Z0-9\.\-]{1,6})\)', ticker)
            if m:
                ticker = m.group(1)
            else:
                ticker = ticker.split()[-1].strip("()")
        t = yf.Ticker(ticker)
        hist = t.history(period=period)['Close']
        return hist
    except Exception:
        return pd.Series(dtype=float)

@st.cache_data(ttl=3600)
def get_etf_returns(etf_tickers: List[str], periods=['1M','3M','6M']) -> pd.DataFrame:
    period_days = {"1M": 21, "3M": 63, "6M": 126}
    rows = []
    for raw in etf_tickers:
        ticker = str(raw).strip()
        hist = get_etf_history_close(ticker, period="1y")
        if hist.empty or len(hist) < 2:
            vals = {p: np.nan for p in periods}
        else:
            latest = hist.iloc[-1]
            vals = {}
            for p in periods:
                days = period_days.get(p, None)
                if days is None or len(hist) <= days:
                    vals[p] = np.nan
                else:
                    past_price = hist.iloc[-days]
                    vals[p] = round((latest / past_price - 1) * 100, 2)
        rows.append((ticker, vals))
    index = [r[0] for r in rows]
    data = {p: [r[1][p] for r in rows] for p in periods}
    df = pd.DataFrame(data, index=index)
    df.index.name = 'Ticker'
    return df

# --- Build sector dict ---
sector_dict = {
    "Video Games": st.session_state.video_game_etfs,
    "Defense": st.session_state.defense_etfs,
    "Luxury": st.session_state.luxury_etfs,
    "Cloud": st.session_state.cloud_etfs,
    "Nuclear": st.session_state.nuclear_etfs,
    "Healthcare": st.session_state.healthcare_etfs,
    "Semiconductors": st.session_state.semiconductor_etfs,
    "Artificial Intelligence": st.session_state.artificial_intelligence_etfs,
}

# --- Build combined dataset for Treemap ---
periods = ['1M','3M','6M']
rows = []

for sector, df in sector_dict.items():
    if ticker_col in df.columns:
        tickers = df[ticker_col].astype(str).tolist()
    elif 'ETF Name' in df.columns:
        tickers = df['ETF Name'].astype(str).tolist()
    else:
        tickers = []

    if not tickers:
        continue

    etf_returns = get_etf_returns(tickers, periods=periods)

    if 'ETF Name' in df.columns and len(etf_returns) == len(df):
        mapping = dict(zip(df[ticker_col].astype(str), df['ETF Name'].astype(str)))
        etf_returns = etf_returns.rename(index=mapping)

    for etf, row in etf_returns.iterrows():
        for p in periods:
            rows.append({
                "Sector": sector,
                "ETF": etf,
                "Period": p,
                "Return (%)": row[p]
            })

combined_df = pd.DataFrame(rows)
combined_df["Size"] = combined_df["Return (%)"].abs().fillna(0) + 1

with st.spinner("Loading data and generating visualizations..."):
    selected_period = st.radio(
    "Select Time Period",
    options=['6M', '3M', '1M'],
    index=0,
    horizontal=True)

# Filter data for selected period
filtered_df = combined_df[combined_df["Period"] == selected_period].copy()

# --- Treemap (now only Sector → ETF, not Period) ---
fig = px.treemap(
    filtered_df,
    path=["Sector", "ETF"],   # Removed Period from hierarchy
    values="Size",
    color="Return (%)",
    color_continuous_scale=px.colors.diverging.RdYlGn,
    color_continuous_midpoint=0,
    hover_data={"Return (%)": True, "Size": False, "Sector": False, "Period": False}
)

fig.update_traces(
    hovertemplate="<b>ETF:</b>%{label}<br><b>Return:</b>%{color:.2f}%<extra></extra>"
)

fig.update_traces(
    textinfo="label+value",      
    textfont=dict(size=20, color="black"),
    marker=dict(line=dict(color="white", width=2)))

fig.update_layout(margin=dict(t=40, l=0, r=0, b=0))

st.plotly_chart(fig, use_container_width=True)

st.markdown("## 📊 ETF Screener")

# --- Filters ---
available_sectors = sorted(combined_df["Sector"].unique())
selected_sectors = st.multiselect(
    "Select Sector(s)",
    options=available_sectors,
    default=available_sectors  # preselect all
)

col1,col2 = st.columns(2)
with col1:
 selected_period = st.radio(
    "Select Time Period",
    options=['1M', '3M', '6M'],
    index=0,
    horizontal=True)
with col2:
 min_return, max_return = st.slider(
    "Return Range (%)",
    min_value=float(combined_df["Return (%)"].min(skipna=True)),
    max_value=float(combined_df["Return (%)"].max(skipna=True)),
    value=(0.0, float(combined_df["Return (%)"].max(skipna=True))),
    step=0.5)

# --- Apply Filters ---
filtered_screen = combined_df[
    (combined_df["Sector"].isin(selected_sectors)) &
    (combined_df["Period"] == selected_period) &
    (combined_df["Return (%)"].between(min_return, max_return))
]

# --- Show Results ---
if filtered_screen.empty:
    st.warning("No ETFs found for the selected criteria.")
else:
    st.data_editor(filtered_screen[["Sector", "ETF", "Return (%)"]].sort_values("Return (%)", ascending=False),hide_index=True,
        use_container_width=True
    )
