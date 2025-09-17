import streamlit as st
import pandas as pd
import altair as alt
import psycopg2
import json
import yfinance as yf
from yahooquery import Ticker
from datetime import timedelta
import plotly.express as px
import plotly.graph_objects as go
import redis
import nest_asyncio

nest_asyncio.apply()
st.set_page_config(layout="wide", page_title="Financial Comparison")

# CSS styling
st.markdown("""
    <style>
        body {
            font-family: 'Inter', sans-serif;
            background-color: #f8f9fa;
            color: #343a40;
        }
        .main .block-container {
            padding-top: 2rem;
            padding-right: 2rem;
            padding-left: 2rem;
            padding-bottom: 2rem;
            max-width: 1280px;
            margin-left: auto;
            margin-right: auto;
        }
        .bold-text {
            font-weight: bold;
            font-size: 0.875rem;
            color: #6c757d;
        }
        .stContainer {
            background-color: #ffffff;
            border-radius: 0.5rem;
            box-shadow: 0 1px 2px 0 rgba(0, 0, 0, 0.05);
            padding: 1.5rem;
            margin-bottom: 1.5rem;
            border: 1px solid #e0e0e0 !important;
        }
        .vs-text {
            color: #6c757d;
            font-weight: 600;
            font-size: 0.9rem;
            text-align: center;
        }
        .data-row {
            display: flex;
            justify-content: space-between;
            padding-top: 0.5rem;
            padding-bottom: 0.5rem;
            border-bottom: 1px solid #e9ecef;
        }
        .data-row:last-child {
            border-bottom: none;
        }
        .data-label {
            color: #6c757d;
        }
        .data-value {
            font-weight: 500;
            color: #343a40;
        }
        .legend-dot {
            width: 8px;
            height: 8px;
            border-radius: 50%;
            display: inline-block;
            margin-right: 0.5rem;
            vertical-align: middle;
        }
        .performance-value-positive {
            color: #28a745;
            font-weight: 500;
        }
        .performance-value-negative {
            color: #dc3545;
            font-weight: 500;
        }
        h1, h2, h3, h4, h5, h6 {
            color: #343a40;
        }
        .data-table {
            width: 100%;
            border-collapse: collapse;
            margin-top: 25px;
            margin-bottom: 30px;
            font-size: 1rem;
            background-color: #ffffff;
            border-radius: 12px;
            overflow: hidden;
            box-shadow: 0 4px 15px rgba(0, 0, 0, 0.05);
            border: 1px solid #e0e6ed;
        }
        .data-table th, .data-table td {
            padding: 15px 20px;
            text-align: left;
            border-bottom: 1px solid #f0f4f8;
        }
        .data-table th {
            background-color: #34495e;
            color: #ffffff;
            font-weight: 600;
            text-transform: none !important;
            letter-spacing: 0.5px;
        }
        .data-table tr:nth-child(even) {
            background-color: #f8f9fa;
        }
        .data-table tr:hover {
            background-color: #eef1f4;
        }
    </style>
""", unsafe_allow_html=True)

# Database configuration
db_config = st.secrets["db_config"]
db_config = {
    "user": db_config["user"],
    "password": db_config["password"],
    "host": db_config["host"],
    "port": db_config["port"],
    "dbname": db_config["dbname"]
}

# Helper functions
def get_nav_data(ticker):
    etf = yf.Ticker(ticker)
    nav_data = etf.history(period="max")
    nav_data.reset_index(inplace=True)
    nav_data['Date'] = pd.to_datetime(nav_data['Date']).dt.strftime('%d-%m-%Y')
    nav_data = nav_data.iloc[:, :-3]
    return nav_data

def get_etf_data(tickerlist):
    raw_data = Ticker(tickerlist)
    raw_data_info = raw_data.fund_holding_info
    sectors = raw_data_info[tickerlist]['sectorWeightings']
    sectors_flattened = [(list(d.keys())[0], list(d.values())[0]) for d in sectors]
    portfolio_composition = {
        'Equity': raw_data_info[tickerlist]['stockPosition'] * 100,
        'Cash': raw_data_info[tickerlist]['cashPosition'] * 100,
        'Bonds': raw_data_info[tickerlist]['bondPosition'] * 100
    }
    return sectors_flattened, portfolio_composition

def fetch_table_data(connection, table_name):
    try:
        query = f'SELECT * FROM "{table_name}"'
        with connection.cursor() as cursor:
            cursor.execute(query)
            columns = [desc[0] for desc in cursor.description]
            rows = cursor.fetchall()
            return pd.DataFrame(rows, columns=columns)
    except psycopg2.Error as e:
        st.error(f"Error fetching data from {table_name}: {e}")
        return pd.DataFrame()

def render_table_card(title, dataframe):
    with st.container():
        st.markdown(f"<h3 style='font-size: 1.25rem; font-weight: 600; color: #495057; margin-bottom: 1rem;'>{title}</h3>", unsafe_allow_html=True)
        table_html = "<table class='data-table'><thead><tr><th>Metric</th><th>Value</th></tr></thead><tbody>"
        for _, row in dataframe.iterrows():
            table_html += f"<tr><td>{row['Metric']}</td><td>{row['Value']}</td></tr>"
        table_html += "</tbody></table>"
        st.markdown(table_html, unsafe_allow_html=True)

def render_performance_table_card(title, dataframe):
    with st.container():
        st.markdown(f"<h3 style='font-size: 1.25rem; font-weight: 600; color: #495057; margin-bottom: 1rem;'>{title}</h3>", unsafe_allow_html=True)
        table_html = "<table class='data-table'><thead><tr><th>Timeframe</th><th>Value</th></tr></thead><tbody>"
        for _, row in dataframe.iterrows():
            value_class = "performance-value-positive" if pd.notnull(row['Value']) and float(row['Value']) >= 0 else "performance-value-negative"
            table_html += f"<tr><td>{row['Timeframe']}</td><td class='{value_class}'>{row['Value']}</td></tr>"
        table_html += "</tbody></table>"
        st.markdown(table_html, unsafe_allow_html=True)

def render_sector_table_card(sectors, column_object):
    with column_object:
        table_html = "<table class='data-table'><thead><tr><th>Sector</th><th>Weight (%)</th></tr></thead><tbody>"
        sorted_sectors = sorted(sectors, key=lambda x: float(x[1].replace('%', '')), reverse=True)
        for sector, weight in sorted_sectors:
            table_html += f"<tr><td>{sector}</td><td>{weight}</td></tr>"
        table_html += "</tbody></table>"
        st.markdown(table_html, unsafe_allow_html=True)

def render_country_exposure_card(dataframe):
    with st.container():
        table_html = "<table class='data-table'><thead><tr><th>Country</th><th>Exposure</th></tr></thead><tbody>"
        for _, row in dataframe.iterrows():
            flag_url = row['flag']
            flag_and_country = f"""
                <div style='display: flex; align-items: center;'>
                    <img src='{flag_url}' alt='flag' style='width: 20px; height: 14px; margin-right: 8px;'>
                    <span>{row['country']}</span>
                </div>
            """
            table_html += f"<tr><td>{flag_and_country}</td><td>{row['exposure']}</td></tr>"
        table_html += "</tbody></table>"
        st.markdown(table_html, unsafe_allow_html=True)

def create_donut_chart(data_df, title_text):
    colors = ['#4a90e2', '#28a745', '#9333ea', '#f0ad4e', '#d9534f', '#6c757d', 
              '#fd7e14', '#17a2b8', '#6610f2', '#20c997']
    data_df['Color'] = [colors[i % len(colors)] for i in range(len(data_df))]
    
    data_df['start_angle'] = data_df['Percentage'].cumsum() - data_df['Percentage']
    data_df['end_angle'] = data_df['Percentage'].cumsum()

    base = alt.Chart(data_df).encode(theta=alt.Theta("Percentage", stack=True))

    pie = base.mark_arc(outerRadius=60, innerRadius=40).encode(
        color=alt.Color("Color", scale=None),
        order=alt.Order("Percentage", sort="descending"),
        tooltip=["Company", "Percentage"]
    )

    center_text_data = pd.DataFrame([{'text': title_text}])
    center_text = alt.Chart(center_text_data).mark_text(
        align='center',
        baseline='middle',
        fontSize=18,
        fontWeight='bold',
        color='black'
    ).encode(text=alt.Text("text:N"))

    chart = (pie + center_text).properties(title=title_text)
    return chart

# ETF Selection
ETF_selected = st.multiselect(
    "Select ETFs to Compare",
    options=[
        "SKYY - First Trust Cloud Computing ETF",
        "SMH - VanEck Semiconductor ETF",
        "ESPO - VanEck Video Gaming and eSports ETF",
        "NLR - VanEck Uranium and Nuclear ETF",
        "CIBR - First Trust NASDAQ Cybersecurity ETF",
        "PPA - Invesco Aerospace & Defense ETF",
        "AIQ - Global X Artificial Intelligence"
    ],
    max_selections=3
)

# Extract tickers and names
selected_etf_tickers = [option.split("-")[0].strip() for option in ETF_selected]
etf_second_name = [option.split("-")[1].strip() for option in ETF_selected]
etf_full_name = ETF_selected

# Only render if at least one ETF is selected
if len(selected_etf_tickers) > 0:
    # Comparison Header Section
    num_etfs = len(selected_etf_tickers)
    cols = st.columns([1] + [0.1, 1] * (num_etfs - 1))
    for i, ticker in enumerate(selected_etf_tickers):
        with cols[i * 2]:
            st.markdown("<div style='border: 2px solid #008080; background-color: #f0f8f8;'>", unsafe_allow_html=True)
            st.markdown("<div style='text-align: center;'>", unsafe_allow_html=True)
            st.markdown(f"<h2 style='font-size: 2.25rem; font-weight: 600; color: #008080;'>{ticker}</h2>", unsafe_allow_html=True)
            st.markdown(f"<p class='bold-text'>{etf_second_name[i]}</p>", unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)
        if i < num_etfs - 1:
            with cols[i * 2 + 1]:
                st.markdown("<div class='vs-text' style='padding-top: 2rem;'>VS.</div>", unsafe_allow_html=True)

    st.markdown("<br><br>", unsafe_allow_html=True)

    # Overview Section
    st.markdown(
        """
        <h2 style='font-size: 2.8rem; font-weight: 900; color: #2c3e50; margin-bottom: 1.5rem; font-family: "Times New Roman";'>
            Overview</h2>""", unsafe_allow_html=True)

    with psycopg2.connect(**db_config) as connection:
        df = fetch_table_data(connection=connection, table_name="US_ETF_OVERVIEW_DATA")
    
    overview_dfs = []
    for i, etf_name in enumerate(etf_full_name):
        try:
            etf_data = json.loads(df[df['ETF_Name'] == etf_name]['ETF Overview'].values[0])
            overview_dfs.append(pd.DataFrame([etf_data]).T.reset_index().rename(columns={'index': 'Metric', 0: 'Value'}))
        except (IndexError, KeyError, json.JSONDecodeError):
            st.error(f"Error loading overview data for {etf_name}")
            overview_dfs.append(pd.DataFrame())

    cols = st.columns(num_etfs)
    for i, ticker in enumerate(selected_etf_tickers):
        if not overview_dfs[i].empty:
            with cols[i]:
                render_table_card(ticker, overview_dfs[i])

    # Performance Section
    st.markdown(
        """
        <h2 style='font-size: 2.8rem; font-weight: 900; color: #2c3e50; margin-bottom: 1.5rem; font-family: "Times New Roman";'>
            Performance</h2>""", unsafe_allow_html=True)

    all_etf_nav_data = {}
    combined_df = pd.DataFrame()
    returns_data = []
    for ticker in selected_etf_tickers:
        nav_data_df = get_nav_data(ticker)
        nav_data_df['Date'] = pd.to_datetime(nav_data_df['Date'], format='%d-%m-%Y')
        all_etf_nav_data[ticker] = nav_data_df
        temp_df = nav_data_df.copy()
        temp_df['Ticker'] = ticker
        combined_df = pd.concat([combined_df, temp_df], ignore_index=True)
        
        nav_data_df.sort_values(by='Date', inplace=True)
        nav_data_df.set_index('Date', inplace=True)
        latest_date = nav_data_df.index.max()
        timeframes = {"1M": 30, "3M": 90, "6M": 180, "1Y": 365, "3Y": 365 * 3}
        returns = {'Ticker': ticker}
        latest_price = nav_data_df['Close'].iloc[-1]
        for label, days in timeframes.items():
            start_date = latest_date - timedelta(days=days)
            df_filtered = nav_data_df[nav_data_df.index <= start_date]
            if not df_filtered.empty:
                start_price = df_filtered['Close'].iloc[-1]
                ret = ((latest_price / start_price) - 1) * 100
                returns[label] = round(ret, 2)
            else:
                returns[label] = None
        returns_data.append(returns)

    if not combined_df.empty:
        combined_df['Date'] = pd.to_datetime(combined_df['Date'], format='%d-%m-%Y')
        comparative_fig = px.line(
            combined_df, x='Date', y='Close', color='Ticker',
            labels={'Date': 'Date', 'Close': 'Net Asset Value', 'Ticker': 'ETF Ticker'}
        )
        comparative_fig.update_xaxes(
            rangeslider_visible=False,
            rangeselector=dict(
                buttons=[
                    dict(count=1, label="1M", step="month", stepmode="backward"),
                    dict(count=3, label="3M", step="month", stepmode="backward"),
                    dict(count=6, label="6M", step="month", stepmode="backward"),
                    dict(count=1, label="YTD", step="year", stepmode="todate"),
                    dict(count=1, label="1Y", step="year", stepmode="backward"),
                    dict(count=3, label="3Y", step="year", stepmode="backward"),
                ]
            )
        )
        comparative_fig.update_traces(
            hovertemplate="<b>Date:</b> %{x|%d-%m-%Y}<br><b>Ticker:</b> %{fullData.name}<br><b>NAV:</b> %{y:.2f}<extra></extra>"
        )
        comparative_fig.update_layout(hovermode="x unified", font=dict(family="Outfit, sans-serif", size=12, color="#333"))
        st.plotly_chart(comparative_fig, use_container_width=True)

    performance_dfs = [
        pd.DataFrame(returns, index=[0]).iloc[:, 1:].T.reset_index().rename(columns={'index': 'Timeframe', 0: 'Value'})
        for returns in returns_data
    ]
    
    cols = st.columns(num_etfs)
    for i, ticker in enumerate(selected_etf_tickers):
        with cols[i]:
            render_performance_table_card(ticker, performance_dfs[i])

    # Tabs for Sectors, Holdings, Country Exposure
    tab_sectors, tab_holdings, tab_country_exposure = st.tabs(["Sectors", "Holdings", "Country Exposure"])

    with tab_sectors:
        st.markdown(
            """
            <h2 style='font-size: 2.8rem; font-weight: 900; color: #2c3e50; margin-bottom: 1.5rem; font-family: "Times New Roman";'>
                Sectors</h2>""", unsafe_allow_html=True)

        all_etf_sectors = {}
        sector_rename_map = {
            'realestate': 'Real Estate', 'consumer_cyclical': 'Consumer Cyclical', 'basic_materials': 'Basic Materials',
            'consumer_defensive': 'Consumer Defensive', 'technology': 'Technology', 'communication_services': 'Communication Services',
            'financial_services': 'Financial Services', 'utilities': 'Utilities', 'industrials': 'Industrials',
            'energy': 'Energy', 'healthcare': 'Healthcare'
        }
        etf_sectors_data = []
        for ticker in selected_etf_tickers:
            try:
                sectors_flattened, _ = get_etf_data(ticker)
                all_etf_sectors[ticker] = [
                    (sector_rename_map.get(sector, sector), f"{float(weight) * 100:.2f}%")
                    for sector, weight in sectors_flattened
                ]
                etf_sectors_data.append(all_etf_sectors[ticker])
            except (KeyError, AttributeError):
                st.error(f"Error fetching sector data for {ticker}")
                etf_sectors_data.append([])

        if etf_sectors_data:
            plot_data = []
            for idx, etf in enumerate(etf_sectors_data):
                if etf:
                    df = pd.DataFrame(etf, columns=['Sector', 'Weight'])
                    df['Weight'] = df['Weight'].str.replace('%', '').astype(float)
                    df['ETF'] = selected_etf_tickers[idx]
                    plot_data.append(df)

            if plot_data:
                combined_sector_df = pd.concat(plot_data, ignore_index=True)
                combined_sector_df = combined_sector_df[combined_sector_df['Weight'] > 0]
                unique_etfs = combined_sector_df['ETF'].unique().tolist()
                color_map = dict(zip(unique_etfs, px.colors.diverging.Temps))
                fig = go.Figure()
                for etf in unique_etfs:
                    df_etf = combined_sector_df[combined_sector_df['ETF'] == etf]
                    fig.add_trace(go.Bar(
                        name=etf, x=df_etf['Sector'], y=df_etf['Weight'],
                        text=[f'{val:.1f}%' for val in df_etf['Weight']],
                        textposition='outside', marker_color=color_map[etf],
                        marker_line=dict(width=0.5, color='white')
                    ))
                fig.update_layout(
                    barmode='group', height=600,
                    font=dict(family='Segoe UI, sans-serif', size=14, color='#333'),
                    margin=dict(t=70, b=100, l=60, r=40), plot_bgcolor='#f9f9f9', paper_bgcolor='#ffffff',
                    xaxis=dict(title=dict(text='Sector', font=dict(weight='bold')), showgrid=False, categoryorder='total descending', tickfont=dict(weight='bold')),
                    yaxis=dict(title=dict(text='Total Sector Allocation(%)', font=dict(weight='bold')), gridcolor='rgba(200, 200, 200, 0.3)', ticksuffix='%', zeroline=True, zerolinewidth=1.3, tickfont=dict(weight='bold')),
                    legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1)
                )
                st.plotly_chart(fig, use_container_width=True)

            cols = st.columns(num_etfs)
            for i, ticker in enumerate(selected_etf_tickers):
                if all_etf_sectors[ticker]:
                    render_sector_table_card(all_etf_sectors[ticker], cols[i])

    with tab_holdings:
        st.markdown(
            """
            <h2 style='font-size: 2.8rem; font-weight: 900; color: #2c3e50; margin-bottom: 1.5rem; font-family: "Times New Roman";'>
                Holdings</h2>""", unsafe_allow_html=True)

        cols = st.columns(num_etfs) if num_etfs > 0 else [st.container()]
        for i, ticker in enumerate(selected_etf_tickers):
            with cols[i]:
                try:
                    raw_data = Ticker(ticker)
                    holdings_data = raw_data.fund_holding_info.get(ticker, {}).get('holdings', [])
                    if not holdings_data:
                        st.warning(f"No holdings data available for {ticker}.")
                        continue
                    
                    holdings_df = pd.DataFrame(holdings_data)[['symbol', 'holdingName', 'holdingPercent']]
                    holdings_df['Percentage'] = (holdings_df['holdingPercent'] * 100).round(2)
                    holdings_df['Company'] = holdings_df['holdingName']
                    holdings_df = holdings_df.sort_values(by='Percentage', ascending=False).head(10)
                    
                    donut_chart = create_donut_chart(holdings_df, ticker)
                    st.altair_chart(donut_chart, use_container_width=True)

                    st.markdown("<ul style='list-style: none; padding: 0;'>", unsafe_allow_html=True)
                    for _, row in holdings_df.iterrows():
                        st.markdown(
                            f"""
                            <li style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 0.5rem;">
                                <div style="display: flex; align-items: center;">
                                    <span class="legend-dot" style="background-color: {row['Color']};"></span>
                                    <span>{row['Company']}</span>
                                </div>
                                <span style="font-weight: 500;">{row['Percentage']:.2f}%</span>
                            </li>
                            """,
                            unsafe_allow_html=True
                        )
                    st.markdown("</ul>", unsafe_allow_html=True)
                    
                except Exception as e:
                    st.error(f"Error fetching holdings data for {ticker}: {str(e)}")

    with tab_country_exposure:
        st.markdown(
            """
            <h2 style='font-size: 2.8rem; font-weight: 900; color: #2c3e50; margin-bottom: 1.5rem; font-family: "Times New Roman";'>
                Country Exposure</h2>""", unsafe_allow_html=True)

        country_data_list = []
        r = redis.Redis(
            host="redis-14320.c305.ap-south-1-1.ec2.redns.redis-cloud.com",
            port=14320,
            username="default",
            password="sZPzjhoSF6Hcz15tX3zReq3zlqIluqJR"
        )
        for ticker in selected_etf_tickers:
            redis_key = f"country_exposure:{ticker}"
            data = r.get(redis_key)
            if data:
                json_string = data.decode('utf-8')
                country_data_list.append(json.loads(json_string))
            else:
                country_data_list.append([])

        if country_data_list:
            combined_data = []
            for i, data in enumerate(country_data_list):
                df = pd.DataFrame(data)
                df['ETF'] = selected_etf_tickers[i]
                combined_data.append(df)

            if combined_data:
                combined_df = pd.concat(combined_data, ignore_index=True)
                combined_df['exposure'] = combined_df['exposure'].astype(str).str.replace('%', '').astype(float)
                pivot_df = combined_df.pivot_table(index='ETF', columns='country', values='exposure', fill_value=0)
                pivot_df['Total'] = pivot_df.sum(axis=1)
                pivot_df = pivot_df.sort_values(by='Total', ascending=False).drop(columns='Total')
                color_palette = ['#8DA0CB', '#A6D854', '#FFD92F', '#E5C494', '#B3B3B3', '#66C2A5', '#FC8D62', 
                                 '#E78AC3', '#A6CEE3', '#B2DF8A', '#FDBF6F', '#CAB2D6', '#FFFF99', '#1F78B4', '#33A02C']
                country_list = pivot_df.columns.tolist()
                colors = {country: color_palette[i % len(color_palette)] for i, country in enumerate(country_list)}
                fig = go.Figure()
                for country in pivot_df.columns:
                    fig.add_trace(go.Bar(
                        name=country, y=pivot_df.index, x=pivot_df[country], orientation='h',
                        width=0.5, marker_color=colors[country]
                    ))
                fig.update_layout(
                    barmode='stack', xaxis_title='Exposure (%)', yaxis_title='ETF',
                    font=dict(family='Segoe UI', size=14, color="#333333", weight='bold'),
                    height=600, margin=dict(l=150, r=50, t=80, b=50),
                    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                    plot_bgcolor='#f5f5f5', paper_bgcolor='#ffffff'
                )
                st.plotly_chart(fig, use_container_width=True)

            cols = st.columns(num_etfs)
            for i, ticker in enumerate(selected_etf_tickers):
                with cols[i]:
                    st.markdown(f"<h3 style='font-size: 1.25rem; font-weight: 600; color: #495057; margin-bottom: 1rem;'>{ticker}</h3>", unsafe_allow_html=True)
                    render_country_exposure_card(pd.DataFrame(country_data_list[i]))

else:
    st.info("Please select at least one ETF to display the comparison.")
