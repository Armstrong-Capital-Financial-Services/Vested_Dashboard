import streamlit as st
import pandas as pd
import altair as alt # For the donut charts
import psycopg2
import json
import yfinance as yf
from yahooquery import Ticker
from datetime import timedelta
from crawl4ai import *
import plotly.colors as pc
import asyncio
import redis as redis
import nest_asyncio 
import plotly.express as px
import plotly.graph_objects as go


nest_asyncio.apply()
st.set_page_config(layout="wide", page_title="Financial Comparison")
st.markdown("""
    <style>
        body {
            font-family: 'Inter', sans-serif;
            background-color: #f8f9fa; /* Light grey background for the overall page */
            color: #343a40; /* Dark grey text */
        }

        /* Streamlit Main Container Styling */
        .main .block-container {
            padding-top: 2rem;
            padding-right: 2rem;
            padding-left: 2rem;
            padding-bottom: 2rem;
            max-width: 1280px; /* Equivalent to max-w-7xl */
            margin-left: auto;
            margin-right: auto;
        }
            
        .bold-text {
           font-weight: bold;
           style='font-size: 0.875rem;
           color: #6c757d;' 
        }    

        /* Card Styling (mimics bg-gray-50, rounded-lg, shadow-sm) */
        /* Changed background to white to make border more visible against page background */
        .stContainer {
            background-color: #ffffff; /* White background for cards */
            border-radius: 0.5rem; /* rounded-lg */
            box-shadow: 0 1px 2px 0 rgba(0, 0, 0, 0.05); /* shadow-sm */
            padding: 1.5rem; /* p-6 */
            margin-bottom: 1.5rem; /* mb-6 for spacing between cards if not in columns */
            border: 1px solid #e0e0e0 !important; /* Added border for cards, with !important for priority */
        }

        /* Custom styling for the comparison header 'VS.' text */
        .vs-text {
            color: #6c757d; /* Muted grey */
            font-weight: 600;
            font-size: 0.9rem;
            text-align: center;
        }

        /* Data Row Styling (mimics data-row, data-label, data-value) - KEPT FOR HOLDINGS SECTION */
        .data-row {
            display: flex;
            justify-content: space-between;
            padding-top: 0.5rem;
            padding-bottom: 0.5rem;
            border-bottom: 1px solid #e9ecef; /* Light border for separation */
        }
        .data-row:last-child {
            border-bottom: none; /* No border for the last item */
        }
        .data-label {
            color: #6c757d; /* Muted grey for labels */
        }
        .data-value {
            font-weight: 500;
            color: #343a40; /* Darker for values */
        }

        /* Legend dots */
        .legend-dot {
            width: 8px;
            height: 8px;
            border-radius: 50%;
            display: inline-block;
            margin-right: 0.5rem;
            vertical-align: middle;
        }

        /* Performance values */
        .performance-value-positive {
            color: #28a745; /* Green for positive values */
            font-weight: 500;
        }
        .performance-value-negative {
            color: #dc3545; /* Red for negative values */
            font-weight: 500;
        }

        /* Adjust Streamlit's default elements */
        h1, h2, h3, h4, h5, h6 {
            color: #343a40; /* Dark grey for headings */
        }
            
        /* Table Styling - Clean and professional */
        .data-table {
            width: 100%;
            border-collapse: collapse;
            margin-top: 25px;
            margin-bottom: 30px;
            font-size: 1rem;
            background-color: #ffffff; /* White background for tables */
            border-radius: 12px; /* Rounded corners for the table */
            overflow: hidden; /* Ensures rounded corners on children */
            box-shadow: 0 4px 15px rgba(0, 0, 0, 0.05); /* Subtle shadow for table */
            border: 1px solid #e0e6ed; /* Light border for the table */
        }

        .data-table th, .data-table td {
            padding: 15px 20px; /* More padding for spacious feel */
            text-align: left;
            border-bottom: 1px solid #f0f4f8; /* Very light border for rows */
        }

        .data-table th {
            background-color: #34495e; /* Dark blue-gray header for tables */
            color: #ffffff;
            font-weight: 600;
            text-transform: none !important;
            letter-spacing: 0.5px;
        }

        .data-table tr:nth-child(even) {
            background-color: #f8f9fa; /* Lighter zebra striping */
        }

        .data-table tr:hover {
            background-color: #eef1f4; /* Subtle hover effect for table rows */
        }
            
            
    </style>
""", unsafe_allow_html=True)

db_config = st.secrets["db_config"]
db_config = {
    "user": db_config["user"],
    "password": db_config["password"],
    "host": db_config["host"],
    "port": db_config["port"],
    "dbname": db_config["dbname"]
}

def display_country_exposure_horizontal_bar_chart(data, chart_title="Country Exposure Breakdown"):
    """
    Generates a Plotly horizontal bar chart for country exposure
    and applies the custom styled wrapper.
    """
    df = pd.DataFrame(data)
    df['Percentage_Value'] = df['exposure'].astype(float)
    df = df.sort_values(by='Percentage_Value', ascending=True)

    fig = px.bar(df,x='Percentage_Value',y='country',orientation='h', 
        text='exposure',color='country',
        color_discrete_sequence=px.colors.diverging.curl,)

    fig.update_layout(
        xaxis_title="Exposure (%)",yaxis_title="",showlegend=False,
        yaxis={'categoryorder': 'total ascending'}, # Order bars by value
        margin=dict(l=100, r=50, t=30, b=50))

    fig.update_traces(
        texttemplate='<b>%{text}%</b>',
        textposition='outside',
        hoverinfo='skip',   hovertemplate=None)

    plotly_inner_height = len(df) * 35 + 150 


    styled_plotly_chart_2(fig, chart_title, height=plotly_inner_height)

    
def get_nav_data(ticker):
    """
    Fetches the NAV data for a given ticker using yfinance.
    Returns a DataFrame with cleaned NAV data.
    """
    etf = yf.Ticker(ticker)
    nav_data = etf.history(period="max")
    nav_data.reset_index(inplace=True)
    nav_data['Date'] = pd.to_datetime(nav_data['Date']).dt.strftime('%d-%m-%Y')
    
    # Remove unnecessary columns
    nav_data = nav_data.iloc[:, :-3]
    
    return nav_data

def get_etf_data(tickerlist):
      raw_data = Ticker(tickerlist)
      raw_data_info = raw_data.fund_holding_info
      sectors=(raw_data_info[tickerlist]['sectorWeightings'])
      sectors_flattened = [(list(d.keys())[0], list(d.values())[0]) for d in sectors]
      portfolio_compostion={'Equity': raw_data_info[tickerlist]['stockPosition']*100,'Cash':raw_data_info[tickerlist]['cashPosition']*100,
      'Bonds': raw_data_info[tickerlist]['bondPosition']*100}
      return sectors_flattened,portfolio_compostion

def fetch_table_data(connection, table_name):
    """Fetch data from a PostgreSQL table and return as a Pandas DataFrame."""
    try:
        query = f'SELECT * FROM "{table_name}";'
        with connection.cursor() as cursor:
            cursor.execute(query)
            columns = [desc[0] for desc in cursor.description]
            rows = cursor.fetchall()
            return pd.DataFrame(rows, columns=columns)
    except psycopg2.Error as e:
        print(f"Error fetching data from {table_name}: {e}")
        return pd.DataFrame()
    
# --- Comparison Header Section ---
ETF_selected=st.multiselect("Select ETFs to Compare", 
                options=["SKYY - First Trust Cloud Computing ETF", "SMH - VanEck Semiconductor ETF", "ESPO - VanEck Video Gaming and eSports ETF","NLR - VanEck Uranium and Nuclear ETF",
            "CIBR - First Trust NASDAQ Cybersecurity ETF",
            "PPA - Invesco Aerospace & Defense ETF"],)

col1, col_vs1, col2, col_vs2, col3 = st.columns([1, 0.1, 1, 0.1, 1])

selected_etf_tickers = []
etf_second_name=[]
etf_full_name = []
for option in ETF_selected:
    selected_etf_tickers.append(option.split("-")[0].strip())
    etf_second_name.append(option.split("-")[1].strip())
    etf_full_name.append(option)

if len(selected_etf_tickers) != 0: 
 with col1:
    st.markdown("""<div style=" border: 2px solid #008080;  background-color: #f0f8f8;">""", unsafe_allow_html=True)
    st.markdown("<div style='text-align: center;'>", unsafe_allow_html=True)
    st.markdown(f"<h2 style='font-size: 2.25rem; font-weight: 600; color: #008080;'>{selected_etf_tickers[0]}</h2>", unsafe_allow_html=True) # Teal green color and bigger size
    st.markdown(f'<p class="bold-text">{etf_second_name[0]}</p>', unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

if len(selected_etf_tickers) > 1: 
 with col_vs1:
    st.markdown("<div class='vs-text' style='padding-top: 2rem;'>VS.</div>", unsafe_allow_html=True)

 with col2:
    st.markdown("""<div style=" border: 2px solid #008080;  background-color: #f0f8f8;">""", unsafe_allow_html=True)
    st.markdown("<div style='text-align: center;'>", unsafe_allow_html=True)
    st.markdown(f"<h2 style='font-size: 2.25rem; font-weight: 600; color: #008080;'>{selected_etf_tickers[1]}</h2>", unsafe_allow_html=True) # Teal green color and bigger size
    st.markdown(f'<p class="bold-text">{etf_second_name[1]}</p>', unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)
if len(selected_etf_tickers) > 2:
 with col_vs2:
    st.markdown("<div class='vs-text' style='padding-top: 2rem;'>VS.</div>", unsafe_allow_html=True)

 with col3:
    st.markdown("""<div style=" border: 2px solid #008080;  background-color: #f0f8f8;">""", unsafe_allow_html=True)
    st.markdown("<div style='text-align: center;'>", unsafe_allow_html=True)
    st.markdown(f"<h2 style='font-size: 2.25rem; font-weight: 600; color: #008080;'>{selected_etf_tickers[2]}</h2>", unsafe_allow_html=True) # Teal green color and bigger size
    st.markdown(f'<p class="bold-text">{etf_second_name[2]}</p>', unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

st.markdown("<br><br>", unsafe_allow_html=True) # Add some space

st.markdown(
    """
    <h2 style='font-size: 2.8rem; font-weight: 900; color: #2c3e50; margin-bottom: 1.5rem; font-family: "Times New Roman";'>
        Overview</h2>""",unsafe_allow_html=True)
etf_1, etf_2, etf_3 = st.columns(3)
with psycopg2.connect(**db_config) as connection:
            df = fetch_table_data(connection=connection, table_name="US_ETF_OVERVIEW_DATA")
if len(selected_etf_tickers) != 0: 
      etf_1_overiview_data = json.loads(df[df['ETF_Name'] == etf_full_name[0]]['ETF Overview'].values[0])   
      etf_1_overiview_data_df= pd.DataFrame([etf_1_overiview_data]).T.reset_index().rename(columns={'index': 'Metric', 0: 'Value'})
if len(selected_etf_tickers) > 1: 
      etf_2_overiview_data = json.loads(df[df['ETF_Name'] == etf_full_name[1]]['ETF Overview'].values[0])   
      etf_2_overiview_data_df= pd.DataFrame([etf_2_overiview_data]).T.reset_index().rename(columns={'index': 'Metric', 0: 'Value'})
if len(selected_etf_tickers) > 2: 
       etf_3_overiview_data = json.loads(df[df['ETF_Name'] == etf_full_name[2]]['ETF Overview'].values[0])   
       etf_3_overiview_data_df= pd.DataFrame([etf_3_overiview_data]).T.reset_index().rename(columns={'index': 'Metric', 0: 'Value'})

def render_table_card(title, dataframe):
        with st.container():
            st.markdown(f"<h3 style='font-size: 1.25rem; font-weight: 600; color: #495057; margin-bottom: 1rem;'>{title}</h3>", unsafe_allow_html=True)
            table_html = "<table class='data-table'><thead><tr><th>Metric</th><th>Value</th></tr></thead><tbody>"
            for index, row in dataframe.iterrows():
                table_html += f"<tr><td>{row['Metric']}</td><td>{row['Value']}</td></tr>"
            table_html += "</tbody></table>"
            st.markdown(table_html, unsafe_allow_html=True)

 
if len(selected_etf_tickers) != 0: 
       with etf_1:
         render_table_card(selected_etf_tickers[0], etf_1_overiview_data_df)

if len(selected_etf_tickers) > 1:
       with etf_2:
        render_table_card(selected_etf_tickers[1], etf_2_overiview_data_df)

if len(selected_etf_tickers) > 2:
        with etf_3:
         render_table_card(selected_etf_tickers[2], etf_3_overiview_data_df)



st.markdown(
    """
    <h2 style='font-size: 2.8rem; font-weight: 900; color: #2c3e50; margin-bottom: 1.5rem; font-family: "Times New Roman";'>
        Performance</h2>""",unsafe_allow_html=True)
 
all_etf_nav_data = {}
with st.container(): 
        for ticker in selected_etf_tickers:
            etf = yf.Ticker(f'{ticker}')
            nav_data = etf.history(period="max")
            nav_data.reset_index(inplace=True)
            nav_data['Date'] = pd.to_datetime(nav_data['Date']).dt.strftime('%d-%m-%Y')
            nav_data_df = get_nav_data(ticker)
            nav_data_df['Date'] = pd.to_datetime(nav_data_df['Date'], format='%d-%m-%Y')
            all_etf_nav_data[ticker] = nav_data_df

        combined_df = pd.DataFrame()
        returns_data = []
        for ticker, df in all_etf_nav_data.items():
           temp_df = df.copy()
           temp_df['Ticker'] = ticker # Add a column to identify the ETF
           combined_df = pd.concat([combined_df, temp_df], ignore_index=True)
           combined_df['Date'] = pd.to_datetime(combined_df['Date'], format='%d-%m-%Y')
           latest_date = combined_df['Date'].max()
           timeframes = {"1M": 30,"3M": 90,"6M": 180,"1Y": 365,"3Y": 365 * 3}
           df.sort_values(by='Date', inplace=True)
           df.set_index('Date', inplace=True)
           returns = {'Ticker': ticker}
           latest_price = df['Close'].iloc[-1]  
           for label, days in timeframes.items():
                start_date = latest_date - timedelta(days=days)
                df_filtered = df[df.index <= start_date]
                if not df_filtered.empty:
                    start_price = df_filtered['Close'].iloc[-1]
                    ret = ((latest_price / start_price) - 1) * 100
                    returns[label] = round(ret, 2)
                else:
                   returns[label] = None  # Not enough data
           returns_data.append(returns)
        if len(combined_df) > 0:
         comparative_fig = px.line( combined_df, x='Date', y='Close', color='Ticker', labels={'Date': 'Date', 'NAV': 'Net Asset Value', 'Ticker': 'ETF Ticker'} )

         comparative_fig.update_xaxes(rangeslider_visible=False, rangeselector=dict(
         buttons=list([
            dict(count=1, label="1M", step="month", stepmode="backward"),
            dict(count=3, label="3M", step="month", stepmode="backward"),
            dict(count=6, label="6M", step="month", stepmode="backward"),
            dict(count=1, label="YTD", step="year", stepmode="todate"),
            dict(count=1, label="1Y", step="year", stepmode="backward"),
            dict(count=3, label="3Y", step="year", stepmode="backward"),
         ])))
         comparative_fig.update_traces(
         hovertemplate=(
            "<b>Date:</b> %{x|%d-%m-%Y}<br>"  
            "<b>Ticker:</b> %{fullData.name}<br>" 
            "<b>NAV:</b> %{y:.2f}<extra></extra>" ))
         comparative_fig.update_layout(hovermode="x unified", font=dict(
            family="Outfit, sans-serif", size=12, color="#333",weight='bold' )) 
         st.plotly_chart(comparative_fig, use_container_width=True)

timeframes = { "1M": 30, "3M": 90, "6M": 180, "1Y": 365, "3Y": 365 * 3}
    
st.markdown("<br>", unsafe_allow_html=True) # Add some space
col_qqq_p, col_smh_p, col_aiq_p = st.columns(3)
if  len(selected_etf_tickers)>0:
      etf_1_performance_df = pd.DataFrame(returns_data[0], index=[0]).iloc[:, 1:].T.reset_index().rename(columns={'index': 'Timeframe'})
      etf_1_performance_df = etf_1_performance_df.rename(columns={0: 'Value'})
      
if len(selected_etf_tickers)>1:
        etf_2_performance_df = pd.DataFrame(returns_data[1], index=[0]).iloc[:, 1:].T.reset_index().rename(columns={'index': 'Timeframe'})
        etf_2_performance_df = etf_2_performance_df.rename(columns={0: 'Value'})

if len(selected_etf_tickers)>2:
       etf_3_performance_df = pd.DataFrame(returns_data[2], index=[0]).iloc[:, 1:].T.reset_index().rename(columns={'index': 'Timeframe'})
       etf_3_performance_df = etf_3_performance_df.rename(columns={0: 'Value'})


def render_performance_table_card(title, dataframe):
            with st.container():
                st.markdown(f"<h3 style='font-size: 1.25rem; font-weight: 600; color: #495057; margin-bottom: 1rem;'>{title}</h3>", unsafe_allow_html=True)
                table_html = "<table class='data-table'><thead><tr><th>Timeframe</th><th>Value</th></tr></thead><tbody>"
                for index, row in dataframe.iterrows():
                    # Determine class for positive/negative values
                    try:
                        numeric_value = float(row['Value'])
                        value_class = "performance-value-positive" if numeric_value >= 0 else "performance-value-negative"
                    except ValueError: 
                        value_class = ""
                    table_html += f"<tr><td>{row['Timeframe']}</td><td class='{value_class}'>{row['Value']}</td></tr>"
                table_html += "</tbody></table>"
                st.markdown(table_html, unsafe_allow_html=True)
if len(selected_etf_tickers)>0:
   with col_qqq_p:
            render_performance_table_card(selected_etf_tickers[0], etf_1_performance_df)
if len(selected_etf_tickers)>1:
   with col_smh_p:
            render_performance_table_card(selected_etf_tickers[1], etf_2_performance_df)
if len(selected_etf_tickers)>2:
    with col_aiq_p:
            render_performance_table_card(selected_etf_tickers[2], etf_3_performance_df)      

tab_sectors , tab_holdings , tab_country_exposure = st.tabs(["Sectors","Holdings","Country Exposure"])

with tab_sectors:
    st.markdown(
    """
    <h2 style='font-size: 2.8rem; font-weight: 900; color: #2c3e50; margin-bottom: 1.5rem; font-family: "Times New Roman";'>
        Sectors
    </h2> """, unsafe_allow_html=True)
 
    all_etf_sectors = {}
    all_etf_portfolio_composition = {}
    
    sector_rename_map = {'realestate':'Real Estate',
                             'consumer_cyclical':'Consumer Cyclical',
                            'basic_materials':'Basic Materials',
                             'consumer_defensive':'Consumer Defensive',
                              'technology':'Technology',
                               'communication_services':'Communication Services',
                                'financial_services':'Financial Services',
                                 'utilities':'Utilities',
                                  'industrials':'Industrials',
                                   'energy':'Energy',
                                    'healthcare':'Healthcare' }  
    etf_sectors_data = []
    with st.container():
        for ticker in selected_etf_tickers:
            sectors_flattened, portfolio_composition = get_etf_data(ticker)
            all_etf_sectors[ticker] = sectors_flattened
            data_to_rename = all_etf_sectors[ticker][0]
            rename_and_format = lambda x: (
            sector_rename_map.get(x[0], x[0]),
            f"{float(x[1]) * 100:.2f}%")
            all_etf_sectors[ticker] = [
            ((lambda s: sector_rename_map.get(s, s))(sector),
                f"{float(weight) * 100:.2f}%")for sector, weight in sectors_flattened ]
            etf_sectors_data.append(all_etf_sectors[ticker])

        plot_data = []

        for idx, etf in enumerate(etf_sectors_data):
             df = pd.DataFrame(etf, columns=['Sector', 'Weight'])
             df['Weight'] = df['Weight'].str.replace('%', '').astype(float) 
             df['ETF'] = selected_etf_tickers[idx] 
             plot_data.append(df)

        combined_sector_df = pd.concat(plot_data, ignore_index=True)
        combined_sector_df = combined_sector_df[combined_sector_df['Weight'] > 0]

        unique_etfs = combined_sector_df['ETF'].unique().tolist()
        color_map = dict(zip(unique_etfs, px.colors.diverging.Temps))
        fig = go.Figure()
        for i, etf in enumerate(unique_etfs):
          df_etf = combined_sector_df[combined_sector_df['ETF'] == etf]
          fig.add_trace(go.Bar( name=etf, x=df_etf['Sector'], y=df_etf['Weight'],
        text=[f'{val:.1f}%' for val in df_etf['Weight']],
        textposition='outside', marker_color=color_map[etf], marker_line=dict(width=0.5, color='white')))

          fig.update_layout(barmode='group',height=600,font=dict(family='Segoe UI, sans-serif', size=14, color='#333'),
    margin=dict(t=70, b=100, l=60, r=40), plot_bgcolor='#f9f9f9', paper_bgcolor='#ffffff',
    xaxis=dict( title=dict(text='Sector', font=dict(weight='bold')),    showgrid=False,    categoryorder='total descending',tickfont=dict(weight='bold')),
    yaxis=dict( title=dict(text='Total Sector Allocation(%)', font=dict(weight='bold')), gridcolor='rgba(200, 200, 200, 0.3)',   ticksuffix='%',zeroline=True,zerolinewidth=1.3,tickfont=dict(weight='bold')),
    legend=dict(  orientation='h',   yanchor='bottom',   y=1.02,   xanchor='right', x=1 ))
        st.plotly_chart(fig, use_container_width=True)        

        col_qqq_s, col_smh_s, col_aiq_s = st.columns([1, 1, 1])
        columns = st.columns(len(selected_etf_tickers))
        def render_sector_table_card(sectors, column_object):
            with column_object:
                table_html = "<table class='data-table'><thead><tr><th>Sector</th><th>Weight (%)</th></tr></thead><tbody>"
                sorted_sectors = sorted(sectors, key=lambda x: x[1], reverse=True)
                for sector, weight in sorted_sectors:
                    table_html += f"<tr><td>{sector}</td><td>{weight}</td></tr>"
                table_html += "</tbody></table>"
                st.markdown(table_html, unsafe_allow_html=True)
      

        for i, ticker in enumerate(selected_etf_tickers):
            if ticker in all_etf_sectors: 
                render_sector_table_card(all_etf_sectors[ticker], columns[i]) 


    with tab_holdings:
       col_qqq_h, col_smh_h, col_aiq_h = st.columns(3)

       holdings_qqq = pd.DataFrame({
        'Company': ['NVIDIA Corporation', 'Microsoft Corporation', 'Apple Inc.', 'Amazon.com, Inc.', 'Broadcom Inc.',
                    'Meta Platforms Inc Class A', 'Netflix, Inc.', 'Tesla, Inc.', 'Costco Wholesale Corporation', 'Alphabet Inc. Class A'],
        'Percentage': [8.58, 8.55, 7.56, 5.44, 5.02, 3.73, 3.23, 2.99, 2.85, 2.43],
        'Color': ['#4a90e2', '#28a745', '#9333ea', '#f0ad4e', '#d9534f', '#6c757d', '#fd7e14', '#17a2b8', '#6610f2', '#20c997']
    })

       holdings_smh = pd.DataFrame({
        'Company': ['NVIDIA Corporation', 'Taiwan Semiconductor Manufactur...', 'Broadcom Inc.', 'Advanced Micro Devices, Inc.', 'ASML Holding NV Sponsored ADR',
                    'Applied Materials, Inc.', 'Texas Instruments Incorporated', 'QUALCOMM Incorporated', 'Micron Technology, Inc.', 'Analog Devices, Inc.'],
        'Percentage': [21.49, 11.37, 10.18, 4.64, 4.63, 4.28, 4.28, 4.26, 4.26, 4.17],
        'Color': ['#4a90e2', '#28a745', '#9333ea', '#f0ad4e', '#d9534f', '#6c757d', '#fd7e14', '#17a2b8', '#6610f2', '#20c997']
    })

       holdings_aiq = pd.DataFrame({
        'Company': ['Tencent Holdings Ltd', 'Netflix, Inc.', 'Palantir Technologies Inc. Class A', 'Samsung Electronics Co., Ltd.', 'Alibaba Group Holding Limited ...',
                    'Broadcom Inc.', 'Meta Platforms Inc Class A', 'Microsoft Corporation', 'Cisco Systems, Inc.', 'International Business Machines...'],
        'Percentage': [3.86, 3.71, 3.52, 3.32, 3.23, 3.21, 3.14, 3.11, 3.10, 2.97],
        'Color': ['#4a90e2', '#28a745', '#9333ea', '#f0ad4e', '#d9534f', '#6c757d', '#fd7e14', '#17a2b8', '#6610f2', '#20c997']
    })

    def create_donut_chart(data_df, title_text):
       data_df['start_angle'] = data_df['Percentage'].cumsum() - data_df['Percentage']
       data_df['end_angle'] = data_df['Percentage'].cumsum()

       base = alt.Chart(data_df).encode(theta=alt.Theta("Percentage", stack=True))

       pie = base.mark_arc(outerRadius=60, innerRadius=40).encode(
        color=alt.Color("Color", scale=None), 
        order=alt.Order("Percentage", sort="descending"),tooltip=["Company", "Percentage"] )

       center_text_data = pd.DataFrame([{'text': title_text}])

       center_text = alt.Chart(center_text_data).mark_text(
        align='center',
        baseline='middle',
        fontSize=18, fontWeight='bold',
        color='black'  ).encode(text=alt.Text("text:N"))
       
       chart = (pie + center_text).properties(title=title_text )
       return chart

    with col_qqq_h:
        with st.container():
          if len(selected_etf_tickers)>0:
            donut_chart_qqq = create_donut_chart(holdings_qqq, selected_etf_tickers[0])
            st.altair_chart(donut_chart_qqq, use_container_width=True)

            st.markdown("<ul style='list-style: none; padding: 0;'>", unsafe_allow_html=True)
            for index, row in holdings_qqq.iterrows():
                st.markdown(f"""
                    <li style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 0.5rem;">
                        <div style="display: flex; align-items: center;">
                            <span class="legend-dot" style="background-color: {row['Color']};"></span> {row['Company']}
                        </div>
                        <span style="font-weight: 500;">{row['Percentage']:.2f}%</span>
                    </li>
                """, unsafe_allow_html=True)
            st.markdown("</ul>", unsafe_allow_html=True)

    with col_smh_h:
        with st.container():
          if len(selected_etf_tickers)>1:  
            donut_chart_smh = create_donut_chart(holdings_smh, selected_etf_tickers[1])
            st.altair_chart(donut_chart_smh, use_container_width=True)

            st.markdown("<ul style='list-style: none; padding: 0;'>", unsafe_allow_html=True)
            for index, row in holdings_smh.iterrows():
                st.markdown(f"""
                    <li style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 0.5rem;">
                        <div style="display: flex; align-items: center;">
                            <span class="legend-dot" style="background-color: {row['Color']};"></span> {row['Company']}
                        </div>
                        <span style="font-weight: 500;">{row['Percentage']:.2f}%</span>
                    </li>
                """, unsafe_allow_html=True)
            st.markdown("</ul>", unsafe_allow_html=True)

    with col_aiq_h:
        with st.container():
          if len(selected_etf_tickers)>2:
            donut_chart_aiq = create_donut_chart(holdings_aiq, selected_etf_tickers[2])
            st.altair_chart(donut_chart_aiq, use_container_width=True)

            st.markdown("<ul style='list-style: none; padding: 0;'>", unsafe_allow_html=True)
            for index, row in holdings_aiq.iterrows():
                st.markdown(f"""
                    <li style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 0.5rem;">
                        <div style="display: flex; align-items: center;">
                            <span class="legend-dot" style="background-color: {row['Color']};"></span> {row['Company']}
                        </div>
                        <span style="font-weight: 500;">{row['Percentage']:.2f}%</span>
                    </li>
                """, unsafe_allow_html=True)
            st.markdown("</ul>", unsafe_allow_html=True)

with tab_country_exposure:
    st.markdown( """
    <h2 style='font-size: 2.8rem; font-weight: 900; color: #2c3e50; margin-bottom: 1.5rem; font-family: "Times New Roman";'>
        Country Exposure
    </h2> """,unsafe_allow_html=True)
    def render_country_exposure_card(dataframe):
        with st.container():
            table_html = "<table class='data-table'><thead><tr><th>Country</th><th>Exposure</th></tr></thead><tbody>"
            for index, row in dataframe.iterrows():
                # Render flag and country name together
                flag_url = row['flag']  # URL to flag image
                flag_and_country = f"""
                    <div style='display: flex; align-items: center;'>
                        <img src='{flag_url}' alt='flag' style='width: 20px; height: 14px; margin-right: 8px;'>
                        <span>{row['country']}</span>
                    </div>
                """

                exposure = row['exposure']

                table_html += f"<tr><td>{flag_and_country}</td><td>{exposure}</td></tr>"
            table_html += "</tbody></table>"
            st.markdown(table_html, unsafe_allow_html=True)
    country_data_list=[]
    for ticker in selected_etf_tickers:
        r = redis.Redis(
            host="redis-14320.c305.ap-south-1-1.ec2.redns.redis-cloud.com",
            port=14320,
            username="default",
            password="sZPzjhoSF6Hcz15tX3zReq3zlqIluqJR"
        )
        redis_key = f"country_exposure:{ticker}"
        data = r.get(redis_key)
        json_string = data.decode('utf-8')
        country_data_list.append(json.loads(json_string))

    combined_data = []
    for i, data in enumerate(country_data_list):
       df = pd.DataFrame(data)
       df['ETF'] = selected_etf_tickers[i]  # Label by ETF
       combined_data.append(df)

    combined_df = pd.concat(combined_data, ignore_index=True)
    combined_df['exposure'] = combined_df['exposure'].astype(str).str.replace('%', '').astype(float)
    pivot_df = combined_df.pivot_table(index='ETF', columns='country', values='exposure', fill_value=0)
    pivot_df['Total'] = pivot_df.sum(axis=1)
    pivot_df = pivot_df.sort_values(by='Total', ascending=False).drop(columns='Total')
    color_palette = pc.diverging.Tropic
    country_list = pivot_df.columns.tolist()
    custom_colors = [
    '#8DA0CB',  # muted blue
    '#A6D854',  # soft lime green
    '#FFD92F',  # soft yellow
    '#E5C494',  # light brown
    '#B3B3B3',  # gray
    '#66C2A5',  # sea green
    '#FC8D62',  # salmon
    '#E78AC3',  # dusty pink
    '#A6CEE3',  # light blue
    '#B2DF8A',  # pale green
    '#FDBF6F',  # peach
    '#CAB2D6',  # light violet
    '#FFFF99',  # soft yellow highlight
    '#1F78B4',  # muted navy blue
    '#33A02C', ] 
    colors = {country: custom_colors[i % len(custom_colors)] for i, country in enumerate(country_list)}
    fig = go.Figure()
    for country in pivot_df.columns:
        fig.add_trace(go.Bar(
        name=country,
        y=pivot_df.index,  
        x=pivot_df[country],
        orientation='h',width=0.5, marker_color=colors[country]))

    fig.update_layout(barmode='stack', xaxis_title='Exposure (%)', yaxis_title='ETF',
    font=dict(family='Segoe UI', size=14, color="#333333",weight='bold'), 
    height=600,  margin=dict(l=150, r=50, t=80, b=50), 
    legend=dict(orientation="h", yanchor="bottom",  y=1.02,  xanchor="right",  x=1 ),plot_bgcolor='#f5f5f5', paper_bgcolor='#ffffff' ) 
    st.plotly_chart(fig)

    col1,col2,col3= st.columns(3)
    if selected_etf_tickers:
      with col1:
         if len(selected_etf_tickers)>0:
          st.markdown(f"<h3 style='font-size: 1.25rem; font-weight: 600; color: #495057; margin-bottom: 1rem;'>{selected_etf_tickers[0]}</h3>", unsafe_allow_html=True)
          df=pd.DataFrame(country_data_list[0])
          render_country_exposure_card(df)
      with col2:
         if len(selected_etf_tickers)>1:
          st.markdown(f"<h3 style='font-size: 1.25rem; font-weight: 600; color: #495057; margin-bottom: 1rem;'>{selected_etf_tickers[1]}</h3>", unsafe_allow_html=True)
          df=pd.DataFrame(country_data_list[1])
          render_country_exposure_card(df)  
      with col3:
           if len(selected_etf_tickers)>2:
            st.markdown(f"<h3 style='font-size: 1.25rem; font-weight: 600; color: #495057; margin-bottom: 1rem;'>{selected_etf_tickers[2]}</h3>", unsafe_allow_html=True)
            df=pd.DataFrame(country_data_list[2])
            render_country_exposure_card(df)
