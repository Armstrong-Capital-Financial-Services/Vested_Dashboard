import streamlit as st
import pandas as pd
import psycopg2
from supabase import create_client
import plotly.express as px
import plotly.graph_objects as go
from yahooquery import Ticker
from streamlit.components.v1 import html
import yfinance as yf
import json
import redis
import quantstats as qs
from datetime import timedelta
st.set_page_config(layout='wide') 

db_config = st.secrets["db_config"]
db_config = {
    "user": db_config["user"],
    "password": db_config["password"],
    "host": db_config["host"],
    "port": db_config["port"],
    "dbname": db_config["dbname"]
}

SUPABASE_URL = st.secrets["SUPABASE_URL"]
SUPABASE_KEY = st.secrets["SUPABASE_KEY"]
supabase = create_client(SUPABASE_URL, SUPABASE_KEY)


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
    
def get_nav_data(ticker):
    """
    Fetches the NAV data for a given ticker from the Supabase database.
    Returns a DataFrame with the NAV data.
    """
    etf = yf.Ticker(ticker=ticker)
    nav_data = etf.history(period="max")
    nav_data.reset_index(inplace=True)
    nav_data['Date'] = pd.to_datetime(nav_data['Date']).dt.strftime('%d-%m-%Y')
    if ticker.startswith('^'):
        nav_data=nav_data.iloc[:,:-3] 
    else:
       nav_data=nav_data.iloc[:,:-4]

    return nav_data

def inject_custom_styles():
    """
    Injects custom CSS styles into the Streamlit application to brand the dashboard.
    This includes styling for the main app background, headers, text, cards, and tables.
    """
   
    st.markdown("""
    <style>
        /* Import Google Fonts for a more refined typography */
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=Outfit:wght@400;500;600;700&display=swap');

        /* Apply background to entire page with a subtle gradient */
        .stApp {
            background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%) !important; /* Soft light blue-gray gradient */
            color: #2c3e50 !important; /* Dark charcoal for primary text */
            font-family: 'Inter', sans-serif; /* General body font */
        }

        /* Adjust Streamlit specific header elements to blend with the light theme */
        header, .css-18ni7ap.e8zbici2, .css-1dp5vir.e8zbici3 {
            background: #ffffff !important; /* White header background */
            color: #2c3e50 !important;
            box-shadow: none !important;
            border-bottom: 1px solid #e0e6ed; /* Subtle separator */
        }

        /* Fix for text and elements inside Streamlit containers for consistent color */
        .stMarkdown, .stText, .stDataFrame, .stTable, .stExpander, .stSelectbox,
        .stButton > button, .stDownloadButton > button, .stFileUploader, .stTextInput > div > div > input,
        .stTextArea > div > div > textarea, .stDateInput > div > div > input, .stTimeInput > div > div > input,
        .stNumberInput > div > div > input, .stSlider > div > div > div > div {
            color: #2c3e50 !important; /* Dark charcoal for text */
        }

        /* Styled card section - More depth and refinement */
        .etf-section {
            background-color: #ffffff; /* Pure white background for cards */
            border-radius: 18px; /* Slightly more rounded corners */
            padding: 30px;
            margin-bottom: 40px;
            /* Multi-layered shadows for depth */
            box-shadow: 0 10px 30px rgba(0, 0, 0, 0.08), 0 4px 12px rgba(0, 0, 0, 0.05);
            border: 1px solid #e0e6ed; /* Subtle border for definition */
            transition: transform 0.3s ease, box-shadow 0.3s ease;
            animation: fadeInUp 0.8s ease-out forwards;
            opacity: 0; /* Initially hidden for animation */
            max-width: 900px;
            width: 100%;
            margin-left: auto;
            margin-right: auto;
        }

        .etf-section:hover {
            transform: translateY(-7px); /* More pronounced lift on hover */
            box-shadow: 0 15px 40px rgba(0, 0, 0, 0.12), 0 6px 18px rgba(0, 0, 0, 0.08); /* Stronger shadow on hover */
        }

        /* ETF Title */
        .etf-section h2 {
            font-family: 'Outfit', sans-serif; /* Distinct font for main titles */
            font-size: 2.5rem;
            color: #3498db; /* Vibrant blue accent */
            margin-top: 0;
            margin-bottom: 20px;
            border-bottom: 2px solid #e0e6ed; /* Lighter, defined border */
            padding-bottom: 10px;
            font-weight: 600;
        }

        /* Sub-sections within ETF */
        .etf-section h3 {
            font-family: 'Outfit', sans-serif;
            font-size: 1.8rem;
            color: #2980b9; /* Slightly darker blue accent */
            margin-top: 30px;
            margin-bottom: 15px;
            font-weight: 600;
        }

        /* Paragraphs within ETF sections */
        .etf-section p {
            margin-bottom: 15px;
            font-size: 1.1rem;
            line-height: 1.7; /* Improved readability */
            color: #4a6572; /* Darker blue-gray text for body */
        }

        /* Lists within ETF sections */
        .etf-section ul {
            list-style: none;
            padding-left: 0;
            margin-bottom: 20px;
        }

        .etf-section ul li {
            position: relative;
            padding-left: 30px; /* More space for custom bullet */
            margin-bottom: 10px; /* More space between list items */
            font-size: 1.05rem;
            color: #5e7f8d; /* Slightly muted text for list items */
        }

        .etf-section ul li::before {
            content: '➤'; /* Checkmark for custom bullet */
            color: #27ae60; /* Fresh green accent */
            position: absolute;
            left: 0;
            font-weight: bold;
            font-size: 1.2em; /* Larger bullet */
            line-height: 1.2;
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
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }

        .data-table tr:nth-child(even) {
            background-color: #f8f9fa; /* Lighter zebra striping */
        }

        .data-table tr:hover {
            background-color: #eef1f4; /* Subtle hover effect for table rows */
        }

    
        /* Responsive Design (keeping existing for completeness) */
        @media (max-width: 768px) {
            .etf-section h2 {
                font-size: 2rem;
            }
            .etf-section h3 {
                font-size: 1.5rem;
            }
            .etf-section {
                padding: 20px;
            }
            .data-table th, .data-table td {
                padding: 10px 15px;
            }
        }

        @media (max-width: 480px) {
            .etf-section h2 {
                font-size: 1.7rem;
            }
            .etf-section h3 {
                font-size: 1.3rem;
            }
            .etf-section {
                padding: 15px;
            }
        }

        /* Keyframe Animations (keeping existing) */
        @keyframes fadeInDown {
            from {
                opacity: 0;
                transform: translateY(-20px);
            }
            to {
                opacity: 1;
                transform: translateY(0);
            }
        }

        @keyframes fadeInUp {
            from {
                opacity: 0;
                transform: translateY(20px);
            }
            to {
                opacity: 1;
                transform: translateY(0);
            }
        }

        /* Delay for sequential animations (keeping existing) */
        .etf-section:nth-child(2) { animation-delay: 0.2s; }
        .etf-section:nth-child(3) { animation-delay: 0.4s; }
        .etf-section:nth-child(4) { animation-delay: 0.6s; }
        .etf-section:nth-child(5) { animation-delay: 0.8s; }
        .etf-section:nth-child(6) { animation-delay: 1.0s; }
        .etf-section:nth-child(7) { animation-delay: 1.2s; }
        .etf-section:nth-child(8) { animation-delay: 1.4s; }
        .etf-section:nth-child(9) { animation-delay: 1.6s; }

        /* Ensure all headings inside main app have the correct color */
        h1, h4, h5, h6 { /* Exclude h2, h3 which are styled within etf-section */
            color: #3498db !important; /* Primary blue for general headings */
            font-family: 'Outfit', sans-serif;
            font-weight: 600;
        }

        /* Specific overrides for etf-section headings */
        .etf-section h2 {
            color: #3498db !important;
        }
        .etf-section h3 {
            color: #2980b9 !important;
        }

        /* Additional styling for Streamlit's built-in elements for consistency */
        .stButton > button {
            background-color: #3498db; /* Blue button */
            color: white;
            border-radius: 8px;
            padding: 10px 20px;
            font-weight: 500;
            transition: background-color 0.2s ease, transform 0.2s ease;
            box-shadow: 0 4px 10px rgba(52, 152, 219, 0.2);
        }

        .stButton > button:hover {
            background-color: #2980b9; /* Darker blue on hover */
            transform: translateY(-2px);
            box-shadow: 0 6px 15px rgba(52, 152, 219, 0.3);
        }

        .stDownloadButton > button {
            background-color: #27ae60; /* Green download button */
            color: white;
            border-radius: 8px;
            padding: 10px 20px;
            font-weight: 500;
            transition: background-color 0.2s ease, transform 0.2s ease;
            box-shadow: 0 4px 10px rgba(39, 174, 96, 0.2);
        }

        .stDownloadButton > button:hover {
            background-color: #229954; /* Darker green on hover */
            transform: translateY(-2px);
            box-shadow: 0 6px 15px rgba(39, 174, 96, 0.3);
        }

        .stSelectbox > div > div {
            border-radius: 8px;
            border: 1px solid #e0e6ed;
            background-color: #ffffff;
            box-shadow: 0 2px 5px rgba(0,0,0,0.03);
        }        
    </style>
    """, unsafe_allow_html=True)

def calcualate_risk_stats(ticker):   
        add_stats = ['volatility', 'sharpe', 'sortino', 'max_drawdown','recovery_factor']
        all_stats = [f for f in dir(qs.stats) if f[0] != '_']
        required_stats = list(set(all_stats) & set(add_stats))
        required_stats_dict = {stat: getattr(qs.stats, stat) for stat in required_stats}
        nav_data=get_nav_data(ticker=ticker)
        nav_data['Date'] = pd.to_datetime(nav_data['Date'], format='%d-%m-%Y')
        nav_data=nav_data.set_index('Date')
        nav_data =pd.DataFrame(nav_data['Close'].iloc[-252:])
        nav_data_reset = nav_data.reset_index()
        benchmark_1_data = get_nav_data("^GSPC") 
        benchmark_2_data = get_nav_data("^NDX")
        if benchmark_1_data['Date'].dtype == 'object':
              benchmark_1_data['Date'] = pd.to_datetime(benchmark_1_data['Date'],format='%d-%m-%Y')
        if benchmark_2_data['Date'].dtype == 'object':
               benchmark_2_data['Date'] = pd.to_datetime(benchmark_2_data['Date'],format='%d-%m-%Y')
        benchmark_combined_df= pd.merge(left=benchmark_1_data,right=benchmark_2_data,on='Date',how='inner',suffixes=('_benchmark-1','_benchmark-2'))[['Date','Close_benchmark-1','Close_benchmark-2']]
        final_combined_df = pd.merge(left=nav_data_reset,right=benchmark_combined_df,on='Date',how='inner')
        final_combined_df['Close'] = final_combined_df['Close'].astype(float)
        final_combined_df['Close_benchmark-1'] = final_combined_df['Close_benchmark-1'].astype(float)
        final_combined_df = final_combined_df.iloc[:,:-1]
        final_combined_df.set_index('Date',inplace=True)
        smallcase_stats_1yr = {stat: required_stats_dict[stat](final_combined_df) for stat in required_stats}
        smallcase_stats_1yr = pd.DataFrame(smallcase_stats_1yr).T
        return smallcase_stats_1yr

          
  
def get_etf_data(tickerlist):
      raw_data = Ticker(tickerlist)
      raw_data_info = raw_data.fund_holding_info
      etf_parameters = raw_data.fund_equity_holdings
      holdings=(raw_data_info[tickerlist]['holdings'])
      sectors=(raw_data_info[tickerlist]['sectorWeightings'])
      sectors_flattened = [(list(d.keys())[0], list(d.values())[0]) for d in sectors]
      portfolio_compostion={'Equity': raw_data_info[tickerlist]['stockPosition']*100,'Cash':raw_data_info[tickerlist]['cashPosition']*100,
      'Bonds': raw_data_info[tickerlist]['bondPosition']*100}
      return holdings,sectors_flattened,etf_parameters,portfolio_compostion
    
def styled_plotly_chart(fig, title, height=350):
    """Wraps Plotly chart in a styled div with embedded title"""
    plot_html = fig.to_html(include_plotlyjs="cdn", full_html=False)
    rounded_chart = f"""
    <div style="
        background: white;
        border-radius: 12px;
        overflow: hidden;
        box-shadow: 0 4px 12px rgba(0,0,0,0.05);
        margin: 5px 0;
    ">
        <h3 style="
            color: #3498db;
            font-family: 'Outfit', sans-serif;
            margin: 0;
            padding: 15px 20px 10px;
            border-bottom: 1px solid #e0e6ed;
            font-size: 1.5rem;
        ">{title}</h3>

        <div style="padding: 25px 25px 10px;">  <!-- Adds space above chart -->
            {plot_html}
        </div>
    </div>
    """
    html(rounded_chart, height=height + 130)


def styled_plotly_chart_2(fig, title, height=350):
    """Wraps Plotly chart in a styled div with embedded title"""
    # Ensure background is transparent so the custom div background shows
    fig.update_layout(
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(
            family="Outfit, sans-serif", # Ensure consistent font if loaded
            size=12,
            color="#333" ))
        
    plot_html = fig.to_html(include_plotlyjs="cdn", full_html=False, config={'displayModeBar': False})
    total_wrapper_height = height + 260
    rounded_chart = f"""
    <div style="
        background: white;
        border-radius: 12px;
        overflow: hidden;
        box-shadow: 0 4px 12px rgba(0,0,0,0.05);
        margin: 5px 0;
    ">
        <h3 style="
            color: #3498db;
            font-family: 'Outfit', sans-serif;
            margin: 0;
            padding: 15px 20px 10px;
            border-bottom: 1px solid #e0e6ed;
            font-size: 1.5rem;
        ">{title}</h3>

        <div style="padding: 25px 25px 10px;">  {plot_html}
        </div>
    </div>
    """
    html(rounded_chart, height=total_wrapper_height)

def styled_plotly_chart_3(fig, title, height=350):
    """Wraps Plotly chart in a styled div with embedded title"""
    # Ensure background is transparent so the custom div background shows
    fig.update_layout(
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(
            family="Outfit, sans-serif", # Ensure consistent font if loaded
            size=12,
            color="#333"
        )
    )
    plot_html = fig.to_html(include_plotlyjs="cdn", full_html=False, config={'displayModeBar': False})
    total_wrapper_height = height + 130 # Use the fixed offset from your original snippet

    rounded_chart = f"""
    <div style="
        background: white;
        border-radius: 12px;
        overflow: hidden;
        box-shadow: 0 4px 12px rgba(0,0,0,0.05);
        margin: 5px 0;
    ">
        <h3 style="
            color: #3498db;
            font-family: 'Outfit', sans-serif;
            margin: 0;
            padding: 15px 20px 10px;
            border-bottom: 1px solid #e0e6ed;
            font-size: 1.5rem;
        ">{title}</h3>

         <div style="padding: 10px 10px 10px;">
            {plot_html}
        </div>

    </div>
    """
    html(rounded_chart, height=total_wrapper_height)    

def create_pie_chart(sectors_flattened=None, holdings=None):
      """Creates a pie chart with custom styling"""
      sectors_df = pd.DataFrame(sectors_flattened, columns=['Sector', 'Weight'])
      sectors_df['Sector'] = sectors_df['Sector'].str.upper()
      sectors_df=sectors_df[sectors_df['Weight'] > 0]  # Filter out sectors with zero weight
      
      fig = go.Figure(data=[go.Pie(
        labels=sectors_df['Sector'],
        values=sectors_df['Weight'],
        hole=0.35,
        marker=dict(colors=px.colors.diverging.Tropic),
        hovertemplate="Sector: <b>%{label}</b><br>Weight: <b>%{value:.2f}%</b><extra></extra>",
         hoverlabel=dict(bgcolor='white'))])

      fig.update_layout(showlegend=True,  legend=dict( orientation="v", yanchor="middle", xanchor="right",x=1.2))

      if holdings is not None:
          holdings_df = pd.DataFrame(holdings).reset_index().rename(columns={'holdingName': 'Name', 'holdingPercent': 'Weight'})
          holdings_df["Weight"] = holdings_df["Weight"].astype(float)
          holdings_df['Weight'] = holdings_df['Weight'] * 100
          others_weight = 100 - holdings_df["Weight"].sum()
          if others_weight > 0:
             holdings_df = pd.concat(
        [holdings_df, pd.DataFrame([{"Name": "Others", "Weight": others_weight}])],
        ignore_index=True)
          holdings_df['Name'] = holdings_df['Name'].str.upper()
          fig2 = go.Figure(data=[go.Pie(labels=holdings_df['Name'], values=holdings_df['Weight'], hole=0.35,
          marker=dict(colors=px.colors.diverging.delta), hovertemplate="Holding Name: <b>%{label}</b> <br>Weight: <b>%{value:.2f}%</b><extra></extra>",
          hoverlabel=dict(bgcolor='white'))])
                  
          fig.update_layout(showlegend=True, legend=dict(
            orientation="v",
            yanchor="middle", 
            xanchor="right", x=1.2) )       
      return fig, fig2

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


def create_line_chart(data,df_benchmark,df_benchmark_2):
    df = pd.DataFrame(data)
    df['Date'] = pd.to_datetime(df['Date'], format='%d-%m-%Y')
    df_benchmark['Date'] = pd.to_datetime(df_benchmark['Date'], format='%d-%m-%Y')
    latest_common_start_date = max(df['Date'].min(), df_benchmark['Date'].min())
    df_benchmark_2['Date'] = pd.to_datetime(df_benchmark_2['Date'], format='%d-%m-%Y')
    df_aligned = df[df['Date'] >= latest_common_start_date].copy()
    df_benchmark_aligned = df_benchmark[df_benchmark['Date'] >= latest_common_start_date].copy()
    df_benchmark_aligned_2 = df_benchmark_2[df_benchmark_2['Date'] >= latest_common_start_date].copy()

    first_close_df = df_aligned['Close'].iloc[0]
    df_aligned['Normalized_Close_Base'] = (df_aligned['Close'] / first_close_df)*100
    first_close_benchmark = df_benchmark_aligned['Close'].iloc[0]
    df_benchmark_aligned['Normalized_Close_Base'] = (df_benchmark_aligned['Close'] / first_close_benchmark)*100

    first_close_benchmark_2 = df_benchmark_aligned_2['Close'].iloc[0]
    df_benchmark_aligned_2['Normalized_Close_Base'] = (df_benchmark_aligned_2['Close'] / first_close_benchmark_2)*100

    fig = px.line(df_aligned, x=df_aligned['Date'], y=df_aligned['Normalized_Close_Base'])
    fig.add_trace(
            go.Scatter(
                x=df_benchmark_aligned['Date'],
                y=df_benchmark_aligned['Normalized_Close_Base'], 
                mode='lines',
                name='S&P 500', # Name for the legend
                line=dict(color='orange') 
            )
        )
    fig.add_trace(
            go.Scatter(x=df_benchmark_aligned_2['Date'],
                y=df_benchmark_aligned_2['Normalized_Close_Base'],
                mode='lines',       
                name='Nasdaq 100', # Name for the legend
                line=dict(color='green')        
            )
        )

    fig.update_layout(
        xaxis_title="Date",
        yaxis_title="NAV",
        height=350,
        width=1000,
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(family="Outfit, sans-serif", size=12, color="#333"),
        hovermode="x unified")
    
    fig.update_xaxes(
        rangeslider_visible=False,
        rangeselector=dict(
        buttons=list([
            dict(count=1, label="1m", step="month", stepmode="backward"),
            dict(count=3, label="3m", step="month", stepmode="backward"),
            dict(count=6, label="6m", step="month", stepmode="backward"),
            dict(count=1, label="1y", step="year", stepmode="backward"),
            dict(count=2, label="2y", step="year", stepmode="backward"),
            dict(count=3, label="3y", step="year", stepmode="backward"),
            dict(step="all") ])))

    styled_plotly_chart_3(fig, "NAV Over Time")


    df_aligned_benchmark_1 = df_aligned.merge(df_benchmark_aligned, on='Date', suffixes=('', '_Benchmark'))
    df_aligned_benchmark_1 = df_aligned_benchmark_1[['Date', 'Normalized_Close_Base', 'Normalized_Close_Base_Benchmark']]
    df_aligned_benchmark_1['Date'] = pd.to_datetime(df_aligned_benchmark_1['Date'], format='%d-%m-%Y')
    df_aligned_benchmark_1.set_index('Date', inplace=True)
    end_date = df_aligned_benchmark_1.index.max()
  
    end_price_base = df_aligned_benchmark_1.loc[end_date, 'Normalized_Close_Base']
    end_price_benchmark = df_aligned_benchmark_1.loc[end_date, 'Normalized_Close_Base_Benchmark']

      
    timeframes = {'1M': 30,'3M': 90,'6M': 180,'1Y': 252,'3Y': 756,'5Y': 1260}

    etf_returns = {}
    benchmark_returns = {}

    for timeframe_name, days_delta in timeframes.items():
        start_date = end_date - timedelta(days=days_delta)
        pos = df_aligned_benchmark_1.index.get_indexer([start_date], method='nearest')[0]
        nearest_start_date = df_aligned_benchmark_1.index[pos]

        start_price_base = df_aligned_benchmark_1.at[nearest_start_date, 'Normalized_Close_Base']
        start_price_benchmark = df_aligned_benchmark_1.at[nearest_start_date, 'Normalized_Close_Base_Benchmark']

        etf_returns[timeframe_name] = round(((end_price_base / start_price_base) - 1) * 100, 2)
        benchmark_returns[timeframe_name] = round(((end_price_benchmark / start_price_benchmark) - 1) * 100, 2)

  
    returns_df = pd.DataFrame([etf_returns, benchmark_returns], index=["ETF", "Benchmark"])
                
    returns_df = returns_df.reset_index().rename(columns={'index': ''})         
    returns_html = returns_df.to_html(
    index=False,classes="data-table",border=0,justify="center",escape=False)

    st.markdown(f"""<div class="etf-section">
                <h3>ETF Returns</h3>
                <div class="data-table-small-container">
            {returns_html}
    </div>       
    </div>""",unsafe_allow_html=True)
                
           
def display_sector_pie_fig(ticker):
    selected_ticker_symbol= ticker.split("-")[0].strip()
    col1, col2,col3 = st.columns([0.3,2.9, 0.3])
    with col2:
        holdings, sectors_flattened, etf_parameters, portfolio_compostion = get_etf_data(selected_ticker_symbol)
        sectors_df=pd.DataFrame(data=sectors_flattened,columns=['Sector', 'Weight']).reset_index()
        sectors_df=sectors_df[sectors_df['Weight'] > 0]  # Filter out sectors with zero weight

        fig, fig2 = create_pie_chart(sectors_flattened=sectors_df, holdings=holdings)
        styled_plotly_chart(fig, f"{ticker} Sector Allocation")
        styled_plotly_chart(fig2, f"{ticker} Holdings Allocation")
        redis_config = st.secrets["redis"]
        r = redis.Redis(
            host=redis_config["host"],
            port=redis_config["port"],
            username=redis_config["username"],
            password=redis_config["password"])
        redis_key = f"country_exposure:{selected_ticker_symbol}"
        data = r.get(redis_key)
        json_string = data.decode('utf-8')   
  
        display_country_exposure_horizontal_bar_chart(data= json.loads(json_string))


def display_nav_chart(selected_ticker):    
        selected_ticker_symbol= selected_ticker.split("-")[0].strip()    
        nav_data = get_nav_data(selected_ticker_symbol)
        nav_dataa = get_nav_data("^GSPC") 
        N100_nav_data = get_nav_data("^NDX")          
        create_line_chart(nav_data, df_benchmark=nav_dataa, df_benchmark_2=N100_nav_data)

    
def create_etf_dashboard_content(selected_ticker):

    with psycopg2.connect(**db_config) as connection:
            df = fetch_table_data(connection=connection, table_name="US_ETF_OVERVIEW_DATA")
            df = df[df['ETF_Name']==selected_ticker]
            why_text = df['Why'].iloc[0]
            why_items = why_text.split('\n') if pd.notna(why_text) else []
            why_html = ""
            if why_items:
                why_html = "<ul>"
                for item in why_items:
                    if item.strip():  
                        why_html += f"<li>{item.strip()}</li>"
                why_html += "</ul>"

            Market_Size_Growth_text=df['Market Size & Growth'].iloc[0]
            Market_Size_Growth_items = Market_Size_Growth_text.split('\n') if pd.notna(Market_Size_Growth_text) else []
            Market_Size_Growth_html = ""
            if Market_Size_Growth_items:
                Market_Size_Growth_html = "<ul>"
                for item in Market_Size_Growth_items:
                    if item.strip():  
                        Market_Size_Growth_html += f"<li>{item.strip()}</li>"
                Market_Size_Growth_html += "</ul>"

            Growth_drivers_text=df['Growth Drivers'].iloc[0]
            Growth_drivers_items = Growth_drivers_text.split('\n') if pd.notna(Growth_drivers_text) else []
            Growth_drivers_html = ""  
            if Growth_drivers_items:
                Growth_drivers_html = "<ul>"
                for item in Growth_drivers_items:   
                    if item.strip():
                        Growth_drivers_html += f"<li>{item.strip()}</li>"
                Growth_drivers_html += "</ul>"

            data_dict = json.loads(df['ETF Overview'].iloc[0])
            data_list = [data_dict]
            etf_overview_df = pd.DataFrame(data_list)
            etf_overview_df_T= etf_overview_df.T.reset_index()
            etf_overview_df_T.columns = ['Metric', 'Value']
            table_html = etf_overview_df_T.to_html(  index=False,  classes="data-table",  border=0,  justify="left",  escape=False)
            st.markdown(f""" <div class="etf-section">     <h2>{df['ETF_Name'].iloc[0]}</h2>     <p> {df['Description'].iloc[0]} </p>     <h3>Why</h3>
            <ul>{why_html} </ul> </div> """, unsafe_allow_html=True)
            risk_statistics=calcualate_risk_stats(ticker=df['ETF_Name'].iloc[0].split("-")[0].strip())  
            risk_statistics = risk_statistics.reset_index()
            metric_mapping = {'sharpe': 'Sharpe Ratio','sortino': 'Sortino Ratio',
             'max_drawdown': 'Max Drawdown', 'volatility': 'Volatility', 'recovery_factor': 'Recovery Factor'}
            risk_statistics['index'] = risk_statistics['index'].replace(metric_mapping)
            risk_statistics['Close'] = risk_statistics['Close'].apply(lambda x: f"{x:.2}")
            risk_statistics['Close_benchmark-1'] = risk_statistics['Close_benchmark-1'].apply(lambda x: f"{x:.2}")
            risk_statistics = risk_statistics.rename(columns={'Close_benchmark-1':'S&P500', 'index': 'Metric', 0: 'Value','Close': 'ETF'})
            risk_statistics = risk_statistics.rename(index={'sharpe': 'Sharpe Ratio', 'sortino': 'Sortino Ratio', 'max_drawdown': 'Max Drawdown', 'volatility': 'Volatility', 'recovery_factor': 'Recovery Factor'})
            risk_statistics_html = risk_statistics.to_html(classes="data-table",  border=0,  justify="left",  escape=False, index=False)
    
    master_etf_data = {}
    if selected_ticker:
        selected_ticker_symbol= selected_ticker.split("-")[0].strip()
        equity_holding_data,sector_holding_data,etf_parameters,etf_portfolio_composition=get_etf_data(tickerlist=selected_ticker_symbol)
        master_etf_data[selected_ticker]= {'Equity_Exposure_Data':equity_holding_data,'Sector_Exposure_Data':sector_holding_data,
                                    'ETF_Parameters':etf_parameters,'ETF_Composition':etf_portfolio_composition}
    

    col1,col2,col3,col4=st.columns([0.2,0.86,0.86,0.2])
    with col2:
        st.markdown(f"""<div class="etf-section">
            <h3>Fund Characteristics</h3>
            <div class="data-table-small-container">
            {table_html}
    </div>
    </div>""",unsafe_allow_html=True)

    with col3:
        st.markdown(f"""<div class="etf-section">
            <h3>Risk Statistics</h3>
            <div class="data-table-small-container">
            {risk_statistics_html}
    </div>""", unsafe_allow_html=True)
 

    st.markdown(f"""
    <div class="etf-section">
        <h3>Market Size & Growth </h3>
        <ul>{Market_Size_Growth_html}</ul>
        <h3>Growth Drivers</h3>
        <ul>{Growth_drivers_html}</ul>
    </div>
    """, unsafe_allow_html=True)
 
   
# --- Main Streamlit App Execution ---
if __name__ == "__main__":
    inject_custom_styles() 
    c1,c2,c3=st.columns([0.3,1.5,0.3])

    with c2:
     selected_etf = st.selectbox(
        "Select an ETF",
        options=[
            "SKYY - First Trust Cloud Computing ETF",
            "SMH - VanEck Semiconductor ETF",
            "ESPO - VanEck Video Gaming and eSports ETF",
            "NLR - VanEck Uranium and Nuclear ETF",
            "CIBR - First Trust NASDAQ Cybersecurity ETF",
            "PPA - Invesco Aerospace & Defense ETF",
            "AIQ - Global X Artificial Intelligence"])
    
    create_etf_dashboard_content(selected_ticker= selected_etf)  
    display_nav_chart(selected_ticker=selected_etf)
    display_sector_pie_fig(ticker=selected_etf)


