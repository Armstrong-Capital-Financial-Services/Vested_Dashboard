import streamlit as st
import pandas as pd
from yahooquery import Ticker
import plotly.graph_objects as go
import redis
import json
import yfinance as yf
from datetime import datetime, timedelta
import plotly.express as px
import quantstats as qs
import matplotlib.pyplot as plt
from streamlit.components.v1 import html


@st.cache_data(ttl=3600)  # Cache for 1 hour
def fetch_comparison_data(ticker: str) -> pd.DataFrame:
    """Fetch and cache comparison data from Yahoo Finance"""
    try:
        data = yf.Ticker(ticker).history(period="3y").reset_index()
        data['Date'] = pd.to_datetime(data['Date'])
        data['Close'] = data['Close'] / data['Close'].iloc[0] * 1000
        return data[['Date', 'Close']]
    except Exception as e:
        st.error(f"Failed to fetch data for {ticker}: {e}")
        return pd.DataFrame()
def styled_plotly_chart_2(fig, height=350):
    """Wraps Plotly chart in a styled div with embedded title"""
    # Ensure background is transparent so the custom div background shows
    fig.update_layout(
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(
            family="Outfit, sans-serif", # Ensure consistent font if loaded
            size=12,
            color="#333",weight='bold' ))
        
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
        <div style="padding: 25px 25px 10px;">  {plot_html}
        </div>
    </div>
    """
    html(rounded_chart, height=total_wrapper_height)
    
@st.fragment
def portfolio_chart_fragment(price_data: pd.DataFrame):
    """Complete portfolio chart with all options - optimized for performance"""
    
    # Create base figure
    fig = go.Figure()
    
    # Base portfolio trace (always shown)
    fig.add_trace(go.Scatter(
        x=price_data['Date'],
        y=price_data['Portfolio_Value'],
        mode='lines',
        name='Portfolio Value',
        line=dict(width=2, color='blue')
    ))
    
    # Create columns for selectors
    col1, col2 = st.columns(2, gap="medium")
    
    with col1:
        index_choice = st.selectbox(
            "Compare with Index:",
            ["None", "Nasdaq 100", "S&P 500"],
            key="compare_index")
    
    with col2:
        stock_choice = st.selectbox(
            "Compare with Stock:",
            ["None", "AAPL", "MSFT", "GOOGL", "AMZN", "TSLA"],
            key="compare_stock")
    
    # Only fetch data if something is actually selected
    # Index comparison
    if index_choice != "None":
        if index_choice == "Nasdaq 100":
            ticker, color, label = "^NDX", "green", "Nasdaq 100"
        elif index_choice == "S&P 500":
            ticker, color, label = "^GSPC", "orange", "S&P 500"
        
        # Fetch cached data
        with st.spinner(f"Loading {label} data..."):
            index_data = fetch_comparison_data(ticker)
        
        if not index_data.empty:
            fig.add_trace(go.Scatter(
                x=index_data['Date'], 
                y=index_data['Close'],
                mode='lines', 
                name=label,
                line=dict(width=2, color=color)
            ))
    
    # Stock comparison
    if stock_choice != "None":
        with st.spinner(f"Loading {stock_choice} data..."):
            stock_data = fetch_comparison_data(stock_choice)
        
        if not stock_data.empty:
            fig.add_trace(go.Scatter(
                x=stock_data['Date'], 
                y=stock_data['Close'],
                mode='lines', 
                name=stock_choice,
                line=dict(width=2, color='red')
            ))
    
    # Apply layout styling
    fig.update_layout(
        xaxis_title='Date', yaxis_title='Portfolio Value', template='plotly_white',hovermode='x unified',height=500)
    
    # Add range selector
    fig.update_xaxes(
        rangeslider_visible=False,
        rangeselector=dict(
            buttons=list([
                dict(count=1, label="1M", step="month", stepmode="backward"),
                dict(count=3, label="3M", step="month", stepmode="backward"),
                dict(count=6, label="6M", step="month", stepmode="backward"),
                dict(count=1, label="1Y", step="year", stepmode="backward"),
                dict(count=2, label="2Y", step="year", stepmode="backward"),
                dict(count=3, label="3Y", step="year", stepmode="backward")
            ])
        )
    )
    
    # Display the chart
    styled_plotly_chart_2(fig)



def get_nav_data(ticker):
    """
    Fetches the NAV data for a given ticker from the Supabase database.
    Returns a DataFrame with the NAV data.
    """
    etf = yf.Ticker(ticker=ticker)
    nav_data = etf.history(period="1y")
    nav_data.reset_index(inplace=True)
    nav_data['Date'] = pd.to_datetime(nav_data['Date']).dt.strftime('%d-%m-%Y')
    if ticker.startswith('^'):
        nav_data=nav_data.iloc[:,:-3] 
    else:
       nav_data=nav_data.iloc[:,:-4]

    return nav_data

def calcualate_risk_stats(df):   
    add_stats = ['volatility', 'sharpe', 'sortino', 'max_drawdown','recovery_factor']
    all_stats = [f for f in dir(qs.stats) if f[0] != '_']
    required_stats = list(set(all_stats) & set(add_stats))
    required_stats_dict = {stat: getattr(qs.stats, stat) for stat in required_stats}

    # Expecting df with ['Date', 'Portfolio_Value']
    df = df[['Date', 'Portfolio_Value']].copy()
    df['Date'] = pd.to_datetime(df['Date'],format='mixed').dt.tz_localize(None)
    df.set_index('Date', inplace=True)
     
    # Restrict to last 252 trading days (~1 year)
    df = df.iloc[-252:]
    # Get benchmark (S&P500 for example)
    benchmark_data = yf.Ticker("^GSPC")
    benchmark_data = benchmark_data.history(period="1y").reset_index()
    benchmark_data['Date'] = pd.to_datetime(benchmark_data['Date'],format='mixed').dt.tz_localize(None)
    benchmark_data = benchmark_data[['Date', 'Close']]
    
    combined_df = pd.merge(df.reset_index(), benchmark_data, on='Date', how='inner')
    combined_df.set_index('Date', inplace=True)


    # Compute stats
    stats_result = {stat: required_stats_dict[stat](combined_df) for stat in required_stats}
    return stats_result


def inject_custom_styles():
    """
    Injects enhanced CSS styles into the Streamlit application for a modern, polished look.
    """
    # [Existing CSS styles remain unchanged]
    st.markdown("""
    <style>
        /* Import modern Google Fonts for better typography */
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=Poppins:wght@400;500;600;700&display=swap');

        /* Main app background with a softer gradient */
        .stApp {
            background: linear-gradient(135deg, #e6ecf5 0%, #d0d9e8 100%) !important;
            color: #1e2a44 !important;
            font-family: 'Inter', sans-serif;
            min-height: 100vh;
        }

        /* Header styling */
        header, .css-18ni7ap.e8zbici2, .css-1dp5vir.e8zbici3 {
            background: #ffffff !important;
            color: #1e2a44 !important;
            box-shadow: 0 2px 4px rgba(0, 0, 0, 0.05) !important;
            border-bottom: 1px solid #e2e8f0;
        }

        /* Consistent text styling for Streamlit elements */
        .stMarkdown, .stText, .stDataFrame, .stTable, .stExpander, .stSelectbox,
        .stButton > button, .stDownloadButton > button, .stFileUploader,
        .stTextInput > div > div > input, .stTextArea > div > div > textarea,
        .stDateInput > div > div > input, .stTimeInput > div > div > input,
        .stNumberInput > div > div > input, .stSlider > div > div > div > div {
            color: #1e2a44 !important;
        }

        /* Enhanced card section */
        .etf-section {
            background-color: #ffffff !important;
            border-radius: 16px;
            padding: 32px;
            margin: 20px auto;
            box-shadow: 0 8px 24px rgba(0, 0, 0, 0.1), 0 2px 8px rgba(0, 0, 0, 0.05);
            border: 1px solid #e2e8f0;
            transition: transform 0.3s ease, box-shadow 0.3s ease;
            animation: fadeInUp 0.7s ease-out forwards;
            opacity: 0;
            max-width: 1000px;
            width: 95%;
        }

        .etf-section:hover {
            transform: translateY(-5px);
            box-shadow: 0 12px 32px rgba(0, 0, 0, 0.15), 0 4px 12px rgba(0, 0, 0, 0.08);
        }

        /* Card title */
        .etf-section h2 {
            font-family: 'Poppins', sans-serif;
            font-size: 2.4rem;
            color: #3b82f6 !important;
            margin: 0 0 20px;
            padding-bottom: 12px;
            border-bottom: 2px solid #dbeafe;
            font-weight: 600;
        }

        /* Sub-sections */
        .etf-section h3 {
            font-family: 'Poppins', sans-serif;
            font-size: 1.7rem;
            color: #2563eb !important;
            margin: 24px 0 12px;
            font-weight: 500;
        }

        /* Paragraphs */
        .etf-section p {
            font-size: 1.05rem;
            line-height: 1.8;
            color: #475569;
            margin-bottom: 16px;
        }

        /* Lists */
        .etf-section ul {
            list-style: none;
            padding-left: 0;
            margin-bottom: 24px;
        }

        .etf-section ul li {
            position: relative;
            padding-left: 28px;
            margin-bottom: 12px;
            font-size: 1rem;
            color: #64748b;
        }

        .etf-section ul li::before {
            content: '✔';
            color: #10b981;
            position: absolute;
            left: 0;
            font-size: 1.2rem;
            line-height: 1.4;
        }

        /* Table styling */
        .data-table {
            width: 100%;
            border-collapse: collapse;
            margin: 24px 0;
            font-size: 0.95rem;
            background-color: #ffffff;
            border-radius: 12px;
            overflow: hidden;
            box-shadow: 0 4px 12px rgba(0, 0, 0, 0.05);
            border: 1px solid #e2e8f0;
        }

        .data-table th, .data-table td {
            padding: 14px 18px;
            text-align: left;
            border-bottom: 1px solid #f1f5f9;
        }

        .data-table th {
            background-color: #1e3a8a;
            color: #ffffff;
            font-weight: 600;
            text-transform: uppercase;
            letter-spacing: 0.6px;
            text-transform: none !important;
                
        }

        .data-table tr:nth-child(even) {
            background-color: #f9fafb;
        }

        .data-table tr:hover {
            background-color: #eff6ff;
        }

        /* Responsive design */
        @media (max-width: 768px) {
            .etf-section {
                padding: 24px;
                width: 100%;
            }
            .etf-section h2 {
                font-size: 2rem;
            }
            .etf-section h3 {
                font-size: 1.5rem;
            }
            .data-table th, .data-table td {
                padding: 12px 14px;
            }
        }

        @media (max-width: 480px) {
            .etf-section {
                padding: 16px;
            }
            .etf-section h2 {
                font-size: 1.8rem;
            }
            .etf-section h3 {
                font-size: 1.3rem;
            }
        }

        /* Animations */
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

        /* Sequential animation delays */
        .etf-section:nth-child(2) { animation-delay: 0.1s; }
        .etf-section:nth-child(3) { animation-delay: 0.2s; }
        .etf-section:nth-child(4) { animation-delay: 0.3s; }

        /* General headings */
        h1, h4, h5, h6 {
            color: #3b82f6 !important;
            font-family: 'Poppins', sans-serif;
            font-weight: 600;
        }

        /* Button styling */
        .stButton > button {
            background-color: #3b82f6;
            color: #ffffff;
            border-radius: 10px;
            padding: 12px 24px;
            font-family: 'Inter', sans-serif;
            font-weight: 500;
            transition: all 0.3s ease;
            box-shadow: 0 3px 8px rgba(59, 130, 246, 0.2);
        }

        .stButton > button:hover {
            background-color: #2563eb;
            transform: translateY(-2px);
            box-shadow: 0 5px 12px rgba(59, 130, 246, 0.3);
        }

        .stDownloadButton > button {
            background-color: #10b981;
            color: #ffffff;
            border-radius: 10px;
            padding: 12px 24px;
            font-family: 'Inter', sans-serif;
            font-weight: 500;
            transition: all 0.3s ease;
            box-shadow: 0 3px 8px rgba(16, 185, 129, 0.2);
        }

        .stDownloadButton > button:hover {
            background-color: #059669;
            transform: translateY(-2px);
            box-shadow: 0 5px 12px rgba(16, 185, 129, 0.3);
        }

        .stSelectbox > div > div {
            border-radius: 10px;
            border: 1px solid #e2e8f0;
            background-color: #ffffff;
            box-shadow: 0 2px 6px rgba(0, 0, 0, 0.05);
        }

        .stTextInput > div > div > input {
            border-radius: 10px;
            border: 1px solid #e2e8f0;
            background-color: #ffffff;
            padding: 10px;
            box-shadow: 0 2px 6px rgba(0, 0, 0, 0.05);
        }

        /* Form divider */
        .stDivider {
            background-color: #e2e8f0 !important;
            height: 1px;
            margin: 16px 0;
        }
    </style>
    """, unsafe_allow_html=True)

redis_config = st.secrets["redis"]

r = redis.Redis(
    host=redis_config["host"],
    port=redis_config["port"],
    username=redis_config["username"],
    password=redis_config["password"])

def display_etf_card_with_form():
    # Initialize session state
    if 'portfolios' not in st.session_state:
        st.session_state.portfolios = {}
    if 'num_etfs' not in st.session_state:
        st.session_state.num_etfs = 1
    if 'current_portfolio_name_input' not in st.session_state:
        st.session_state.current_portfolio_name_input = ""
    if 'edit_mode' not in st.session_state:
        st.session_state.edit_mode = False
    if 'edit_portfolio_name' not in st.session_state:
        st.session_state.edit_portfolio_name = None
    if 'selected_portfolio_selector' not in st.session_state:
        st.session_state.selected_portfolio_selector = None

    ETF_OPTIONS = [
        "Select an ETF",
        "SKYY - First Trust Cloud Computing ETF",
        "SMH - VanEck Semiconductor ETF",
        "ESPO - VanEck Video Gaming and eSports ETF",
        "NLR - VanEck Uranium and Nuclear ETF",
        "CIBR - First Trust NASDAQ Cybersecurity ETF",
        "PPA - Invesco Aerospace & Defense ETF",
        "AIQ - Global X Artificial Intelligence"
    ]

    col0, col01 = st.columns([0.8, 1], gap="medium")

    with col0:
        form_key = "portfolio_form_edit" if st.session_state.edit_mode else "portfolio_form"
        with st.form(key=form_key):
            portfolio_name = st.text_input(
                "Portfolio Name",
                value=st.session_state.current_portfolio_name_input,
                key="actual_current_portfolio_name_widget",
                placeholder="Enter portfolio name",
                disabled=st.session_state.edit_mode)

            for i in range(st.session_state.num_etfs):
                st.markdown(f"<h2>ETF {i + 1}</h2>", unsafe_allow_html=True)
                col1, col2 = st.columns([1, 1], gap="small")

                with col1:
                    default_index = 0
                    if st.session_state.edit_mode and st.session_state.edit_portfolio_name:
                        portfolio_data = st.session_state.portfolios.get(st.session_state.edit_portfolio_name, [])
                        if i < len(portfolio_data):
                            current_etf = f"{portfolio_data[i]['Symbol']} - {portfolio_data[i]['ETF Name']}"
                            default_index = ETF_OPTIONS.index(current_etf) if current_etf in ETF_OPTIONS else 0
                    st.selectbox(
                        "Select an ETF",
                        options=ETF_OPTIONS,
                        index=default_index,
                        key=f"etf_{i}"
                    )

                with col2:
                    default_allocation = 100.0 / st.session_state.num_etfs if not st.session_state.edit_mode else 0.0
                    if st.session_state.edit_mode and st.session_state.edit_portfolio_name:
                        portfolio_data = st.session_state.portfolios.get(st.session_state.edit_portfolio_name, [])
                        if i < len(portfolio_data):
                            default_allocation = portfolio_data[i]["Allocation"]
                    st.number_input(
                        "% Allocation",
                        min_value=0.01,
                        max_value=100.0,
                        step=0.01,
                        format="%.2f",
                        value=default_allocation,
                        key=f"allocation_{i}"
                    )
                if i < st.session_state.num_etfs - 1:
                    st.divider()

            st.markdown("---")
            col_add, col_reset, col_submit = st.columns([1, 1, 1])

            with col_add:
                add_more = st.form_submit_button("Add Another ETF", use_container_width=True)

            with col_reset:
                reset_form = st.form_submit_button("Reset Form", use_container_width=True)

            with col_submit:
                submit_label = "Update Portfolio" if st.session_state.edit_mode else "Submit Portfolio"
                submitted = st.form_submit_button(submit_label, use_container_width=True)

            if add_more:
                st.session_state.num_etfs += 1
                st.rerun()

            if reset_form:
                st.session_state.num_etfs = 1
                st.session_state.current_portfolio_name_input = ""
                st.session_state.edit_mode = False
                st.session_state.edit_portfolio_name = None
                for i in range(10):
                    for key in [f'etf_{i}', f'allocation_{i}']:
                        if key in st.session_state:
                            del st.session_state[key]
                st.rerun()

            if submitted:
                if not portfolio_name:
                    st.error("Please enter a **Portfolio Name** before submitting.")
                elif not st.session_state.edit_mode and portfolio_name in st.session_state.portfolios:
                    st.error(f"Portfolio '{portfolio_name}' already exists. Choose a different name or edit the existing portfolio.")
                else:
                    valid_entries = []
                    errors = []
                    selected_etfs = set()
                    total_allocation = 0.0

                    for i in range(st.session_state.num_etfs):
                        selected_etf = st.session_state.get(f'etf_{i}', ETF_OPTIONS[0])
                        allocation = st.session_state.get(f'allocation_{i}', 0.0)

                        if selected_etf == ETF_OPTIONS[0]:
                            errors.append(f"ETF #{i + 1}: Please select a valid ETF")
                        elif selected_etf in selected_etfs:
                            errors.append(f"ETF #{i + 1}: Duplicate ETF '{selected_etf}' selected")
                        elif allocation <= 0:
                            errors.append(f"ETF #{i + 1}: Allocation must be greater than 0%")
                        else:
                            selected_etfs.add(selected_etf)
                            total_allocation += allocation
                            ticker, etf_name = selected_etf.split(' - ', 1)
                            entry = {
                                "Symbol": ticker,
                                "ETF Name": etf_name,
                                "Allocation": allocation
                            }
                            valid_entries.append(entry)

                    # Validate total allocation only on submission
                    if not errors and abs(total_allocation - 100.0) > 0.01:
                        errors.append(f"Total allocation must sum to 100%. Current total: {total_allocation:.2f}%")

                    if errors:
                        st.error("Please fix the following errors:")
                        for error in errors:
                            st.error(f"• {error}")
                    else:
                        st.session_state.portfolios[portfolio_name] = valid_entries
                        action = "updated" if st.session_state.edit_mode else "created"
                        st.success(f"Successfully {action} portfolio **'{portfolio_name}'** with **{len(valid_entries)}** ETF(s)!")

                        st.session_state.num_etfs = 1
                        st.session_state.current_portfolio_name_input = ""
                        st.session_state.edit_mode = False
                        st.session_state.edit_portfolio_name = None
                        for i in range(10):
                            for key in [f'etf_{i}', f'allocation_{i}']:
                                if key in st.session_state:
                                    del st.session_state[key]
                        st.rerun()

    with col01:
      with st.container(border=True):  
        if st.session_state.portfolios:

            portfolio_names = list(st.session_state.portfolios.keys())
            if "selected_portfolio_selector" not in st.session_state or st.session_state.selected_portfolio_selector not in portfolio_names:
                st.session_state.selected_portfolio_selector = portfolio_names[0]
            selected_portfolio_name = st.selectbox(
                "Select a portfolio to view",
                options=portfolio_names,
                key="selected_portfolio_selector",
                placeholder="Choose a portfolio"
            )

            if selected_portfolio_name:
                selected_portfolio_data = st.session_state.portfolios.get(selected_portfolio_name, [])
                if selected_portfolio_data:
                    df = pd.DataFrame(selected_portfolio_data)
                    # Format Allocation column to include % symbol
                    df['Allocation'] = df['Allocation'].apply(lambda x: f"{x:.2f}%")
        
                    st.dataframe(df, use_container_width=True, hide_index=True)
                    st.info(f"Portfolio **'{selected_portfolio_name}'** has **{len(selected_portfolio_data)}** ETF(s).")

                    col_edit, col_delete = st.columns(2)
                    with col_edit:
                        if st.button(f"Edit Portfolio '{selected_portfolio_name}'", use_container_width=True):
                            st.session_state.edit_mode = True
                            st.session_state.edit_portfolio_name = selected_portfolio_name
                            st.session_state.current_portfolio_name_input = selected_portfolio_name
                            st.session_state.num_etfs = len(selected_portfolio_data)
                            st.rerun()

                    with col_delete:
                        if st.button(f"Delete Portfolio '{selected_portfolio_name}'", use_container_width=True):
                            del st.session_state.portfolios[selected_portfolio_name]
                            if selected_portfolio_name == st.session_state.selected_portfolio_selector:
                                st.session_state.selected_portfolio_selector = portfolio_names[0] if portfolio_names else None
                            st.rerun()
                else:
                    st.info(f"Portfolio **'{selected_portfolio_name}'** is empty.")
        else:
            st.info("No portfolios created yet. Use the form to add one!")

            

def display_country_exposure_horizontal_bar_chart(data):
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
        texttemplate='<b>%{text:.2f}%</b>',
        textposition='outside',
        hoverinfo='skip',   hovertemplate=None)

    plotly_inner_height = len(df) * 35 + 150 


    styled_plotly_chart_2(fig, height=plotly_inner_height)

def display_selected_portfolio():
    """
    Displays only the currently selected portfolio outside the form loop.
    """    
    selected_portfolio_name = st.session_state.get('selected_portfolio_selector', None)
    if selected_portfolio_name and st.session_state.get('portfolios', {}):

        portfolio_data = st.session_state.portfolios.get(selected_portfolio_name, [])
        if portfolio_data:
            portfolio_df = pd.DataFrame(portfolio_data)
            raw_data_df = pd.read_csv('US ETF UNIVERSE.csv')
            cols_to_merge = ['Symbol', 'ETF Database Category', '# of Holdings', '% In Top 10']
            merged_df = pd.merge(portfolio_df, raw_data_df[cols_to_merge], on='Symbol', how='left')
            merged_df['Allocation'] = merged_df['Allocation'].apply(lambda x: f"{x:.2f}%")
            merged_df_html = merged_df.to_html(classes="data-table",  border=0,  justify="left",  escape=False, index=False)
            st.markdown("""
            <h2 style='font-size: 2.8rem; font-weight: 900; color: #2c3e50; margin-bottom: 1.5rem; font-family: "Times New Roman";'>
            Portfolio Details </h2>""",unsafe_allow_html=True)
            st.markdown(f"""<div class="etf-section">
            {merged_df_html}</div>""", unsafe_allow_html=True)

            sector_rename_map = {
                'realestate': 'Real Estate',
                'consumer_cyclical': 'Consumer Cyclical',
                'basic_materials': 'Basic Materials',
                'consumer_defensive': 'Consumer Defensive',
                'technology': 'Technology',
                'communication_services': 'Communication Services',
                'financial_services': 'Financial Services',
                'utilities': 'Utilities',
                'industrials': 'Industrials',
                'energy': 'Energy',
                'healthcare': 'Healthcare'}

            all_etf_sectors = {}
            all_country_exposures = []
            etf_sectors_data = []
            allocations = {} 
            price_data = pd.DataFrame()
            for entry in portfolio_data:
                ticker = entry['Symbol']
                allocation = entry['Allocation'] / 100 
                allocations[ticker] = allocation
                raw_entry = raw_data_df[raw_data_df['Symbol'] == ticker]

                redis_key = f"country_exposure:{ticker}"
                data = r.get(redis_key)
                  
                json_string = data.decode('utf-8')
                all_country_df = pd.DataFrame(json.loads(json_string))
                all_country_df['exposure'] = all_country_df['exposure'].astype(float) * allocation
                all_country_df["ticker"] = ticker
                all_country_exposures.append(all_country_df)
        
                if not raw_entry.empty:
                    etf = yf.Ticker(ticker=ticker)
                    nav_data = etf.history(period="max")
                    nav_data.reset_index(inplace=True)
                    nav_data['Date'] = pd.to_datetime(nav_data['Date']).dt.tz_localize(None)
                    nav_data = nav_data[['Date', 'Close']]
                    start_date = (datetime.today() - timedelta(days=3*252))
                    nav_data = nav_data[nav_data['Date'] >= start_date]
                    if price_data.empty:
                         price_data['Date'] = nav_data['Date']
        
                    price_data = price_data.merge(nav_data.rename(columns={'Close': ticker}), on='Date', how='inner')
                    price_data.set_index('Date', inplace=True)
                    data = Ticker(ticker).fund_holding_info
                    sectors = data.get(ticker, {}).get('sectorWeightings', [])
                    sectors_flattened = [(list(d.keys())[0], list(d.values())[0]) for d in sectors]
                    all_etf_sectors[ticker] = [(sector_rename_map.get(sector, sector), float(weight) * allocation)
                                             for sector, weight in sectors_flattened]
                    etf_sectors_data.append(all_etf_sectors[ticker])
            alloc_series = pd.Series(allocations)     
            price_data['Portfolio_Value'] = price_data.dot(alloc_series)
            base_value = price_data['Portfolio_Value'].iat[0]   
            price_data['Portfolio_Value'] = (price_data['Portfolio_Value'] / base_value) * 1000
            price_data.reset_index(inplace=True)

            risk_df = calcualate_risk_stats(df=price_data)
            risk_df['max_drawdown'] = (risk_df['max_drawdown'] * 100).round(2).astype(str) + '%'
            risk_df['volatility'] = (risk_df['volatility'] * 100).round(2).astype(str) + '%'
            risk_df['sortino'] = risk_df['sortino'].round(2)
            risk_df['sharpe'] = risk_df['sharpe'].round(2)
            risk_df['recovery_factor'] = risk_df['recovery_factor'].round(2)
            risk_df = pd.DataFrame(risk_df,)
            risk_df = risk_df.rename(columns={'volatility': 'Volatility (Ann.)', 'sharpe': 'Sharpe Ratio', 'sortino': 'Sortino Ratio',
                                              'max_drawdown': 'Max Drawdown', 'recovery_factor': 'Recovery Factor'})
            risk_df = risk_df.rename(index = {'Portfolio_Value': 'Portfolio','Close': 'S&P 500'})        
            risk_statistics_html = risk_df.to_html(classes="data-table",  border=0,  justify="left",  escape=False)
            st.markdown("""
            <h2 style='font-size: 2.8rem; font-weight: 900; color: #2c3e50; margin-bottom: 1.5rem; font-family: "Times New Roman";'>
            Risk Statistics </h2>""",unsafe_allow_html=True)
            st.markdown(f"""<div class="etf-section">
            {risk_statistics_html}</div>""", unsafe_allow_html=True)
    
            
            st.markdown("""
            <h2 style='font-size: 2.8rem; font-weight: 900; color: #2c3e50; margin-bottom: 1.5rem; font-family: "Times New Roman";'>
                        Performance Chart </h2>""",unsafe_allow_html=True)
            portfolio_chart_fragment(price_data)

            price_data = price_data.copy()
            price_data = price_data.set_index("Date").sort_index()
            timeframes = {'1M': 22,'3M': 66,'6M': 132,'1Y': 252, '3Y': 756, }
            portfolio_returns = {}
            end_date = price_data.index[-1]
            end_price = price_data["Portfolio_Value"].iloc[-1]

            for tf_name, days in timeframes.items():
                   start_date = end_date - timedelta(days=days)
                   pos = price_data.index.get_indexer([start_date], method="nearest")[0]
                   nearest_start_date = price_data.index[pos]

                   start_price = price_data.at[nearest_start_date, "Portfolio_Value"]
                   portfolio_returns[tf_name] = round(((end_price / start_price) - 1) * 100, 2)

            returns_df = pd.DataFrame([portfolio_returns], index=["Portfolio"])
            returns_df_html = returns_df.to_html(classes="data-table",  border=0,  justify="left",  escape=False)
            st.markdown("""
            <h2 style='font-size: 2.8rem; font-weight: 900; color: #2c3e50; margin-bottom: 1.5rem; font-family: "Times New Roman";'>
            Returns</h2>""",unsafe_allow_html=True)
            st.markdown(f"""<div class="etf-section">
            {returns_df_html}</div>""", unsafe_allow_html=True)
            price_data = price_data.reset_index()
         
            price_series = price_data.set_index("Date")["Portfolio_Value"]
            drawdown_fig = qs.plots.drawdowns_periods( price_series,show=False,title="Portfolio")
            st.markdown("""
            <h2 style='font-size: 2.8rem; font-weight: 900; color: #2c3e50; margin-bottom: 1.5rem; font-family: "Times New Roman";'>
            Portfolio Drawdowns </h2>""",unsafe_allow_html=True)
            st.pyplot(drawdown_fig)
            
            returns = price_series.pct_change().dropna()
            drawdown_dates = qs.stats.to_drawdown_series(returns)
            dd_details = qs.stats.drawdown_details(drawdown_dates).sort_values(by='max drawdown').head(5).iloc[:, :-1]
            dd_details['start'] = pd.to_datetime(dd_details['start']).dt.strftime('%d-%m-%Y')
            dd_details['end'] = pd.to_datetime(dd_details['end']).dt.strftime('%d-%m-%Y')
            dd_details.rename(columns={
                'max drawdown': 'Max Drawdown',
                'start': 'Start Date',
                'end': 'End Date',
                'valley': 'Recovery Date',
                'days': 'Duration (Days)'
            }, inplace=True)
            dd_details = dd_details.reset_index(drop=True)
            dd_details['Max Drawdown'] = (dd_details['Max Drawdown']).round(2).astype(str) + '%'

            dd_details_html = dd_details.to_html(classes="data-table",  border=0,  justify="left",  escape=False, index=False)
            st.markdown(f"""<div class="etf-section">
            {dd_details_html}</div>""", unsafe_allow_html=True)

            if etf_sectors_data:
                rows = []
                for ticker, sectors in all_etf_sectors.items():
                    for sector, weight in sectors:
                        rows.append({"Ticker": ticker, "Sector": sector, "Weight": f"{weight * 100:.2f}%"})

                df = pd.DataFrame(rows)
                pivot_df = df.pivot_table(index="Ticker", columns="Sector", values="Weight",
                                        aggfunc=lambda x: x, fill_value="0.00%")
                pivot_df.loc['Total'] = pivot_df.replace('%', '', regex=True).astype(float).sum().apply(lambda x: f"{x:.2f}%")
                labels = pivot_df.columns.tolist()
                values = pivot_df.loc['Total'].str.replace('%', '').astype(float).tolist()
                filtered_labels, filtered_values = zip( *[(l, v) for l, v in zip(labels, values) if v > 0])
                fig = go.Figure(data=[go.Pie(labels=filtered_labels, values=filtered_values, hole=.3,marker=dict(colors=px.colors.qualitative.Pastel),
                                             hovertemplate="<b>Sector:</b> %{label}<br><b>Allocation:</b> %{percent}<extra></extra>")])
                fig.update_traces(hoverinfo='label+percent', textfont_size=15)
 
             
                keys = r.keys("market_cap_constituents:*")
                all_data = []

                for key in keys:
                  data = r.get(key)
                  if data:
                      ticker = key.decode("utf-8").split(":")[-1]  # extract ticker from key
                      record = json.loads(data)
                      record["Ticker"] = ticker
                      all_data.append(record)          
                      df = pd.DataFrame(all_data)
                      df = df[["Ticker"] + [col for col in df.columns if col != "Ticker"]]
                      df = df.set_index("Ticker")
                      df = df.drop(columns=["Wgt avg mkt cap (mns)"], errors='ignore')
                      for d in portfolio_data:
                          ticker= d['Symbol']
                          allocation = d['Allocation'] / 100
                          if ticker in df.index:
                             df.loc[ticker] = df.loc[ticker].astype(float) * allocation
                             df = df[df.index.isin([d['Symbol'] for d in portfolio_data])]
                             df.loc["Total"] = df.sum(numeric_only=True)
                mkt_cp_labels = df.columns.tolist()
                mkt_cp_values = df.loc["Total"].tolist()   
                fig2 = go.Figure(data=[go.Pie(labels=mkt_cp_labels, values=mkt_cp_values, hole=.3, marker=dict(colors=px.colors.diverging.Spectral_r),
                                              hovertemplate="<b>Market Cap:</b> %{label}<br><b>Allocation:</b> %{percent}<extra></extra>")])
                fig2.update_traces(hoverinfo='label+percent', textfont_size=15)
                
                st.markdown("""
                 <h2 style='font-size: 2.8rem; font-weight: 900; color: #2c3e50; margin-bottom: 1.5rem; font-family: "Times New Roman";'>
                 Sector Breakdown </h2>""",unsafe_allow_html=True)
                styled_plotly_chart_2(fig)

                st.markdown("""
                   <h2 style='font-size: 2.8rem; font-weight: 900; color: #2c3e50; margin-bottom: 1.5rem; font-family: "Times New Roman";'>
                   Market Cap Breakdown </h2>""",unsafe_allow_html=True)
                styled_plotly_chart_2(fig2)
                data = pivot_df.loc['Total'][1:].reset_index().rename(columns={0:'country', 'Total':'exposure'}).to_dict(orient='records')
                for item in data:
                    item['exposure'] = float(item['exposure'].replace('%', ''))
                combined_df = pd.concat(all_country_exposures, ignore_index=True)
                all_country_pivot_df = pd.pivot_table(combined_df, index="ticker",columns="country",values="exposure", fill_value=0).reset_index()
                all_country_pivot_df.loc['Total'] = all_country_pivot_df.sum()
                all_country_exposure_data = all_country_pivot_df.loc['Total'][1:].reset_index().rename(columns={0:'country', 'Total':'exposure'}).to_dict(orient='records')
                st.markdown("""
             <h2 style='font-size: 2.8rem; font-weight: 900; color: #2c3e50; margin-bottom: 1.5rem; font-family: "Times New Roman";'>
             Country Exposure Breakdown</h2>""",unsafe_allow_html=True)
                display_country_exposure_horizontal_bar_chart(data=all_country_exposure_data)


            else:
                st.warning("No sector data available for the selected ETFs.")
        else:
            st.info(f"Portfolio **'{selected_portfolio_name}'** is empty.")
    else:
        st.info("No portfolio selected or no portfolios available. Select or create one using the form above.")

if __name__ == "__main__":
    st.set_page_config(layout="wide")
    inject_custom_styles()
    display_etf_card_with_form()
    st.markdown("---")
    display_selected_portfolio()
