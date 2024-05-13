import requests
import pandas as pd
import numpy as np
import logging
import matplotlib.pyplot as plt
import yfinance as yf
from alpha_vantage.timeseries import TimeSeries
import quandl
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Bidirectional
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing import image
import fitz  # PyMuPDF for PDF processing
import os
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
import spacy
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from spacy.lang.en.stop_words import STOP_WORDS
from PIL import Image
import matplotlib.pyplot as plt
import ta
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import os
from hashlib import sha256
import json
from time import time
import pytz
from datetime import time as dt_time
import time
from datetime import datetime, timedelta
import networkx as nx
import dwave_networkx as dnx
import aiohttp
import asyncio
import alpaca_trade_api as tradeapi
from trading_ig import IGService
import dash
from dash import dcc, html, Input, Output, State, callback_context
from dash.exceptions import PreventUpdate
import plotly.graph_objs as go
import dash_bootstrap_components as dbc
import plotly.express as px
from dash.dependencies import Input, Output, State
import pandas as pd
from text_preprocessing import preprocess_text
import threading
import io
from transformers import pipeline
import qiskit
from qiskit_aer import AerSimulator
  # Ensure 'qiskit-aer' is installed.
from qiskit.circuit.library import ZZFeatureMap
from sklearn.decomposition import KernelPCA
from qiskit.primitives import Sampler
import plotly.graph_objects as go

#from config import APCA_API_BASE_URL, APCA_API_KEY_ID, APCA_API_SECRET_KEY
# Initialize the Alpaca API
# Setup logging and API keys
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
ALPHA_VANTAGE_API_KEY = os.getenv(' ')  # Ensure this is correctly set in your environment
QUANDL_API_KEY = os.getenv(' ')  # Ensure this is correctly set in your environment
# Set environment variables
# Replace these with your actual IG credentials
os.environ['IG_API_KEY'] = ''
os.environ['IG_IDENTIFIER'] = '  '
os.environ['IG_PASSWORD'] = ' '
IG_API_BASE_URL = "https://demo-api.ig.com/gateway/deal"
ALPACA_API_KEY = ' '
ALPACA_SECRET_KEY = ' '
ALPACA_BASE_URL = 'https://paper-api.alpaca.markets'  # Use the paper trading URL for practice
quandl.ApiConfig.api_key = QUANDL_API_KEY
# Set your environment variables
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Initialize the IGService
pi = tradeapi.REST(ALPACA_API_KEY, ALPACA_SECRET_KEY, ALPACA_BASE_URL, api_version='v2')
ig_service = IGService(os.getenv('IG_SERVICE_USERNAME'), os.getenv('IG_SERVICE_PASSWORD'), os.getenv('IG_SERVICE_API_KEY'), os.getenv('IG_SERVICE_ACC_TYPE', 'DEMO'))
api = tradeapi.REST(ALPACA_API_KEY, ALPACA_SECRET_KEY, ALPACA_BASE_URL, api_version='v2')
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])




# Load VGG16 model preloaded with weights trained on ImageNet
base_model = VGG16(weights='imagenet')
model_vgg16 = VGG16(weights='imagenet', include_top=False)
model_vgg16 = Model(inputs=base_model.input, outputs=base_model.get_layer('fc1').output)


# Remove the output layer
model = Model(inputs=base_model.input, outputs=base_model.get_layer('fc1').output)
nlp = spacy.load('en_core_web_sm')  # Make sure this is after importing spacy
# Additional callback functions for other dynamic graphs
# Default to 'DEMO' if 'IG_SERVICE_ACC_TYPE' is not set
# Ensure the correct Spacy model is loaded
# Example call within extract_text_from_pdfs function

nlp = spacy.load("en_core_web_sm", disable=["parser", "ner"])  # Disable unnecessary pipeline components for efficiency
nlp.max_length = 2100000  # Adjust based on your document size, but be cautious of memory use

account_type = os.getenv('IG_SERVICE_ACC_TYPE', 'DEMO').upper()
ig_service = IGService(
    os.getenv('IG_SERVICE_USERNAME'),
    os.getenv('IG_SERVICE_PASSWORD'),
    os.getenv('IG_SERVICE_API_KEY'),
    account_type  # Use the validated account type here
)
cache_dir = "C:\\Users\\Adhiraj Singh\\OneDrive\\Desktop\\trading_project\\alpha_vantage_cache"
if not os.path.exists(cache_dir):
    os.makedirs(cache_dir)

# Ensure account_type is either 'LIVE' or 'DEMO'
if account_type not in ['LIVE', 'DEMO']:
    raise ValueError("Invalid account type. Please set 'IG_SERVICE_ACC_TYPE' to either 'LIVE' or 'DEMO'.")

app.layout = html.Div([
    html.H1('Hello, Dash!'),
    html.Div('Dash: A web application framework for Python.')
])
app = dash.Dash(__name__, suppress_callback_exceptions=True)

app.layout = html.Div([
    html.Div([
        html.H2("Real-Time Trading Dashboard"),
        dcc.Dropdown(
            id='symbol-selector',
            options=[
                {'label': 'Apple (AAPL)', 'value': 'AAPL'},
                {'label': 'Google (GOOGL)', 'value': 'GOOGL'},
                {'label': 'Microsoft (MSFT)', 'value': 'MSFT'},
            ],
            value='AAPL',
            clearable=False,
            style={"width":  "40%"}
        ),
    ], style={'padding': '20px', 'width': '50%'}),

    html.Div([
        dcc.Graph(id='live-trading-graph'),
        dcc.Graph(id='accuracy-stats-graph'),
        dcc.Graph(id='performance-metrics-graph'),
        dcc.Graph(id='additional-metrics-graph'),
    ]),
    
    dcc.Interval(
        id='interval-component',
        interval=1*60000,  # in milliseconds
        n_intervals=0
    ),
    html.Button("Update Now", id="update-button", n_clicks=0)
])


@app.callback(
    [Output('live-trading-graph', 'figure'),
     Output('accuracy-stats-graph', 'figure'),
     Output('performance-metrics-graph', 'figure'),
     Output('additional-metrics-graph', 'figure')],
    [Input('interval-component', 'n_intervals'),
     Input('update-button', 'n_clicks')],
    [State('symbol-selector', 'value')]
)
def update_graphs(n_intervals, n_clicks, selected_symbol):
    data_dict = fetch_and_process_data([selected_symbol], "2014-01-01", "2024-12-31")
    df_selected = data_dict.get(selected_symbol, pd.DataFrame())
    
    # Initialize placeholders for the figures
    trading_figure = go.Figure()
    accuracy_figure = go.Figure()
    performance_figure = go.Figure()
    additional_metrics_figure = go.Figure()

    if df_selected.empty:
        trading_figure.update_layout(title="No Data Available", xaxis_title="Time", yaxis_title="Price")
    else:
        trading_figure.add_trace(go.Scatter(x=df_selected.index, y=df_selected['Close'], mode='lines', name='Close'))
        if 'SMA_20' in df_selected.columns:
            trading_figure.add_trace(go.Scatter(x=df_selected.index, y=df_selected['SMA_20'], mode='lines', name='SMA 20', line=dict(color='orange')))
        if 'EMA_20' in df_selected.columns:
            trading_figure.add_trace(go.Scatter(x=df_selected.index, y=df_selected['EMA_20'], mode='lines', name='EMA 20', line=dict(color='red')))
        trading_figure.update_layout(title=f"Live Trading Data for {selected_symbol}", xaxis_title="Time", yaxis_title="Price")

    # Hypothetical analysis results for accuracy, performance, and additional metrics
    accuracy_stats = {'Model 1': 0.9, 'Model 2': 0.85}  # Example accuracy stats
    performance_metrics = {'Profit/Loss': [100, 200, 150, 300]}  # Example performance metrics
    additional_metrics = {'Metric 1': [0.1, 0.2, 0.3, 0.4]}  # Example additional metrics

    # Update accuracy figure
    accuracy_figure.add_trace(go.Bar(x=list(accuracy_stats.keys()), y=list(accuracy_stats.values()), marker_color='lightsalmon'))
    accuracy_figure.update_layout(title="Model Accuracy Stats", xaxis_title="Model", yaxis_title="Accuracy")

    # Update performance figure
    for metric, values in performance_metrics.items():
        performance_figure.add_trace(go.Scatter(x=list(range(len(values))), y=values, mode='lines+markers', name=metric))
    performance_figure.update_layout(title="Performance Metrics Over Time", xaxis_title="Time", yaxis_title="Metric Value")

    # Update additional metrics figure
    for metric, values in additional_metrics.items():
        additional_metrics_figure.add_trace(go.Scatter(x=list(range(len(values))), y=values, mode='lines', name=metric))
    additional_metrics_figure.update_layout(title="Additional Metrics", xaxis_title="Metric", yaxis_title="Value")

    return trading_figure, accuracy_figure, performance_figure, additional_metrics_figure


def update_output(input_value):
    """
    This function is triggered by a Dash input component. It processes the input value,
    fetches and processes data accordingly, and then updates the output components
    dynamically. The actual implementation of data fetching and processing will depend
    on the specific requirements and data structure of your application.
    """
    
    # Determine which input triggered the callback
    ctx = callback_context
    if not ctx.triggered:
        # If not triggered by any input, prevent the update
        raise PreventUpdate
    
    # Placeholder for fetching and processing data based on input_value
    # This could interact with your financial data sources, machine learning models, or blockchain
    data = fetch_data_based_on_input(input_value)  # This should be defined elsewhere in your code
    
    # Example processing to generate figures or metrics for Dash components
    if data is not None:
        # Process data to generate metrics or figures
        processed_data = process_data(data)  # This should be defined based on your application's logic
        
        # Generate a Plotly figure as an example output
        figure = go.Figure(data=[go.Scatter(x=processed_data['x'], y=processed_data['y'])])
        figure.update_layout(title="Dynamic Data Visualization")
        
        # Calculate additional metrics as examples
        metric_1 = processed_data['metric_1']
        metric_2 = processed_data['metric_2']
        
        return figure, metric_1, metric_2
    else:
        raise PreventUpdate

def fetch_data_based_on_input(input_value, start_date='2024-01-01', periods=100):
    """
    Fetch data based on the input_value. This function can be tailored to fetch data from different sources.
    """
    np.random.seed(0)
    df = pd.DataFrame({
        'x': pd.date_range(start=start_date, periods=periods),
        'y': np.random.randn(periods).cumsum(),
        'metric_1': np.random.rand(periods),
        'metric_2': np.random.rand(periods)
    })
    return df

def is_valid_symbol(symbol):
    """Check if the symbol is a valid ticker symbol."""
    return isinstance(symbol, str) and symbol.strip() and not symbol.isdigit() and symbol.isupper()

def fetch_and_process_data(symbols, start_date="2014-01-01", end_date="2024-12-31"):
    data_dict = {}
    for symbol in symbols:
        if not is_valid_symbol(symbol):
            print(f"Skipping invalid symbol: {symbol}")
            continue

        try:
            df = yf.download(symbol, start=start_date, end=end_date)
            if df.empty:
                print(f"No data available for {symbol}.")
                continue

            # Calculate indicators
            df['SMA_20'] = ta.trend.sma_indicator(df['Close'], window=20)
            df['SMA_50'] = ta.trend.sma_indicator(df['Close'], window=50)
            df['EMA_20'] = ta.trend.ema_indicator(df['Close'], window=20)
            df['RSI'] = ta.momentum.rsi(df['Close'], window=14)
            macd = ta.trend.MACD(df['Close'])
            df['MACD'] = macd.macd()
            df['MACD_signal'] = macd.macd_signal()
            bollinger = ta.volatility.BollingerBands(df['Close'])
            df['Bollinger_hband'] = bollinger.bollinger_hband()
            df['Bollinger_lband'] = bollinger.bollinger_lband()
            
            data_dict[symbol] = df
        
        except Exception as e:
            print(f"An error occurred while processing {symbol}: {e}")
            # Handle error appropriately

    return data_dict  # Return the complete dictionary after processing all symbols
def generate_placeholder_figure(text):
    fig = go.Figure()
    fig.add_annotation(text=text, x=0.5, y=0.5, showarrow=False, yanchor='middle', xanchor='center')
    fig.update_layout(title=text)
    return fig

def preprocess_text(text, *args, **kwargs):
    """
    Preprocess text using Spacy: tokenization, lemmatization, removal of stop words, and punctuation.
    This version ignores additional positional and keyword arguments.
    
    Args:
        text (str): The text to preprocess.
        
    Returns:
        str: The preprocessed text. Returns an empty string if an error occurs during preprocessing.
    """
    # Convert input to string if not already
    text = str(text)

    try:
        doc = nlp(text)
        preprocessed_text = ' '.join(token.lemma_ for token in doc if not token.is_stop and not token.is_punct)
        return preprocessed_text.lower()
    except Exception as e:
        print(f"An error occurred during text preprocessing: {e}")
        return ""  # Return an empty string in case of an error

def extract_images_from_page(page):
    images = []
    image_list = page.get_images(full=True)
    if image_list:
        for image in image_list:
            xref = image[0]
            base_image = page.get_pixmap(matrix=fitz.Matrix(2, 2), xref=xref)
            image_bytes = base_image.tobytes()
            image = Image.open(io.BytesIO(image_bytes))
            images.append(image)
    return images

# Function to preprocess and extract features from an image
def preprocess_and_extract_features_from_image(image, model_vgg16):
    image_resized = image.resize((224, 224))
    image_array = img_to_array(image_resized)
    image_preprocessed = preprocess_input(np.expand_dims(image_array, axis=0))
    features = model_vgg16.predict(image_preprocessed).flatten()
    return features


# Placeholder function for combining text and image features
pdf_paths = [
    r"C:\Users\Adhiraj Singh\OneDrive\Desktop\trading_project\41775536-Market-Wizards.pdf",
    r"C:\Users\Adhiraj Singh\OneDrive\Desktop\trading_project\Steve-Nison-Japanese-Candlestick-Charting-Techniques-Prentice-Hall-Press-2001.pdf",
    r"C:\Users\Adhiraj Singh\OneDrive\Desktop\trading_project\The Intelligent Investor - BENJAMIN GRAHAM.pdf",
    r"C:\Users\Adhiraj Singh\OneDrive\Desktop\trading_project\John J Murphy - Technical Analysis Of The Financial Markets.pdf",
]

def extract_features_from_pdf(pdf_paths, model_vgg16):
    all_text_features = []
    all_image_features = []
    
    for pdf_path in pdf_paths:
        text_content = []
        image_features = []
        
        try:
            with fitz.open(pdf_path) as doc:
                for page in doc:
                    text_content.append(page.get_text())
                    images = extract_images_from_page(page)
                    for image in images:
                        features = preprocess_and_extract_features_from_image(image, model_vgg16)
                        image_features.append(features)
                        
            processed_text = preprocess_text(" ".join(text_content))
            all_text_features.append(processed_text)
            all_image_features.extend(image_features)
        
        except Exception as e:
            print(f"Error processing {pdf_path}: {e}")
    
    # Here, you might want to combine or further process text and image features
    # For simplicity, we're just returning them
    return all_text_features, all_image_features

    # Your logic to process text and image features
async def login_to_ig_v3_async():
    url = "https://demo-api.ig.com/gateway/deal/session"
    headers = {
        "Content-Type": "application/json",
        "Accept": "application/json; charset=UTF-8",
        "X-IG-API-KEY": "f6724afe141b88edba5d9e4f15cb5f537af73242",  # Your API key
        "Version": "3"  # Specifying version 3 of the API
    }
    payload = {
        "identifier": os.getenv('IG_IDENTIFIER'),  # Use environment variable for the identifier
        "password": os.getenv('IG_PASSWORD')  # Use environment variable for the password
    }
    async with aiohttp.ClientSession() as session:
        try:
            async with session.post(url, headers=headers, json=payload) as response:
                if response.status == 200:
                    data = await response.json()
                    logging.info("Successfully logged in to IG API with v3.")
                    # Extract and return OAuth tokens
                    return {
                        "access_token": data['oauthToken']['access_token'],
                        "refresh_token": data['oauthToken']['refresh_token'],
                        "token_type": data['oauthToken']['token_type'],
                        "expires_in": data['oauthToken']['expires_in'],
                        "account_id": "Z5ISFT"  # Your account ID, may be used later
                    }
                else:
                    logging.error(f"Failed to log in to IG API with v3: HTTP {response.status}")
                    return None
        except Exception as e:
            logging.error(f"An error occurred during IG API v3 login: {e}")
            return None
        # and returns {"access_token": "<YOUR_ACCESS_TOKEN>"}
    
async def async_get_watchlists(access_token, account_id):
    url = "https://demo-api.ig.com/gateway/deal/watchlists"
    headers = {
        "Content-Type": "application/json; charset=UTF-8",
        "Accept": "application/json; charset=UTF-8",
        "Authorization": f"Bearer {access_token}",  # Use the OAuth access token for authorization
        "IG-ACCOUNT-ID": "Z5ISFT",  # Specify the account ID for which to retrieve watchlists
        "X-IG-API-KEY": "f6724afe141b88edba5d9e4f15cb5f537af73242"  # Your API key
    }

    async with aiohttp.ClientSession() as session:
        async with session.get(url, headers=headers) as response:
            try:
                response.raise_for_status()  # Raises an error for 4XX/5XX responses
                watchlists = await response.json()
                logging.info("List of watchlists retrieved successfully.")
                return watchlists
            except aiohttp.ClientResponseError as e:
                logging.error(f"Failed to retrieve watchlists: HTTP {e.status} - {e.reason}")
                return None
            except Exception as e:
                logging.error(f"An unexpected error occurred while retrieving watchlists: {e}")
                return None

# Example of how to call this function
# import asyncio
# asyncio.run(async_get_watchlists('your_cst_token', 'your_x_security_token'))
async def fetch_data_alpaca_async(session, symbol, start_date, end_date, timeframe='1D'):
    url = f"{ALPACA_BASE_URL}/v2/stocks/{symbol}/bars"
    # Ensure the 'start' and 'end' parameters are passed as strings in YYYY-MM-DD format
    params = {
        'start': start_date,  # Correctly formatted start date
        'end': end_date,    # Correctly formatted end date
        'timeframe': timeframe  # Valid timeframe value ('1D' for daily bars)
    }
    headers = {
        'APCA-API-KEY-ID': '  ',
        'APCA-API-SECRET-KEY': ' ',
    }

    try:
        async with session.get(url, headers=headers, params=params) as response:
            if response.status == 404:
                logging.error(f"Failed to fetch data from Alpaca for {symbol}: HTTP Status 404, Message: Not Found")
                return pd.DataFrame()
            response.raise_for_status()  # Will raise an HTTPError if the status is 4xx, 5xx
            data = await response.json()
            if 'bars' in data:
                df = pd.DataFrame(data['bars'])
                logging.info(f"Data fetched successfully for {symbol}.")
                return df
            else:
                logging.error(f"No 'bars' data found in response for {symbol}.")
                return pd.DataFrame()
    except Exception as e:
        logging.error(f"Error fetching data for {symbol}: {e}")
        return pd.DataFrame()

async def fetch_data_async(symbol, timeframe='1d', period='1mo'):
    """
    Asynchronously fetch historical data for the given symbol.
    """
    # Implement fetching data from your data source (e.g., Alpaca, Yahoo Finance)
    return pd.DataFrame()  # Return a DataFrame with historical data
# Improved function to apply technical indicators with error handling
def apply_technical_indicators(df):
    required_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
    if not all(column in df.columns for column in required_columns):
        logging.error("One or more required columns are missing in the data.")
        return df  # Return the original DataFrame without modifications

    try:
        # RSI
        df['RSI'] = ta.momentum.rsi(df['Close'])

        # MACD
        macd = ta.trend.MACD(df['Close'])
        df['MACD'] = macd.macd()
        df['MACD_signal'] = macd.macd_signal()

        # Check for candlestick attribute support and apply if available
        if hasattr(ta, 'candlestick'):
            # Example: Hammer pattern
            df['Hammer'] = ta.candlestick.cdl_hammer(df['Open'], df['High'], df['Low'], df['Close'])
        else:
            logging.warning("The 'ta' library does not support 'candlestick' attribute. Skipping candlestick patterns.")

        # Additional indicators as per your strategy...
        # Bollinger Bands
        bollinger = ta.volatility.BollingerBands(df['Close'])
        df['Bollinger_High'] = bollinger.bollinger_hband()
        df['Bollinger_Low'] = bollinger.bollinger_lband()

        # Stochastic Oscillator
        df['Stochastic_Oscillator'] = ta.momentum.stoch(df['High'], df['Low'], df['Close'])

    except Exception as e:
        logging.error(f"Error applying technical indicators: {e}")

    return df
def apply_technical_indicators(data):
    """
    Apply various technical indicators on the data.
    """
    # MACD and Signal Line
    macd_indicator = ta.trend.MACD(data['Close'])
    data['macd'] = macd_indicator.macd()
    data['macd_signal'] = macd_indicator.macd_signal()

    # RSI
    data['rsi'] = ta.momentum.rsi(data['Close'])

    # DEMA
    data['dema'] = ta.trend.dema_indicator(data['Close'], window=21)

    # SuperTrend
    st_indicator = ta.trend.STCIndicator(data['Close'], data['High'], data['Low'])
    data['supertrend'] = st_indicator.stc()

    return data

def generate_trade_signal(data):
    """
    Generate trade signal based on technical indicators and market conditions.
    """
    fig = go.Figure()
    latest_data = data.iloc[-1]
    signal = 0  # No action by default

    # Buy signal conditions
    if latest_data['macd'] > latest_data['macd_signal'] and latest_data['rsi'] < 70:
        signal = 1  # Buy
    
    # Sell signal conditions
    elif latest_data['macd'] < latest_data['macd_signal'] and latest_data['rsi'] > 30:
        signal = -1  # Sell
    
    return signal, fig
    

async def fetch_cfd_market_data_async(access_token, epic, resolution='D', num_points=10, account_id=None):
    """
    Asynchronously fetch historical market data for a given CFD market.

    Args:
        access_token (str): The OAuth access token for authentication.
        epic (str): The market epic identifier.
        resolution (str): The data resolution ('D' for daily, 'H' for hourly, etc.).
        num_points (int): Number of data points to fetch.
        account_id (str, optional): The account ID if required for the endpoint.

    Returns:
        A JSON object containing the historical market data.
    """
    url = f"{IG_API_BASE_URL}/prices/{epic}/{resolution}/{num_points}"
    headers = {
        "X-IG-API-KEY": " ",  # Your API key
        "Authorization": f"Bearer {access_token}",  # OAuth token for authentication
        "Content-Type": "application/json",
        "Accept": "application/json",
    }
    # Include IG-ACCOUNT-ID header if an account ID is provided
    if account_id:
        headers["IG-ACCOUNT-ID"] = " ",

    async with aiohttp.ClientSession() as session:
        try:
            async with session.get(url, headers=headers) as response:
                if response.status == 200:
                    data = await response.json()
                    logging.info(f"Successfully fetched market data for {epic}.")
                    return data  # Adjust based on the actual response structure
                else:
                    logging.error(f"Failed to fetch market data for {epic}: Status {response.status}, Message: {await response.text()}")
                    return None
        except Exception as e:
            logging.error(f"An error occurred while fetching market data for {epic}: {e}")
            return None

# Example of how to call this function
# Assuming your trading logic is implemented in another function
def fetch_cfd_market_data(ig_service, epic, resolution='D', num_points=10):
    """
    Fetch historical market data for a given CFD market.

    :param ig_service: The authenticated IG service instance.
    :param epic: The market epic identifier.
    :param resolution: The data resolution ('D' for daily, 'H' for hourly, etc.).
    :param num_points: Number of data points to fetch.
    :return: Historical market data.
    """
    historical_prices = ig_service.fetch_historical_prices_by_epic_and_num_points(epic, resolution, num_points)
    return historical_prices['prices']
async def place_cfd_trade_async(access_token, account_id, epic, direction, size, strategy='MARKET', limit_price=None, stop_loss=None, take_profit=None):
    """
    Asynchronously place a CFD trade on the IG platform with GBP as the currency.

    :param access_token: The OAuth access token for authentication.
    :param account_id: The IG account ID.
    :param epic: The market epic identifier.
    :param direction: 'BUY' or 'SELL'.
    :param size: The size of the trade.
    :param strategy: Trading strategy ('MARKET', 'LIMIT', etc.). Default is 'MARKET'.
    :param limit_price: Limit price for 'LIMIT' orders.
    :param stop_loss: Stop loss value.
    :param take_profit: Take profit value.
    """
    url = f"{IG_API_BASE_URL}/positions/otc"
    headers = {
        "X-IG-API-KEY": "f6724afe141b88edba5d9e4f15cb5f537af73242",
        "Authorization": f"Bearer {access_token}",
        "Content-Type": "application/json",
        "Accept": "application/json",
        "IG-ACCOUNT-ID": "Z5ISFT",  # Needed for operations on specific accounts
    }
    payload = {
        "currencyCode": "GBP",
        "direction": direction.upper(),
        "epic": epic,
        "orderType": strategy,
        "size": size,
    }
    # Correct handling for limit and take profit
    if strategy == 'LIMIT' and limit_price:
        payload["limitLevel"] = limit_price
    if take_profit:
        payload["takeProfitLevel"] = take_profit
    if stop_loss:
        payload["stopLossLevel"] = stop_loss

    data = json.dumps(payload)

    async with aiohttp.ClientSession() as session:
        async with session.post(url, headers=headers, data=data) as response:
            if response.status in [200, 201]:
                response_data = await response.json()
                logging.info(f"Trade executed successfully: {response_data}")
            else:
                response_text = await response.text()
                logging.error(f"Failed to execute trade. Status code: {response.status}, Response: {response_text}")

# Example of how you might call this function in an async context
# import asyncio
# asyncio.run(place_cfd_trade_async('EPIC_CODE', 'BUY', 10, strategy='LIMIT', limit_price=1.250, stop_loss=1.240, take_profit=1.260))
async def fetch_open_cfd_positions():
    # Assume async_login_to_ig now returns an access token for OAuth
    oauth_tokens = await async_login_to_ig()
    access_token = oauth_tokens.get("access_token")
    
    if not access_token:
        logging.error("Failed to obtain access token")
        return None

    headers = {
        "X-IG-API-KEY": os.getenv(""),
        "Authorization": f"Bearer {access_token}",
        "Content-Type": "application/json",
        "Accept": "application/json",
    }

    url = f"{IG_API_BASE_URL}/positions"

    async with aiohttp.ClientSession() as session:
        async with session.get(url, headers=headers) as response:
            if response.status == 200:
                data = await response.json()
                return data['positions']  # Assuming the response structure contains a 'positions' key
            else:
                logging.error(f"Failed to fetch open CFD positions. Status code: {response.status}")
                return None

async def async_close_cfd_position(deal_id, direction, size, strategy='MARKET', stop_loss=None, take_profit=None, limit_price=None):
    """
    Asynchronously close an open CFD position on the IG platform.
    """
    # Obtain OAuth tokens via login
    oauth_tokens = await async_login_to_ig()
    access_token = oauth_tokens.get("access_token")

    if not access_token:
        logging.error("Failed to obtain access token")
        return

    headers = {
        "X-IG-API-KEY": " ",
        "Authorization": f"Bearer {access_token}",
        "Content-Type": "application/json",
        "Accept": "application/json",
    }

    payload = {
        "dealId": deal_id,
        "direction": "SELL" if direction.upper() == "BUY" else "BUY",
        "size": size,
        "orderType": strategy.upper(),
    }

    if stop_loss:
        payload["stopLoss"] = {"value": stop_loss}
    if take_profit:
        payload["takeProfit"] = {"value": take_profit}
    if limit_price and strategy.upper() == "LIMIT":
        payload["limitPrice"] = limit_price

    async with aiohttp.ClientSession() as session:
        async with session.post(f"{IG_API_BASE_URL}/positions/otc", headers=headers, json=payload) as response:
            if response.status in [200, 201]:
                response_data = await response.json()
                logging.info(f"Position closed successfully: {response_data}")
            else:
                response_text = await response.text()
                logging.error(f"Failed to close position. Status code: {response.status}, Response: {response_text}")

async def fetch_data_async(session, symbol, start_date, end_date):
    # Alpha Vantage endpoint for daily historical data
    url = f"https://www.alphavantage.co/query?function=TIME_SERIES_DAILY&symbol={symbol}&apikey={ALPHA_VANTAGE_API_KEY}&datatype=json&outputsize=full"

    async with session.get(url) as response:
        if response.status == 200:
            data = await response.json()
            # Extract the "Time Series (Daily)" data from the response
            time_series = data.get('Time Series (Daily)', {})
            # Convert the time series data to a pandas DataFrame
            df = pd.DataFrame.from_dict(time_series, orient='index', dtype=float)
            # Rename the columns to more recognizable names
            df.columns = ['Open', 'High', 'Low', 'Close', 'Volume']
            # Convert the index to datetime
            df.index = pd.to_datetime(df.index)
            # Optionally, filter the DataFrame based on the start_date and end_date
            df_filtered = df.loc[start_date:end_date]
            return df_filtered
        else:
            logging.error(f"Failed to fetch data for {symbol}")
            return pd.DataFrame()
async def fetch_market_data(symbols, period="60d", interval="1d", fields=None):
    """
    Asynchronously fetches market data for the specified symbols over a given period and interval.

    Parameters:
    - symbols (list of str): The stock symbols to fetch data for.
    - period (str): The period over which to fetch historical market data (default "60d" for 60 days).
    - interval (str): The data interval (default "1d" for daily data).
    - fields (list of str): Specific data fields to include in the output. If None, all available fields are included.

    Returns:
    - pd.DataFrame: A DataFrame containing the concatenated fetched market data for all symbols.
    """
    loop = asyncio.get_event_loop()

    with ThreadPoolExecutor() as executor:
        tasks = [
            loop.run_in_executor(
                executor,
                fetch_data_for_symbol,  # This is a synchronous wrapper function we'll define next
                symbol,
                period,
                interval,
                fields
            )
            for symbol in symbols
        ]
        results = await asyncio.gather(*tasks)
    
    # Concatenate all fetched dataframes
    all_data = pd.concat(results, axis=0)
    return all_data

def fetch_data_for_symbol(symbol, period, interval, fields):
    """
    Synchronous function to fetch market data for a single symbol.
    """
    try:
        # Fetch data for the symbol
        stock_data = yf.Ticker(symbol)
        df = stock_data.history(period=period, interval=interval)
        
        # Check if specific fields are requested
        if fields is not None and isinstance(fields, list):
            df = df[fields]
        
        # Add a 'Symbol' column to distinguish between different stocks
        df['Symbol'] = symbol
        
    except Exception as e:
        print(f"Error fetching data for {symbol}: {e}")
        df = pd.DataFrame()  # Return an empty DataFrame in case of error
    
    return df

async def fetch_real_time_price(symbol):
    url = f"{ALPACA_BASE_URL}/v2/stocks/{symbol}/bars/latest"
    headers = {
        "APCA-API-KEY-ID": ALPACA_API_KEY,
        "APCA-API-SECRET-KEY": ALPACA_SECRET_KEY
    }
    async with aiohttp.ClientSession() as session:
        try:
            async with session.get(url, headers=headers) as response:
                if response.status == 200:
                    data = await response.json()
                    current_price = data['bar']['c']  # Adjust based on the actual JSON response structure
                    return current_price
                else:
                    logging.error(f"Failed to fetch real-time price for {symbol}. Status code: {response.status}")
                    return None
        except Exception as e:
            logging.error(f"An error occurred while fetching real-time price for {symbol}: {e}")
            return None

def fetch_data_multiple_symbols(symbols, start_date='2014-01-01', end_date='2024-12-31', interval='1d'):
    """
    Fetch data for given symbols from Yahoo Finance for a specific date range.

    Args:
    - symbols (list of str): List of ticker symbols for the stocks.
    - start_date (str, optional): Start date for the data fetch in 'YYYY-MM-DD' format. Defaults to '2014-01-01'.
    - end_date (str, optional): End date for the data fetch in 'YYYY-MM-DD' format. Defaults to '2024-12-31'.
    - interval (str): The interval between data points. Defaults to '1d' for daily data.

    Returns:
    - dict: A dictionary with symbols as keys and fetched data as pandas DataFrame values.
    """
    results = {}
    for symbol in symbols:
        logging.info(f"Fetching data from Yahoo Finance for {symbol} over the period {start_date} to {end_date}...")
        data = yf.download(symbol, start=start_date, end=end_date, interval=interval)
        if not data.empty:
            logging.info(f"Data fetched successfully for {symbol}.")
            results[symbol] = data
        else:
            logging.error(f"Failed to fetch data for {symbol}.")
    return results

# Example usage
symbols = ['AAPL', 'GOOGL', 'MSFT']  # Apple, Google, Microsoft
data = fetch_data_multiple_symbols(symbols)
for symbol, df in data.items():
    if not df.empty:
        print(f"Data for {symbol} from 2014-01-01 to 2024-12-31:", df.head())
    else:
        print(f"Data fetching was unsuccessful for {symbol}.")

def fetch_alpha_vantage_data(symbol, interval='5min'):
    if not ALPHA_VANTAGE_API_KEY:
        logging.error("ALPHA_VANTAGE_API_KEY environment variable not set.")
        return None

    logging.info(f"Fetching data from Alpha Vantage for {symbol} with interval {interval}...")
    ts = TimeSeries(key='18RCKNOU5MMQHXJU', output_format='pandas')

    try:
        data, meta_data = ts.get_intraday(symbol=symbol, interval=interval, outputsize='compact')
        if not data.empty:
            # Alpha Vantage sometimes changes its column names, ensure they are standardized for your application
            data.columns = ['Open', 'High', 'Low', 'Close', 'Volume']
            logging.info(f"Data fetched successfully from Alpha Vantage for {symbol}.")
            return data
        else:
            logging.warning(f"No data received from Alpha Vantage for {symbol}. Check if the symbol is correct.")
            return None
    except Exception as e:  # Catching a more general exception
        logging.error(f"Error fetching data from Alpha Vantage for {symbol}: {e}")
        return None
def fetch_latest_news_sentiment(ticker):
    """
    Fetch the latest news sentiment for a given ticker using Alpha Vantage's News Sentiment API and process it into a sentiment matrix.

    Parameters:
        ticker (str): The stock ticker symbol to analyze.

    Returns:
        dict: A dictionary containing the count of 'POSITIVE' and 'NEGATIVE' sentiments, or None if data is unavailable.
    """
    url = "https://www.alphavantage.co/query"
    params = {
        "function": "NEWS_SENTIMENT",
        "tickers": ticker,
        "apikey": '18RCKNOU5MMQHXJU',
    }

    # Initialize sentiment matrix
    sentiment_matrix = {'POSITIVE': 0, 'NEGATIVE': 0}

    try:
        response = requests.get(url, params=params)
        response.raise_for_status()  # Raises a HTTPError for bad responses
        data = response.json()

        # Extract and process sentiment data
        for article in data.get('feed', []):
            sentiment_score = article.get('sentiment_score', 0)  # Assuming a neutral default score
            if sentiment_score >= 0:
                sentiment_matrix['POSITIVE'] += 1
            else:
                sentiment_matrix['NEGATIVE'] += 1

        logging.info(f"Sentiment data processed for {ticker}.")
        return sentiment_matrix

    except requests.exceptions.HTTPError as http_err:
        logging.error(f"HTTP error occurred while fetching sentiment for {ticker}: {http_err}")
    except Exception as err:
        logging.error(f"An unexpected error occurred while fetching sentiment for {ticker}: {err}")

    return sentiment_matrix

# Example usage for multiple tickers
tickers = ["AAPL", "MSFT", "GOOGL"]
for ticker in tickers:
    sentiment_matrix = fetch_latest_news_sentiment(ticker)
    print(f"Sentiment Matrix for {ticker}: {sentiment_matrix}\n")

def fetch_live_data(symbol, interval='5min', outputsize='compact'):
    ts = TimeSeries(key=ALPHA_VANTAGE_API_KEY, output_format='pandas')
    data, _ = ts.get_intraday(symbol=symbol, interval=interval, outputsize=outputsize)
    logging.info(f"Fetched live data for {symbol}")
    data = data.iloc[::-1]
    data.columns = ['Open', 'High', 'Low', 'Close', 'Volume']
    return data
def fetch_quandl_data(symbol):
    end_date = datetime.now()
    start_date = end_date - timedelta(days=365.25 * 10)  # Adjust the duration as needed
    try:
        data = quandl.get(symbol, start_date=start_date.date(), end_date=end_date.date())
        logging.info(f"Data fetched for {symbol}")
        return data
    except quandl.errors.quandl_error.LimitExceededError as e:
        logging.error("Quandl API limit exceeded. Please check your API key and limit.")
    except Exception as e:
        logging.error(f"An error occurred while fetching Quandl data: {e}")        
def fetch_data(symbol='MSFT', interval='5min', period='1y'):
    cached_data = read_cache(symbol)
    if cached_data:
        logging.info("Using cached data.")
        return pd.DataFrame(cached_data)  # Ensure this line correctly converts cached data back to DataFrame
    else:
        logging.info("Fetching new data.")
        try:
            data = fetch_alpha_vantage_data(symbol, interval)
            if data is not None and not data.empty:
                cache_data(symbol, data.to_dict('records'))  # Cache newly fetched data
                return data
            else:
                raise ValueError("Empty data from Alpha Vantage.")
        except Exception as e:
            logging.error(f"Error or rate limit hit with Alpha Vantage: {e}")
            data = (symbol, period, '1d')  # Adjusted for Yahoo Finance
            if data is not None and not data.empty:
                cache_data(symbol, data.to_dict('records'))  # Cache this as fallback data
            return data
async def fetch_data_alpha_vantage(symbol):
    ts = TimeSeries(key='18RCKNOU5MMQHXJU', output_format='pandas')
    loop = asyncio.get_running_loop()
    try:
        # Check cache first
        cached_data = read_cache(symbol)
        if cached_data:
            logging.info(f"Using cached data for {symbol}.")
            return cached_data
        # Fetch new data
        data, _ = await loop.run_in_executor(None, ts.get_intraday, symbol, '5min', 'compact')
        # Cache the new data
        cache_data(symbol, data.to_dict('records'))
        return data.to_dict('records')
    except Exception as e:
        logging.warning(f"Fetching from Alpha Vantage failed for {symbol}: {e}")
        # Fallback to Yahoo Finance
        return await fetch_data_multiple_symbols(symbol)
async def fetch_data_alpaca(symbol):
    """
    Fetch data from Alpaca for a given symbol.
    """
    url = f"{ALPACA_BASE_URL}/v2/stocks/{symbol}/bars"
    headers = {
        'APCA-API-KEY-ID': 'PKHZ7M5JSIIAX1PCB3FI',
        'APCA-API-SECRET-KEY': 'NATavl5vfvPQrr7QRSEaPr2LB5wAQUyoxJDBRhkM',
    }
    async with aiohttp.ClientSession() as session:
        async with session.get(url, headers=headers) as response:
            data = await response.json()
    return data

async def fetch_new_data():
    logging.info("Fetching new data...")
    symbols = ['AAPL', 'MSFT', 'GOOGL']
    tasks = [fetch_data_alpha_vantage(symbol) for symbol in symbols]
    results = await asyncio.gather(*tasks)
    logging.info("Data fetched successfully.")
    # Print or process the results as needed
    print(results)
    return results # Placeholder for actual fetched data


def process_diagram(image_path):
    """Process diagram/image using Pillow."""
    img = Image.open(image_path)
    # Placeholder for image processing steps
    return img

    """
    Prepares and preprocesses data for LSTM models and other models, handling sequence creation, feature scaling, and dynamic feature selection.

    Args:
        data (pandas.DataFrame): The input data.
        sequence_length (int): The length of the sequences for LSTM training.
        target_look_ahead (int): The number of steps to look ahead for setting the target.
        feature_columns (list of str, optional): Specific columns to use as features for LSTM. Defaults to ['Close'] if None.

    Returns:
        X (numpy.ndarray): Feature sequences for LSTM.
        y (numpy.ndarray): Target sequences for LSTM.
        features (pandas.DataFrame): Selected features for other models.
        target (pandas.Series): The target for other models.
        scaler (object): Scaler used for LSTM feature scaling.
    """
def prepare_and_preprocess_data(data, sequence_length=60, target_look_ahead=1, feature_columns=None):
    """
    Prepares and preprocesses the data, adapting for LSTM inputs and scaling.
    """
    if 'Close' not in data.columns:
        raise ValueError("'Close' column is missing from the data.")
    
    feature_columns = feature_columns or [col for col in data.columns if data[col].dtype in [np.float64, np.int64]]
    missing_columns = set(feature_columns).difference(data.columns)
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")

    # Scale features for LSTM input
    lstm_scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_features = lstm_scaler.fit_transform(data[feature_columns])
    
    # Create sequences for LSTM
    X_lstm, y_lstm = [], []
    for i in range(sequence_length, len(data) - target_look_ahead + 1):
        X_lstm.append(scaled_features[i-sequence_length:i])
        y_lstm.append(scaled_features[i + target_look_ahead - 1, feature_columns.index('Close')])
    X_lstm, y_lstm = np.array(X_lstm), np.array(y_lstm)
    
    # Scale features for other models
    scaler_other = StandardScaler()
    features_scaled = scaler_other.fit_transform(data[feature_columns])
    features = pd.DataFrame(features_scaled, columns=feature_columns, index=data.index)
    
    # Generate target based on 'Close' price movement
    target = data['Close'].shift(-target_look_ahead) > data['Close']
    target = target.iloc[sequence_length-1:-target_look_ahead+1].reset_index(drop=True)
    
    return X_lstm, y_lstm, features, target.astype(int), lstm_scaler

def update_and_optimize_model(model_path, data, sequence_length=60, batch_size=32, epochs=10, validation_split=0.1, retrain_threshold=0.01, learning_rate_adjustment=True):
    """
    Single function handling model update: preprocessing, training, evaluation, and saving the updated model.
    """
    logging.basicConfig(level=logging.INFO)
    
    # Load the existing model
    model = load_model(model_path)

    # Preprocess the new data
    X_lstm, y_lstm, features, target, lstm_scaler = prepare_and_preprocess_data(data, sequence_length)

    # Adjust the learning rate if necessary
    if learning_rate_adjustment:
        initial_lr = 0.001
        decay_rate = initial_lr / epochs
        optimizer = Adam(learning_rate=initial_lr, decay=decay_rate)
        model.compile(optimizer=optimizer, loss='mean_squared_error', metrics=['accuracy'])
        logging.info(f"Learning rate adjusted to {initial_lr} with decay {decay_rate}.")

    # Evaluate the model on new data before deciding on retraining
    mse = model.evaluate(X_lstm, y_lstm, verbose=0)[0]
    logging.info(f"Evaluation MSE: {mse}")
    
    # Retrain if the model performance is not satisfactory
    if mse > retrain_threshold:
        logging.info("Retraining model due to performance below the threshold...")
        model.fit(X_lstm, y_lstm, batch_size=batch_size, epochs=epochs, validation_split=validation_split, callbacks=[EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)])
        logging.info("Model retraining completed.")
    else:
        logging.info("Model performance is satisfactory. No retraining necessary.")
    
    # Save the updated model
    updated_model_path = model_path.replace('.h5', '_updated.keras')  # Use '.keras' extension for updated model
    model.save(r"C:\Users\Adhiraj Singh\OneDrive\Desktop\trading_project\saved_models\lstm_model.keras")
    logging.info(f"Updated model saved to {updated_model_path}")

# Example usage
# data = pd.read_csv("your_new_data.csv")  # Load your new market data here
# model_path = 'path/to/your/model.h5'
# update_and_optimize_model(model_path, data)

def enhance_features(data, symbol, news_data=None, other_market_data=None):
    if data.empty:
        raise ValueError(f"No data to enhance for {symbol}.")
    
    required_columns = ['Close', 'Volume', 'High', 'Low', 'Open']
    if not all(column in data.columns for column in required_columns):
        raise ValueError("Missing one or more required columns in the data.")

    try:
        data = data.copy()
        data['RSI'] = ta.momentum.rsi(data['Close'])
        data['MACD'] = ta.trend.MACD(data['Close']).macd_diff()
        data['Volume_Rate_Change'] = ta.volume.volume_price_trend(data['Close'], data['Volume'])
        data['CCI'] = ta.trend.CCIIndicator(data['High'], data['Low'], data['Close']).cci()
        
        dynamic_features_based_on_feedback(data)
        
        if news_data is not None:
            data['Sentiment_Score'] = apply_sentiment_analysis(data, news_data)
        
        if other_market_data is not None:
            data = apply_intermarket_analysis(data, other_market_data)
        
        return data
    except Exception as e:
        raise ValueError(f"An error occurred while enhancing features for {symbol}: {e}")
    
def execute_advanced_trading_strategy(data, symbol, trading_api):
    """
    Execute trades based on an advanced strategy that includes SMA crossovers and additional conditions.

    Parameters:
    - data: DataFrame containing market data and indicators.
    - symbol: The trading symbol (e.g., 'AAPL').
    - trading_api: An instance of the trading API class for executing trades.
    """
    # Check for the presence of required columns/indicators
    required_columns = ['Close', 'SMA_20', 'SMA_50', 'Volume']
    if not all(column in data.columns for column in required_columns):
        logging.error("Missing one or more required columns in the data.")
        return

    # Calculate buy and sell signals based on SMA crossover and additional conditions
    buy_signals = (data['Close'] > data['SMA_20']) & (data['Close'].shift(1) <= data['SMA_20'].shift(1))
    sell_signals = (data['Close'] < data['SMA_20']) & (data['Close'].shift(1) >= data['SMA_20'].shift(1))

    # Additional logic for refining signals, e.g., based on volume or other indicators
    buy_signals = buy_signals & (data['Volume'] > data['Volume'].rolling(window=10).mean())
    sell_signals = sell_signals & (data['Volume'] > data['Volume'].rolling(window=10).mean())

    # Execute trades based on signals
    for index, signal in buy_signals.iteritems():
        if signal:
            # Implement your trade execution logic here, e.g., place a buy order
            logging.info(f"Placing BUY order for {symbol} on {data.index[index]}")
            # Example: trading_api.place_order(symbol=symbol, quantity=10, order_type='buy')
            api.submit_order(symbol=symbol, qty=quantity, side='sell', type='market', stop_loss=stop_loss)
    for index, signal in sell_signals.iteritems():
        if signal:
            # Implement your trade execution logic here, e.g., place a sell order
            logging.info(f"Placing SELL order for {symbol} on {data.index[index]}")
            # Example: trading_api.place_order(symbol=symbol, quantity=10, order_type='sell')
            api.submit_order(symbol=symbol, qty=quantity, side='sell', type='market', stop_loss=stop_loss)
    logging.info(f"Strategy executed for {symbol}. Buy signals: {buy_signals.sum()}, Sell signals: {sell_signals.sum()}.")

# Example usage
# This assumes 'market_data' is your DataFrame containing the necessary columns and 'trading_api' is an instance of your trading API class.
# execute_advanced_trading_strategy(market_data, 'AAPL', trading_api)    
def calculate_sentiment_score(news_data, date):
    return np.random.rand()

def calculate_intermarket_indicators(data, other_market_data):
    return pd.DataFrame({'Date': data['Date'], 'Intermarket_Feature': np.random.rand(len(data))})
    
def calculate_advanced_macd(data, fast_period=12, slow_period=26, signal_period=9):
    # Calculate MACD using TA-Lib
    data['macd'], data['macd_signal'], data['macd_hist'] = ta.trend.macd(data['Close'],
                                                                         window_slow=slow_period,
                                                                         window_fast=fast_period,
                                                                         window_sign=signal_period)

    # Modify the histogram to identify strong trends
    data['macd_hist_color'] = np.where(data['macd_hist'] > 0, 'green', 'red')
    return data

def calculate_impulse_macd(data):
    # Calculate the MACD and signal line
    macd_indicator = ta.trend.MACD(data['Close'])
    data['macd'] = macd_indicator.macd()
    data['macd_signal'] = macd_indicator.macd_signal()
    data['macd_diff'] = macd_indicator.macd_diff()

    # Calculate RSI for dynamic zones
    data['rsi'] = ta.momentum.rsi(data['Close'])

    # Define market condition
    data['market_condition'] = np.where(data['rsi'] > 70, 'overbought', np.where(data['rsi'] < 30, 'oversold', 'neutral'))

    # Generate Impulse MACD signals
    data['impulse_signal'] = 0
    data.loc[(data['macd'] > data['macd_signal']) & (data['market_condition'] == 'neutral'), 'impulse_signal'] = 1
    data.loc[(data['macd'] < data['macd_signal']) & (data['market_condition'] == 'neutral'), 'impulse_signal'] = -1

    return data

def execute_trades_based_on_advanced_macd(df, symbol, api, max_risk_per_trade=0.01, available_capital=10000):
    df = calculate_impulse_macd(df)
    atr = ta.volatility.AverageTrueRange(df['High'], df['Low'], df['Close']).average_true_range()
    
    for index, row in df.iterrows():
        current_price = row['Close']
        atr_value = atr.loc[index]
        risk_per_share = atr_value * 2  # Example: setting risk per share as twice the ATR value
        
        # Calculate quantity based on risk management
        quantity = int((available_capital * max_risk_per_trade) / risk_per_share)
        
        # Define the stop loss based on ATR
        stop_loss = current_price - risk_per_share if row['impulse_signal'] == 1 else current_price + risk_per_share
        
        # Buy signals
        if row['impulse_signal'] == 1:
            print(f"BUY {quantity} of {symbol} at {current_price}, SL: {stop_loss}, on {index.strftime('%Y-%m-%d')}")
            api.submit_order(symbol=symbol, qty=quantity, side='buy', type='market', stop_loss=stop_loss)
        
        # Sell signals
        elif row['impulse_signal'] == -1:
            print(f"SELL {quantity} of {symbol} at {current_price}, SL: {stop_loss}, on {index.strftime('%Y-%m-%d')}")
            api.submit_order(symbol=symbol, qty=quantity, side='sell', type='market', stop_loss=stop_loss)

async def execute_trade(symbol, signal, quantity):
    """
    Execute trade based on the signal.
    """
    if signal == 1:
        logging.info(f"Buying {quantity} shares of {symbol}")
        api.submit_order(symbol=symbol, qty=quantity, side='buy', type='market')
    elif signal == -1:
        logging.info(f"Selling {quantity} shares of {symbol}")
        api.submit_order(symbol=symbol, qty=quantity, side='sell', type='market')    
def extract_features(img_path, model):
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    expanded_img_array = np.expand_dims(img_array, axis=0)
    preprocessed_img = preprocess_input(expanded_img_array)
    features = model.predict(preprocessed_img)
    return features.flatten()
                
def build_and_train_advanced_lstm_model(X_train_text, y_train, X_val_text, y_val, X_train_image, X_val_image):
    try:
        # Ensure correct reference to X_train for the text input shape
        text_input = Input(shape=(X_train_text.shape[1], X_train_text.shape[2]), name='text_input')

        # Convolutional layer for text feature extraction
        conv1_text = Conv1D(filters=64, kernel_size=3, activation='relu')(text_input)
        bn1_text = BatchNormalization()(conv1_text)

        # Applying Bidirectional LSTM on text data
        lstm1_text = Bidirectional(LSTM(100, return_sequences=True))(bn1_text)
        dropout1_text = Dropout(0.3)(lstm1_text)
        
        # Image input path
        image_input = Input(shape=(X_train_image.shape[1],), name='image_input')
        dense_img = Dense(256, activation='relu')(image_input)
        dropout_img = Dropout(0.3)(dense_img)

        # Combine text and image pathways
        combined_features = Concatenate()([dropout1_text, dropout_img])

        # Additional LSTM Layer to combine features
        lstm2_combined = LSTM(50, return_sequences=False)(combined_features)
        dropout2_combined = Dropout(0.3)(lstm2_combined)

        # Output layer
        output = Dense(1, activation='sigmoid')(dropout2_combined)

        # Creating the model
        model = Model(inputs=[text_input, image_input], outputs=output)

        # Compilation of the model
        optimizer = Adam(learning_rate=0.001)
        model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])

        # Implementing early stopping to prevent overfitting
        early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

        # Model training
        history = model.fit([X_train_text, X_train_image], y_train, epochs=100, batch_size=32, validation_data=([X_val_text, X_val_image], y_val), callbacks=[early_stopping], verbose=1)

        # Save the model
        model.save(r'C:\Users\Adhiraj Singh\OneDrive\Desktop\trading_project\saved_models\lstm_model.keras')  # Adjust the path as necessary

        return model, history

    except Exception as e:
        logging.error(f"Error in advanced model training: {e}")
        return None, None
def execute_strategy_with_dynamic_tp(data, symbol):
    for index, row in data.iterrows():
        if row['impulse_signal'] == 1:
            print(f"Buy signal at {index} for {symbol}")
            api.submit_order(symbol=symbol, qty=quantity, side='buy', type='market')
        elif row['impulse_signal'] == -1:
            print(f"Sell signal at {index} for {symbol}")
            api.submit_order(symbol=symbol, qty=quantity, side='sell', type='market')

        # Dynamic Take Profit Logic
        # This could be a placeholder. Implement according to your risk management strategy.
        # Example: Exit when MACD crosses below signal line after a buy signal
        if 'last_buy_index' in locals() and row['macd'] < row['macd_signal']:
            print(f"Dynamic Sell for Take Profit at {index} for {symbol}")
            api.submit_order(symbol=symbol, qty=quantity, side='sell', type='market')
            del last_buy_index  # Reset last buy index
def calculate_dema(data, period=200):
    ema1 = data['Close'].ewm(span=period, adjust=False).mean()
    ema2 = ema1.ewm(span=period, adjust=False).mean()
    dema = 2 * ema1 - ema2
    return dema
def calculate_supertrend(df, period=7, atr_multiplier=3):
    hl2 = (df['High'] + df['Low']) / 2
    atr = hl2.rolling(period).apply(lambda x: x.diff().abs().mean()) * atr_multiplier
    final_upperband = hl2 + atr
    final_lowerband = hl2 - atr
    supertrend = [True] * len(df)

    for i in range(1, len(df)):
        if df['Close'][i] > final_upperband[i - 1]:
            supertrend[i] = True
        elif df['Close'][i] < final_lowerband[i - 1]:
            supertrend[i] = False
        else:
            supertrend[i] = supertrend[i - 1]
            if supertrend[i] and final_lowerband[i] < final_lowerband[i - 1]:
                final_lowerband[i] = final_lowerband[i - 1]
            if not supertrend[i] and final_upperband[i] > final_upperband[i - 1]:
                final_upperband[i] = final_upperband[i - 1]

    df['SuperTrend'] = final_lowerband.where(supertrend, final_upperband)
    return df
def calculate_trade_size(current_price, signal, account_balance=100000, risk_per_trade=0.01):
    """
    Calculate trade size based on current price, signal, and account balance.
    """
    # This is a placeholder calculation; adjust according to your risk management strategy
    capital_at_risk = account_balance * risk_per_trade
    trade_size = int(capital_at_risk / current_price)  # Simplified calculation
    return trade_size
def generate_signals(data, account_balance=100000, risk_per_trade=0.01):
    """Generates trading signals based on a simple moving average strategy."""
    fig = go.Figure()
    signals = pd.Series(index=data.index, data=np.zeros(len(data)))
    signals[data['EMA_20'] > data['EMA_50']] = 1  # Buy signal
    signals[data['EMA_20'] < data['EMA_50']] = -1  # Sell signal
    
    # Loop through signals and execute trades
    for date, signal in signals.iteritems():
        if signal != 0:  # If there's a trading signal
            current_price = data.loc[date, 'Close']  # Assuming 'Close' price is available
            trade_size = asyncio.run(calculate_trade_size(current_price, signal, account_balance))
            
            # Placeholder for trade execution, adapt as per your API
            print(f"On {date}, executing {'BUY' if signal > 0 else 'SELL'} for {trade_size} units of asset at price {current_price}.")

    return signals, fig

# Example usage
# Make sure your data DataFrame includes 'EMA_20' and 'EMA_50' columns
# data = pd.DataFrame(...)  # Your DataFrame with 'EMA_20', 'EMA_50', and 'Close'
# account_balance = 10000  # Example account balance
# signals, fig = generate_signals(data, account_balance)
# fig.show()

from sklearn.preprocessing import MinMaxScaler 
def define_trading_signals(df):
    df['DEMA'] = calculate_dema(df)
    df = calculate_supertrend(df, period=12, atr_multiplier=3)
    df['Buy_Signal'] = (df['Close'] > df['DEMA']) & (df['SuperTrend'] < df['Close'])
    df['Sell_Signal'] = (df['Close'] < df['DEMA']) | (df['SuperTrend'] > df['Close'])
    return df

async def execute_trade_ig_v3(symbol, action, quantity, access_token, stop_loss=None, take_profit=None, guaranteed_stop=False, limit_distance=None):
    """
    Execute a trade using IG's API v3 with OAuth authentication, including additional parameters.

    Parameters:
    - symbol: The market's EPIC identifier.
    - action: 'BUY' or 'SELL'.
    - quantity: The number of units to trade.
    - access_token: OAuth access token for authentication.
    - stop_loss: (Optional) Stop loss level.
    - take_profit: (Optional) Take profit level.
    - guaranteed_stop: (Optional) Whether the stop loss is guaranteed.
    - limit_distance: (Optional) Distance for limit order.
    """
    url = "https://api.ig.com/gateway/deal/positions/otc"
    headers = {
        "Content-Type": "application/json",
        "Accept": "application/json",
        "Authorization": f"Bearer {access_token}",
        "X-IG-API-KEY": os.getenv( '5FA056D2706634F2B7C6FC66FE17517B'),
        "Version": "3"
    }

    # Prepare the payload with the order details and additional parameters.
    payload = {
        "currencyCode": "GBP",
        "direction": action.upper(),
        "epic": symbol,
        "size": quantity,
        "orderType": "MARKET",
        # Additional parameters based on strategy and IG's API requirements
    }

    if stop_loss:
        payload["stopDistance"] = stop_loss
        payload["guaranteedStop"] = guaranteed_stop
    if take_profit:
        payload["limitDistance"] = take_profit
    if limit_distance:
        payload["orderType"] = "LIMIT"
        payload["level"] = limit_distance  # Specify the price level for the limit order

    async with aiohttp.ClientSession() as session:
        async with session.post(url, headers=headers, json=payload) as response:
            if response.status in [200, 201]:
                response_data = await response.json()
                logging.info(f"Trade executed successfully for {symbol}: {action} {quantity} units. Response: {response_data}")
                return response_data
            else:
                response_text = await response.text()
                logging.error(f"Failed to execute trade for {symbol}: {response.status} {response_text}")
                return None

def execute_trade_alpaca(symbol, qty, trade_type, limit_price=None):
    try:
        api.submit_order(
            symbol=symbol,
            qty=qty,
            side=trade_type,
            type='limit' if limit_price else 'market',
            limit_price=limit_price,
            time_in_force='gtc'
        )
        logging.info(f"{trade_type.capitalize()} order submitted for {qty} shares of {symbol}")
    except tradeapi.rest.APIError as e:
        logging.error(f"Alpaca API error executing {trade_type} trade for {symbol}: {e}")
    except Exception as e:
        logging.error(f"Unexpected error executing {trade_type} trade for {symbol}: {e}")

def execute_short_sell(symbol, qty):
    """
    Execute a short sell order.
    """
    try:
        api.submit_order(
            symbol=symbol,
            qty=qty,
            side='sell',
            type='market',
            time_in_force='day'
        )
        logging.info(f"Short sell order executed for {qty} shares of {symbol}")
    except Exception as e:
        logging.error(f"Failed to execute short sell for {symbol}: {e}")
def execute_trade_via_blockchain(action):
    """
    Execute trade using blockchain technology to record transactions.

    Parameters:
        action (dict): Details of the trade action, including sender, recipient, and amount.

    Returns:
        bool: True if the trade is successfully executed and recorded on the blockchain, False otherwise.
    """
    # Assuming you have a blockchain instance named 'blockchain'
    try:
        # Validate required keys are in the action
        if all(key in action for key in ['sender', 'recipient', 'amount']):
            logging.info(f"Executing trade via blockchain: {action}")
            # Simulate adding a transaction to the blockchain
            blockchain.new_transaction(action['sender'], action['recipient'], action['amount'])
            logging.info(f"Trade recorded on the blockchain. Action details: {action}")
        else:
            logging.error("Trade action missing required details.")
    except Exception as e:
        logging.error(f"Error executing trade via blockchain: {e}")    
def place_order(symbol, qty, side, type='market', time_in_force='gtc', limit_price=None):
    """
    Place an order on Alpaca.
    """
    try:
        if type == 'limit' and limit_price is not None:
            api.submit_order(symbol=symbol, qty=qty, side=side, type=type, time_in_force=time_in_force, limit_price=limit_price)
        else:
            api.submit_order(symbol=symbol, qty=qty, side=side, type=type, time_in_force=time_in_force)
        print(f"Order submitted: {qty} {side} {symbol}")
    except Exception as e:
        print(f"An error occurred while placing the order: {e}")
def place_daily_trades(cst_token, x_security_token):
    # Dictionary to hold the EPIC codes for each instrument
    epic_codes = {
        'Tesla': 'EPIC_FOR_TESLA',
        'Crude Oil': 'EPIC_FOR_CRUDE_OIL',
        'US Tech 100': 'EPIC_FOR_US_TECH_100'
    }
    
    # Loop through each instrument to perform operations
    for instrument_name, epic_code in epic_codes.items():
        # Fetch daily market data for the instrument
        # You need to replace this with a function that fetches the actual market data for the given epic
        # For demonstration, creating a dummy DataFrame
        data = pd.DataFrame({'Close': np.random.rand(100) * 100})
        
        # Calculate trading signals using your strategy
        data_with_signals = calculate_impulse_macd(data)
        
        # Check the last trading signal
        last_signal = data_with_signals.iloc[-1]['impulse_signal']
        if last_signal == 1:
            execute_trade_ig(epic_code, "BUY", 1, cst_token, x_security_token)
            print(f"Placed BUY order for {instrument_name}")
        elif last_signal == -1:
            execute_trade_ig(epic_code, "SELL", 1, cst_token, x_security_token)
            print(f"Placed SELL order for {instrument_name}")
        else:
            print(f"No trading signal for {instrument_name} based on the latest data")   
def execute_trades_based_on_signals(df, symbol, quantity=10):
    df = define_trading_signals(df)

    for index, row in df.iterrows():
        if row['Buy_Signal']:
            print(f"Executing BUY for {symbol} at {row['Close']}, on {index}")
            # Uncomment the line below to execute buy order
            api.submit_order(symbol=symbol, qty=quantity, side='buy', type='market')
        elif row['Sell_Signal']:
            print(f"Executing SELL for {symbol} at {row['Close']}, on {index}")
            # Uncomment the line below to execute sell order
            api.submit_order(symbol=symbol, qty=quantity, side='sell', type='market')

async def main():
    fig = go.Figure()
    # Login and authenticate with your trading platforms here
    access_token_ig, refresh_token_ig = await login_to_ig_v3()  # Placeholder function
    access_token_alpaca = "your_alpaca_access_token"  # Assume already available or fetched elsewhere

    while True:
        if not is_time_to_run():
            logging.info("Market is closed. Waiting for the market to open...")
            await asyncio.sleep(3600)  # Sleep for an hour before checking again
            continue
        
        logging.info("Market is open. Running trading system...")

        # Define your trading symbols and any other configurations
        symbols = ['AAPL', 'MSFT', 'GOOGL']  # Example symbols
        

        # Fetch and process data for each symbol, then execute trades based on your strategy
        for symbol in symbols:
            # Fetch data (placeholder function)
            data = await fetch_market_data(symbol, period="1mo", interval="1d")  # You need to define this function to fetch market data
            print(data)

            # Process the data (apply your strategy)
            if data is not None and not data.empty:
                data_with_indicators = apply_technical_indicators(data)  # Placeholder for your indicator calculations
                signal = generate_trade_signal(data_with_indicators)  # Placeholder for your signal generation logic
                
                if signal != 0:
                    quantity = calculate_trade_size(data_with_indicators, signal)  # Placeholder for your trade size calculation
                    
                    # Execute trades based on the platform
                    if symbol in ['EPIC_FOR_GOOGLE', 'EPIC_FOR_APPLE', 'EPIC_FOR_MICROSOFT']:  # IG symbols
                        if access_token_ig:
                            await execute_trade_ig_v3(access_token_ig, symbol, "BUY" if signal > 0 else "SELL", quantity)
                            logging.info(f"Executed IG trade for {symbol} based on signal: {signal}")
                    else:  # Alpaca or other platforms
                        await execute_trade_alpaca(symbol, signal, quantity, access_token_alpaca)  # Placeholder function
                        logging.info(f"Executed Alpaca trade for {symbol} based on signal: {signal}")

        logging.info("Trading iteration complete. Waiting before next iteration...")
        await asyncio.sleep(800)  # Wait before the next iteration of the trading loop

# This assumes is_time_to_run, fetch_data_alpaca_async, apply_technical_indicators,
# generate_trade_signal, fetch_real_time_price, login_to_ig_async, execute_trade, 
# and calculate_trade_size are all defined and implemented elsewhere in your code.
#def run_dash_app():
#app.run_server(debug=True)
async def periodic_task(interval, task_func, *args, **kwargs):
    while True:
        try:
            await task_func(*args, **kwargs)
            await asyncio.sleep(interval)
        except Exception as e:
            logging.error(f"Error during the execution of {task_func.__name__}: {str(e)}")
async def update_trading_model():
    # Placeholder for the update trading model function
    logging.info("Updating trading model...")        

async def main_background_loop():
    # Define the interval in seconds for each task
    data_fetch_interval = 600  # Example: 10 minutes
    model_update_interval = 3600  # Example: 1 hour
    await asyncio.gather(
        periodic_task(data_fetch_interval, fetch_new_data),
        periodic_task(model_update_interval, update_trading_model),
    ) #You can add more periodic tasks here as needed

def start_background_tasks():
    def start_loop():
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        loop.run_until_complete(main_background_loop())
    
    threading.Thread(target=start_loop, daemon=True).start()
    asyncio.run(main_background_loop())

def main():
    logging.info("Trading process started.")
    data = fetch_data()
    if not data.empty:
        data = enhance_features(data, symbol='MSFT')  # Assuming 'MSFT' is your symbol of interest
        prepared_data = prepare_data(data)
        if prepared_data is not None:
            X_train, y_train = prepared_data
            # Continue with your process
        else:
            logging.error("Prepared data is None. Exiting...")
    else:
        logging.info("No data fetched. Exiting...")
def run_trading_strategy():
    symbols = ["AAPL", "MSFT", "GOOGL"]
    for symbol in symbols:
        data = fetch_market_data(symbol)
        signals = analyze_data(data)
        execute_trades(signals)

def run_advanced_trading_system():
    try:
        if not is_time_to_run():
            print("Market is not open in the UK. Exiting...")
            return
        symbol = 'AAPL'
        qty = 10  # Define your quantity
        # Example strategy: Place a buy order if not already holding the stock
        positions = api.list_positions()
        if not any(position.symbol == symbol for position in positions):
            place_order(symbol, qty, 'buy')
        else:
            print("Already holding the stock. No action taken.")
    except Exception as e:
        print(f"An error occurred in the trading system: {e}")
def execute_day_trading_strategy(symbol, quantity=10):
    now = datetime.now()
    start_of_day = now - timedelta(hours=now.hour, minutes=now.minute, seconds=now.second, microseconds=now.microsecond)
    data = api.get_barset(symbol, 'minute', start=start_of_day.isoformat()).df
    data['SMA'] = data[symbol]['Close'].rolling(window=15).mean()
    if data.iloc[-1][symbol]['Close'] > data.iloc[-1][symbol]['SMA']:  # Example condition for buying
        api.submit_order(symbol=symbol, qty=quantity, side='buy', type='market')
        logging.info(f"Executed DAY TRADE BUY order for {quantity} shares of {symbol}")
    elif data.iloc[-1][symbol]['Close'] < data.iloc[-1][symbol]['SMA']:  # Example condition for selling
        api.submit_order(symbol=symbol, qty=quantity, side='sell', type='market')
        logging.info(f"Executed DAY TRADE SELL order for {quantity} shares of {symbol}")
def execute_scalping_strategy(symbol, quantity=10):
    current_price = asyncio.run(fetch_real_time_price(symbol))
    logging.info(f"Current price of {symbol} is {current_price}")
    # Placeholder for your buying/selling condition
    if some_condition:  # Define your condition
        api.submit_order(symbol=symbol, qty=quantity, side='buy', type='market')
        logging.info(f"Executed BUY order for {quantity} shares of {symbol} at {current_price}")
    elif another_condition:  # Define your condition
        api.submit_order(symbol=symbol, qty=quantity, side='sell', type='market')
        logging.info(f"Executed SELL order for {quantity} shares of {symbol} at {current_price}")

def cache_data(symbol, data):
    """Cache data to a file."""
    if not os.path.exists(cache_dir):
        os.makedirs(cache_dir)
    filepath = os.path.join(cache_dir, f"{symbol}.json")
    with open(filepath, 'w') as file:
        json.dump(data, file)

def read_cache(symbol, max_age_hours=24):
    """Read data from cache if it's not older than max_age_hours."""
    filepath = os.path.join(cache_dir, f"{symbol}.json")
    if os.path.exists(filepath):
        file_mod_time = datetime.fromtimestamp(os.path.getmtime(filepath))
        if datetime.now() - file_mod_time < timedelta(hours=max_age_hours):
            with open(filepath, 'r') as file:
                return json.load(file)
    return None

    """
    Perform sentiment analysis on a list of news texts.

    Parameters:
        news_texts (list of str): The news texts to analyze.
        detailed (bool): Whether to return detailed sentiment analysis results. Default is False.
        model (str): Model identifier to be used for sentiment analysis. Default is a fine-tuned DistilBERT model.

    Returns:
        numpy.ndarray: An array of sentiment scores or detailed analysis results.
                       Scores > 0.5 indicate positive sentiment, while < 0.5 indicate negative sentiment.
                       Detailed results include both the label and score for each text.
    """
def sentiment_analysis(news_texts, detailed=False, model="distilbert-base-uncased-finetuned-sst-2-english"):
    logging.info("Initializing refined sentiment analysis pipeline...")
    sentiment_pipeline = pipeline("sentiment-analysis", model=model)

    try:
        logging.info("Performing refined sentiment analysis on the provided texts...")
        results = sentiment_pipeline(news_texts)
        if detailed:
            return [(result['label'], result['score']) for result in results]
        else:
            # Adjust sentiment scores for direct utility in trade decisions.
            return [result['score'] if result['label'] == 'POSITIVE' else -(1 - result['score']) for result in results]
    except Exception as e:
        logging.error(f"Refined sentiment analysis failed: {e}")
        return [("NEUTRAL", 0)] * len(news_texts) if detailed else [0] * len(news_texts)

def dynamic_trade_decision_based_on_sentiment(symbol, sentiment_scores, intermarket_indicators):
    base_quantity = 10
    sentiment_factor = np.mean(sentiment_scores)
    intermarket_factor = np.mean([indicator['Intermarket_Feature'] for indicator in intermarket_indicators])

    quantity = base_quantity + round(sentiment_factor * 20 + intermarket_factor * 10)
    trade_action = "hold"
    if sentiment_factor > 0.5 and intermarket_factor > 0.1:
        trade_action = "buy"
    elif sentiment_factor < -0.5 or intermarket_factor < -0.1:
        trade_action = "sell"
    
    return trade_action, quantity

def execute_trade_based_on_adaptive_analysis(symbol, news_texts, intermarket_indicators):
    sentiment_results = refined_sentiment_analysis(news_texts, detailed=False)
    trade_action, quantity = dynamic_trade_decision_based_on_sentiment(symbol, sentiment_results, intermarket_indicators)
    
    # Fetch the last closing price for the stock for completeness
    stock_data = yf.download(symbol, period="1d")
    last_close_price = stock_data['Close'].iloc[-1]

    if trade_action == "buy":
        logging.info(f"Adaptive Decision: Placing BUY order for {quantity} units of {symbol} at {last_close_price}")
        # Integrate with your trading system's buy function
        api.submit_order(symbol=symbol, qty=1, side='buy', type='market', time_in_force='gtc')
    elif trade_action == "sell":
        logging.info(f"Adaptive Decision: Placing SELL order for {quantity} units of {symbol} at {last_close_price}")
        # Integrate with your trading system's sell function
        api.submit_order(symbol=symbol, qty=1, side='buy', type='market', time_in_force='gtc')
    else:
        logging.info("Adaptive Decision: Holding position based on current analysis.")

# Placeholder for quantum_ml_model
def advanced_quantum_ml_model(data, num_qubits=3, feature_map_depth=2, shots=1024, seed=12345):
    """
    Enhances the trading system with quantum computing capabilities by constructing
    an advanced quantum circuit as a feature map. This method computes the kernel matrix
    on a quantum simulator, leveraging entanglement and variational parameters.

    Args:
        data (np.ndarray): Input data for generating quantum features. Shape: (num_samples, num_features).
        num_qubits (int): Number of qubits, should match the number of features in the data.
        feature_map_depth (int): Complexity of the quantum state, affecting data encoding.
        shots (int): Number of measurements for estimating the outcome of quantum circuits.
        seed (int): Seed for reproducibility of results.

    Returns:
        np.ndarray: Kernel matrix representing quantum-enhanced features of the input data.
    """
    if data.shape[1] != num_qubits:
        raise ValueError("Data feature size must match the number of qubits.")

    # Define a quantum feature map for data encoding
    feature_map = ZZFeatureMap(feature_dimension=num_qubits, reps=feature_map_depth, entanglement='full')

    # Configure the quantum instance using the Aer simulator backend
    backend = Aer.get_backend('qasm_simulator')
    quantum_instance = QuantumInstance(backend, shots=shots, seed_simulator=seed, seed_transpiler=seed)

    # Initialize the quantum kernel with the specified feature map and quantum instance
    quantum_kernel = QuantumKernel(feature_map=feature_map, quantum_instance=quantum_instance)

    # Compute the kernel matrix for the input data
    kernel_matrix = quantum_kernel.evaluate(x_vec=data)

    return kernel_matrix

# Example usage
# Ensure your data is in the correct format: a NumPy array where rows are samples and columns are features
# data = np.random.rand(100, 3)  # Example data with 100 samples and 3 features
# kernel_matrix = advanced_quantum_ml_model(data)
# print(kernel_matrix)
def reinforcement_learning_model(env_config):
    logging.info("Determining trading actions from RL model placeholder...")
    # Ensure the action dictionary includes 'sender', 'recipient', and 'amount'
    simulated_action = [{'sender': 'Trader', 'recipient': 'Exchange', 'action': 'buy', 'amount': 100, 'symbol': 'MSFT'}]
    return simulated_action
#def reinforcement_learning_model(env_config):
    logging.info("Placeholder for reinforcement learning model.")
    # Placeholder implementation
    return [{'action': 'buy', 'amount': 100}]  # Dummy action
def advanced_risk_management(features):
    logging.info("Performing advanced risk management...")
    # Placeholder for advanced risk management
    return {'risk_level': 'low'}
def advanced_risk_management(portfolio):
    # Implement Advanced Risk Management
    pass
def continuous_learning_update(model_path, data, sequence_length=60, batch_size=32, epochs=10, validation_split=0.1, retrain_threshold=0.01, learning_rate_adjustment=True):
    logging.basicConfig(level=logging.INFO)
    
    # Load the existing model
    model = load_model(model_path)

    # Preprocess the new data
    X_lstm, y_lstm, _, _, lstm_scaler = prepare_and_preprocess_data(data, sequence_length)

    # Adjust the learning rate if necessary
    if learning_rate_adjustment:
        initial_lr = 0.001
        decay_rate = initial_lr / epochs
        optimizer = Adam(lr=initial_lr, decay=decay_rate)
        model.compile(optimizer=optimizer, loss='mean_squared_error', metrics=['accuracy'])
        logging.info(f"Learning rate adjusted to {initial_lr} with decay {decay_rate}.")

    # Evaluate the model on new data before deciding on retraining
    mse = model.evaluate(X_lstm, y_lstm, verbose=0)[0]
    logging.info(f"Evaluation MSE: {mse}")
    
    # Retrain if the model performance is not satisfactory
    if mse > retrain_threshold:
        logging.info("Retraining model due to performance below the threshold...")
        model.fit(X_lstm, y_lstm, batch_size=batch_size, epochs=epochs, validation_split=validation_split, callbacks=[EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)])
        logging.info("Model retraining completed.")
    else:
        logging.info("Model performance is satisfactory. No retraining necessary.")
    
    # Save the updated model
    updated_model_path = model_path.replace('.keras', '_updated.keras')
    model.save(r'C:\Users\Adhiraj Singh\OneDrive\Desktop\trading_project\saved_models\lstm_model.keras')
    logging.info(f"Updated model saved to {updated_model_path}")

    return model
class Blockchain:
    def __init__(self):
        self.chain = []
        self.current_transactions = []
        # Create the genesis block
        self.new_block(previous_hash='1', proof=100)

    def new_block(self, proof, previous_hash=None):
        """
        Create a new Block in the Blockchain.

        :param proof: The proof given by the Proof of Work algorithm.
        :param previous_hash: (Optional) Hash of previous Block.
        :return: New Block.
        """
        block = {
            'index': len(self.chain) + 1,
            'timestamp': time.time(),
            'transactions': self.current_transactions,
            'proof': proof,
            'previous_hash': previous_hash or self.hash(self.chain[-1]),
        }
        self.current_transactions = []  # Reset the current list of transactions
        self.chain.append(block)
        return block

    def new_transaction(self, sender, recipient, amount):
        """
        Creates a new transaction to go into the next mined Block.

        :param sender: Address of the Sender.
        :param recipient: Address of the Recipient.
        :param amount: Amount.
        :return: The index of the Block that will hold this transaction.
        """
        self.current_transactions.append({
            'sender': sender,
            'recipient': recipient,
            'amount': amount,
        })
        return self.last_block['index'] + 1

    @staticmethod
    def hash(block):
        """
        Creates a SHA-256 hash of a Block.

        :param block: Block.
        """
        block_string = json.dumps(block, sort_keys=True).encode()
        return hashlib.sha256(block_string).hexdigest()

    @property
    def last_block(self):
        """
        Returns the last Block in the chain.
        """
        return self.chain[-1]

    def proof_of_work(self, last_proof):
        """
        Simple Proof of Work Algorithm:
         - Find a number p' such that hash(pp') contains 4 leading zeroes, where p is the previous p'.
         - p is the previous proof, and p' is the new proof.

        :param last_proof: Previous proof.
        :return: New proof.
        """
        proof = 0
        while self.valid_proof(last_proof, proof) is False:
            proof += 1
        return proof

    @staticmethod
    def valid_proof(last_proof, proof):
        """
        Validates the Proof: Does hash(last_proof, proof) contain 4 leading zeroes?

        :param last_proof: Previous proof.
        :param proof: Current proof.
        :return: True if correct, False if not.
        """
        guess = f'{last_proof}{proof}'.encode()
        guess_hash = hashlib.sha256(guess).hexdigest()
        return guess_hash[:4] == "0000"

# Initialize blockchain
blockchain = Blockchain()
def is_time_to_run():
    # Define the UK timezone
    uk_timezone = pytz.timezone('Europe/London')

    # Get the current time in UK timezone
    now_uk = datetime.now(uk_timezone).time()

    # Define market open and Close times in UK time
    market_open_time = dt_time(14, 30)  # 2:30 PM
    market_Close_time = dt_time(21, 30)  # 9:30 PM

    # Check if current UK time is within trading hours
    return market_open_time <= now_uk <= market_Close_time
def check_arbitrage_opportunity(symbol, exchange_a_price, exchange_b_price):
    if exchange_a_price < exchange_b_price:
        execute_buy_order(symbol, exchange_a_price)
        execute_sell_order(symbol, exchange_b_price)
        logging.info(f"Arbitrage executed: Bought {symbol} on Exchange A and sold on Exchange B")
    elif exchange_b_price < exchange_a_price:
        execute_buy_order(symbol, exchange_b_price)
        execute_sell_order(symbol, exchange_a_price)
        logging.info(f"Arbitrage executed: Bought {symbol} on Exchange B and sold on Exchange A")

# Load the data
file_path = r"C:\Users\Adhiraj Singh\OneDrive\Desktop\trading_project\merged_etf_subset.csv"  # Ensure this path is correct
try:
    df = pd.read_csv(file_path)
    print("File loaded successfully.")
except FileNotFoundError as e:
    print(f"File not found: {e}")

# Preprocess data for LSTM
if 'Close' in df.columns:
    data = df[['Close']].values
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data)

    sequence_length = 60
    X, y = [], []
    for i in range(sequence_length, len(scaled_data)):
        X.append(scaled_data[i-sequence_length:i, 0])
        y.append(scaled_data[i, 0])
    X, y = np.array(X), np.array(y)
    X = X.reshape((X.shape[0], X.shape[1], 1))  # Reshape for LSTM

    # Split into training and validation sets
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
    print("Data prepared for LSTM.")
else:
    print("Close column not found in data.")

def check_market_open():
    """
    Check if the market is currently open.
    """
    logging.info
    clock = api.get_clock()
    return clock.is_open

def check_positions_alpaca():
    return alpaca_api.list_positions()

def get_account_info_alpaca():
    logging.info("Account info get from alpaca")
    return alpaca_api.get_account()

def volume_rate_of_change(volume, window=3):
    """Calculate the Volume Rate of Change."""
    volume_roc = volume.pct_change(periods=window) * 100
    return volume_roc

def visualize_trading_decisions_with_price(data, signals):
    fig, ax1 = plt.subplots(figsize=(14, 9))

    # Plot the Close price
    ax1.plot(data.index, data['Close'], label='Close Price', color='blue', alpha=0.5)
    ax1.set_title('Stock Price, Trading Signals, and Sentiment Score')
    ax1.set_xlabel('Date')
    ax1.set_ylabel('Price')
    ax1.legend(loc='upper left')

    # Standardize the sentiment score
    sentiment_scores = data['Sentiment_Score'].values.reshape(-1, 1)
    scaler = MinMaxScaler(feature_range=(data['Close'].min(), data['Close'].max()))
    scaled_sentiment_scores = scaler.fit_transform(sentiment_scores).flatten()

    # Create a second y-axis for the scaled sentiment scores
    ax2 = ax1.twinx()
    ax2.plot(data.index, scaled_sentiment_scores, label='Sentiment Score', color='grey', alpha=0.3)
    ax2.set_ylabel('Scaled Sentiment Score')
    ax2.legend(loc='upper right')

    plt.show()
def load_gnn_model(filepath):
    """
    Load a pre-trained GNN model from a specified filepath.
    """
    try:
        model = torch.load(filepath)
        model.eval()  # Set the model to evaluation mode
        logging.info("GNN model loaded successfully.")
        return model
    except Exception as e:
        logging.error(f"Error loading GNN model: {e}")
        return None    
def graph_to_pyg(G):
    """
    Convert a NetworkX graph into a PyTorch Geometric Data object.
    """
    pyg_graph = from_networkx(G)
    return pyg_graph    
def extract_gnn_features(pyg_graph, model):
    """
    Extract node features from a graph using a GNN model.
    """
    with torch.no_grad():  # Ensure no gradients are computed to save memory
        node_embeddings = model(pyg_graph.x, pyg_graph.edge_index)
    return node_embeddings.detach().cpu().numpy()    
def graph_neural_network_model(data, model_path):
    """
    Main workflow to process graph data, extract features using a GNN model, and return those features.
    """
    if 'source' not in data.columns or 'target' not in data.columns:
        logging.warning("'source' or 'target' columns not found in data. Skipping GNN feature generation.")
        return None

    # Create graph from DataFrame
    G = nx.from_pandas_edgelist(data, 'source', 'target')
    logging.info("Graph created from data.")

    # Convert NetworkX graph to PyTorch Geometric graph
    pyg_graph = graph_to_pyg(G)

    # Load the pre-trained GNN model
    gnn_model = load_gnn_model(model_path)
    if gnn_model is None:
        return None

    # Extract features using the GNN model
    gnn_features = extract_gnn_features(pyg_graph, gnn_model)
    logging.info("GNN features extracted successfully.")

    return gnn_features

# Example usage
data_path = r"C:\Users\Adhiraj Singh\OneDrive\Desktop\trading_project\merged_etf_subset.csv" # Adjust as per your data file location
model_path = r"C:\Users\Adhiraj Singh\OneDrive\Desktop\trading_project\saved_models\lstm_model.keras"  # Adjust as per your model file location

try:
    data = pd.read_csv(data_path)
    gnn_features = graph_neural_network_model(data, model_path)
    if gnn_features is not None:
        print("Extracted GNN Features:", gnn_features.shape)
    else:
        print("Failed to extract GNN features.")
except Exception as e:
    logging.error(f"An error occurred: {e}")


if __name__ == "__main__":
    
    
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    start_background_tasks()
    
# Assuming you want to run the async main function and possibly start a Dash app as well
    extracted_text = extract_text_from_pdfs(pdf_paths)
    print(extracted_text)

    while True:
        if is_time_to_run():
            print("Market is open in the UK. Running trading system...")
            run_advanced_trading_system()     
        else:
            print("Market is Closed in the UK. Waiting for market to open..."), 
        # Wait for 60 seconds before checking again
        time.sleep(800)
        asyncio.run(main, fetch_new_data, main_background_loop())
        loop = asyncio.get_event_loop()
        loop.run_until_complete(fetch_new_data())
        app.run_server(debug=True)
