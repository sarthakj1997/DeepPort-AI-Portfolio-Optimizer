# src/app.py

import dash
from dash import html, dcc
from dash.dependencies import Input, Output
import plotly.graph_objs as go
from src.database import get_engine, get_session, AssetData, SentimentData, OptimizationResult
from src.prediction.lstm_model import LSTMPredictor
import pandas as pd
import os

app = dash.Dash(__name__)
engine = get_engine()
session = get_session(engine)

tickers = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'FB']

def get_price_data(ticker):
    query = session.query(AssetData).filter(AssetData.ticker == ticker)
    df = pd.read_sql(query.statement, session.bind)
    df['date'] = pd.to_datetime(df['date'])
    df.set_index('date', inplace=True)
    return df

def get_sentiment_data(ticker):
    query = session.query(SentimentData).filter(SentimentData.ticker == ticker)
    df = pd.read_sql(query.statement, session.bind)
    df['date'] = pd.to_datetime(df['date'])
    return df

app.layout = html.Div(children=[
    html.H1(children='AI-Driven Portfolio Optimization'),

    dcc.Dropdown(
        id='ticker-dropdown',
        options=[{'label': ticker, 'value': ticker} for ticker in tickers],
        value='AAPL',
        multi=False
    ),
    dcc.Graph(id='price-chart'),
    dcc.Graph(id='sentiment-chart'),
    dcc.Graph(id='prediction-chart'),
    dcc.Graph(id='optimization-comparison-chart')
])

@app.callback(
    [Output('price-chart', 'figure'),
     Output('sentiment-chart', 'figure'),
     Output('prediction-chart', 'figure'),
     Output('optimization-comparison-chart', 'figure')],
    [Input('ticker-dropdown', 'value')]
)
def update_charts(selected_ticker):
    price_df = get_price_data(selected_ticker)
    sentiment_df = get_sentiment_data(selected_ticker)

    price_fig = {
        'data': [go.Scatter(
            x=price_df.index,
            y=price_df['adj_close'],
            mode='lines',
            name='Adjusted Close'
        )],
        'layout': go.Layout(title=f'{selected_ticker} Price Data', xaxis={'title': 'Date'}, yaxis={'title': 'Price'})
    }

    sentiment_fig = {
        'data': [go.Bar(
            x=sentiment_df['date'],
            y=sentiment_df['sentiment_score'],
            name='Sentiment Score'
        )],
        'layout': go.Layout(title=f'{selected_ticker} Sentiment Analysis', xaxis={'title': 'Date'}, yaxis={'title': 'Sentiment Score'})
    }

    # Prediction
    lstm_predictor = LSTMPredictor()
    # Ensure the processed data directory exists and contains the data
    data_dir = 'data/processed_data/'
    if not os.path.exists(f"{data_dir}{selected_ticker}_processed.csv"):
        # Handle missing data
        prediction_fig = {'data': [], 'layout': go.Layout(title='No prediction available')}
    else:
        model, scaler = lstm_predictor.train_model(selected_ticker)
        df = lstm_predictor.load_data(selected_ticker)
        last_sequence = df['adj_close'].values[-60:]
        scaled_sequence = scaler.transform(last_sequence.reshape(-1,1))
        X_pred = np.array([scaled_sequence])
        X_pred = X_pred.reshape((X_pred.shape[0], X_pred.shape[1], 1))
        predicted_scaled_price = model.predict(X_pred)
        predicted_price = scaler.inverse_transform(predicted_scaled_price)[0][0]

        prediction_fig = {
            'data': [go.Scatter(
                x=price_df.index,
                y=price_df['adj_close'],
                mode='lines',
                name='Adjusted Close'
            ), go.Scatter(
                x=[price_df.index[-1] + pd.Timedelta(days=1)],
                y=[predicted_price],
                mode='markers',
                name='Predicted Price'
            )],
            'layout': go.Layout(title=f'{selected_ticker} Price Prediction', xaxis={'title': 'Date'}, yaxis={'title': 'Price'})
        }

    # Optimization comparison
    query = session.query(OptimizationResult).filter(OptimizationResult.ticker == selected_ticker)
    df_opt = pd.read_sql(query.statement, session.bind)
    fig_opt = {
        'data': [go.Bar(
            x=df_opt['method'],
            y=df_opt['weight'],
            name='Optimization Weights'
        )],
        'layout': go.Layout(title=f'{selected_ticker} Optimization Comparison', xaxis={'title': 'Method'}, yaxis={'title': 'Weight'})
    }

    return price_fig, sentiment_fig, prediction_fig, fig_opt

if __name__ == '__main__':
    app.run_server(debug=True)
