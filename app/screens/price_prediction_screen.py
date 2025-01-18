import streamlit as st
import plotly.graph_objects as go
import pandas as pd
from datetime import datetime, timedelta
from enums import InvestmentSymbol
from env_loader import GOLD_DATA_SET, BITCOIN_DATA_SET, SILVER_DATA_SET, ETH_DATA_SET, DOGE_DATA_SET
from fischerAI.price_prediction.price_prediction_AI import PricePredictionAI
from fischerAI.price_prediction.data_processing import prepare_data_set, get_sequences, get_min_max_for_sequences_and_target_sequences_from_saved_models
from fischerAI.utils.input_data_normalization import min_max_normalization_with_min_max_params


investment_data_sets = {
    InvestmentSymbol.GOLD.value: GOLD_DATA_SET,
    InvestmentSymbol.BITCOIN.value: BITCOIN_DATA_SET,
    InvestmentSymbol.ETHEREUM.value: ETH_DATA_SET,
}


min_days_to_predict = 20
max_days_to_predict = 600


def price_prediction_screen():
    col1, col2, col3 = st.columns([3, 1, 1])

    with col1:
        investment_options = [investment_symbol.value for investment_symbol in InvestmentSymbol
                              if investment_symbol != InvestmentSymbol.POLSKAGORACOIN
                              and investment_symbol != InvestmentSymbol.DOGECOIN
                              and investment_symbol != InvestmentSymbol.SILVER]

        selected_investment = st.selectbox(
            'Choose investment: ', investment_options
        )

    data = prepare_data_set(investment_data_sets[selected_investment], is_training_data=False)
    sequence = get_sequences(data, min_days_to_predict, is_graph=True)

    today = datetime.today()
    dates = [today - timedelta(days=i) for i in range(len(sequence) - 1, -1, -1)]

    with col2:
        days_to_predict = st.number_input('Days for predict: ', value=min_days_to_predict, step=1)

    predict_button_disabled = not (min_days_to_predict <= days_to_predict <= max_days_to_predict)

    with col3:
        predict_button = st.button('Predict', use_container_width=True, disabled=predict_button_disabled)
        back_to_curr_prices_button = st.button('Back to current', use_container_width=True)

    df = pd.DataFrame({
        'date': dates,
        'price': sequence
    })

    df['open'] = df['price'].shift(1)
    df['high'] = df[['price', 'open']].max(axis=1)
    df['low'] = df[['price', 'open']].min(axis=1)

    fig = go.Figure()

    fig.add_trace(
        go.Candlestick(
            x=df['date'],
            open=df['open'],
            high=df['high'],
            low=df['low'],
            close=df['price'],
            increasing_line_color='green',
            decreasing_line_color='red',
            name='Current prices'
        )
    )

    if predict_button and not predict_button_disabled:
        predictions = predict_for_n_days(sequence, days_to_predict, selected_investment)

        prediction_dates = [
            df['date'].iloc[-1] + timedelta(days=i + 1)
            for i in range(len(predictions))
        ]

        pred_df = pd.DataFrame({
            'date': prediction_dates,
            'price': predictions
        })

        pred_df['open'] = pred_df['price'].shift(1)
        pred_df.loc[pred_df.index[0], 'open'] = df['price'].iloc[-1]
        pred_df['high'] = pred_df[['price', 'open']].max(axis=1)
        pred_df['low'] = pred_df[['price', 'open']].min(axis=1)

        fig.add_trace(
            go.Candlestick(
                x=pred_df['date'],
                open=pred_df['open'],
                high=pred_df['high'],
                low=pred_df['low'],
                close=pred_df['price'],
                increasing_line_color='lightgreen',
                decreasing_line_color='lightcoral',
                name='Predictions',
                visible=True
            )
        )

        frames = []
        for i in range(len(pred_df)):
            frame = go.Frame(
                data=[
                    go.Candlestick(
                        x=df['date'],
                        open=df['open'],
                        high=df['high'],
                        low=df['low'],
                        close=df['price'],
                        increasing_line_color='green',
                        decreasing_line_color='red',
                        name='Historical'
                    ),
                    go.Candlestick(
                        x=pred_df['date'][:i + 1],
                        open=pred_df['open'][:i + 1],
                        high=pred_df['high'][:i + 1],
                        low=pred_df['low'][:i + 1],
                        close=pred_df['price'][:i + 1],
                        increasing_line_color='lightgreen',
                        decreasing_line_color='lightcoral',
                        name='Predictions'
                    )
                ],
                name=f'frame{i}'
            )
            frames.append(frame)

        fig.frames = frames

        fig.update_layout(
            updatemenus=[
                dict(
                    type="buttons",
                    showactive=False,
                    buttons=[
                        dict(
                            label="Play",
                            method="animate",
                            args=[None,
                                  {"frame": {"duration": 500, "redraw": True},
                                   "fromcurrent": True,
                                   "transition": {"duration": 300}}
                                  ]
                        )
                    ],
                    x=0.9,
                    y=1.1,
                )
            ]
        )


    if back_to_curr_prices_button:
        fig.data = [fig.data[0]]
        fig.frames = []

    fig.update_layout(
        xaxis_rangeslider_visible=False,
        yaxis=dict(
            tickprefix="$",
            tickformat=".2f"
        ),
        showlegend=True,
        xaxis_range=[
            df['date'].min(),
            (df['date'].max() + timedelta(days=days_to_predict)) if predict_button else df['date'].max()
        ]
    )

    st.plotly_chart(fig, use_container_width=True)


def predict_for_n_days(sequence, n_days, selected_investment_symbol):
    lstm_model_info, dense_model_info, sequences_min, sequences_max, target_min, target_max = get_min_max_for_sequences_and_target_sequences_from_saved_models(None, None, selected_investment_symbol)

    normalized_sequence = min_max_normalization_with_min_max_params(sequence, sequences_min, sequences_max)

    normalized_sequence = normalized_sequence.reshape(1, len(normalized_sequence), 1)

    price_prediction_ai = PricePredictionAI(investment_symbol=selected_investment_symbol, batch_size=normalized_sequence.shape[0],
                                                sequence_length=normalized_sequence.shape[1],
                                                input_sequence_length=normalized_sequence.shape[2],
                                                hidden_lstm_layers=None,
                                                hidden_dense_layers=None, epochs=None,
                                                learning_rate_lstm=None,
                                                learning_rate_dense=None,
                                                decay_rate_lstm=0.999,
                                                sequences_min=sequences_min, sequences_max=sequences_max,
                                                target_sequences_min=target_min,
                                                target_sequences_max=target_max)

    predictions = price_prediction_ai.predict_for_n_days(normalized_sequence, lstm_model_info, dense_model_info, n_days)

    return predictions.flatten()