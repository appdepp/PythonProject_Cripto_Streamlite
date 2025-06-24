import os
import logging
import streamlit as st
from dotenv import load_dotenv
from binance.client import Client
import matplotlib.pyplot as plt
from io import BytesIO

# Загружаем переменные окружения
load_dotenv()
BINANCE_API_KEY = os.getenv('BINANCE_API_KEY')
BINANCE_API_SECRET = os.getenv('BINANCE_API_SECRET')

# Binance клиент
client = Client(BINANCE_API_KEY, BINANCE_API_SECRET)

# Топ-10 криптовалютных пар
TOP_CRYPTO_PAIRS = [
    'BTCUSDT', 'ETHUSDT', 'BNBUSDT', 'SOLUSDT', 'ADAUSDT', 'LTCUSDT',
    'XRPUSDT', 'DOGEUSDT', 'DOTUSDT', 'AVAXUSDT'
]

# Периоды для анализа
PERIODS = ['5m', '15m', '30m', '1h', '6h', '12h', '1d', '3d', '1w', '1M']


# Функция для вычисления RSI
def calculate_rsi(prices, period=14):
    if len(prices) < period:
        raise ValueError("Недостаточно данных для расчета RSI")
    gains, losses = [], []
    for i in range(1, len(prices)):
        change = prices[i] - prices[i - 1]
        gains.append(max(change, 0))
        losses.append(-min(change, 0))
    avg_gain = sum(gains[:period]) / period
    avg_loss = sum(losses[:period]) / period
    for i in range(period, len(prices)):
        change = prices[i] - prices[i - 1]
        gain = max(change, 0)
        loss = -min(change, 0)
        avg_gain = (avg_gain * (period - 1) + gain) / period
        avg_loss = (avg_loss * (period - 1) + loss) / period
    if avg_loss == 0:
        return 100
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))


# Функция для вычисления MACD
def calculate_macd(prices, short_period=12, long_period=26, signal_period=9):
    def ema(prices, period):
        multiplier = 2 / (period + 1)
        ema_values = [sum(prices[:period]) / period]
        for price in prices[period:]:
            ema_values.append((price - ema_values[-1]) * multiplier + ema_values[-1])
        return ema_values

    short_ema = ema(prices, short_period)
    long_ema = ema(prices, long_period)
    macd = [s - l for s, l in zip(short_ema[len(short_ema) - len(long_ema):], long_ema)]
    signal_line = ema(macd, signal_period)
    return macd[-1], signal_line[-1]


# Генерация графика
def generate_chart(pair):
    klines = client.get_klines(symbol=pair, interval='1d', limit=30)
    dates = [kline[0] for kline in klines]
    prices = [float(kline[4]) for kline in klines]
    from datetime import datetime
    dates = [datetime.utcfromtimestamp(date / 1000).strftime('%Y-%m-%d') for date in dates]
    max_price = max(prices)
    min_price = min(prices)
    max_index = prices.index(max_price)
    min_index = prices.index(min_price)
    plt.figure(figsize=(10, 6))
    plt.plot(dates, prices, label=f'{pair} Price', color='b', marker='o')
    plt.scatter(dates[max_index], max_price, color='g', label="Максимум")
    plt.scatter(dates[min_index], min_price, color='r', label="Минимум")
    plt.annotate(f"Max: {max_price:.2f}", (dates[max_index], max_price), textcoords="offset points", xytext=(0, 10),
                 ha='center', fontsize=10, color='g')
    plt.annotate(f"Min: {min_price:.2f}", (dates[min_index], min_price), textcoords="offset points", xytext=(0, -10),
                 ha='center', fontsize=10, color='r')
    plt.title(f'График цены для {pair} за последний месяц')
    plt.xlabel('Дата')
    plt.ylabel('Цена (USDT)')
    plt.xticks(rotation=45)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    buf = BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    return buf


# Streamlit интерфейс
st.title("📈 Анализ Криптовалют")
pair = st.selectbox("Выберите валютную пару", TOP_CRYPTO_PAIRS)
interval = st.selectbox("Выберите период анализа", PERIODS)

if st.button("Показать график"):
    chart_buf = generate_chart(pair)
    st.image(chart_buf, caption=f"📉 График цены для {pair}")

    # Получаем исторические данные
    klines = client.get_klines(symbol=pair, interval=interval)
    closes = [float(entry[4]) for entry in klines]

    # RSI и MACD
    rsi = calculate_rsi(closes)
    macd, signal_line = calculate_macd(closes)

    st.subheader(f"RSI для {pair} ({interval}): {rsi:.2f}")
    st.write("📊 Сигнал RSI:",
             "🟢 Сигнал на покупку" if rsi < 40 else "🔴 Сигнал на продажу" if rsi > 70 else "🔷 Нейтральная зона")
    st.subheader(f"MACD для {pair} ({interval}): {macd:.2f} (Сигнальная линия: {signal_line:.2f})")
    st.write("📊 Сигнал MACD:",
             "🟢 Сигнал на покупку" if macd > signal_line else "🔴 Сигнал на продажу" if macd < signal_line else "🔷 Нейтральная зона")