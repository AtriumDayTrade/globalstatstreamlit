# ========== IMPORTAÇÕES ==========================
import os
import time
import random
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import plotly.graph_objs as go
import yfinance as yf

import dash
from dash import dcc, html, State
from dash.dependencies import Input, Output
from plotly.subplots import make_subplots

# ========== CACHE (TTL) + DOWNLOAD EM LOTE PARA YFINANCE ==========
class YFCache:
    def __init__(self, ttl_seconds=180):
        self.ttl = timedelta(seconds=ttl_seconds)
        self.store = {}  # key -> (timestamp, data)

    def get(self, key):
        rec = self.store.get(key)
        if not rec:
            return None
        ts, data = rec
        if datetime.utcnow() - ts > self.ttl:
            self.store.pop(key, None)
            return None
        return data

    def set(self, key, data):
        self.store[key] = (datetime.utcnow(), data)

YF_CACHE = YFCache(ttl_seconds=180)  # 3 min

def _yf_key(symbols, period, interval):
    if isinstance(symbols, (list, tuple, set)):
        symbols = ','.join(sorted(list(symbols)))
    return f"{symbols}|{period}|{interval}"

def yf_download_cached(symbols, period, interval):
    """
    Faz UMA chamada ao Yahoo para vários símbolos e guarda em cache (TTL).
    Retorna: {symbol: DataFrame OHLC}
    """
    key = _yf_key(symbols, period, interval)
    cached = YF_CACHE.get(key)
    if cached is not None:
        return cached

    try:
        df = yf.download(
            symbols, period=period, interval=interval,
            auto_adjust=False, progress=False, threads=False, group_by='ticker'
        )
    except Exception as e:
        print(f"[YF ERROR] download {symbols} {period} {interval}: {e}")
        df = pd.DataFrame()

    result = {}
    if isinstance(symbols, str):
        result[symbols] = df.copy() if isinstance(df, pd.DataFrame) and not df.empty else pd.DataFrame()
    else:
        if isinstance(df, pd.DataFrame) and not df.empty and isinstance(df.columns, pd.MultiIndex):
            for sym in symbols:
                try:
                    part = df.xs(sym, axis=1, level=0, drop_level=False)
                    part.columns = part.columns.get_level_values(1)
                    result[sym] = part.copy()
                except Exception:
                    result[sym] = pd.DataFrame()
        else:
            for sym in symbols:
                result[sym] = df.copy() if isinstance(df, pd.DataFrame) else pd.DataFrame()

    YF_CACHE.set(key, result)
    return result

def _resample_ohlc_to_4h(df_1h):
    if df_1h.empty:
        return df_1h
    # se vier com timezone, remove para resample sem erro
    if hasattr(df_1h.index, "tz") and df_1h.index.tz is not None:
        df_1h = df_1h.tz_convert(None)

    cols = [c for c in ['Open','High','Low','Close','Adj Close','Volume'] if c in df_1h.columns]
    agg = {c: ('first' if c=='Open' else
               'max' if c=='High' else
               'min' if c=='Low' else
               'sum' if c=='Volume' else
               'last') for c in cols}
    return df_1h.resample('4H').agg(agg).dropna(how='all')

# ===================== CONFIGURAÇÃO INICIAL ====================
cryptos = [
    "GC=F", "BTC-USD", "ETH-USD", "BNB-USD", "SOL-USD", "ADA-USD",
    "XRP-USD", "DOGE-USD", "AVAX-USD", "DOT-USD", "LTC-USD", "LINK-USD", "ATOM-USD"
]

SPAN_VOL = 10
interval_ms = 300000
interval_sec = interval_ms // 1000

timeframes_dict = {'D1': 'D1', 'H4': 'H4'}

YF_MAP = {
    "EURUSD.r": "EURUSD=X", "GBPUSD.r": "GBPUSD=X", "USDJPY.r": "JPY=X",
    "USDCAD.r": "CAD=X", "USDCHF.r": "CHF=X", "USDSEK.r": "SEK=X",
    "EURGBP.r": "EURGBP=X", "EURJPY.r": "EURJPY=X", "EURCHF.r": "EURCHF=X", "EURAUD.r": "EURAUD=X",
    "GBPJPY.r": "GBPJPY=X", "GBPEUR.r": "GBPEUR=X", "GBPCHF.r": "GBPCHF=X",
    "EURCAD.r": "EURCAD=X", "GBPCAD.r": "GBPCAD=X", "EURSEK.r": "EURSEK=X",
    "US500": "^GSPC", "USDX": "DX-Y.NYB", "VIX": "^VIX", "TLT": "TLT",
    "GC=F": "GC=F", "BTC-USD": "BTC-USD", "ETH-USD": "ETH-USD", "BNB-USD": "BNB-USD",
    "SOL-USD": "SOL-USD", "ADA-USD": "ADA-USD", "XRP-USD": "XRP-USD", "DOGE-USD": "DOGE-USD",
    "AVAX-USD": "AVAX-USD", "DOT-USD": "DOT-USD", "LTC-USD": "LTC-USD",
    "LINK-USD": "LINK-USD", "ATOM-USD": "ATOM-USD",
    "SPY": "SPY", "QQQ": "QQQ", "IWM": "IWM"
}

TF_TO_YF = {
    "D1": ("1y", "1d"),
    "H4": ("60d", "1h"),  # baixa 1h e reamostra para 4h
}

ativos_comp = {
    'EURUSD.r': 1.0000, 'GBPUSD.r': 1.2000, 'USDJPY.r': 140.00,
    'USDCAD.r': 1.3500, 'USDCHF.r': 0.9000, 'USDSEK.r': 10.5000
}

pares_fx = ['EURUSD.r', 'GBPUSD.r', 'USDJPY.r', 'USDCAD.r', 'USDCHF.r', 'USDSEK.r']

cores_ativos = {
    'EURUSD.r': 'red',
    'GBPUSD.r': 'fuchsia',
    'USDJPY.r': 'green',
    'USDCAD.r': 'yellow',
    'USDCHF.r': 'aqua',
    'USDSEK.r': 'purple'
}

ativos = ['SPY', 'QQQ', 'IWM']

# ========================= Funções Auxiliares ======================
def calcular_b2(df, periodo=20):
    df = df.copy()
    if 'Close' not in df.columns:
        raise ValueError("Dados sem coluna 'Close'")
    close = df['Close']
    if isinstance(close, pd.DataFrame):
        close = close.iloc[:, 0]
    close = pd.to_numeric(close, errors='coerce')

    out = pd.DataFrame(index=close.index)
    out['Close'] = close

    ma = close.rolling(window=periodo, min_periods=periodo).mean()
    std = close.rolling(window=periodo, min_periods=periodo).std()
    upper = ma + 2 * std
    lower = ma - 2 * std
    delta = (upper - lower).replace(0, np.nan)

    out['B2'] = (close - lower) / delta
    out['Date'] = out.index
    return out[['Close', 'B2', 'Date']].dropna()

def coletar(ticker, timeframe):
    timeframe_label = timeframe if isinstance(timeframe, str) else 'D1'
    period, interval = TF_TO_YF.get(str(timeframe_label), ("1y", "1d"))
    yf_symbol = YF_MAP.get(ticker, ticker)

    data_map = yf_download_cached([yf_symbol], period, interval)
    df = data_map.get(yf_symbol, pd.DataFrame())
    if df is None or df.empty:
        return pd.DataFrame()

    if 'Close' not in df.columns and 'close' in df.columns:
        df = df.rename(columns={'close': 'Close'})
    if 'Close' not in df.columns:
        return pd.DataFrame()

    df = df.copy()
    df.index = pd.to_datetime(df.index)
    df = df[~df.index.duplicated(keep='last')]

    if timeframe_label == 'H4' and interval == '1h':
        df = _resample_ohlc_to_4h(df)

    s = df['Close'].dropna()
    return pd.DataFrame({'time': s.index, 'close': s.values})

def get_options_data(ticker, tentativas=3, espera=2):
    try:
        stock = yf.Ticker(ticker)
        expirations = list(stock.options or [])
        for _ in range(tentativas):
            if expirations:
                break
            time.sleep(espera)
            expirations = list(stock.options or [])
        if not expirations:
            return pd.DataFrame(), pd.DataFrame(), np.nan, None

        calls = puts = None
        chosen_exp = None
        for exp_date in expirations[:6]:
            try:
                chain = stock.option_chain(exp_date)
                if chain and not chain.calls.empty and not chain.puts.empty:
                    calls = chain.calls.copy()
                    puts  = chain.puts.copy()
                    chosen_exp = exp_date
                    break
            except Exception:
                continue

        if calls is None or puts is None:
            return pd.DataFrame(), pd.DataFrame(), np.nan, None

        exp_ts = pd.to_datetime(chosen_exp, utc=True, errors='coerce')
        calls['expiration'] = exp_ts
        puts['expiration'] = exp_ts

        spot_hist = stock.history(period='1d', auto_adjust=False)
        spot = float(spot_hist['Close'].iloc[-1]) if not spot_hist.empty else np.nan

        return calls, puts, spot, chosen_exp

    except Exception as e:
        print(f"[EXCEÇÃO] get_options_data({ticker}): {e}")
        return pd.DataFrame(), pd.DataFrame(), np.nan, None

def gamma_exposure(options, spot, rate=0.05):
    try:
        opts = options.copy()
        opts['expiration'] = pd.to_datetime(opts['expiration'], utc=True, errors='coerce')
        now = pd.Timestamp.utcnow()
        T = (opts['expiration'] - now).dt.total_seconds() / (365 * 24 * 3600)
        T = T.clip(lower=1/365)

        K = pd.to_numeric(opts['strike'], errors='coerce')
        sigma = pd.to_numeric(opts.get('impliedVolatility', 0.0), errors='coerce').fillna(0.0)
        sigma = sigma.clip(lower=0.001)
        oi = pd.to_numeric(opts.get('openInterest', 0), errors='coerce').fillna(0.0)

        S = float(spot)
        d1 = (np.log(S / K) + (rate + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
        gamma = np.exp(-0.5 * d1**2) / (S * sigma * np.sqrt(2 * np.pi * T))

        exposure = gamma * oi * 100 * S * 0.01
        return exposure.fillna(0.0)
    except Exception as e:
        print(f"Erro ao calcular gamma exposure: {e}")
        return pd.Series(np.zeros(len(options)))

# ========================= Gráficos ===============================
def gerar_range_index_plotly(timeframe_label, ativos_visiveis=None):
    # cores para índices sintéticos + USDX
    cores_idx = {'USDX': 'white', 'EURX': 'red', 'GBPX': 'fuchsia',
                 'JPYX': 'green', 'CADX': 'yellow', 'SEKX': 'aqua', 'CHFX': 'purple'}

    # cestas usadas nos índices sintéticos
    indices_sinteticos = {
        'EURX': (['EURUSD.r', 'EURGBP.r', 'EURJPY.r', 'EURCHF.r', 'EURAUD.r'],
                 [0.315, 0.305, 0.189, 0.111, 0.080]),
        'GBPX': (['GBPUSD.r', 'GBPJPY.r', 'GBPEUR.r', 'GBPCHF.r'],
                 [0.347, 0.207, 0.339, 0.107]),
        'JPYX': (['USDJPY.r', 'EURJPY.r', 'GBPJPY.r'],
                 [0.5, 0.3, 0.2]),
        'CADX': (['USDCAD.r', 'EURCAD.r', 'GBPCAD.r'],
                 [0.5, 0.3, 0.2]),
        'SEKX': (['USDSEK.r', 'EURSEK.r'],
                 [0.6, 0.4]),
        'CHFX': (['USDCHF.r', 'EURCHF.r', 'GBPCHF.r'],
                 [0.5, 0.3, 0.2]),
    }

    # (1) lista única de tickers (USDX + cestas)
    all_app_tickers = ['USDX']
    for pares, _ in indices_sinteticos.values():
        all_app_tickers.extend(pares)
    all_app_tickers = sorted(set(all_app_tickers))
    yf_symbols = [YF_MAP.get(t, t) for t in all_app_tickers]

    # (2) período/intervalo do Yahoo
    period, interval = TF_TO_YF.get(str(timeframe_label), ("1y", "1d"))

    # (3) UMA chamada ao Yahoo (com cache)
    data_map = yf_download_cached(yf_symbols, period=period, interval=interval)

    # helper: transforma DataFrame YF -> df{'time','close'}
    def df_close(sym):
        df = data_map.get(sym, pd.DataFrame())
        if df is None or df.empty:
            return pd.DataFrame()
        if 'Close' not in df.columns and 'close' in df.columns:
            df = df.rename(columns={'close': 'Close'})
        if 'Close' not in df.columns:
            return pd.DataFrame()
        out = df[['Close']].copy()
        out.index = pd.to_datetime(out.index)
        out = out[~out.index.duplicated(keep='last')]
        if timeframe_label == 'H4' and interval == '1h':
            out = _resample_ohlc_to_4h(out)
        return pd.DataFrame({'time': out.index, 'close': out['Close'].values})

    # (4) USDX direto
    df_final = pd.DataFrame()
    usdxy = YF_MAP.get('USDX', 'USDX')
    df_usdx = df_close(usdxy)
    if not df_usdx.empty:
        df_usdx = df_usdx.rename(columns={'close': 'USDX'})
        df_final = df_usdx[['time', 'USDX']]

    # (5) índices sintéticos calculados
    def calc_index(pares, pesos):
        base_df = pd.DataFrame()
        for par in pares:
            sym = YF_MAP.get(par, par)
            d = df_close(sym)
            if not d.empty:
                d = d.rename(columns={'close': par})
                base_df = d if base_df.empty else pd.merge(base_df, d, on='time', how='outer')
        if base_df.empty:
            return None
        base_df = base_df.sort_values('time').ffill().bfill()

        acc = None
        for par, w in zip(pares, pesos):
            if par in base_df.columns:
                series = base_df[par]
                base = series.iloc[0]
                if pd.isna(base) or base == 0:
                    return None
                term = w * np.log(series / base)
                acc = term if acc is None else acc + term
        if acc is None:
            return None
        base_df['Index'] = 100 * np.exp(acc)
        return base_df[['time', 'Index']]

    for nome, (pares, pesos) in indices_sinteticos.items():
        df_idx = calc_index(pares, pesos)
        if df_idx is not None:
            df_idx = df_idx.rename(columns={'Index': nome})
            df_final = df_idx if df_final.empty else pd.merge(df_final, df_idx, on='time', how='outer')

    if df_final.empty:
        return go.Figure()

    # (6) métricas e plot
    df_final = df_final.sort_values('time').ffill().bfill()
    ativos_cols = [c for c in df_final.columns if c != 'time']
    df_final['MediaGrupo'] = df_final[ativos_cols].mean(axis=1)
    df_final['FaixaAltaSuave'] = df_final['MediaGrupo'] + 10
    df_final['FaixaBaixaSuave'] = df_final['MediaGrupo'] - 10
    df_final['FaixaAltaInternaSuave2'] = df_final['MediaGrupo'] + 9
    df_final['FaixaBaixaInternaSuave2'] = df_final['MediaGrupo'] - 9

    span = 100
    for col in ['MediaGrupo', 'FaixaAltaSuave', 'FaixaBaixaSuave',
                'FaixaAltaInternaSuave2', 'FaixaBaixaInternaSuave2']:
        df_final[f'{col}_smooth'] = df_final[col].ewm(span=span).mean().ewm(span=span).mean()

    fig = go.Figure()
    ativos_visiveis = (ativos_visiveis or ativos_cols)
    for ativo in ativos_visiveis:
        if ativo not in df_final.columns:
            continue
        fig.add_trace(go.Scatter(
            x=df_final['time'], y=df_final[ativo],
            mode='lines', name=ativo,
            line=dict(color=cores_idx.get(ativo, 'gray')),
            showlegend=False
        ))
        fig.add_trace(go.Scatter(
            x=[df_final['time'].iloc[-1]], y=[df_final[ativo].iloc[-1]],
            mode='text', text=[ativo],
            textposition='middle right', showlegend=False,
            textfont=dict(size=12, color=cores_idx.get(ativo, 'white'))
        ))

    fig.add_trace(go.Scatter(x=df_final['time'], y=df_final['MediaGrupo_smooth'],
                             mode='lines', line=dict(color='silver', width=2, dash='dash'),
                             showlegend=False))
    fig.add_trace(go.Scatter(x=df_final['time'], y=df_final['FaixaAltaSuave_smooth'],
                             mode='lines', line=dict(color='gray', width=2),
                             showlegend=False))
    fig.add_trace(go.Scatter(x=df_final['time'], y=df_final['FaixaBaixaSuave_smooth'],
                             mode='lines', line=dict(color='gray', width=2),
                             showlegend=False))
    fig.add_trace(go.Scatter(x=df_final['time'], y=df_final['FaixaAltaInternaSuave2_smooth'],
                             mode='lines', line=dict(color='gray', dash='dash'),
                             showlegend=False))
    fig.add_trace(go.Scatter(x=df_final['time'], y=df_final['FaixaBaixaInternaSuave2_smooth'],
                             mode='lines', line=dict(color='gray', dash='dash'),
                             showlegend=False))

    today_str = datetime.now().strftime("%Y-%m-%d")
    fig.add_shape(type="line", x0=today_str, x1=today_str, y0=0, y1=1,
                  xref='x', yref='paper', line=dict(color="yellow", width=0.5, dash="dash"))
    fig.add_annotation(x=today_str, y=1.1, xref="x", yref="paper",
                       text=datetime.now().strftime("%d/%m/%Y"),
                       showarrow=False, font=dict(color="yellow", size=8))

    fig.update_layout(
        title=dict(text=f"Index ({timeframe_label})", font=dict(size=10, color='white'),
                   x=0.5, y=0.95, xanchor='center'),
        plot_bgcolor='black', paper_bgcolor='black', font=dict(color='white'),
        xaxis=dict(showgrid=True, gridcolor='#1c1c1c'),
        yaxis=dict(showgrid=True, gridcolor='#1c1c1c'),
        margin=dict(l=30, r=20, t=40, b=30),
        showlegend=False
    )
    return fig

def gerar_range_comparativo_plotly(timeframe_label, ativos_visiveis=None):
    df_final = pd.DataFrame()
    for ativo, base in ativos_comp.items():
        d = coletar(ativo, timeframe_label)
        if not d.empty:
            d = d.rename(columns={'close': 'close'})
            d['norm'] = 100 + 100 * np.log(d['close'] / base)
            d = d[['time', 'norm']].rename(columns={'norm': ativo})
            df_final = d if df_final.empty else pd.merge(df_final, d, on='time', how='outer')

    if df_final.empty:
        return go.Figure()

    df_final = df_final.sort_values('time').ffill().bfill()
    ativos_cols = list(ativos_comp.keys())

    df_final['MediaGrupo'] = df_final[ativos_cols].mean(axis=1)
    df_final['FaixaAltaSuave'] = df_final['MediaGrupo'] + 10
    df_final['FaixaBaixaSuave'] = df_final['MediaGrupo'] - 10
    df_final['FaixaAltaInternaSuave2'] = df_final['MediaGrupo'] + 9
    df_final['FaixaBaixaInternaSuave2'] = df_final['MediaGrupo'] - 9

    span = 100
    for col in ['MediaGrupo', 'FaixaAltaSuave', 'FaixaBaixaSuave',
                'FaixaAltaInternaSuave2', 'FaixaBaixaInternaSuave2']:
        df_final[f'{col}_smooth'] = df_final[col].ewm(span=span).mean()

    fig = go.Figure()
    ativos_visiveis = ativos_visiveis or ativos_cols

    for ativo in ativos_visiveis:
        if ativo not in df_final.columns:
            continue
        fig.add_trace(go.Scatter(
            x=df_final['time'], y=df_final[ativo], mode='lines',
            name=ativo, line=dict(color=cores_ativos.get(ativo, 'white')),
            hovertemplate=f"<b>{ativo}</b><br>Data: %{{x|%d/%m/%Y}}<br>Valor: %{{y:.2f}}",
            showlegend=False
        ))
        fig.add_trace(go.Scatter(
            x=[df_final['time'].iloc[-1]], y=[df_final[ativo].iloc[-1]],
            mode='text', text=[ativo], textposition='middle right',
            showlegend=False, textfont=dict(color=cores_ativos.get(ativo, 'white'), size=12)
        ))

    fig.add_trace(go.Scatter(x=df_final['time'], y=df_final['MediaGrupo_smooth'],
                             mode='lines', line=dict(color='silver', width=2, dash='dash'),
                             showlegend=False))
    fig.add_trace(go.Scatter(x=df_final['time'], y=df_final['FaixaAltaSuave_smooth'],
                             mode='lines', line=dict(color='gray', width=2),
                             showlegend=False))
    fig.add_trace(go.Scatter(x=df_final['time'], y=df_final['FaixaBaixaSuave_smooth'],
                             mode='lines', line=dict(color='gray', width=2),
                             showlegend=False))
    fig.add_trace(go.Scatter(x=df_final['time'], y=df_final['FaixaAltaInternaSuave2_smooth'],
                             mode='lines', line=dict(color='gray', dash='dash'),
                             showlegend=False))
    fig.add_trace(go.Scatter(x=df_final['time'], y=df_final['FaixaBaixaInternaSuave2_smooth'],
                             mode='lines', line=dict(color='gray', dash='dash'),
                             showlegend=False))

    today_str = datetime.now().strftime("%Y-%m-%d")
    fig.add_shape(type="line", x0=today_str, x1=today_str, y0=0, y1=1,
                  xref='x', yref='paper', line=dict(color="yellow", width=0.5, dash="dash"))
    fig.add_annotation(x=today_str, y=1.1, xref="x", yref="paper",
                       text=datetime.now().strftime("%d/%m/%Y"),
                       showarrow=False, font=dict(color="yellow", size=8))

    fig.update_layout(
        title=dict(text=f"Pares({timeframe_label})", font=dict(size=10, color='white'), x=0.5, xanchor='center'),
        plot_bgcolor='black', paper_bgcolor='black', font=dict(color='white'),
        xaxis=dict(showgrid=True, gridcolor='#1c1c1c', zeroline=False, tickcolor='white'),
        yaxis=dict(showgrid=True, gridcolor='#1c1c1c', zeroline=False, tickcolor='white'),
        showlegend=True,
        legend=dict(orientation="h", yanchor="top", y=-0.2, xanchor="center", x=0.5),
        margin=dict(l=30, r=20, t=40, b=17)
    )
    return fig

def calcular_hotelling_t2(_, timeframe_label):
    df_final = pd.DataFrame()
    for ativo in pares_fx:
        d = coletar(ativo, timeframe_label)
        if not d.empty:
            d = d.rename(columns={'close': ativo})
            df_final = d if df_final.empty else pd.merge(df_final, d, on='time', how='outer')

    if df_final.empty:
        return go.Figure()

    df_final = df_final.sort_values('time').ffill().bfill()
    for ativo in pares_fx:
        df_final[f"{ativo}_ret"] = np.log(df_final[ativo] / df_final[ativo].shift(1))

    df_ret = df_final[[f"{ativo}_ret" for ativo in pares_fx]].dropna()
    if df_ret.empty:
        return go.Figure()

    means = df_ret.mean()
    stds = df_ret.std().replace(0, np.nan)
    zscores = (df_ret - means) / stds
    T2 = (zscores.fillna(0) ** 2).sum(axis=1)

    df_plot = df_final.loc[df_ret.index].copy()
    df_plot['T2'] = T2
    idx_red = df_plot[df_plot['T2'] > 30].index

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df_plot['time'], y=df_plot['T2'], mode='lines',
                             line=dict(color='rgb(200,200,200)', width=1), showlegend=False))
    fig.add_trace(go.Scatter(x=df_plot.loc[idx_red, 'time'], y=df_plot.loc[idx_red, 'T2'],
                             mode='markers', marker=dict(color='red', size=3),
                             showlegend=False,
                             hovertemplate="<b>T2</b><br>Data: %{x|%d/%m/%Y}<br>Valor: %{y:.2f}"))

    today_str_iso = datetime.now().strftime("%Y-%m-%d")
    fig.add_shape(type="line", x0=today_str_iso, x1=today_str_iso, y0=0, y1=1,
                  xref='x', yref='paper', line=dict(color="yellow", width=0.5, dash="dash"))
    fig.add_annotation(x=today_str_iso, y=1.10, xref='x', yref='paper',
                       text=datetime.now().strftime("%d/%m/%Y"),
                       showarrow=False, font=dict(color="yellow", size=8))
    fig.add_shape(type="line", x0=df_plot['time'].min(), x1=df_plot['time'].max(),
                  y0=30, y1=30, line=dict(color="green", width=0.7, dash="dashdot"))
    fig.update_layout(title=dict(text="Ponto de Interesse", font=dict(size=10, color='white'),
                                 x=0.5, xanchor='center'),
                      plot_bgcolor='black', paper_bgcolor='black', font=dict(color='white'),
                      xaxis=dict(showgrid=True, gridcolor='#1c1c1c'),
                      yaxis=dict(showgrid=True, gridcolor='#1c1c1c'),
                      margin=dict(l=30, r=20, t=40, b=17), showlegend=False)
    return fig

def gerar_grafico_vix(timeframe, divisor_macro):
    df_vix = coletar("VIX", timeframe)
    df_sp = coletar("US500", timeframe)
    df_usdx = coletar("USDX", timeframe)
    df_tlt = coletar("TLT", timeframe)

    if any(df is None or df.empty for df in [df_vix, df_sp, df_usdx, df_tlt]):
        return go.Figure()

    df = df_vix.copy()
    df['media'] = df['close'].ewm(span=742).mean()
    roll_std = df['close'].rolling(200).std()
    df['z'] = (df['close'] - df['media']) / roll_std.replace(0, np.nan)

    df_sp['ret'] = np.log(df_sp['close']).diff()
    df_sp['vol'] = df_sp['ret'].rolling(20).std() * np.sqrt(252) * 100
    df['vol_realizada'] = df_sp['vol'] / divisor_macro

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df['time'], y=df['close'], mode='lines',
                             name='VIX (Preço)', line=dict(color='orange'), showlegend=False))
    fig.add_trace(go.Scatter(x=df['time'], y=df['media'], mode='lines',
                             name='Média Exponencial (742)', line=dict(color='white', dash='dash'),
                             showlegend=False))
    fig.add_trace(go.Bar(x=df['time'], y=df['vol_realizada'], name='Vol. Realizada (S&P 500) ÷ divisor',
                         marker_color='tan', showlegend=False))

    df['marker'] = np.where(df['z'] > 3.5, 'acima_3.5', np.where(df['z'] > 3, 'acima_3', ''))
    df_z3 = df[df['marker'] == 'acima_3']
    df_z35 = df[df['marker'] == 'acima_3.5']
    fig.add_trace(go.Scatter(x=df_z3['time'], y=df_z3['close'], mode='markers',
                             marker=dict(size=3, color='yellow'), showlegend=False))
    fig.add_trace(go.Scatter(x=df_z35['time'], y=df_z35['close'], mode='markers',
                             marker=dict(size=3, color='red'), showlegend=False))

    fig.add_trace(go.Scatter(x=df_usdx['time'], y=df_usdx['close'] / divisor_macro, mode='lines',
                             name='DXY', line=dict(color='lime')))
    fig.add_trace(go.Scatter(x=df_tlt['time'], y=df_tlt['close'] / divisor_macro, mode='lines',
                             name='TLT', line=dict(color='magenta')))

    today_str = datetime.now().strftime("%Y-%m-%d")
    fig.add_shape(type="line", x0=today_str, x1=today_str, y0=0, y1=1,
                  xref='x', yref='paper', line=dict(color="yellow", width=0.5, dash="dash"))
    fig.add_annotation(x=today_str, y=1.03, xref="x", yref="paper",
                       text=datetime.now().strftime("%d/%m/%Y"), showarrow=False,
                       font=dict(color="yellow", size=10))
    fig.update_layout(title=dict(text='VIX - Contexto Macro', font=dict(size=10, color='white'),
                                 x=0.5, xanchor='center'),
                      plot_bgcolor='black', paper_bgcolor='black', font=dict(color='white'),
                      xaxis=dict(showgrid=True, gridcolor='#1c1c1c'),
                      yaxis=dict(showgrid=True, gridcolor='#1c1c1c'),
                      margin=dict(l=30, r=20, t=40, b=17), showlegend=False)
    return fig

# ===================== APP / SERVER =====================
app = dash.Dash(__name__)
server = app.server

# Health check do Render
@server.route("/healthz", methods=["GET"])
def healthz():
    return "ok", 200

# ===================== LAYOUT =====================
app.layout = html.Div(style={"backgroundColor": "black", "padding": "10px"}, children=[
    # TOPO
    html.Div([
        html.Div([
            html.Img(src='/assets/globalstat_logo.png', style={'height': '120px', 'margin': '10px'}),
        ], style={'display': 'inline-block', 'verticalAlign': 'top', 'marginRight': '20px'}),
        html.Div([
            html.P("Buy and Hold", style={"color": "white", "marginBottom": "5px"}),
            dcc.Interval(id='atualizacao-temporizada', interval=3*60*1000, n_intervals=0),
            html.Div(id="mini-graficos", style={"whiteSpace": "nowrap", "overflowX": "auto"}),
            html.Div(id="mensagem-erro", style={"color": "red", "marginTop": "10px"})
        ], style={'display': 'inline-block', 'width': 'calc(100% - 100px)'})
    ], style={'display': 'flex', 'alignItems': 'center', 'marginBottom': '10px'}),

    html.Hr(style={"margin": "20px 0", "borderColor": "#555"}),

    # CONTROLES
    html.Div([
        html.Button("Atualizar", id="refresh-button", style={"marginRight": "10px"}),
        dcc.Dropdown(id="timeframe-dropdown",
                     options=[{"label": k, "value": k} for k in timeframes_dict],
                     value="D1", clearable=False,
                     style={"width": "100px", "display": "inline-block"}),
        html.Div(id="last-update", style={"color": "gray", "marginLeft": "20px", "display": "inline-block"})
    ]),

    # 4 GRÁFICOS
    html.Div([
        html.Div([
            dcc.Checklist(
                id="checklist-index",
                options=[{"label": a, "value": a} for a in ['USDX', 'EURX', 'GBPX', 'JPYX', 'CADX', 'SEKX', 'CHFX']],
                value=['USDX', 'EURX', 'GBPX', 'JPYX', 'CADX', 'SEKX', 'CHFX'],
                labelStyle={'display': 'inline-block', 'marginRight': '12px', 'color': 'white', 'fontSize': '12px'},
                style={'marginBottom': '5px', 'textAlign': 'center'}
            ),
            dcc.Graph(id='graph-index', style={'height': '38vh', 'width': '100%'})
        ], style={'width': '25%', 'padding': '5px', 'display': 'flex', 'flexDirection': 'column', 'alignItems': 'center'}),

        html.Div([
            dcc.Checklist(
                id='checklist-comparativo',
                options=[{"label": k, "value": k} for k in ativos_comp.keys()],
                value=list(ativos_comp.keys()),
                labelStyle={'display': 'inline-block', 'marginRight': '12px', 'color': 'white', 'fontSize': '12px'},
                style={'marginBottom': '5px', 'textAlign': 'center'}
            ),
            dcc.Graph(id='graph-comparativo', style={'height': '38vh', 'width': '100%'})
        ], style={'width': '25%', 'padding': '5px', 'display': 'flex', 'flexDirection': 'column', 'alignItems': 'center'}),

        html.Div([
            html.Div(' ', style={'height': '27px'}),
            dcc.Graph(id='graph-t2', style={'height': '38vh', 'width': '100%'})
        ], style={'width': '25%', 'padding': '5px', 'display': 'flex', 'flexDirection': 'column', 'alignItems': 'center'}),

        html.Div([
            html.Div([
                dcc.Dropdown(
                    id='divisor-dropdown',
                    options=[{'label': '3', 'value': 3}, {'label': '4', 'value': 4}],
                    value=3, clearable=False,
                    style={'width': '50px', 'height': '20px', 'borderRadius': '0px',
                           'fontSize': '12px', 'padding': '0px', 'textAlign': 'center',
                           'marginBottom': '2px', 'float': 'right'}
                )
            ], style={'width': '100%', 'display': 'inline-block', 'textAlign': 'right'}),
            dcc.Graph(id='grafico-vix', style={'height': '38vh', 'width': '100%'})
        ], style={'width': '25%', 'padding': '5px', 'display': 'flex', 'flexDirection': 'column', 'alignItems': 'center'})
    ], style={'display': 'flex'}),

    html.Hr(style={"margin": "20px 0", "borderColor": "#555"}),

    # GAMMA
    html.Div(id='gamma-charts-container', style={
        'display': 'flex', 'flexDirection': 'column', 'alignItems': 'center',
        'gap': '20px', 'padding': '10px 0'
    }),

    html.Hr(style={"margin": "20px 0", "borderColor": "#555"}),

    # CRYPTO MINI-CHARTS
    html.Div(id='crypto-charts-container', style={
        'display': 'flex', 'flexWrap': 'wrap', 'justifyContent': 'center',
        'gap': '10px', 'padding': '10px 0'
    }),

    html.Div(id="contador", style={"color": "gray", "marginTop": "20px"}),

    dcc.Interval(id="interval-refresh", interval=interval_ms, n_intervals=0),
    dcc.Interval(id="interval-1s", interval=1000, n_intervals=0),
    dcc.Interval(id='interval-component', interval=300*1000, n_intervals=0)
])

# ===================== CALLBACKS =====================
@app.callback(
    [Output("mini-graficos", "children"), Output("mensagem-erro", "children")],
    Input("atualizacao-temporizada", "n_intervals")
)
def atualizar_mini_graficos(n):
    if n == 0:
        time.sleep(random.uniform(0.6, 1.4))

    graficos, erros = [], []
    symbols = [YF_MAP.get(t, t) for t in cryptos]
    data_map = yf_download_cached(symbols, period="90d", interval="1d")

    for cripto in cryptos:
        yf_symbol = YF_MAP.get(cripto, cripto)
        df = data_map.get(yf_symbol, pd.DataFrame())
        if df is None or df.empty or 'Close' not in df.columns:
            erros.append("XAU" if cripto == "GC=F" else cripto.replace("-USD","").replace("=F","").replace("X",""))
            continue

        df_b2 = calcular_b2(df)
        if df_b2.empty:
            erros.append("XAU" if cripto == "GC=F" else cripto.replace("-USD","").replace("=F","").replace("X",""))
            continue

        nome = "XAU" if cripto == "GC=F" else cripto.replace("-USD","").replace("=F","").replace("X","")
        cor_base = "#FFD700" if cripto == "GC=F" else "white"

        fig = make_subplots(specs=[[{"secondary_y": True}]])
        fig.add_trace(go.Scatter(x=df_b2['Date'], y=df_b2['B2'], mode='lines',
                                 line=dict(color=cor_base, width=1, dash='dot'),
                                 name=nome, showlegend=False, connectgaps=True),
                      secondary_y=False)

        df_sinal = df_b2[df_b2['B2'] < 0]
        fig.add_trace(go.Scatter(x=df_sinal['Date'], y=df_sinal['B2'], mode='markers',
                                 marker=dict(color='lime', size=8, symbol='triangle-up'),
                                 name="sinal", showlegend=False),
                      secondary_y=False)

        fig.add_trace(go.Scatter(x=df_b2['Date'], y=df_b2['Close'], mode='lines',
                                 line=dict(color='yellow', width=1), name='Preço', showlegend=False),
                      secondary_y=True)

        data_atual = datetime.today().replace(hour=0, minute=0, second=0, microsecond=0)
        fig.add_vline(x=data_atual, line_width=1, line_dash="dash", line_color="#FFFACD")
        fig.add_annotation(x=data_atual, y=0, xref="x", yref="paper",
                           text=data_atual.strftime("%Y-%m-%d"), showarrow=False,
                           font=dict(color="#FFFACD", size=8), align="center", yanchor="top")

        fig.update_layout(title=nome, title_x=0.5, height=120, width=200,
                          margin=dict(l=5, r=5, t=25, b=20),
                          plot_bgcolor="black", paper_bgcolor="black",
                          font=dict(color="white", size=8),
                          xaxis=dict(showticklabels=False, showgrid=False, zeroline=False),
                          yaxis=dict(showticklabels=False, showgrid=False, zeroline=False),
                          yaxis2=dict(showticklabels=False, showgrid=False, zeroline=False))

        graficos.append(html.Div(dcc.Graph(figure=fig),
                                 style={"display": "inline-block", "marginRight": "10px",
                                        "border": "1px solid #444", "borderRadius": "10px",
                                        "padding": "5px", "backgroundColor": "#111"}))

    mensagem_erro = "Falha ao carregar: " + ", ".join(erros) if erros else ""
    return graficos, mensagem_erro

@app.callback(
    [Output('graph-index', 'figure'),
     Output('graph-comparativo', 'figure'),
     Output('graph-t2', 'figure'),
     Output('grafico-vix', 'figure'),
     Output('last-update', 'children')],
    [Input('refresh-button', 'n_clicks'),
     Input('interval-refresh', 'n_intervals'),
     Input('checklist-index', 'value'),
     Input('checklist-comparativo', 'value')],
    [State('timeframe-dropdown', 'value'),
     State('divisor-dropdown', 'value')]
)
def atualizar_todos(n_clicks, n_intervals, index_sel, comp_sel, timeframe_label, divisor_macro):
    # pequeno "stagger" na 1ª carga / manual refresh
    if (n_clicks is None and n_intervals == 0) or (n_clicks and n_clicks > 0):
        time.sleep(random.uniform(0.6, 1.8))

    try:
        fig_index = gerar_range_index_plotly(timeframe_label, index_sel)
    except Exception as e:
        print("[index] erro:", e)
        fig_index = go.Figure()
    time.sleep(0.15)

    try:
        fig_comp = gerar_range_comparativo_plotly(timeframe_label, comp_sel)
    except Exception as e:
        print("[comparativo] erro:", e)
        fig_comp = go.Figure()
    time.sleep(0.15)

    try:
        fig_t2 = calcular_hotelling_t2(None, timeframe_label)
    except Exception as e:
        print("[t2] erro:", e)
        fig_t2 = go.Figure()
    time.sleep(0.15)

    try:
        fig_vix = gerar_grafico_vix(timeframe_label, divisor_macro)
    except Exception as e:
        print("[vix] erro:", e)
        fig_vix = go.Figure()

    now = datetime.now().strftime("Última atualização: %d/%m/%Y %H:%M:%S")
    return fig_index, fig_comp, fig_t2, fig_vix, now

@app.callback(Output('contador', 'children'), Input('interval-1s', 'n_intervals'))
def atualizar_contador(n):
    tempo_restante = interval_sec - (n % interval_sec)
    return f'Atualização automática em: {tempo_restante} segundos'

@app.callback(Output('gamma-charts-container', 'children'),
              Input('interval-component', 'n_intervals'))
def update_dashboard(n):
    if n == 0:
        time.sleep(random.uniform(0.6, 1.4))

    columns = []
    for ativo in ativos:
        calls, puts, spot, exp_date = get_options_data(ativo)
        print(f"[{datetime.now()}] Atualizando Gamma Exposure para {ativo}")

        if calls.empty or puts.empty or np.isnan(spot):
            fig_strike = go.Figure()
            fig_profile = go.Figure()
        else:
            calls_gamma = gamma_exposure(calls, spot)
            puts_gamma = gamma_exposure(puts, spot)

            strikes_calls = calls['strike']; strikes_puts = puts['strike']
            call_wall = calls.iloc[np.argmax(calls_gamma)]['strike'] if not calls_gamma.empty else spot
            put_wall  = puts .iloc[np.argmax(puts_gamma)]['strike'] if not puts_gamma.empty  else spot

            fig_strike = go.Figure()
            fig_strike.add_trace(go.Bar(x=strikes_calls, y=calls_gamma, name='Call Gamma', marker_color='blue'))
            fig_strike.add_trace(go.Bar(x=strikes_puts, y=-puts_gamma, name='Put Gamma', marker_color='red'))
            fig_strike.add_vline(x=call_wall, line_color='green', line_dash='dot')
            fig_strike.add_vline(x=put_wall,  line_color='red',   line_dash='dot')
            fig_strike.add_vline(x=spot,      line_color='yellow')
            fig_strike.update_layout(template='plotly_dark', height=400,
                                     title=f'{ativo} Gamma Exposure Strike',
                                     xaxis_title='Strike', yaxis_title='Gamma Exposure')

            price_range = np.linspace(spot * 0.90, spot * 1.20, 100)
            gamma_profile = [gamma_exposure(calls, p).sum() - gamma_exposure(puts, p).sum() for p in price_range]
            gamma_flip_idx = np.where(np.diff(np.sign(gamma_profile)))[0]
            gamma_flip = price_range[gamma_flip_idx[0]] if len(gamma_flip_idx) > 0 else spot

            fig_profile = go.Figure()
            fig_profile.add_trace(go.Scatter(x=price_range, y=gamma_profile, mode='lines', name='Gamma Exposure'))
            if not np.isnan(gamma_flip):
                fig_profile.add_vline(x=gamma_flip, line_color='red', line_dash='dot')
            fig_profile.add_vline(x=call_wall, line_color='green', line_dash='dot')
            fig_profile.add_vline(x=put_wall,  line_color='blue',  line_dash='dot')
            fig_profile.add_vline(x=spot,      line_color='yellow')
            fig_profile.update_layout(template='plotly_dark', height=400,
                                      title=f'{ativo} Gamma Exposure Profile',
                                      xaxis_title='Preço Simulado', yaxis_title='Gamma Exposure')

        column = html.Div(style={'display': 'flex', 'flexDirection': 'row', 'padding': '10px',
                                 'width': '100%', 'justifyContent': 'space-between', 'alignItems': 'flex-end'},
                          children=[
                              html.Div(style={'width': '48%', 'padding': '10px'},
                                       children=[html.H4(ativo, style={'textAlign': 'center'}),
                                                 dcc.Graph(figure=fig_strike, config={'displayModeBar': False})]),
                              html.Div(style={'width': '48%', 'padding': '10px'},
                                       children=[dcc.Graph(figure=fig_profile, config={'displayModeBar': False})])
                          ])
        columns.append(column)

    return columns

# ===================== EXECUÇÃO =====================
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8085))
    host = "0.0.0.0" if "PORT" in os.environ else "127.0.0.1"
    app.run(host=host, port=port, debug=False)  # Dash 3+: use run, não run_server
