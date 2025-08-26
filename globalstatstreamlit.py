# ==========================================================
#  Globalstat - Streamlit (login opcional via secrets)
#  Telas: 📊 Global  |  🔮 Previsão (MLQ: MLP / SL-LSTM / TLS-LSTM)
# ==========================================================

# ================= IMPORTAÇÕES ============================
import os
import time
import random
import socket
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import plotly.graph_objs as go
from plotly.subplots import make_subplots
import yfinance as yf
import streamlit as st

# MLQ helpers
import json
import subprocess
import tempfile

# ========== CONFIG GERAL ================
np.seterr(all="ignore")
st.set_page_config(
    page_title="Globalstat Dashboard",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Altura menor nos 4 gráficos principais
CHART_HEIGHT = 360

# =============== CSS (tema escuro + controles menores) ===============
# =============== CSS (tema escuro + controles menores + SCROLL nos multiselects) ===============
st.markdown(
    """
    <style>
      :root { --fs-base: 14.5px; } /* fonte base levemente menor */
      html, body, .stApp { background:#000 !important; color:#fff !important; font-size:var(--fs-base) !important; }
      .block-container { padding-top:.6rem; padding-bottom: 1.0rem; }
      div[data-testid="stDecoration"]{ display:none !important; height:0 !important; }
      header[data-testid="stHeader"]{ background:transparent !important; box-shadow:none !important; }

      /* Sidebar fosca */
      section[data-testid="stSidebar"]{
        background:#0b0b0b !important; border-right:1px solid #1a1a1a !important;
        box-shadow: inset 0 0 0 1px rgba(255,255,255,.03);
      }
      section[data-testid="stSidebar"] *{ color:#e6e6e6 !important; }
      section[data-testid="stSidebar"] label[data-testid="stWidgetLabel"]{
        color:#f0f0f0 !important; font-weight:600; font-size:13px !important;
      }
      section[data-testid="stSidebar"] .stButton > button{
        background:#1a1a1a !important; color:#fff !important; border:1px solid #2a2a2a !important;
        border-radius:8px !important; font-weight:600 !important; height:34px !important;
      }
      section[data-testid="stSidebar"] .stButton > button:hover{ background:#2a2a2a !important; }

      /* Caixas dos charts */
      .stPlotlyChart{ border:1px solid #333; border-radius:10px; background:#111; }

      /* Rótulos dos widgets (conteúdo) */
      label[data-testid="stWidgetLabel"]{ color:#fff !important; font-weight:600; font-size:13px !important; }

      /* Inputs / selects menores (conteúdo) */
      div[data-baseweb="input"] input{
        background:#f2f2f2 !important; color:#111 !important; height:32px !important; padding:4px 8px !important; font-size:13px !important;
      }
      div[data-baseweb="select"] div[role="combobox"]{
        background:#f2f2f2 !important; color:#111 !important; min-height:32px !important; padding:2px 8px !important; font-size:13px !important;
      }

      /* ===== Multiselect: UMA LINHA com SCROLL HORIZONTAL ===== */
      /* container interno das tags do BaseWeb Select */
      div[data-testid="stMultiSelect"] div[data-baseweb="select"] > div{
        display:flex !important;
        flex-wrap: nowrap !important;          /* não quebrar linha */
        overflow-x: auto !important;           /* scroll horizontal */
        overflow-y: hidden !important;
        -webkit-overflow-scrolling: touch;     /* scroll suave no mobile */
        scrollbar-width: thin;                 /* Firefox */
      }
      /* espessura da barra (WebKit) — opcional */
      div[data-testid="stMultiSelect"] div[data-baseweb="select"] > div::-webkit-scrollbar{ height:8px; }
      div[data-testid="stMultiSelect"] div[data-baseweb="select"] > div::-webkit-scrollbar-thumb{ background:#333; border-radius:6px; }

      /* tags/chips menores e mais juntinhas */
      div[data-testid="stMultiSelect"] div[data-baseweb="tag"]{
        transform: scale(.90);                 /* compacto */
        margin-right:4px !important;
      }
      div[data-testid="stMultiSelect"] span{ font-size:12px !important; }
      div[data-testid="stMultiSelect"] label{ font-size:12.5px !important; }
      div[data-testid="stSelectbox"] label{ font-size:12.5px !important; }

      .mini-card{ display:inline-block; margin-right:10px; border:1px solid #444; border-radius:10px; padding:6px; background:#111; }
      .muted{ color:#aaa; }

      /* Botões (conteúdo) */
      .stButton > button{
        background:#222 !important; color:#fff !important; border:1px solid #444 !important;
        border-radius:6px !important; font-weight:600 !important; height:34px !important;
      }
      .stButton > button:hover{ background:#444 !important; }

      /* ===== Login ===== */
      div[data-testid="stForm"] form{ max-width:420px; margin-left:auto !important; margin-right:auto !important; }
      div[data-testid="stForm"] .stTextInput, div[data-testid="stForm"] .stPassword{ max-width:420px; margin-left:auto; margin-right:auto; width:100%; }
      div[data-testid="stForm"] input{ height:36px !important; padding:8px 10px !important; }
      div[data-testid="stForm"] .stButton{ max-width:420px; margin-left:auto; margin-right:auto; text-align:center; }
      div[data-testid="stForm"] .stButton > button,
      div[data-testid="stForm"] button[type="submit"],
      div[data-testid="stForm"] button[data-testid*="FormSubmit"]{
        display:block !important; width:100% !important; max-width:420px !important; height:36px !important; margin:0 auto !important;
        background:#1976d2 !important; border:1px solid #1976d2 !important; color:#fff !important;
      }
    </style>
    """,
    unsafe_allow_html=True,
)

# ==================== LOGIN OPCIONAL ======================
def _maybe_login():
    users = st.secrets.get("users", None)
    if not users:
        return
    if "auth_ok" not in st.session_state:
        st.session_state.auth_ok = False
    if "auth_user" not in st.session_state:
        st.session_state.auth_user = ""

    if st.session_state.auth_ok:
        with st.sidebar:
            st.markdown(f"👋 Olá, **{st.session_state.auth_user}**")
            if st.button("Sair", key="logout_btn", use_container_width=True):
                st.session_state.auth_ok = False
                st.session_state.auth_user = ""
                st.rerun()
        return

    st.markdown("<h3>🔒 Globalstat — Login</h3>", unsafe_allow_html=True)
    with st.form("login_form", clear_on_submit=False):
        u = st.text_input("Usuário", key="login_user", placeholder="seu usuário")
        p = st.text_input("Senha", key="login_pass", placeholder="••••••", type="password")
        enviar = st.form_submit_button("Entrar")
        if enviar:
            if u in users and str(users[u]) == str(p):
                st.session_state.auth_ok = True
                st.session_state.auth_user = u
                st.rerun()
            else:
                st.error("Usuário/senha incorretos.")
    st.stop()

_maybe_login()

# ===== Navegação lateral =====
with st.sidebar:
    st.markdown("### 🧭 Navegação")
    page = st.radio("Escolha a tela", ["📊 Global", "🔮 Previsão"], index=0)
    st.divider()
    # (sem textos/links adicionais)

# ========== CACHE YFINANCE ==========
class YFCache:
    def __init__(self, ttl_seconds=180):
        self.ttl = timedelta(seconds=ttl_seconds)
        self.store = {}
    def get(self, key):
        rec = self.store.get(key)
        if not rec: return None
        ts, data = rec
        if datetime.utcnow() - ts > self.ttl:
            self.store.pop(key, None)
            return None
        return data
    def set(self, key, data):
        self.store[key] = (datetime.utcnow(), data)

YF_CACHE = YFCache(ttl_seconds=180)

def _yf_key(symbols, period, interval):
    if isinstance(symbols, (list, tuple, set)):
        symbols = ','.join(sorted(list(symbols)))
    return f"{symbols}|{period}|{interval}"

def _flatten_multiindex_batch(df_batch, symbols):
    out = {}
    if isinstance(df_batch, pd.DataFrame) and not df_batch.empty:
        if hasattr(df_batch, "columns") and isinstance(df_batch.columns, pd.MultiIndex):
            for sym in symbols:
                try:
                    part = df_batch.xs(sym, axis=1, level=0, drop_level=False)
                    part.columns = part.columns.get_level_values(1)
                    out[sym] = part.copy()
                except Exception:
                    out[sym] = pd.DataFrame()
        else:
            for sym in symbols:
                out[sym] = df_batch.copy()
    else:
        for sym in symbols:
            out[sym] = pd.DataFrame()
    return out

def _chunks(lst, size):
    for i in range(0, len(lst), size):
        yield lst[i:i+size]

def yf_download_cached(symbols, period, interval, max_retries=2, backoff=1.5):
    key = _yf_key(symbols, period, interval)
    cached = YF_CACHE.get(key)
    if cached is not None:
        return cached
    sym_list = [symbols] if isinstance(symbols, str) else list(symbols)
    last_exc = None
    df = pd.DataFrame()
    for att in range(max_retries + 1):
        try:
            df = yf.download(sym_list, period=period, interval=interval,
                             auto_adjust=False, progress=False, threads=False, group_by='ticker')
            break
        except Exception as e:
            last_exc = e
            time.sleep((backoff ** att) + random.random() * 0.3)
    if isinstance(df, pd.DataFrame) and not df.empty:
        result = _flatten_multiindex_batch(df, sym_list)
        YF_CACHE.set(key, result)
        return result
    if last_exc:
        print(f"[YF ERROR] batch download {sym_list} {period} {interval}: {last_exc}")
    result = {}
    for chunk in _chunks(sym_list, 6):
        df_chunk = pd.DataFrame()
        err = None
        for att in range(max_retries + 1):
            try:
                df_chunk = yf.download(chunk, period=period, interval=interval,
                                       auto_adjust=False, progress=False, threads=False, group_by='ticker')
                break
            except Exception as e:
                err = e
                time.sleep((backoff ** att) + random.random() * 0.3)
        if isinstance(df_chunk, pd.DataFrame) and not df_chunk.empty:
            result.update(_flatten_multiindex_batch(df_chunk, chunk))
        else:
            if err:
                print(f"[YF ERROR] chunk download {chunk} {period} {interval}: {err}")
            for s in chunk:
                result[s] = pd.DataFrame()
        time.sleep(0.2 + random.random() * 0.2)
    YF_CACHE.set(key, result)
    return result

def _resample_ohlc_to_4h(df_1h):
    if df_1h.empty:
        return df_1h
    if hasattr(df_1h.index, "tz") and df_1h.index.tz is not None:
        df_1h = df_1h.tz_convert(None)
    cols = [c for c in ['Open','High','Low','Close','Adj Close','Volume'] if c in df_1h.columns]
    agg = {c: ('first' if c=='Open' else 'max' if c=='High' else 'min' if c=='Low'
               else 'sum' if c=='Volume' else 'last') for c in cols}
    try:
        out = df_1h.resample('4H').agg(agg).dropna(how='all')
    except Exception:
        s = df_1h['Close'] if 'Close' in df_1h.columns else df_1h.iloc[:, 0]
        out = s.resample('4H').last().to_frame('Close').dropna()
    return out

# ===================== CONFIG INICIAL ====================
cryptos = [
    "GC=F","BTC-USD","ETH-USD","BNB-USD","SOL-USD","ADA-USD",
    "XRP-USD","DOGE-USD","AVAX-USD","DOT-USD","LTC-USD","LINK-USD","ATOM-USD"
]
SPAN_VOL = 10
interval_ms = 300000
interval_sec = interval_ms // 1000
timeframes_dict = {'D1': 'D1', 'H4': 'H4'}

YF_MAP = {
    "EURUSD.r":"EURUSD=X","GBPUSD.r":"GBPUSD=X","USDJPY.r":"JPY=X",
    "USDCAD.r":"CAD=X","USDCHF.r":"CHF=X","USDSEK.r":"SEK=X",
    "EURGBP.r":"EURGBP=X","EURJPY.r":"EURJPY=X","EURCHF.r":"EURCHF=X","EURAUD.r":"EURAUD=X",
    "GBPJPY.r":"GBPJPY=X","GBPEUR.r":"GBPEUR=X","GBPCHF.r":"GBPCHF=X",
    "EURCAD.r":"EURCAD=X","GBPCAD.r":"GBPCAD=X","EURSEK.r":"EURSEK=X",

    "US500":"^GSPC","USDX":"DX-Y.NYB","VIX":"^VIX","TLT":"TLT",

    "GC=F":"GC=F","BTC-USD":"BTC-USD","ETH-USD":"ETH-USD","BNB-USD":"BNB-USD",
    "SOL-USD":"SOL-USD","ADA-USD":"ADA-USD","XRP-USD":"XRP-USD","DOGE-USD":"DOGE-USD",
    "AVAX-USD":"AVAX-USD","DOT-USD":"DOT-USD","LTC-USD":"LTC-USD","LINK-USD":"LINK-USD","ATOM-USD":"ATOM-USD",

    "SPY":"SPY","QQQ":"QQQ","IWM":"IWM"
}
TF_TO_YF = {"D1":("1y","1d"), "H4":("60d","1h")}
ativos_comp = {'EURUSD.r':1.0000,'GBPUSD.r':1.2000,'USDJPY.r':140.00,'USDCAD.r':1.3500,'USDCHF.r':0.9000,'USDSEK.r':10.5000}
pares_fx = ['EURUSD.r','GBPUSD.r','USDJPY.r','USDCAD.r','USDCHF.r','USDSEK.r']
cores_ativos = {'EURUSD.r':'red','GBPUSD.r':'fuchsia','USDJPY.r':'green','USDCAD.r':'yellow','USDCHF.r':'aqua','USDSEK.r':'purple'}
ativos = ['SPY','QQQ','IWM']

# ========================= Auxiliares/Coleta ======================
def calcular_b2(df, periodo=20):
    df = df.copy()
    if 'Close' not in df.columns:
        raise ValueError("Dados sem coluna 'Close'")
    close = pd.to_numeric(df['Close'] if 'Close' in df.columns else df.iloc[:,0], errors='coerce')
    out = pd.DataFrame(index=close.index)
    out['Close'] = close
    ma = close.rolling(window=periodo, min_periods=periodo).mean()
    std = close.rolling(window=periodo, min_periods=periodo).std()
    upper = ma + 2*std
    lower = ma - 2*std
    delta = (upper - lower).replace(0, np.nan)
    out['B2'] = (close - lower) / delta
    out['Date'] = out.index
    return out[['Close','B2','Date']].dropna()

def coletar(ticker, timeframe):
    label = timeframe if isinstance(timeframe, str) else 'D1'
    period, interval = TF_TO_YF.get(str(label), ("1y","1d"))
    yf_symbol = YF_MAP.get(ticker, ticker)
    data_map = yf_download_cached([yf_symbol], period, interval)
    df = data_map.get(yf_symbol, pd.DataFrame())
    if df is None or df.empty: return pd.DataFrame()
    if 'Close' not in df.columns and 'close' in df.columns:
        df = df.rename(columns={'close':'Close'})
    if 'Close' not in df.columns: return pd.DataFrame()
    df = df.copy()
    df.index = pd.to_datetime(df.index)
    df = df[~df.index.duplicated(keep='last')]
    if label == 'H4' and interval == '1h':
        df = _resample_ohlc_to_4h(df)
    s = df['Close'].dropna()
    return pd.DataFrame({'time': s.index, 'close': s.values})

def get_options_data(ticker, tentativas=3, espera=2):
    try:
        stock = yf.Ticker(ticker)
        expirations = list(stock.options or [])
        for _ in range(tentativas):
            if expirations: break
            time.sleep(espera); expirations = list(stock.options or [])
        if not expirations: return pd.DataFrame(), pd.DataFrame(), np.nan, None
        calls = puts = None; chosen_exp = None
        for exp_date in expirations[:6]:
            try:
                chain = stock.option_chain(exp_date)
                if chain and hasattr(chain,"calls") and hasattr(chain,"puts"):
                    if not chain.calls.empty and not chain.puts.empty:
                        calls = chain.calls.copy(); puts = chain.puts.copy(); chosen_exp = exp_date; break
            except Exception:
                time.sleep(0.2); continue
        if calls is None or puts is None:
            return pd.DataFrame(), pd.DataFrame(), np.nan, None
        exp_ts = pd.to_datetime(chosen_exp, utc=True, errors='coerce')
        calls['expiration'] = exp_ts; puts['expiration'] = exp_ts
        try:
            spot_hist = stock.history(period='1d', auto_adjust=False)
            spot = float(spot_hist['Close'].iloc[-1]) if not spot_hist.empty else np.nan
        except Exception:
            spot = np.nan
        return calls, puts, spot, chosen_exp
    except Exception as e:
        print(f"[EXCEÇÃO] get_options_data({ticker}): {e}")
        return pd.DataFrame(), pd.DataFrame(), np.nan, None

def gamma_exposure(options, spot, rate=0.05):
    try:
        opts = options.copy()
        opts['expiration'] = pd.to_datetime(opts['expiration'], utc=True, errors='coerce')
        now = pd.Timestamp.utcnow()
        T = (opts['expiration'] - now).dt.total_seconds() / (365*24*3600)
        T = T.clip(lower=1/365)
        K = pd.to_numeric(opts['strike'], errors='coerce')
        sigma = pd.to_numeric(opts.get('impliedVolatility', 0.0), errors='coerce').fillna(0.0)
        sigma = sigma.clip(lower=0.001)
        oi = pd.to_numeric(opts.get('openInterest', 0), errors='coerce').fillna(0.0)
        S = float(spot)
        d1 = (np.log(S/K) + (rate + 0.5*sigma**2)*T) / (sigma*np.sqrt(T))
        gamma = np.exp(-0.5*d1**2) / (S*sigma*np.sqrt(2*np.pi*T))
        exposure = gamma * oi * 100 * S * 0.01
        return exposure.fillna(0.0)
    except Exception as e:
        print(f"Erro ao calcular gamma exposure: {e}")
        return pd.Series(np.zeros(len(options)))

# ========================= MLQ (Previsão) ======================
def _normalize_mlq_output(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normaliza o dataframe de saída do MLQ para colunas padrão:
      time (datetime), yhat (float), yhat_lower (opt), yhat_upper (opt)
    """
    if df is None or len(df) == 0:
        return pd.DataFrame()
    df = df.copy()

    for c in ['time','timestamp','ds','date','datetime']:
        if c in df.columns:
            df['time'] = pd.to_datetime(df[c], errors='coerce')
            break
    if 'time' not in df.columns:
        return pd.DataFrame()

    for c in ['yhat','y_pred','pred','forecast','value','y']:
        if c in df.columns:
            df['yhat'] = pd.to_numeric(df[c], errors='coerce')
            break

    for lo in ['yhat_lower','y_lower','lower','lo95','lo']:
        if lo in df.columns:
            df['yhat_lower'] = pd.to_numeric(df[lo], errors='coerce'); break
    for up in ['yhat_upper','y_upper','upper','hi95','hi']:
        if up in df.columns:
            df['yhat_upper'] = pd.to_numeric(df[up], errors='coerce'); break

    out = df[['time','yhat']].copy()
    if 'yhat_lower' in df.columns: out['yhat_lower'] = df['yhat_lower']
    if 'yhat_upper' in df.columns: out['yhat_upper'] = df['yhat_upper']
    out = out.dropna(subset=['time','yhat']).sort_values('time')
    return out

def run_mlq_forecast(ticker: str, horizon_days: int, timeframe_label: str,
                     extra_params: dict | None = None, model_name: str | None = None) -> pd.DataFrame:
    """
    Tenta:
      1) Pacote Python `mlq` com mlq.predict(..., model=...).
      2) CLI externo configurado em st.secrets['mlq_cli'] (aceita --model).
      3) Se nada, retorna vazio.
    """
    extra_params = extra_params or {}
    # 1) Pacote Python
    try:
        import mlq  # seu pacote
        raw = mlq.predict(
            ticker=ticker,
            horizon=horizon_days,
            timeframe=timeframe_label,
            model=model_name,
            **extra_params
        )
        df_raw = pd.DataFrame(raw)
        return _normalize_mlq_output(df_raw)
    except Exception:
        pass

    # 2) CLI externo
    mlq_cli = st.secrets.get("mlq_cli")
    if mlq_cli:
        with tempfile.TemporaryDirectory() as td:
            out_csv = os.path.join(td, "mlq_forecast.csv")
            params_path = os.path.join(td, "params.json")
            with open(params_path, "w", encoding="utf-8") as f:
                json.dump(
                    {"ticker": ticker, "horizon": horizon_days, "timeframe": timeframe_label,
                     "model": model_name, **extra_params},
                    f
                )
            cmd = [
                mlq_cli, "--ticker", ticker, "--horizon", str(horizon_days),
                "--timeframe", str(timeframe_label), "--model", str(model_name or ""),
                "--out", out_csv, "--params", params_path
            ]
            try:
                subprocess.run(cmd, check=True, capture_output=True)
                df_raw = pd.read_csv(out_csv)
                return _normalize_mlq_output(df_raw)
            except Exception as e:
                print("[MLQ CLI] erro:", e)

    return pd.DataFrame()

# ========================= Gráficos =====================
def _apply_height(fig, h=CHART_HEIGHT):
    fig.update_layout(height=h, margin=dict(l=28, r=18, t=34, b=26))
    return fig

def gerar_range_index_plotly(timeframe_label, ativos_visiveis=None):
    # CORES CORRETAS para índices sintéticos
    cores_idx = {'USDX':'white','EURX':'red','GBPX':'fuchsia','JPYX':'green','CADX':'yellow','SEKX':'aqua','CHFX':'purple'}
    indices_sinteticos = {
        'EURX': (['EURUSD.r','EURGBP.r','EURJPY.r','EURCHF.r','EURAUD.r'], [0.315,0.305,0.189,0.111,0.080]),
        'GBPX': (['GBPUSD.r','GBPJPY.r','GBPEUR.r','GBPCHF.r'], [0.347,0.207,0.339,0.107]),
        'JPYX': (['USDJPY.r','EURJPY.r','GBPJPY.r'], [0.5,0.3,0.2]),
        'CADX': (['USDCAD.r','EURCAD.r','GBPCAD.r'], [0.5,0.3,0.2]),
        'SEKX': (['USDSEK.r','EURSEK.r'], [0.6,0.4]),
        'CHFX': (['USDCHF.r','EURCHF.r','GBPCHF.r'], [0.5,0.3,0.2]),
    }
    all_app_tickers = ['USDX']
    for pares,_ in indices_sinteticos.values(): all_app_tickers.extend(pares)
    all_app_tickers = sorted(set(all_app_tickers))
    yf_symbols = [YF_MAP.get(t,t) for t in all_app_tickers]
    period, interval = TF_TO_YF.get(str(timeframe_label), ("1y","1d"))
    data_map = yf_download_cached(yf_symbols, period=period, interval=interval)

    def df_close(sym):
        df = data_map.get(sym, pd.DataFrame())
        if df is None or df.empty: return pd.DataFrame()
        if 'Close' not in df.columns and 'close' in df.columns: df = df.rename(columns={'close':'Close'})
        if 'Close' not in df.columns: return pd.DataFrame()
        out = df[['Close']].copy()
        out.index = pd.to_datetime(out.index)
        out = out[~out.index.duplicated(keep='last')]
        if timeframe_label == 'H4' and interval == '1h':
            out = _resample_ohlc_to_4h(out)
        return pd.DataFrame({'time': out.index, 'close': out['Close'].values})

    df_final = pd.DataFrame()
    usdxy = YF_MAP.get('USDX','USDX')
    df_usdx = df_close(usdxy)
    if not df_usdx.empty:
        df_usdx = df_usdx.rename(columns={'close':'USDX'})
        df_final = df_usdx[['time','USDX']]

    def calc_index(pares,pesos):
        base_df = pd.DataFrame()
        for par in pares:
            sym = YF_MAP.get(par, par)
            d = df_close(sym)
            if not d.empty:
                d = d.rename(columns={'close':par})
                base_df = d if base_df.empty else pd.merge(base_df, d, on='time', how='outer')
        if base_df.empty: return None
        base_df = base_df.sort_values('time').ffill().bfill()
        acc = None
        for par,w in zip(pares,pesos):
            if par in base_df.columns:
                series = base_df[par]; base = series.iloc[0]
                if pd.isna(base) or base == 0: return None
                term = w * np.log(series/base)
                acc = term if acc is None else acc + term
        if acc is None: return None
        base_df['Index'] = 100*np.exp(acc)
        return base_df[['time','Index']]

    for nome,(pares,pesos) in indices_sinteticos.items():
        df_idx = calc_index(pares,pesos)
        if df_idx is not None:
            df_idx = df_idx.rename(columns={'Index':nome})
            df_final = df_idx if df_final.empty else pd.merge(df_final, df_idx, on='time', how='outer')

    if df_final.empty: return _apply_height(go.Figure())

    df_final = df_final.sort_values('time').ffill().bfill()
    ativos_cols = [c for c in df_final.columns if c!='time']
    df_final['MediaGrupo'] = df_final[ativos_cols].mean(axis=1)
    df_final['FaixaAltaSuave'] = df_final['MediaGrupo'] + 10
    df_final['FaixaBaixaSuave'] = df_final['MediaGrupo'] - 10
    df_final['FaixaAltaInternaSuave2'] = df_final['MediaGrupo'] + 9
    df_final['FaixaBaixaInternaSuave2'] = df_final['MediaGrupo'] - 9
    span = 100
    for col in ['MediaGrupo','FaixaAltaSuave','FaixaBaixaSuave','FaixaAltaInternaSuave2','FaixaBaixaInternaSuave2']:
        df_final[f'{col}_smooth'] = df_final[col].ewm(span=span).mean().ewm(span=span).mean()

    fig = go.Figure()
    ativos_visiveis = (ativos_visiveis or ativos_cols)
    for ativo in ativos_visiveis:
        if ativo not in df_final.columns: continue
        cor = cores_idx.get(ativo, 'white')
        fig.add_trace(go.Scatter(x=df_final['time'], y=df_final[ativo], mode='lines',
                                 name=ativo, line=dict(color=cor), showlegend=False))
        fig.add_trace(go.Scatter(x=[df_final['time'].iloc[-1]], y=[df_final[ativo].iloc[-1]], mode='text',
                                 text=[ativo], textposition='middle right', showlegend=False,
                                 textfont=dict(size=11, color=cor)))
    fig.add_trace(go.Scatter(x=df_final['time'], y=df_final['MediaGrupo_smooth'], mode='lines',
                             line=dict(color='silver', width=2, dash='dash'), showlegend=False))
    for c in ['FaixaAltaSuave_smooth','FaixaBaixaSuave_smooth','FaixaAltaInternaSuave2_smooth','FaixaBaixaInternaSuave2_smooth']:
        fig.add_trace(go.Scatter(x=df_final['time'], y=df_final[c], mode='lines', line=dict(color='gray', dash='dash'), showlegend=False))
    today_str = datetime.now().strftime("%Y-%m-%d")
    fig.add_shape(type="line", x0=today_str, x1=today_str, y0=0, y1=1, xref='x', yref='paper',
                  line=dict(color="yellow", width=0.5, dash="dash"))
    fig.add_annotation(x=today_str, y=1.08, xref="x", yref="paper",
                       text=datetime.now().strftime("%d/%m/%Y"), showarrow=False, font=dict(color="yellow", size=8))
    fig.update_layout(title=dict(text=f"Index ({timeframe_label})", font=dict(size=10, color='white'),
                                 x=0.5, y=0.96, xanchor='center'),
                      plot_bgcolor='black', paper_bgcolor='black', font=dict(color='white'),
                      xaxis=dict(showgrid=True, gridcolor='#1c1c1c'),
                      yaxis=dict(showgrid=True, gridcolor='#1c1c1c'),
                      showlegend=False)
    return _apply_height(fig)

def gerar_range_comparativo_plotly(timeframe_label, ativos_visiveis=None):
    df_final = pd.DataFrame()
    for ativo, base in ativos_comp.items():
        d = coletar(ativo, timeframe_label)
        if not d.empty:
            d['norm'] = 100 + 100*np.log(d['close']/base)
            d = d[['time','norm']].rename(columns={'norm':ativo})
            df_final = d if df_final.empty else pd.merge(df_final, d, on='time', how='outer')
    if df_final.empty: return _apply_height(go.Figure())
    df_final = df_final.sort_values('time').ffill().bfill()
    ativos_cols = list(ativos_comp.keys())
    df_final['MediaGrupo'] = df_final[ativos_cols].mean(axis=1)
    df_final['FaixaAltaSuave'] = df_final['MediaGrupo'] + 10
    df_final['FaixaBaixaSuave'] = df_final['MediaGrupo'] - 10
    df_final['FaixaAltaInternaSuave2'] = df_final['MediaGrupo'] + 9
    df_final['FaixaBaixaInternaSuave2'] = df_final['MediaGrupo'] - 9
    span = 100
    for col in ['MediaGrupo','FaixaAltaSuave','FaixaBaixaSuave','FaixaAltaInternaSuave2','FaixaBaixaInternaSuave2']:
        df_final[f'{col}_smooth'] = df_final[col].ewm(span=span).mean()
    fig = go.Figure()
    ativos_visiveis = ativos_visiveis or ativos_cols
    for ativo in ativos_visiveis:
        if ativo not in df_final.columns: continue
        cor = cores_ativos.get(ativo,'white')
        fig.add_trace(go.Scatter(x=df_final['time'], y=df_final[ativo], mode='lines',
                                 name=ativo, line=dict(color=cor), showlegend=False))
        fig.add_trace(go.Scatter(x=[df_final['time'].iloc[-1]], y=[df_final[ativo].iloc[-1]], mode='text',
                                 text=[ativo], textposition='middle right',
                                 showlegend=False, textfont=dict(color=cor, size=11)))
    fig.add_trace(go.Scatter(x=df_final['time'], y=df_final['MediaGrupo_smooth'], mode='lines',
                             line=dict(color='silver', width=2, dash='dash'), showlegend=False))
    for c in ['FaixaAltaSuave_smooth','FaixaBaixaSuave_smooth','FaixaAltaInternaSuave2_smooth','FaixaBaixaInternaSuave2_smooth']:
        fig.add_trace(go.Scatter(x=df_final['time'], y=df_final[c], mode='lines', line=dict(color='gray', dash='dash'), showlegend=False))
    today_str = datetime.now().strftime("%Y-%m-%d")
    fig.add_shape(type="line", x0=today_str, x1=today_str, y0=0, y1=1, xref='x', yref='paper',
                  line=dict(color="yellow", width=0.5, dash="dash"))
    fig.add_annotation(x=today_str, y=1.08, xref="x", yref="paper",
                       text=datetime.now().strftime("%d/%m/%Y"), showarrow=False, font=dict(color="yellow", size=8))
    fig.update_layout(title=dict(text=f"Pares ({timeframe_label})", font=dict(size=10, color='white'), x=0.5, xanchor='center'),
                      plot_bgcolor='black', paper_bgcolor='black', font=dict(color='white'),
                      xaxis=dict(showgrid=True, gridcolor='#1c1c1c', zeroline=False, tickcolor='white'),
                      yaxis=dict(showgrid=True, gridcolor='#1c1c1c', zeroline=False, tickcolor='white'),
                      showlegend=False)
    return _apply_height(fig)

def calcular_hotelling_t2(_, timeframe_label):
    df_final = pd.DataFrame()
    for ativo in pares_fx:
        d = coletar(ativo, timeframe_label)
        if not d.empty:
            d = d.rename(columns={'close': ativo})
            df_final = d if df_final.empty else pd.merge(df_final, d, on='time', how='outer')
    if df_final.empty: return _apply_height(go.Figure())
    df_final = df_final.sort_values('time').ffill().bfill()
    for ativo in pares_fx:
        df_final[f"{ativo}_ret"] = np.log(df_final[ativo] / df_final[ativo].shift(1))
    df_ret = df_final[[f"{ativo}_ret" for ativo in pares_fx]].dropna()
    if df_ret.empty: return _apply_height(go.Figure())
    means = df_ret.mean(); stds = df_ret.std().replace(0, np.nan)
    zscores = (df_ret - means) / stds
    T2 = (zscores.fillna(0)**2).sum(axis=1)
    df_plot = df_final.loc[df_ret.index].copy(); df_plot['T2'] = T2
    idx_red = df_plot[df_plot['T2'] > 30].index
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df_plot['time'], y=df_plot['T2'], mode='lines',
                             line=dict(color='rgb(200,200,200)', width=1), showlegend=False))
    fig.add_trace(go.Scatter(x=df_plot.loc[idx_red,'time'], y=df_plot.loc[idx_red,'T2'],
                             mode='markers', marker=dict(color='red', size=3), showlegend=False))
    today_str_iso = datetime.now().strftime("%Y-%m-%d")
    fig.add_shape(type="line", x0=today_str_iso, x1=today_str_iso, y0=0, y1=1, xref='x', yref='paper',
                  line=dict(color="yellow", width=0.5, dash="dash"))
    fig.add_annotation(x=today_str_iso, y=1.06, xref='x', yref='paper',
                       text=datetime.now().strftime("%d/%m/%Y"), showarrow=False, font=dict(color="yellow", size=8))
    fig.add_shape(type="line", x0=df_plot['time'].min(), x1=df_plot['time'].max(),
                  y0=30, y1=30, line=dict(color="green", width=0.7, dash="dashdot"))
    fig.update_layout(title=dict(text="Ponto de Interesse", font=dict(size=10, color='white'),
                                 x=0.5, xanchor='center'),
                      plot_bgcolor='black', paper_bgcolor='black', font=dict(color='white'),
                      xaxis=dict(showgrid=True, gridcolor='#1c1c1c'),
                      yaxis=dict(showgrid=True, gridcolor='#1c1c1c'),
                      showlegend=False)
    return _apply_height(fig)

def gerar_grafico_vix(timeframe, divisor_macro):
    df_vix = coletar("VIX", timeframe)
    df_sp = coletar("US500", timeframe)
    df_usdx = coletar("USDX", timeframe)
    df_tlt = coletar("TLT", timeframe)
    if any(df is None or df.empty for df in [df_vix, df_sp, df_usdx, df_tlt]):
        return _apply_height(go.Figure())
    df = df_vix.copy()
    df['media'] = df['close'].ewm(span=742).mean()
    roll_std = df['close'].rolling(200).std()
    df['z'] = (df['close'] - df['media']) / roll_std.replace(0, np.nan)
    df_sp['ret'] = np.log(df_sp['close']).diff()
    df_sp['vol'] = df_sp['ret'].rolling(20).std()*np.sqrt(252)*100
    df['vol_realizada'] = df_sp['vol'] / divisor_macro
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df['time'], y=df['close'], mode='lines', line=dict(color='orange'), showlegend=False))
    fig.add_trace(go.Scatter(x=df['time'], y=df['media'], mode='lines', line=dict(color='white', dash='dash'), showlegend=False))
    fig.add_trace(go.Bar(x=df['time'], y=df['vol_realizada'], marker_color='tan', showlegend=False))
    df['marker'] = np.where(df['z'] > 3.5,'acima_3.5', np.where(df['z'] > 3,'acima_3',''))
    df_z3 = df[df['marker']=='acima_3']; df_z35 = df[df['marker']=='acima_3.5']
    fig.add_trace(go.Scatter(x=df_z3['time'], y=df_z3['close'], mode='markers', marker=dict(size=3, color='yellow'), showlegend=False))
    fig.add_trace(go.Scatter(x=df_z35['time'], y=df_z35['close'], mode='markers', marker=dict(size=3, color='red'), showlegend=False))
    fig.add_trace(go.Scatter(x=df_usdx['time'], y=df_usdx['close']/divisor_macro, mode='lines', line=dict(color='lime')))
    fig.add_trace(go.Scatter(x=df_tlt['time'], y=df_tlt['close']/divisor_macro, mode='lines', line=dict(color='magenta')))
    today_str = datetime.now().strftime("%Y-%m-%d")
    fig.add_shape(type="line", x0=today_str, x1=today_str, y0=0, y1=1, xref='x', yref='paper',
                  line=dict(color="yellow", width=0.5, dash="dash"))
    fig.add_annotation(x=today_str, y=1.02, xref="x", yref="paper",
                       text=datetime.now().strftime("%d/%m/%Y"), showarrow=False, font=dict(color="yellow", size=9))
    fig.update_layout(title=dict(text='VIX - Contexto Macro', font=dict(size=10, color='white'),
                                 x=0.5, xanchor='center'),
                      plot_bgcolor='black', paper_bgcolor='black', font=dict(color='white'),
                      xaxis=dict(showgrid=True, gridcolor='#1c1c1c'),
                      yaxis=dict(showgrid=True, gridcolor='#1c1c1c'),
                      showlegend=False)
    return _apply_height(fig)

# ===================== UI / EXECUÇÃO =====================

if "last_update_dt" not in st.session_state:
    st.session_state.last_update_dt = datetime.now()

# --- PÁGINA PREVISÃO (MLQ) ---
if page == "🔮 Previsão":
    st.title("🔮 Previsão — MLQ")

    left, right = st.columns([2, 1])
    with left:
        ticker = st.text_input("Ticker", "SPY")
        horizon_days = st.number_input("Horizonte (dias à frente)", 1, 365, 30)
        timeframe_prev = st.selectbox("Timeframe de referência", list(timeframes_dict.keys()), index=0)
        modelo = st.selectbox("Modelo", ["MLP", "SL-LSTM", "TLS-LSTM"], index=0)
        run_btn = st.button("Rodar MLQ", use_container_width=True)
    with right:
        st.caption("Parâmetros extras (JSON, opcional)")
        extra = st.text_area("{}", height=110, label_visibility="collapsed")
        try:
            extra_params = json.loads(extra) if extra.strip() else {}
        except Exception:
            st.warning("JSON inválido nos parâmetros extras. Usando vazio.")
            extra_params = {}

    if run_btn:
        with st.spinner("Rodando MLQ..."):
            df_pred = run_mlq_forecast(
                ticker=ticker,
                horizon_days=horizon_days,
                timeframe_label=timeframe_prev,
                extra_params=extra_params,
                model_name=modelo
            )

        if df_pred.empty:
            st.info("MLQ indisponível neste ambiente ou sem retorno.")
            st.stop()

        # preço recente para contexto
        df_hist = coletar(ticker, timeframe_prev)
        if not df_hist.empty:
            cutoff = df_hist['time'].max() - pd.Timedelta(days=90)
            df_hist = df_hist[df_hist['time'] >= cutoff]

        fig = go.Figure()
        if not df_hist.empty:
            fig.add_trace(go.Scatter(x=df_hist['time'], y=df_hist['close'], mode='lines',
                                     name='Histórico', line=dict(width=1.2)))
        fig.add_trace(go.Scatter(x=df_pred['time'], y=df_pred['yhat'], mode='lines+markers',
                                 name=f'{modelo} (yhat)', line=dict(width=2)))
        if {'yhat_lower','yhat_upper'}.issubset(df_pred.columns):
            fig.add_trace(go.Scatter(
                x=pd.concat([df_pred['time'], df_pred['time'][::-1]]),
                y=pd.concat([df_pred['yhat_upper'], df_pred['yhat_lower'][::-1]]),
                fill='toself', mode='lines', line=dict(width=0),
                name='IC', opacity=0.2
            ))
        fig.update_layout(template='plotly_dark', height=560,
                          title=f"Previsão — {ticker} (+{horizon_days}d) | Modelo: {modelo}",
                          xaxis_title="Data", yaxis_title="Preço/Valor")
        st.plotly_chart(fig, use_container_width=True)
        st.dataframe(df_pred.reset_index(drop=True))

    st.stop()

# --- Barra superior com logo e mini-gráficos (tela Global)
top_cols = st.columns([1, 6])
with top_cols[0]:
    logo_path = "assets/globalstat_logo.png"
    if os.path.exists(logo_path): st.image(logo_path, width=180)  # <<< corrigido (sem use_container_width)
    else: st.markdown("### Globalstat")
with top_cols[1]:
    st.markdown("**Buy and Hold**")
    erros = []
    symbols = [YF_MAP.get(t, t) for t in cryptos]
    data_map = yf_download_cached(symbols, period="90d", interval="1d")
    mini_cols = st.columns(6); i_col = 0
    for cripto in cryptos:
        yf_symbol = YF_MAP.get(cripto, cripto)
        df = data_map.get(yf_symbol, pd.DataFrame())
        if df is None or df.empty or 'Close' not in df.columns:
            erros.append("XAU" if cripto=="GC=F" else cripto.replace("-USD","").replace("=F","").replace("X","")); continue
        try:
            df_b2 = calcular_b2(df)
        except Exception:
            df_b2 = pd.DataFrame()
        if df_b2.empty:
            erros.append("XAU" if cripto=="GC=F" else cripto.replace("-USD","").replace("=F","").replace("X","")); continue
        nome = "XAU" if cripto=="GC=F" else cripto.replace("-USD","").replace("=F","").replace("X","")
        cor_base = "#FFD700" if cripto=="GC=F" else "white"
        fig = make_subplots(specs=[[{"secondary_y": True}]])
        fig.add_trace(go.Scatter(x=df_b2['Date'], y=df_b2['B2'], mode='lines',
                                 line=dict(color=cor_base, width=1, dash='dot'),
                                 name=nome, showlegend=False, connectgaps=True), secondary_y=False)
        df_sinal = df_b2[df_b2['B2'] < 0]
        fig.add_trace(go.Scatter(x=df_sinal['Date'], y=df_sinal['B2'], mode='markers',
                                 marker=dict(color='lime', size=7, symbol='triangle-up'),
                                 name="sinal", showlegend=False), secondary_y=False)
        fig.add_trace(go.Scatter(x=df_b2['Date'], y=df_b2['Close'], mode='lines',
                                 line=dict(color='yellow', width=1), name='Preço', showlegend=False), secondary_y=True)
        data_atual = datetime.today().replace(hour=0, minute=0, second=0, microsecond=0)
        fig.add_vline(x=data_atual, line_width=1, line_dash="dash", line_color="#FFFACD")
        fig.add_annotation(x=data_atual, y=0, xref="x", yref="paper",
                           text=data_atual.strftime("%Y-%m-%d"), showarrow=False,
                           font=dict(color="#FFFACD", size=8), align="center", yanchor="top")
        fig.update_layout(title=nome, title_x=0.5, height=130, width=240,
                          margin=dict(l=5, r=5, t=24, b=18),
                          plot_bgcolor="black", paper_bgcolor="black",
                          font=dict(color="white", size=8),
                          xaxis=dict(showticklabels=False, showgrid=False, zeroline=False),
                          yaxis=dict(showticklabels=False, showgrid=False, zeroline=False),
                          yaxis2=dict(showticklabels=False, showgrid=False, zeroline=False))
        with mini_cols[i_col % len(mini_cols)]:
            st.plotly_chart(fig, use_container_width=True)
        i_col += 1
    if erros:
        st.markdown(f"<span class='muted'>Falha ao carregar: {', '.join(erros)}</span>", unsafe_allow_html=True)

st.markdown("""<hr style="border-color:#555">""", unsafe_allow_html=True)

# --- Controles
ctrl_cols = st.columns([1, 1, 2, 4])
with ctrl_cols[0]:
    refresh = st.button("Atualizar", use_container_width=True)
with ctrl_cols[1]:
    timeframe_label = st.selectbox("Timeframe", list(timeframes_dict.keys()), index=0)
with ctrl_cols[2]:
    auto_refresh = st.toggle(f"Auto-refresh (a cada {interval_sec}s)", value=False)
with ctrl_cols[3]:
    st.markdown(
        f"""
        <div id="contador" class="muted" style="margin-top:20px; font-size:12px;">
            Atualização automática em: <span id="sec">{interval_sec}</span> segundos
        </div>
        <script>
        var s = {interval_sec};
        setInterval(function(){{
            if(s<=0) s = {interval_sec};
            var el = document.getElementById('sec');
            if (el) {{ el.innerText = s; }}
            s -= 1;
        }}, 1000);
        </script>
        """,
        unsafe_allow_html=True
    )
if auto_refresh:
    st.markdown(f"<meta http-equiv='refresh' content='{interval_sec}'>", unsafe_allow_html=True)
if refresh:
    st.session_state.last_update_dt = datetime.now()
st.caption(f"Última atualização: {st.session_state.last_update_dt.strftime('%d/%m/%Y %H:%M:%S')}")

# --- Multiselects + divisor (controles menores)
sel_cols = st.columns([3, 3, 3, 3])
with sel_cols[0]:
    index_sel = st.multiselect("Index - séries visíveis",
                               ['USDX','EURX','GBPX','JPYX','CADX','SEKX','CHFX'],
                               default=['USDX','EURX','GBPX','JPYX','CADX','SEKX','CHFX'])
with sel_cols[1]:
    comp_sel = st.multiselect("Pares FX - visíveis", list(ativos_comp.keys()), default=list(ativos_comp.keys()))
with sel_cols[2]:
    st.write("&nbsp;", unsafe_allow_html=True)
with sel_cols[3]:
    divisor_macro = st.selectbox("Divisor Macro (VIX)", [3, 4], index=0)

# --- 4 gráficos
gcols = st.columns(4)
with gcols[0]:
    try: fig_index = gerar_range_index_plotly(timeframe_label, index_sel)
    except Exception as e: print("[index] erro:", e); fig_index = go.Figure()
    st.plotly_chart(fig_index, use_container_width=True, theme=None)
with gcols[1]:
    try: fig_comp = gerar_range_comparativo_plotly(timeframe_label, comp_sel)
    except Exception as e: print("[comparativo] erro:", e); fig_comp = go.Figure()
    st.plotly_chart(fig_comp, use_container_width=True, theme=None)
with gcols[2]:
    try: fig_t2 = calcular_hotelling_t2(None, timeframe_label)
    except Exception as e: print("[t2] erro:", e); fig_t2 = go.Figure()
    st.plotly_chart(fig_t2, use_container_width=True, theme=None)
with gcols[3]:
    try: fig_vix = gerar_grafico_vix(timeframe_label, divisor_macro)
    except Exception as e: print("[vix] erro:", e); fig_vix = go.Figure()
    st.plotly_chart(fig_vix, use_container_width=True, theme=None)

st.markdown("""<hr style="border-color:#555">""", unsafe_allow_html=True)

# ===================== Seção GAMMA =====================
st.subheader("Gamma Exposure", divider="gray")
for ativo in ativos:
    calls, puts, spot, exp_date = get_options_data(ativo)
    st.caption(f"[{datetime.now().strftime('%d/%m/%Y %H:%M:%S')}] Atualizando Gamma Exposure para **{ativo}**")
    if calls.empty or puts.empty or np.isnan(spot):
        fig_strike = go.Figure(); fig_profile = go.Figure()
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
        fig_strike.update_layout(template='plotly_dark', height=CHART_HEIGHT,
                                 title=f'{ativo} Gamma Exposure Strike',
                                 xaxis_title='Strike', yaxis_title='Gamma Exposure')
        price_range = np.linspace(spot * 0.90, spot * 1.20, 100)
        gamma_profile = [gamma_exposure(calls, p).sum() - gamma_exposure(puts, p).sum() for p in price_range]
        gamma_flip_idx = np.where(np.diff(np.sign(gamma_profile)))[0]
        gamma_flip = price_range[gamma_flip_idx[0]] if len(gamma_flip_idx) > 0 else spot
        fig_profile = go.Figure()
        fig_profile.add_trace(go.Scatter(x=price_range, y=gamma_profile, mode='lines', name='Gamma Exposure'))
        if not np.isnan(gamma_flip): fig_profile.add_vline(x=gamma_flip, line_color='red', line_dash='dot')
        fig_profile.add_vline(x=call_wall, line_color='green', line_dash='dot')
        fig_profile.add_vline(x=put_wall,  line_color='blue',  line_dash='dot')
        fig_profile.add_vline(x=spot,      line_color='yellow')
        fig_profile.update_layout(template='plotly_dark', height=CHART_HEIGHT,
                                  title=f'{ativo} Gamma Exposure Profile',
                                  xaxis_title='Preço Simulado', yaxis_title='Gamma Exposure')
    c1, c2 = st.columns(2)
    with c1: st.plotly_chart(fig_strike, use_container_width=True)
    with c2: st.plotly_chart(fig_profile, use_container_width=True)

st.markdown("""<hr style="border-color:#555">""", unsafe_allow_html=True)
st.caption("Cripto (mini-charts acima).")
# ===================== FIM ==============================
