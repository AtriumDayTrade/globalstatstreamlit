# ===================== GLOBALSTAT 3.0 - APP COMPLETO =====================

import dash
from dash import dcc, html, State
import plotly.graph_objs as go
from plotly.subplots import make_subplots
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime

from dash.dependencies import Input, Output



# ---- MetaTrader5: tornar opcional no servidor ----
try:
    import MetaTrader5 as mt5
    MT5_OK = True
except Exception:
    MT5_OK = False



# PARTE 1
# ===================== CONFIGURAÇÃO INICIAL =====================

cryptos = [
    "GC=F", "BTC-USD", "ETH-USD", "BNB-USD", "SOL-USD", "ADA-USD",
    "XRP-USD", "DOGE-USD", "AVAX-USD", "DOT-USD", "LTC-USD", "LINK-USD", "ATOM-USD"
]

SPAN_VOL = 10
interval_ms = 300000
interval_sec = interval_ms // 1000
N_BARRAS = 365  # ← usado para definir o número de candles ao puxar dados do MT5

if MT5_OK:
    timeframes_dict = {
        'D1': mt5.TIMEFRAME_D1,
        'H4': mt5.TIMEFRAME_H4
    }
else:
    # Sem MT5 (ex.: no Render) usamos apenas rótulos
    timeframes_dict = {'D1': 'D1', 'H4': 'H4'}

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




# PARTE 2
#==============================================================================
#Todas as Funções
# ===================================================================



# ========================= Funções Auxiliares ======

def calcular_b2(df, periodo=20):
    df = df.copy()
    df['MA'] = df['Close'].rolling(window=periodo).mean()
    df['STD'] = df['Close'].rolling(window=periodo).std()
    df['Upper'] = df['MA'] + 2 * df['STD']
    df['Lower'] = df['MA'] - 2 * df['STD']
    delta = df['Upper'] - df['Lower']
    delta = delta.replace(0, np.nan)
    df['B2'] = (df['Close'] - df['Lower']) / delta
    df['Date'] = df.index
    return df[['Close', 'B2', 'Date']].dropna()

def obter_dados_cripto(ticker):
    try:
        df = yf.download(ticker, period="90d", interval="1d", auto_adjust=False)
        df.columns = df.columns.get_level_values(0) if isinstance(df.columns, pd.MultiIndex) else df.columns
        if df.empty or 'Close' not in df.columns:
            raise ValueError("Dados vazios ou sem 'Close'")
        return calcular_b2(df)
    except Exception as e:
        print(f"Erro ao processar {ticker}: {e}")
        return pd.DataFrame()

def coletar(simbolo, timeframe):
    if not MT5_OK:
        return None
    if not mt5.initialize():
        print("Erro ao inicializar MT5")
        return None
    hoje = datetime.now()
    df = mt5.copy_rates_from(simbolo, timeframe, hoje, 365)
    if df is None or len(df) == 0:
        return None
    df = pd.DataFrame(df)
    df['time'] = pd.to_datetime(df['time'], unit='s')
    return df


import time

def get_options_data(ticker, tentativas=3, espera=2):
    try:
        stock = yf.Ticker(ticker)

        # 🔁 Tenta buscar datas de expiração
        expirations = stock.options
        tentativas_restantes = tentativas

        while not expirations and tentativas_restantes > 0:
            print(f"[RETRY] Tentando novamente obter opções para {ticker} ({tentativas - tentativas_restantes + 1}/{tentativas})")
            time.sleep(espera)
            expirations = stock.options
            tentativas_restantes -= 1

        if not expirations:
            print(f"[AVISO] Nenhuma data de expiração encontrada para {ticker} após {tentativas} tentativas.")
            return pd.DataFrame(), pd.DataFrame(), np.nan, None

        # ▶️ Tenta com a primeira expiração
        exp_date = expirations[0]
        try:
            opt_chain = stock.option_chain(exp_date)
            calls = opt_chain.calls
            puts = opt_chain.puts
        except Exception as e:
            print(f"[FALHA] Primeira expiração falhou para {ticker} ({exp_date}). Tentando próxima...")
            if len(expirations) > 1:
                exp_date = expirations[1]
                try:
                    opt_chain = stock.option_chain(exp_date)
                    calls = opt_chain.calls
                    puts = opt_chain.puts
                except Exception as e:
                    print(f"[ERRO] Segunda tentativa também falhou para {ticker}: {e}")
                    return pd.DataFrame(), pd.DataFrame(), np.nan, None
            else:
                return pd.DataFrame(), pd.DataFrame(), np.nan, None

        # 🔒 Validação final
        if calls is None or puts is None or calls.empty or puts.empty:
            print(f"[ERRO] Option chain inválida para {ticker}")
            return pd.DataFrame(), pd.DataFrame(), np.nan, None

        # 🧠 Data de expiração e preço spot
        calls['expiration'] = pd.to_datetime(exp_date)
        puts['expiration'] = pd.to_datetime(exp_date)

        spot_data = stock.history(period='1d')
        spot = spot_data['Close'].iloc[0] if not spot_data.empty else np.nan

        return calls, puts, spot, exp_date

    except Exception as e:
        print(f"[EXCEÇÃO] Erro ao obter dados de {ticker}: {e}")
        return pd.DataFrame(), pd.DataFrame(), np.nan, None

def gamma_exposure(options, spot, rate=0.05):
    try:
        S = spot
        K = options['strike']
        sigma = options['impliedVolatility']
        T = options['expiration'].apply(lambda x: max((x - pd.Timestamp.today()).days / 365, 1/365))
        d1 = (np.log(S/K) + (rate + sigma**2 / 2)*T) / (sigma*np.sqrt(T))
        gamma = np.exp(-d1**2/2) / (S * sigma * np.sqrt(2 * np.pi * T))
        gamma_exposure = gamma * options['openInterest'] * 100 * S * 0.01
        return gamma_exposure.fillna(0)
    except Exception as e:
        print(f"Erro ao calcular gamma exposure: {e}")
        return pd.Series([0]*len(options))
# ========================= Funções Principais ===============================


# ===================================================================
# Função Range Index
# ===================================================================
def gerar_range_index_plotly(timeframe_label, ativos_visiveis=None):
    if not MT5_OK:
        return go.Figure()  # sem MT5 no servidor

    timeframe = timeframes_dict[timeframe_label]
    df_final = pd.DataFrame()
    ...


    cores_ativos = {'USDX': 'white'}
    cores_ativos.update({
        'EURX': 'red', 'GBPX': 'fuchsia',
        'JPYX': 'green', 'CADX': 'yellow',
        'SEKX': 'aqua', 'CHFX': 'purple'
    })

    rates_usdx = mt5.copy_rates_from_pos('USDX', timeframe, 0, N_BARRAS)
    if rates_usdx is not None and len(rates_usdx) > 0:
        df_usdx = pd.DataFrame(rates_usdx)[['time', 'close']]
        df_usdx['time'] = pd.to_datetime(df_usdx['time'], unit='s')
        df_usdx = df_usdx.rename(columns={'close': 'USDX'})
        df_final = df_usdx

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
                 [0.5, 0.3, 0.2])
    }

    def calcular_index_sintetico(pares, pesos):
        df_index = pd.DataFrame()
        for par in pares:
            rates = mt5.copy_rates_from_pos(par, timeframe, 0, N_BARRAS)
            if rates is not None and len(rates) > 0:
                df = pd.DataFrame(rates)[['time', 'close']]
                df['time'] = pd.to_datetime(df['time'], unit='s')
                df = df.rename(columns={'close': par})
                df_index = df if df_index.empty else pd.merge(df_index, df, on='time', how='outer')
        if df_index.empty:
            return None
        df_index = df_index.sort_values('time').ffill().bfill()
        index_val = np.prod(
            [df_index[par] ** peso for par, peso in zip(pares, pesos) if par in df_index.columns],
            axis=0
        )
        df_index['Index'] = index_val
        return df_index[['time', 'Index']]

    for nome, (pares, pesos) in indices_sinteticos.items():
        df_idx = calcular_index_sintetico(pares, pesos)
        if df_idx is not None:
            df_idx = df_idx.rename(columns={'Index': nome})
            df_final = df_idx if df_final.empty else pd.merge(df_final, df_idx, on='time', how='outer')

    if df_final.empty:
        return go.Figure()

    df_final = df_final.sort_values('time').ffill().bfill()

    ativos_cols = [col for col in df_final.columns if col != 'time']
    for ativo in ativos_cols:
        df_final[ativo] = 100 + 100 * np.log(df_final[ativo] / df_final[ativo].iloc[0])

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
    ativos_visiveis = ativos_visiveis or ativos_cols

    for ativo in ativos_visiveis:
        if ativo not in df_final.columns:
            continue
        fig.add_trace(go.Scatter(
            x=df_final['time'], y=df_final[ativo],
            mode='lines', name=ativo,
            line=dict(color=cores_ativos.get(ativo, 'gray')),
            showlegend=False
        ))
        fig.add_trace(go.Scatter(
            x=[df_final['time'].iloc[-1]], y=[df_final[ativo].iloc[-1]],
            mode='text', text=[ativo],
            textposition='middle right', showlegend=False,
            textfont=dict(size=12, color=cores_ativos.get(ativo, 'white'))
        ))

    fig.add_trace(go.Scatter(x=df_final['time'], y=df_final['MediaGrupo_smooth'],
                             mode='lines', line=dict(color='silver', width=2, dash='dash'),
                             name='MediaGrupo', showlegend=False))
    fig.add_trace(go.Scatter(x=df_final['time'], y=df_final['FaixaAltaSuave_smooth'],
                             mode='lines', line=dict(color='gray', width=2),
                             name='FaixaAlta', showlegend=False))
    fig.add_trace(go.Scatter(x=df_final['time'], y=df_final['FaixaBaixaSuave_smooth'],
                             mode='lines', line=dict(color='gray', width=2),
                             name='FaixaBaixa', showlegend=False))
    fig.add_trace(go.Scatter(x=df_final['time'], y=df_final['FaixaAltaInternaSuave2_smooth'],
                             mode='lines', line=dict(color='gray', dash='dash'),
                             name='FaixaAltaIn', showlegend=False))
    fig.add_trace(go.Scatter(x=df_final['time'], y=df_final['FaixaBaixaInternaSuave2_smooth'],
                             mode='lines', line=dict(color='gray', dash='dash'),
                             name='FaixaBaixaIn', showlegend=False))

    today_str = datetime.now().strftime("%Y-%m-%d")
    fig.add_shape(type="line", x0=today_str, x1=today_str, y0=0, y1=1,
                  xref='x', yref='paper',
                  line=dict(color="yellow", width=0.5, dash="dash"))
    fig.add_annotation(x=today_str, y=1.1, xref="x", yref="paper",
                       text=datetime.now().strftime("%d/%m/%Y"),
                       showarrow=False, font=dict(color="yellow", size=8))

    fig.update_layout(
        title=dict(
            text=f"Index ({timeframe_label})",
            font=dict(size=10, color='white'),
            x=0.5,       # centralizado no eixo x
            y=0.95,       # sobe no eixo y (valor entre 0 e 1 — quanto maior, mais para cima)
            xanchor='center'
        ),
        plot_bgcolor='black',
        paper_bgcolor='black',
        font=dict(color='white'),
        xaxis=dict(showgrid=True, gridcolor='#1c1c1c'),
        yaxis=dict(showgrid=True, gridcolor='#1c1c1c'),
        margin=dict(
            l=30,  # l = left (esquerda)
            r=20,  # r = right (direita)
            t=40,  # t = top (topo)
            b=30   # b = bottom (base)
            ),
        showlegend=False
        
    )


    return fig



#===================================================================
# Função Range Comparativo (com suavizacao EMA)
#===================================================================
def gerar_range_comparativo_plotly(timeframe_label, ativos_visiveis=None):
    if not MT5_OK:
        return go.Figure()  # servidor sem MT5 → evita NameError

    timeframe = timeframes_dict[timeframe_label]
    df_final = pd.DataFrame()


    for ativo, base in ativos_comp.items():
        rates = mt5.copy_rates_from_pos(ativo, timeframe, 0, N_BARRAS)
        if rates is not None and len(rates) > 0:
            df = pd.DataFrame(rates)[['time', 'close']]
            df['time'] = pd.to_datetime(df['time'], unit='s')
            df['norm'] = 100 + 100 * np.log(df['close'] / base)
            df = df[['time', 'norm']].rename(columns={'norm': ativo})
            df_final = df if df_final.empty else pd.merge(df_final, df, on='time', how='outer')

    if df_final.empty:
        return go.Figure()

    df_final = df_final.sort_values('time').ffill().bfill()
    ativos_cols = list(ativos_comp.keys())

    # ----- métricas de grupo -----
    df_final['MediaGrupo'] = df_final[ativos_cols].mean(axis=1)
    df_final['FaixaAltaSuave'] = df_final['MediaGrupo'] + 10
    df_final['FaixaBaixaSuave'] = df_final['MediaGrupo'] - 10
    df_final['FaixaAltaInternaSuave2'] = df_final['MediaGrupo'] + 9
    df_final['FaixaBaixaInternaSuave2'] = df_final['MediaGrupo'] - 9

    # ----- suavização -----
    span = 100
    for col in ['MediaGrupo', 'FaixaAltaSuave', 'FaixaBaixaSuave',
                'FaixaAltaInternaSuave2', 'FaixaBaixaInternaSuave2']:
        df_final[f'{col}_smooth'] = df_final[col].ewm(span=span).mean()

    # ----- figura -----
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
            mode='text', text=[ativo],
            textposition='middle right', showlegend=False,
            textfont=dict(color=cores_ativos.get(ativo, 'white'), size=12)
        ))

    # ----- linhas de grupo -----
    fig.add_trace(go.Scatter(x=df_final['time'], y=df_final['MediaGrupo_smooth'],
                             mode='lines', line=dict(color='silver', width=2, dash='dash'),
                             name='MediaGrupo', showlegend=False))
    fig.add_trace(go.Scatter(x=df_final['time'], y=df_final['FaixaAltaSuave_smooth'],
                             mode='lines', line=dict(color='gray', width=2),
                             name='FaixaAlta', showlegend=False))
    fig.add_trace(go.Scatter(x=df_final['time'], y=df_final['FaixaBaixaSuave_smooth'],
                             mode='lines', line=dict(color='gray', width=2),
                             name='FaixaBaixa', showlegend=False))
    fig.add_trace(go.Scatter(x=df_final['time'], y=df_final['FaixaAltaInternaSuave2_smooth'],
                             mode='lines', line=dict(color='gray', dash='dash'),
                             name='FaixaAltaIn', showlegend=False))
    fig.add_trace(go.Scatter(x=df_final['time'], y=df_final['FaixaBaixaInternaSuave2_smooth'],
                             mode='lines', line=dict(color='gray', dash='dash'),
                             name='FaixaBaixaIn', showlegend=False))

    # ----- marcador de hoje -----
    today_str = datetime.now().strftime("%Y-%m-%d")
    fig.add_shape(type="line", x0=today_str, x1=today_str, y0=0, y1=1,
                  xref='x', yref='paper',
                  line=dict(color="yellow", width=0.5, dash="dash"))
    fig.add_annotation(x=today_str, y=1.1, xref="x", yref="paper",
                       text=datetime.now().strftime("%d/%m/%Y"),
                       showarrow=False, font=dict(color="yellow", size=8))

    fig.update_layout(
        title=dict(
            text=f"Pares({timeframe_label})",
            font=dict(size=10, color='white'),
            x=0.5,
            xanchor='center'
        ),
        plot_bgcolor='black',
        paper_bgcolor='black',
        font=dict(color='white'),

        xaxis=dict(
            showgrid=True,
            gridcolor='#1c1c1c',
            zeroline=False,
            tickcolor='white'
        ),
        yaxis=dict(
            title='',  # ← Removido para ganhar espaço horizontal
            showgrid=True,
            gridcolor='#1c1c1c',
            zeroline=False,
            tickcolor='white'
        ),
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="top",
            y=-0.2,
            xanchor="center",
            x=0.5
        ),
        margin=dict(
            l=30,  # l = left (esquerda)
            r=20,  # r = right (direita)
            t=40,  # t = top (topo)
            b=17   # b = bottom (base)
        ))   
  
    return fig



# =============================================== 
# Função calcular Hotelling T2
# ===============================================

def calcular_hotelling_t2(_, timeframe):
    pares_fx = ['EURUSD.r', 'GBPUSD.r', 'USDJPY.r', 'USDCAD.r', 'USDCHF.r', 'USDSEK.r']
    df_final = pd.DataFrame()

    for ativo in pares_fx:
        rates = mt5.copy_rates_from_pos(ativo, timeframe, 0, N_BARRAS)
        if rates is not None and len(rates) > 0:
            df = pd.DataFrame(rates)[['time', 'close']]
            df['time'] = pd.to_datetime(df['time'], unit='s')
            df = df.rename(columns={'close': ativo})
            df_final = df if df_final.empty else pd.merge(df_final, df, on='time', how='outer')
        else:
            print(f"Aviso: {ativo} sem dados.")

    if df_final.empty:
        print("Nenhum dado disponível.")
        return go.Figure()

    df_final = df_final.sort_values('time').ffill().bfill()

    for ativo in pares_fx:
        df_final[f"{ativo}_ret"] = np.log(df_final[ativo] / df_final[ativo].shift(1))

    df_ret = df_final[[f"{ativo}_ret" for ativo in pares_fx]].dropna()

    means = df_ret.mean()
    stds = df_ret.std()

    zscores = (df_ret - means) / stds
    T2 = (zscores ** 2).sum(axis=1)

    df_plot = df_final.loc[df_ret.index].copy()
    df_plot['T2'] = T2

    idx_red = df_plot[df_plot['T2'] > 30].index

    fig = go.Figure()

    # Curva branca
    fig.add_trace(go.Scatter(
        x=df_plot['time'], y=df_plot['T2'],
        mode='lines',
        line=dict(color='rgb(200, 200, 200)', width=1),
        name='Curva T2',
        showlegend=False
    ))

    # Bolinhas vermelhas
    fig.add_trace(go.Scatter(
        x=df_plot.loc[idx_red, 'time'], y=df_plot.loc[idx_red, 'T2'],
        mode='markers',
        marker=dict(color='red', size=3),
        name='Ponto Crítico',
        hovertemplate="<b>T2</b><br>Data: %{x|%d/%m/%Y}<br>Valor: %{y:.2f}",
        showlegend=False
    ))

    # Linha vertical amarela (hoje)
    today_str_iso = datetime.now().strftime("%Y-%m-%d")
    fig.add_shape(type="line", x0=today_str_iso, x1=today_str_iso, y0=0, y1=1,
                  xref='x', yref='paper',
                  line=dict(color="yellow", width=0.5, dash="dash"))
    fig.add_annotation(x=today_str_iso, y=1.10, xref='x', yref='paper',
                       text=datetime.now().strftime("%d/%m/%Y"),
                       showarrow=False, font=dict(color="yellow", size=8))

    # Linha do nível 30
    fig.add_shape(
        type="line",
        x0=df_plot['time'].min(), x1=df_plot['time'].max(),
        y0=30, y1=30,
        line=dict(color="green", width=0.7, dash="dashdot"),
        name="Nível 30"
    )

    fig.update_layout(
        title=dict(
            text="Ponto de Interesse",
            font=dict(size=10, color='white'),
            x=0.5,
            xanchor='center'
        ),
        plot_bgcolor='black',
        paper_bgcolor='black',
        font=dict(color='white'),
        xaxis=dict(
            showgrid=True,
            gridcolor='#1c1c1c',
            tickformat='%b %Y',
            tickangle=0
        ),
        yaxis=dict(
            showgrid=True,
            gridcolor='#1c1c1c'
        ),
        margin=dict(l=30, r=20, t=40, b=17),
        showlegend=False
    )

    return fig



#=============================================== 
# Função calcular VIX
#=============================================== 
def gerar_grafico_vix(timeframe, divisor_macro):
    df_vix = coletar("VIX", timeframe)
    df_sp = coletar("US500", timeframe)
    df_usdx = coletar("USDX", timeframe)
    df_tlt = coletar("TLT", timeframe)

    if any(df is None or df.empty for df in [df_vix, df_sp, df_usdx, df_tlt]):
        return go.Figure()

    df = df_vix.copy()
    df['media'] = df['close'].ewm(span=742).mean()
    df['z'] = (df['close'] - df['media']) / df['close'].rolling(200).std()

    df_sp['ret'] = np.log(df_sp['close']).diff()
    df_sp['vol'] = df_sp['ret'].rolling(20).std() * np.sqrt(252) * 100
    df['vol_realizada'] = df_sp['vol'] / divisor_macro

    fig = go.Figure()

    # VIX (preço)
    fig.add_trace(go.Scatter(
        x=df['time'], y=df['close'],
        mode='lines',
        name='VIX (Preço)',
        line=dict(color='orange'),
        showlegend=False
    ))

    # Média Exponencial
    fig.add_trace(go.Scatter(
        x=df['time'], y=df['media'],
        mode='lines',
        name='Média Exponencial (742)',
        line=dict(color='white', dash='dash'),
        showlegend=False
    ))

    # Vol. Realizada
    fig.add_trace(go.Bar(
        x=df['time'], y=df['vol_realizada'],
        name='Vol. Realizada (S&P 500) ×10',
        marker_color='tan',
        showlegend=False
    ))

    # Z > 3
    df['marker'] = np.where(df['z'] > 3.5, 'acima_3.5', np.where(df['z'] > 3, 'acima_3', ''))
    df_z3 = df[df['marker'] == 'acima_3']
    df_z35 = df[df['marker'] == 'acima_3.5']

    fig.add_trace(go.Scatter(
        x=df_z3['time'], y=df_z3['close'],
        mode='markers',
        name='Z > 3',
        marker=dict(size=3, color='yellow'),
        showlegend=False
    ))

    fig.add_trace(go.Scatter(
        x=df_z35['time'], y=df_z35['close'],
        mode='markers',
        name='Z > 3.5',
        marker=dict(size=3, color='red'),
        showlegend=False
    ))

    # USDX
    fig.add_trace(go.Scatter(
        x=df_usdx['time'], y=df_usdx['close'] / divisor_macro,
        mode='lines',
        name='DXY',
        line=dict(color='lime')
    ))

    # TLT
    fig.add_trace(go.Scatter(
        x=df_tlt['time'], y=df_tlt['close'] / divisor_macro,
        mode='lines',
        name='TLT',
        line=dict(color='magenta')
    ))

    # Linha vertical do dia atual
    today_str = datetime.now().strftime("%Y-%m-%d")
    fig.add_shape(
        type="line",
        x0=today_str, x1=today_str,
        y0=0, y1=1,
        xref='x', yref='paper',
        line=dict(color="yellow", width=0.5, dash="dash")
    )

    # Data no topo
    fig.add_annotation(
        x=today_str,
        y=1.03,
        xref="x",
        yref="paper",
        text=datetime.now().strftime("%d/%m/%Y"),
        showarrow=False,
        font=dict(color="yellow", size=10)
    )

    # Layout geral
    fig.update_layout(
        title=dict(
            text='VIX - Contexto Macro',
            font=dict(size=10, color='white'),
            x=0.5,
            xanchor='center'
        ),
        plot_bgcolor='black',
        paper_bgcolor='black',
        font=dict(color='white'),
        xaxis=dict(showgrid=True, gridcolor='#1c1c1c'),
        yaxis=dict(showgrid=True, gridcolor='#1c1c1c'),
        margin=dict(
            l=30,  # l = left (esquerda)
            r=20,  # r = right (direita)
            t=40,  # t = top (topo)
            b=17   # b = bottom (base)
            ),
        showlegend=False
        )


    return fig






# ===================== LAYOUT =====================
app = dash.Dash(__name__)
server = app.server

app.layout = html.Div(style={"backgroundColor": "black", "padding": "10px"}, children=[

    # ========= TOPO =========
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

    # ========= CONTROLES =========
    html.Div([
        html.Button("Atualizar", id="refresh-button", style={"marginRight": "10px"}),
        dcc.Dropdown(
            id="timeframe-dropdown",
            options=[{"label": k, "value": k} for k in timeframes_dict],
            value="D1", clearable=False,
            style={"width": "100px", "display": "inline-block"}
        ),
        html.Div(id="last-update", style={"color": "gray", "marginLeft": "20px", "display": "inline-block"})
    ]),

    # ========= GRÁFICOS MACRO =========
    html.Div([

        # INDEX
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

        # COMPARATIVO
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

        # T²
        html.Div([
            html.Div(' ', style={'height': '27px'}),
            dcc.Graph(id='graph-t2', style={'height': '38vh', 'width': '100%'})
        ], style={'width': '25%', 'padding': '5px', 'display': 'flex', 'flexDirection': 'column', 'alignItems': 'center'}),

        # VIX
        html.Div([
            html.Div([
                dcc.Dropdown(
                    id='divisor-dropdown',
                    options=[{'label': '3', 'value': 3}, {'label': '4', 'value': 4}],
                    value=3, clearable=False,
                    style={
                        'width': '50px', 'height': '20px', 'borderRadius': '0px',
                        'fontSize': '12px', 'padding': '0px', 'textAlign': 'center',
                        'marginBottom': '2px', 'float': 'right'
                    }
                )
            ], style={'width': '100%', 'display': 'inline-block', 'textAlign': 'right'}),
            dcc.Graph(id='grafico-vix', style={'height': '38vh', 'width': '100%'})
        ], style={'width': '25%', 'padding': '5px', 'display': 'flex', 'flexDirection': 'column', 'alignItems': 'center'})

    ], style={'display': 'flex'}),

    html.Hr(style={"margin": "20px 0", "borderColor": "#555"}),

    # ========= GAMMA EXPOSURE =========
    html.Div(id='gamma-charts-container', style={
        'display': 'flex',
        'flexDirection': 'column',
        'alignItems': 'center',
        'gap': '20px',
        'padding': '10px 0'
    }),

    html.Hr(style={"margin": "20px 0", "borderColor": "#555"}),

    # ========= MINI CHARTS CRYPTOS =========
    html.Div(id='crypto-charts-container', style={
        'display': 'flex',
        'flexWrap': 'wrap',
        'justifyContent': 'center',
        'gap': '10px',
        'padding': '10px 0'
    }),

    # ========= RODAPÉ =========
    html.Div(id="contador", style={"color": "gray", "marginTop": "20px"}),

    dcc.Interval(id="interval-refresh", interval=interval_ms, n_intervals=0),
    dcc.Interval(id="interval-1s", interval=1000, n_intervals=0),
    dcc.Interval(id='interval-component', interval=300*1000, n_intervals=0)
])




# ===================== CALLBACKS UNIFICADOS =====================
# 🔁 Atualiza os mini-gráficos de cripto + ouro
@app.callback(
    [Output("mini-graficos", "children"), Output("mensagem-erro", "children")],
    Input("atualizacao-temporizada", "n_intervals")
)
def atualizar_mini_graficos(n):
    graficos = []
    erros = []

    for cripto in cryptos:
        df = obter_dados_cripto(cripto)
        if df.empty:
            erros.append("XAU" if cripto == "GC=F" else cripto.replace("-USD", "").replace("=F", "").replace("X", ""))
            continue

        nome = "XAU" if cripto == "GC=F" else cripto.replace("-USD", "").replace("=F", "").replace("X", "")
        cor_base = "#FFD700" if cripto == "GC=F" else "white"

        fig = make_subplots(specs=[[{"secondary_y": True}]])
        fig.add_trace(go.Scatter(x=df['Date'], y=df['B2'], mode='lines',
                                 line=dict(color=cor_base, width=1, dash='dot'),
                                 name=nome, showlegend=False, connectgaps=True),
                      secondary_y=False)

        df_sinal = df[df['B2'] < 0]
        fig.add_trace(go.Scatter(x=df_sinal['Date'], y=df_sinal['B2'], mode='markers',
                                 marker=dict(color='lime', size=8, symbol='triangle-up'),
                                 name="sinal", showlegend=False),
                      secondary_y=False)

        fig.add_trace(go.Scatter(x=df['Date'], y=df['Close'], mode='lines',
                                 line=dict(color='yellow', width=1), name='Preço', showlegend=False),
                      secondary_y=True)

        data_atual = datetime.today().replace(hour=0, minute=0, second=0, microsecond=0)
        fig.add_vline(x=data_atual, line_width=1, line_dash="dash", line_color="#FFFACD")
        fig.add_annotation(x=data_atual, y=0, xref="x", yref="paper",
                           text=data_atual.strftime("%Y-%m-%d"), showarrow=False,
                           font=dict(color="#FFFACD", size=8), align="center", yanchor="top")

        fig.update_layout(
            title=nome, title_x=0.5,
            height=120, width=200,
            margin=dict(l=5, r=5, t=25, b=20),
            plot_bgcolor="black", paper_bgcolor="black",
            font=dict(color="white", size=8),
            xaxis=dict(showticklabels=False, showgrid=False, zeroline=False),
            yaxis=dict(showticklabels=False, showgrid=False, zeroline=False),
            yaxis2=dict(showticklabels=False, showgrid=False, zeroline=False)
        )

        graficos.append(html.Div(dcc.Graph(figure=fig),
                                 style={"display": "inline-block", "marginRight": "10px",
                                        "border": "1px solid #444", "borderRadius": "10px",
                                        "padding": "5px", "backgroundColor": "#111"}))

    mensagem_erro = "Falha ao carregar: " + ", ".join(erros) if erros else ""
    print(f"[{datetime.now()}] Atualizando gráficos de Gamma Exposure")

    return graficos, mensagem_erro





# 🔁 Atualiza os 4 gráficos principais
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
def atualizar_todos(n_clicks, n_intervals, index_selecionados, comparativo_selecionados, timeframe_label, divisor_macro):
    fig_index = gerar_range_index_plotly(timeframe_label, index_selecionados)
    fig_comp = gerar_range_comparativo_plotly(timeframe_label, comparativo_selecionados)
    fig_t2 = calcular_hotelling_t2(None, timeframes_dict[timeframe_label])
    fig_vix = gerar_grafico_vix(timeframes_dict[timeframe_label], divisor_macro)
    now = datetime.now().strftime("Última atualização: %d/%m/%Y %H:%M:%S")
    return fig_index, fig_comp, fig_t2, fig_vix, now





# 🔁 Atualiza o contador de tempo regressivo
@app.callback(
    Output('contador', 'children'),
    Input('interval-1s', 'n_intervals')
)
def atualizar_contador(n):
    tempo_restante = interval_sec - (n % interval_sec)
    return f'Atualização automática em: {tempo_restante} segundos'





# 🔁 Atualiza os gráficos de Gamma Exposure SPY, QQQ, IWM
@app.callback(
    Output('gamma-charts-container', 'children'),
    Input('interval-component', 'n_intervals')
)


def update_dashboard(n):
    columns = []
    for ativo in ativos:
        calls, puts, spot, exp_date = get_options_data(ativo)
        print(f"[{datetime.now()}] Atualizando Gamma Exposure para {ativo}")
        last_update = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

        if calls.empty or puts.empty or np.isnan(spot):
            fig_strike = go.Figure()
            fig_profile = go.Figure()
        else:
            calls_gamma = gamma_exposure(calls, spot)
            puts_gamma = gamma_exposure(puts, spot)

            strikes_calls = calls['strike']
            strikes_puts = puts['strike']

            call_wall = calls.iloc[np.argmax(calls_gamma)]['strike'] if not calls_gamma.empty else spot
            put_wall = puts.iloc[np.argmax(puts_gamma)]['strike'] if not puts_gamma.empty else spot

            regime = 'Risk-On' if spot > call_wall else 'Risk-Off' if spot < put_wall else 'Neutra'

            # 🎯 Strike Chart
            fig_strike = go.Figure()
            fig_strike.add_trace(go.Bar(x=strikes_calls, y=calls_gamma, name='Call Gamma', marker_color='blue'))
            fig_strike.add_trace(go.Bar(x=strikes_puts, y=-puts_gamma, name='Put Gamma', marker_color='red'))
            fig_strike.add_vline(x=call_wall, line_color='green', line_dash='dot')
            fig_strike.add_vline(x=put_wall, line_color='red', line_dash='dot')
            fig_strike.add_vline(x=spot, line_color='yellow')
            fig_strike.add_annotation(text=f'Call Wall: {call_wall}', x=call_wall, y=1.15, xref='x', yref='paper', showarrow=False, font=dict(color='green', size=11))
            fig_strike.add_annotation(text=f'Put Wall: {put_wall}', x=put_wall, y=1.10, xref='x', yref='paper', showarrow=False, font=dict(color='red', size=11))
            fig_strike.add_annotation(text=f'Spot: {spot:.2f}', x=spot, y=1.00, xref='x', yref='paper', showarrow=False, font=dict(color='yellow', size=11))
            fig_strike.update_layout(template='plotly_dark', height=400, title=f'{ativo} Gamma Exposure Strike', xaxis_title='Strike', yaxis_title='Gamma Exposure')

            # 📈 Profile Chart
            price_range = np.linspace(spot * 0.90, spot * 1.20, 100)
            gamma_profile = [gamma_exposure(calls, p).sum() - gamma_exposure(puts, p).sum() for p in price_range]
            gamma_flip_idx = np.where(np.diff(np.sign(gamma_profile)))[0]
            gamma_flip = price_range[gamma_flip_idx[0]] if len(gamma_flip_idx) > 0 else spot

            fig_profile = go.Figure()
            fig_profile.add_trace(go.Scatter(x=price_range, y=gamma_profile, mode='lines', name='Gamma Exposure'))
            fig_profile.add_shape(type="rect", x0=spot - 2, x1=spot + 2, y0=min(gamma_profile), y1=max(gamma_profile), fillcolor='rgba(0,0,255,0.1)', line_width=0, layer="below")
            if not np.isnan(gamma_flip):
                fig_profile.add_vline(x=gamma_flip, line_color='red', line_dash='dot')

            fig_profile.add_vline(x=call_wall, line_color='green', line_dash='dot')
            fig_profile.add_vline(x=put_wall, line_color='blue', line_dash='dot')
            fig_profile.add_vline(x=spot, line_color='yellow')
            fig_profile.update_layout(template='plotly_dark', height=400, title=f'{ativo} Gamma Exposure Profile', xaxis_title='Preço Simulado', yaxis_title='Gamma Exposure')

        # Junta os dois gráficos do ativo (Strike + Profile)
        column = html.Div(style={'display': 'flex', 'flexDirection': 'row', 'padding': '10px', 'width': '100%', 'justifyContent': 'space-between', 'alignItems': 'flex-end'}, children=[
            html.Div(style={'width': '48%', 'padding': '10px'}, children=[
                html.H4(ativo, style={'textAlign': 'center'}),
                dcc.Graph(figure=fig_strike, config={'displayModeBar': False})
            ]),
            html.Div(style={'width': '48%', 'padding': '10px'}, children=[
                dcc.Graph(figure=fig_profile, config={'displayModeBar': False})
            ])
        ])
        columns.append(column)

    return columns


# ===================== EXECUÇÃO DO APP =====================
import os

if __name__ == "__main__":
    app.run_server(
        host="0.0.0.0",
        port=int(os.environ.get("PORT", 10000)),
        debug=False
    )





































































