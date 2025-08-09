#CROSSCORRALATION


import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from dash import Dash, dcc, html, Input, Output, State, ctx

# ================== ATIVOS ORGANIZADOS POR FAMÍLIA ==================
ativos_familia = {
    'Equities': {
        'SPY': 'S&P 500', 'QQQ': 'Nasdaq 100', 'IWM': 'Russell 2000',
        'DIA': 'Dow Jones', 'XLF': 'Financeiro', 'XLE': 'Energia', 'XLK': 'Tecnologia'
    },
    'Currencies': {
        'EURUSD=X': 'EUR/USD', 'USDBRL=X': 'USD/BRL', 'USDJPY=X': 'USD/JPY',
        'GBPUSD=X': 'GBP/USD', 'USDCAD=X': 'USD/CAD', 'AUDUSD=X': 'AUD/USD'
    },
    'Commodities': {
        'GLD': 'Ouro (GLD)', 'SLV': 'Prata (SLV)', 'USO': 'Petróleo (USO)',
        'CL=F': 'Petróleo WTI (CL=F)', 'BZ=F': 'Petróleo Brent (BZ=F)',
        'UNG': 'Gás Natural (UNG)', 'ZS=F': 'Soja', 'ZC=F': 'Milho', 'ZW=F': 'Trigo'
    },
    'Rates/Credit': {
        'IEF': 'Treasury 10Y', 'LQD': 'IG Bonds', 'HYG': 'High Yield', 'TLT': 'Long Bonds'
    },
    'Crypto': {
        'BTC-USD': 'Bitcoin', 'ETH-USD': 'Ethereum', 'BNB-USD': 'BNB',
        'SOL-USD': 'Solana', 'ADA-USD': 'Cardano', 'XRP-USD': 'XRP (Ripple)',
        'DOGE-USD': 'Dogecoin', 'AVAX-USD': 'Avalanche', 'DOT-USD': 'Polkadot',
        'MATIC-USD': 'Polygon', 'LTC-USD': 'Litecoin', 'TRX-USD': 'TRON',
        'LINK-USD': 'Chainlink', 'UNI-USD': 'Uniswap', 'ATOM-USD': 'Cosmos'
    }
}

# ================== DASH APP ==================
app = Dash(__name__)
server = app.server

def gerar_checklist_familia(familia, ativos_dict):
    return html.Div(style={'minWidth': '200px'}, children=[
        html.H4(familia, style={'color': 'white', 'marginBottom': '5px'}),
        dcc.Checklist(
            id=f'checklist-{familia}',
            options=[{'label': nome, 'value': ticker} for ticker, nome in ativos_dict.items()],
            value=[],
            labelStyle={'display': 'block', 'color': 'white'},
            inputStyle={'marginRight': '10px'}
        )
    ])

app.layout = html.Div(style={'backgroundColor': 'black', 'padding': '20px'}, children=[
    html.H2("📊 Matriz de Correlação - Seleção Global", style={'color': 'white'}),

    html.Details([
        html.Summary("Selecionar Ativos por Família", style={
            'color': 'white', 'fontSize': '16px', 'cursor': 'pointer',
            'backgroundColor': '#333333', 'padding': '10px', 'borderRadius': '6px',
            'marginBottom': '10px'
        }),
        html.Div([
            html.Button("Selecionar Todos os Ativos", id='selecionar-todos', n_clicks=0,
                        style={'marginRight': '10px'}),
            html.Button("Desmarcar Todos os Ativos", id='desmarcar-todos', n_clicks=0)
        ], style={'margin': '10px 0'}),
        html.Div([
            html.Div(
                [gerar_checklist_familia(fam, ativos) for fam, ativos in ativos_familia.items()],
                style={'display': 'flex', 'flexWrap': 'wrap', 'gap': '30px', 'padding': '10px'}
            )
        ], style={'backgroundColor': '#1a1a1a', 'borderRadius': '6px'})
    ], open=False, style={'marginBottom': '30px'}),

    html.Button('Atualizar Gráfico', id='atualizar-btn', n_clicks=0,
                style={'marginTop': '10px', 'marginBottom': '30px'}),

    html.Details([
        html.Summary("🔍 Visualizar Matriz de Correlação", style={
            'color': 'white', 'fontSize': '16px', 'cursor': 'pointer',
            'backgroundColor': '#333333', 'padding': '10px', 'borderRadius': '6px',
            'marginBottom': '10px'
        }),
        dcc.Graph(id='grafico-correlacao')
    ], open=False, style={'marginBottom': '30px'}),

    html.Details([
        html.Summary("📈 Visualizar Gráfico de Range", style={
            'color': 'white', 'fontSize': '16px', 'cursor': 'pointer',
            'backgroundColor': '#333333', 'padding': '10px', 'borderRadius': '6px',
            'marginBottom': '10px'
        }),
        html.Button('Atualizar Range', id='atualizar-range-btn', n_clicks=0,
                    style={'margin': '10px 0'}),
        dcc.Graph(id='grafico-range')
    ], open=False, style={'marginBottom': '30px'})
])

# ================== CALLBACK: SELECIONAR / DESMARCAR ==================
@app.callback(
    [Output(f'checklist-{fam}', 'value') for fam in ativos_familia],
    Input('selecionar-todos', 'n_clicks'),
    Input('desmarcar-todos', 'n_clicks'),
    prevent_initial_call=True
)
def selecionar_ou_limpar_todos(n_select, n_clear):
    triggered_id = ctx.triggered_id
    if triggered_id == 'selecionar-todos':
        return [list(ativos.keys()) for ativos in ativos_familia.values()]
    elif triggered_id == 'desmarcar-todos':
        return [[] for _ in ativos_familia]
    return dash.no_update

# ================== CALLBACK: MATRIZ DE CORRELAÇÃO ==================
@app.callback(
    Output('grafico-correlacao', 'figure'),
    Input('atualizar-btn', 'n_clicks'),
    [State(f'checklist-{fam}', 'value') for fam in ativos_familia]
)
def atualizar_matriz_corr(n_clicks, *selecoes):
    ativos_selecionados = [ticker for sublist in selecoes for ticker in sublist]

    if len(ativos_selecionados) < 2:
        return go.Figure().update_layout(
            title='Selecione pelo menos dois ativos',
            paper_bgcolor='black', plot_bgcolor='black',
            font=dict(color='white')
        )

    df_final = pd.DataFrame()
    for ticker in ativos_selecionados:
        df = yf.download(ticker, period='6mo', interval='1d', progress=False)['Close']
        if df.empty:
            continue
        df = 100 + 100 * np.log(df / df.iloc[0])
        df_final[ticker] = df

    df_final = df_final.dropna()
    corr = df_final.corr()
    z = corr.values.copy()
    text = np.round(z, 2).astype(str)

    mask_diag = np.eye(len(z), dtype=bool)
    z[mask_diag] = None
    text[mask_diag] = ''

    mask_relevante = (z > 0.7) | (z < -0.7)
    z_exibir = np.where(mask_relevante, z, np.nan)
    text_exibir = np.where(mask_relevante, text, '')

    shapes = []
    for j, y_label in enumerate(corr.index):
        for i, x_label in enumerate(corr.columns):
            if not mask_diag[j, i]:
                shapes.append(dict(
                    type='rect', xref='x', yref='y',
                    x0=x_label, x1=x_label, y0=y_label, y1=y_label,
                    line=dict(color='white', width=1, dash='dot'),
                    fillcolor='rgba(0,0,0,0)')
                )

    fig = go.Figure(data=go.Heatmap(
        z=z_exibir,
        x=corr.columns,
        y=corr.index,
        text=text_exibir,
        texttemplate="%{text}",
        textfont={"size": 12, "color": "white"},
        colorscale='Viridis',
        zmin=-1, zmax=1,
        showscale=False,
        xgap=0, ygap=0
    ))

    fig.update_layout(
        title='Matriz de Correlação',
        paper_bgcolor='black',
        plot_bgcolor='black',
        font=dict(color='white'),
        shapes=shapes,
        xaxis=dict(showgrid=False),
        yaxis=dict(showgrid=False),
        margin=dict(l=40, r=40, t=40, b=40),
        height=600
    )

    return fig

# ================== CALLBACK: GRÁFICO DE RANGE ==================
@app.callback(
    Output('grafico-range', 'figure'),
    Input('atualizar-range-btn', 'n_clicks'),
    [State(f'checklist-{fam}', 'value') for fam in ativos_familia]
)
def atualizar_grafico_range(n_clicks, *selecoes):
    ativos_selecionados = [ticker for sublist in selecoes for ticker in sublist]

    if not ativos_selecionados:
        return go.Figure().update_layout(
            title='Nenhum ativo selecionado',
            paper_bgcolor='black',
            plot_bgcolor='black',
            font=dict(color='white')
        )

    df_final = pd.DataFrame()
    for ticker in ativos_selecionados:
        df = yf.download(ticker, period='6mo', interval='1d', progress=False)['Close']
        if df.empty:
            continue
        df = 100 + 100 * np.log(df / df.iloc[0])
        df_final[ticker] = df

    fig = go.Figure()
    for ticker in df_final.columns:
        fig.add_trace(go.Scatter(
            x=df_final.index, y=df_final[ticker],
            mode='lines', name=ticker
        ))

    fig.update_layout(
        title='Range Normalizado dos Ativos',
        paper_bgcolor='black',
        plot_bgcolor='black',
        font=dict(color='white'),
        xaxis_title='Data',
        yaxis_title='Preço Normalizado (log)',
        height=500
    )

    return fig

# ================== EXECUÇÃO ==================
if __name__ == '__main__':
    app.run(debug=True, port=8071)