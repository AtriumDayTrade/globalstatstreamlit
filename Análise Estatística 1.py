# ===================== Analise Estatística  =====================

import dash
from dash import dcc, html, Input, Output
import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from fitter import Fitter
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.stats import gaussian_kde, anderson, normaltest
from fitter import Fitter
from plotly.subplots import make_subplots
import plotly.graph_objects as go
from scipy.stats import pearsonr


app = dash.Dash(__name__)
app.title = "Análise Beta Alpha"

nativos = ['BTC-USD', 'ETH-USD', 'SPY', 'QQQ', 'GLD', 'XAUUSD=X']

PERIODOS = {
    "6M": "6mo", "YTD": "ytd", "1Y": "1y",
    "2Y": "2y", "5Y": "5y", "Max": "max"
}

FREQUENCIAS = {
    "Diária": "1d", "Semanal": "1wk", "Mensal": "1mo"
}

def sugerir_benchmark(ativo):
    if ativo.endswith('-USD') and ativo != 'BTC-USD':
        return 'BTC-USD'
    elif ativo == 'BTC-USD':
        return 'SPY'
    elif ativo.endswith('=X'):
        return 'DXY'
    else:
        return '^GSPC'

def baixar_serie(ticker, periodo, frequencia):
    df = yf.download(ticker, period=periodo, interval=frequencia, progress=False)
    if isinstance(df.columns, pd.MultiIndex):
        df = df.xs('Adj Close', axis=1, level=0) if 'Adj Close' in df.columns.get_level_values(0) else df.xs('Close', axis=1, level=0)
        return df.iloc[:, 0]
    else:
        return df['Adj Close'] if 'Adj Close' in df.columns else df['Close']

def analisar_lag(df, max_lag=5):
    """Retorna o lag com maior correlação entre ret_ativo e ret_bench"""
    corrs = {}
    for lag in range(-max_lag, max_lag + 1):
        if lag < 0:
            cor = df['ret_ativo'].shift(-lag).corr(df['ret_bench'])
        else:
            cor = df['ret_ativo'].corr(df['ret_bench'].shift(lag))
        corrs[lag] = cor

    melhor_lag = max(corrs, key=lambda x: abs(corrs[x]))
    melhor_corr = corrs[melhor_lag]

    interpretacao = (
        f"O ativo se antecipa ao benchmark em {melhor_lag} períodos."
        if melhor_lag > 0 else
        f"O ativo reage ao benchmark com atraso de {abs(melhor_lag)} períodos."
        if melhor_lag < 0 else
        "Sem defasagem observável entre ativo e benchmark."
    )

    return melhor_lag, melhor_corr, interpretacao

def analisar_lag(df, max_lag=10):
    """
    Testa correlações entre os retornos do ativo e benchmark em diferentes defasagens.
    Retorna o lag ótimo e a maior correlação encontrada.
    """
    melhor_corr = -np.inf
    melhor_lag = 0

    for lag in range(-max_lag, max_lag + 1):
        if lag < 0:
            corr = df['ret_ativo'][:lag].corr(df['ret_bench'][-lag:])
        elif lag > 0:
            corr = df['ret_ativo'][lag:].corr(df['ret_bench'][:-lag])
        else:
            corr = df['ret_ativo'].corr(df['ret_bench'])

        if pd.notna(corr) and abs(corr) > abs(melhor_corr):
            melhor_corr = corr
            melhor_lag = lag

    if melhor_lag > 0:
        interpretacao = f"O ativo tende a seguir o benchmark com {melhor_lag} período(s) de atraso."
    elif melhor_lag < 0:
        interpretacao = f"O ativo tende a antecipar o benchmark em {abs(melhor_lag)} período(s)."
    else:
        interpretacao = "Sem defasagem detectada (lag = 0)."

    return melhor_lag, melhor_corr, interpretacao





app.layout = html.Div(style={'backgroundColor': 'black', 'color': 'white', 'padding': '10px'}, children=[

    html.H2("Análise Beta / Alpha", style={
        'textAlign': 'center',
        'color': 'white',
        'marginBottom': '5px'
    }),

    # Barra superior com controles
    html.Div([
        html.Div([
            html.Label("Ativo:", style={'fontSize': '13px'}),
            dcc.Dropdown(
                id='ativo-dropdown',
                options=[{'label': i, 'value': i} for i in nativos],
                value='ETH-USD',
                style={'fontSize': '13px'}
            )
        ], style={'width': '20%', 'display': 'inline-block'}),

        html.Div([
            html.Label("Benchmark:", style={'fontSize': '13px'}),
            dcc.Dropdown(id='benchmark-dropdown', style={'fontSize': '13px'}),
            html.Div(id='sugestao-texto', style={'fontSize': '10px', 'color': 'gray'})
        ], style={'width': '20%', 'display': 'inline-block', 'paddingLeft': '10px'}),

        html.Div([
            html.Label("Período:", style={'fontSize': '13px'}),
            dcc.RadioItems(
                id='periodo-radio',
                options=[{'label': k, 'value': v} for k, v in PERIODOS.items()],
                value='2y',
                labelStyle={'display': 'inline-block', 'marginRight': '8px'},
                inputStyle={'marginRight': '4px'},
                style={'fontSize': '12px'}
            )
        ], style={'width': '30%', 'display': 'inline-block', 'paddingLeft': '20px'}),

        html.Div([
            html.Label("Frequência:", style={'fontSize': '13px'}),
            dcc.Dropdown(
                id='freq-dropdown',
                options=[{'label': k, 'value': v} for k, v in FREQUENCIAS.items()],
                value='1wk',
                style={'fontSize': '13px'}
            )
        ], style={'width': '20%', 'display': 'inline-block', 'paddingLeft': '10px'})
    ], style={'marginBottom': '5px'}),

    # Gráfico e estatísticas lado a lado
    html.Div([
        dcc.Graph(
            id='scatter-grafico',
            style={'width': '74%', 'height': '1000px', 'display': 'inline-block'}
        ),

        html.Div(id='metricas-beta', style={
            'width': '25%',
            'display': 'inline-block',
            'verticalAlign': 'top',
            'paddingLeft': '10px',
            'position': 'relative'
        })
    ])
])



@app.callback(
    [Output('benchmark-dropdown', 'options'),
     Output('benchmark-dropdown', 'value'),
     Output('sugestao-texto', 'children')],
    Input('ativo-dropdown', 'value')
)
def atualizar_benchmark_sugestao(ativo):
    sugestao = sugerir_benchmark(ativo)
    opcoes = [{'label': i, 'value': i} for i in nativos if i != ativo]
    return opcoes, sugestao, f"Sugestão de benchmark: {sugestao}"


@app.callback(
    [Output('scatter-grafico', 'figure'),
     Output('metricas-beta', 'children')],
    [Input('ativo-dropdown', 'value'),
     Input('benchmark-dropdown', 'value'),
     Input('periodo-radio', 'value'),
     Input('freq-dropdown', 'value')]
)
def gerar_regressao(ativo, benchmark, periodo, frequencia):
    if not ativo or not benchmark:
        return dash.no_update

    df1 = baixar_serie(ativo, periodo, frequencia)
    df2 = baixar_serie(benchmark, periodo, frequencia)

    df = pd.DataFrame({ativo: df1, benchmark: df2}).dropna()
    df['ret_ativo'] = df[ativo].pct_change()
    df['ret_bench'] = df[benchmark].pct_change()
    df = df.dropna()

    df['norm_ativo'] = df[ativo] / df[ativo].iloc[0] * 100
    df['norm_bench'] = df[benchmark] / df[benchmark].iloc[0] * 100

    X = df['ret_bench'].values.reshape(-1, 1)
    y = df['ret_ativo'].values
    reg = LinearRegression().fit(X, y)
    beta = reg.coef_[0]
    alpha = reg.intercept_
    r2 = reg.score(X, y)
    y_pred = reg.predict(X)
    corr = np.corrcoef(df['ret_bench'], df['ret_ativo'])[0, 1]

    _, p_value = normaltest(y)
    ad_test = anderson(y)
    ad_text = f"Normal (A² = {ad_test.statistic:.4f})"

    f = Fitter(y, distributions=['norm', 't', 'laplace', 'lognorm', 'expon', 'gamma', 'beta', 'gennorm'])
    f.fit()
    best_dist = list(f.get_best(method='sumsquare_error').keys())[0]

    kde_x = np.linspace(df['ret_bench'].min(), df['ret_bench'].max(), 200)
    kde_y = gaussian_kde(df['ret_bench'])(kde_x)
    kde_y_v = gaussian_kde(df['ret_ativo'])(kde_x)

    ci_95_x = np.percentile(df['ret_bench'], [2.5, 97.5])
    ci_95_y = np.percentile(df['ret_ativo'], [2.5, 97.5])

    # Lag ótimo
    melhor_lag, melhor_corr_lag, interpretacao = analisar_lag(df)

    fig = make_subplots(
        rows=4, cols=3,
        specs=[
            [None, None, None],
            [{"type": "histogram"}, {"type": "scatter"}, None],
            [None, {"type": "histogram"}, None],
            [None, {"type": "scatter"}, None]
        ],
        column_widths=[0.15, 0.85, 0.0],
        row_heights=[0.0, 0.9, 0.2, 0.5],
        vertical_spacing=0.04,
        horizontal_spacing=0.02
    )

    # Histogramas
    fig.add_trace(go.Histogram(y=df['ret_ativo'], nbinsy=30,
        marker=dict(color='orange', line=dict(color='white', width=1)), opacity=0.7, showlegend=False), row=2, col=1)
    fig.add_trace(go.Scatter(x=kde_y_v, y=kde_x, mode='lines',
        line=dict(color='cyan', width=2), showlegend=False), row=2, col=1)

    fig.add_trace(go.Histogram(x=df['ret_bench'], nbinsx=30,
        marker=dict(color='orange', line=dict(color='white', width=1)), opacity=0.7, showlegend=False), row=3, col=2)
    fig.add_trace(go.Scatter(x=kde_x, y=kde_y, mode='lines',
        line=dict(color='cyan', width=2), showlegend=False), row=3, col=2)

    # Intervalos
    fig.add_trace(go.Scatter(x=[ci_95_x[0]]*2, y=[0, max(kde_y)], mode='lines',
        line=dict(color='cyan', width=3, dash='dot'), showlegend=False), row=3, col=2)
    fig.add_trace(go.Scatter(x=[ci_95_x[1]]*2, y=[0, max(kde_y)], mode='lines',
        line=dict(color='cyan', width=3, dash='dot'), showlegend=False), row=3, col=2)
    fig.add_trace(go.Scatter(x=[0, max(kde_y_v)], y=[ci_95_y[0]]*2, mode='lines',
        line=dict(color='cyan', width=3, dash='dot'), showlegend=False), row=2, col=1)
    fig.add_trace(go.Scatter(x=[0, max(kde_y_v)], y=[ci_95_y[1]]*2, mode='lines',
        line=dict(color='cyan', width=3, dash='dot'), showlegend=False), row=2, col=1)

    # Dispersão + Regressão
    fig.add_trace(go.Scatter(x=df['ret_bench'], y=df['ret_ativo'], mode='markers',
        marker=dict(color='gold', size=6), name='Retornos'), row=2, col=2)
    fig.add_trace(go.Scatter(x=df['ret_bench'], y=y_pred, mode='lines',
        line=dict(color='red', width=4), name=f"Y = {beta:.3f} X + {alpha:.3f}"), row=2, col=2)

    # Linhas
    fig.add_trace(go.Scatter(x=df.index, y=df['norm_ativo'], mode='lines+text',
        name=ativo, text=[None]*(len(df)-1)+[f'<span style="color:yellow">{ativo}</span>'],
        textposition='middle right', line=dict(color='yellow', width=2, dash='dot'), showlegend=False), row=4, col=2)

    fig.add_trace(go.Scatter(x=df.index, y=df['norm_bench'], mode='lines+text',
        name=benchmark, text=[None]*(len(df)-1)+[f'<span style="color:white">{benchmark}</span>'],
        textposition='middle right', line=dict(color='white', width=2, dash='dot'), showlegend=False), row=4, col=2)

    # Cor da anotação
    if corr >= 0.7:
        cor_corr = '#3CB371'
    elif corr <= -0.7:
        cor_corr = 'red'
    else:
        cor_corr = 'cyan'

    fig.add_annotation(
        xref="x4", yref="y4", x=df.index[0],
        y=df[['norm_ativo', 'norm_bench']].max().max() * 1.15,
        text=f"<b>{corr:.4f}</b>",
        showarrow=False,
        font=dict(size=16, color='black'),
        align="center",
        bordercolor=cor_corr,
        borderwidth=2,
        borderpad=4,
        bgcolor=cor_corr,
        opacity=1
    )


    # Linha vertical com data
    fig.add_trace(go.Scatter(
        x=[df.index[-1], df.index[-1]],
        y=[min(df['norm_ativo'].min(), df['norm_bench'].min()) - 10,
           max(df['norm_ativo'].max(), df['norm_bench'].max()) + 10],
        mode='lines+text',
        line=dict(color='gray', width=1, dash='dot'),
        text=[df.index[-1].strftime('%d/%m/%Y')],
        textposition='top center',
        textfont=dict(color='gray'),
        showlegend=False
    ), row=4, col=2)

    fig.update_layout(
        title=f"Regressão de {ativo}",
        plot_bgcolor='black', paper_bgcolor='rgb(5,10,30)', font=dict(color='white'),
        margin=dict(l=40, r=20, t=40, b=40),
        legend=dict(orientation='h', x=0.1, y=1.15)
    )

    fig.update_xaxes(row=2, col=2, title_text=f"Retorno {benchmark}", showgrid=True,
        gridcolor='rgba(255,255,255,0.2)', griddash='dot', zeroline=False,
        range=[df['ret_bench'].min(), df['ret_bench'].max()])
    fig.update_yaxes(row=2, col=2, title_text=f"Retorno {ativo}", showgrid=True,
        gridcolor='rgba(255,255,255,0.2)', griddash='dot', zeroline=False)
    fig.update_xaxes(row=3, col=2, showgrid=False, zeroline=False)
    fig.update_yaxes(row=3, col=2, showgrid=False, zeroline=False)
    fig.update_yaxes(row=2, col=1, showgrid=False, zeroline=False)
    fig.update_xaxes(row=2, col=1, showgrid=False, zeroline=False, showticklabels=False)
    fig.update_xaxes(row=4, col=2, title="Data", showgrid=False)
    fig.update_yaxes(row=4, col=2, title="Preço (Base 100)", showgrid=False)

    # Retângulo lateral: Estatísticas + Lag
    metricas = html.Div([
        html.H4("📊 Estatísticas", style={'borderBottom': '1px solid gray'}),
        html.P(f"Alpha (intercepto): {alpha:.4f}"),
        html.P(f"Beta (coef.): {beta:.4f}"),
        html.P(f"R²: {r2:.4f}"),
        html.P(f"Correlação: {corr:.4f}"),
        html.P(f"P-valor (Normalidade): {p_value:.4f}"),
        html.P(f"Anderson-Darling: {ad_text}"),
        html.P(f"Melhor distribuição: {best_dist}"),
        html.P(f"N pontos: {len(df)}"),

        html.Div([
            html.Hr(style={"marginTop": "20px", "marginBottom": "10px"}),
            html.H5("⏱️ Análise de Lag", style={"marginBottom": "5px"}),
            html.P(f"Lag ótimo: {melhor_lag}"),
            html.P(f"Correlação no lag ótimo: {melhor_corr_lag:.4f}"),
            html.P(interpretacao)
        ], style={
            "position": "absolute",
            "bottom": "25px",
            "left": "10px",
            "fontSize": "12px"
        })
    ], style={
        "position": "relative",
        "paddingBottom": "80px",
        "minHeight": "420px"
    })

    return fig, metricas


if __name__ == '__main__':
    app.run(debug=True, port=8069)
