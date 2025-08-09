#Lag de Atraso
import dash
from dash import dcc, html, Input, Output
import yfinance as yf
import pandas as pd
import plotly.graph_objs as go
from datetime import datetime, timedelta

# ===================== FUNÇÃO DE DADOS =====================
def calcular_dados_boxplot():
    ativos = [
        "BTC-USD", "ETH-USD", "BNB-USD", "SOL-USD", "ADA-USD",
        "XRP-USD", "DOGE-USD", "AVAX-USD", "DOT-USD", "LTC-USD", "LINK-USD", "ATOM-USD"
    ]
    fim = datetime.today()
    inicio = fim - timedelta(weeks=52)

    dados = yf.download(ativos, start=inicio, end=fim, interval='1wk', auto_adjust=True)['Close'].dropna()

    semana_atual = dados.iloc[-1]
    semana_passada = dados.iloc[-2]
    semana_retrasada = dados.iloc[-3]

    retorno_passado = ((semana_passada - semana_retrasada) / semana_retrasada) * 100
    retorno_atual = ((semana_atual - semana_passada) / semana_passada) * 100
    preco_base = semana_passada

    correlacoes = dados.pct_change().corr()['BTC-USD'].drop('BTC-USD')

    return ativos, retorno_passado, retorno_atual, preco_base, correlacoes

# ===================== DASH APP =====================
app = dash.Dash(__name__)
server = app.server

ativos_default = [
    "BTC-USD", "ETH-USD", "BNB-USD", "SOL-USD", "ADA-USD",
    "XRP-USD", "DOGE-USD", "AVAX-USD", "DOT-USD", "LTC-USD", "LINK-USD", "ATOM-USD"
]

app.layout = html.Div(style={"backgroundColor": "black", "padding": "20px"}, children=[
    html.H2("Lag de Atraso", style={"color": "white", "textAlign": "center"}),
    html.Div([
        html.Div([
            html.Div([
                dcc.Checklist(
                    id="dropdown-ativos",
                    options=[{"label": a.replace("-USD", ""), "value": a} for a in ativos_default],
                    value=ativos_default,
                    labelStyle={"display": "inline-block", "marginRight": "10px", "color": "white"},
                    inputStyle={"marginRight": "5px"},
                    style={"marginBottom": "5px"}
                ),
            ], style={"display": "flex", "flexWrap": "wrap", "justifyContent": "center", "alignItems": "center", "gap": "10px"}),

            html.Div(id="texto-correlacoes", style={"textAlign": "center", "marginBottom": "10px"})
        ]),
        html.Button("Atualizar", id="botao", n_clicks=0, style={"marginBottom": "10px", "marginTop": "10px"})
    ], style={"textAlign": "center"}),
    dcc.Graph(id="grafico-boxplot")
])

@app.callback(
    Output("grafico-boxplot", "figure"),
    Output("texto-correlacoes", "children"),
    Input("botao", "n_clicks"),
    Input("dropdown-ativos", "value")
)
def atualizar_grafico(n, ativos):
    ativos_all, ret_passado, ret_atual, preco_base, correlacoes = calcular_dados_boxplot()
    fig = go.Figure()

    ativos_filtrados = [a for a in ativos_all if a in ativos]
    espacamento = 1.5

    for i, ativo in enumerate(ativos_filtrados):
        nome = ativo.replace("-USD", "")
        y_index = i * espacamento
        base = ret_passado[ativo]
        variacao = ret_atual[ativo]
        preco = preco_base[ativo]

        fig.add_trace(go.Scatter(
            x=[0, base, base, 0, 0],
            y=[y_index - 0.45, y_index - 0.45, y_index + 0.45, y_index + 0.45, y_index - 0.45],
            mode='lines',
            line=dict(color='white', width=2, dash='dot'),
            fill='toself',
            fillcolor='rgba(255,255,255,0.15)',
            hovertemplate=(f"<b>{nome}</b><br>"
                           f"Preço semana passada: ${preco:.2f}<br>"
                           f"Retorno semana passada: {base:.2f}%<br>"
                           f"Retorno esta semana: {variacao:+.2f}%"),
            showlegend=False
        ))

        fig.add_trace(go.Scatter(
            x=[base / 2],
            y=[y_index],
            text=[f"{base:+.1f}%"],
            mode='text',
            textfont=dict(color='yellow', size=14),
            showlegend=False,
            hoverinfo='skip'
        ))

        x0 = base
        x1 = base + variacao
        texto = f"{variacao:+.1f}%"

        fig.add_trace(go.Scatter(
            x=[x0, x1],
            y=[y_index, y_index],
            mode='lines',
            line=dict(color='white', width=10),
            showlegend=False,
            hoverinfo='skip'
        ))

        fig.add_trace(go.Scatter(
            x=[x1],
            y=[y_index + 0.45],
            text=[texto],
            mode='text',
            textfont=dict(color='white', size=14),
            showlegend=False,
            hoverinfo='skip'
        ))

        fig.add_trace(go.Scatter(
            x=[x0, x1],
            y=[y_index, y_index],
            mode='markers',
            marker=dict(
                symbol='line-ns-open',
                color='white',
                size=16,
                line=dict(width=2)
            ),
            hoverinfo='skip',
            showlegend=False
        ))

        if ativo != "BTC-USD":
            diff = ret_atual["BTC-USD"] - variacao

            cenarios = {
                "otimista": (variacao + diff, "lime"),
                "realista": (variacao + 0.5 * diff, "deepskyblue"),
                "pessimista": (variacao - 0.3 * abs(diff), "red")
            }

            for label, (projecao, cor) in cenarios.items():
                proj_x = base + projecao
                fig.add_trace(go.Scatter(
                    x=[proj_x],
                    y=[y_index],
                    mode='markers',
                    marker=dict(symbol='line-ew-open', color=cor, size=10, line=dict(width=2)),
                    hovertemplate=f"{label.capitalize()}: {projecao:.2f}%",
                    showlegend=False
                ))

    fig.add_shape(type='line', x0=0, x1=0, y0=-1, y1=len(ativos_filtrados) * espacamento,
                line=dict(color='white', width=1, dash='dot'), layer='above')

    fig.update_layout(
        title=dict(
            text="Histograma = Semana Passada | Linha = Semana Atual",
            font=dict(size=14, color='white'),
            x=0.5
        ),
        plot_bgcolor='black',
        paper_bgcolor='black',
        font=dict(color='white'),
        yaxis_title="Criptoativo",
        margin=dict(l=50, r=30, t=50, b=40),
        height=700,
        xaxis=dict(showgrid=False, showline=False, zeroline=False, ticks='', showticklabels=False),
        yaxis=dict(showline=True, linecolor='white', linewidth=1.5, side='left', showgrid=False,
            tickmode='array',
            tickvals=[i * espacamento for i in range(len(ativos_filtrados))],
            ticktext=[a.replace("-USD", "") for a in ativos_filtrados]
        )
    )

    texto_corr = html.Div([
        html.Span(f"{a.replace('-USD', '')}: {correlacoes[a]:+.2f}  ", style={
            'margin': '0 8px',
            'color': 'lime' if correlacoes[a] >= 0 else 'red'
        })
        for a in ativos_filtrados if a != "BTC-USD"
    ], style={"display": "flex", "justifyContent": "center", "flexWrap": "wrap"})

    return fig, texto_corr

# ===================== EXECUTAR =====================
if __name__ == "__main__":
    app.run(debug=True, port=8054)
