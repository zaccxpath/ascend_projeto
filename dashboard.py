"""
Dashboard Dash para o projeto ASCEND
Pode ser executado como um aplicativo independente ou integrado ao Flask

Para executar como aplicativo independente:
    python dashboard.py

Para usar integrado ao Flask:
    from dashboard import app as dash_app, set_flask_server
    dash_app = set_flask_server(flask_app)
"""

from dash import dcc, html, callback, Input, Output, State
import dash_bootstrap_components as dbc
from flask import Flask, Blueprint
import dash
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np
import json
import os
from datetime import datetime
import logging

# Configurar logger
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Definição de cores para manter consistência visual
COLORS = {
    'background': '#f9f9f9',
    'positive': '#28a745',
    'negative': '#dc3545',
    'neutral': '#6c757d',
    'text': '#212529',
    'light-bg': '#ffffff',
    'border': '#dee2e6',
    'primary': '#007bff',
    'secondary': '#6c757d'
}

# Arquivo para carregar os dados do dashboard
ANALISES_JSON_PATH = os.path.join('data', 'analises.json')
HISTORICO_JSON_PATH = os.path.join('data', 'historico.json')

def load_dashboard_data():
    """
    Carrega os dados necessários para o dashboard a partir dos arquivos JSON.
    IMPORTANTE: Todos os dados devem vir exclusivamente dos arquivos JSON.
    """
    try:
        logger.info(f"Tentando carregar dados dos arquivos {ANALISES_JSON_PATH} e {HISTORICO_JSON_PATH}")
        
        # Verificar se os arquivos existem
        if not os.path.exists(ANALISES_JSON_PATH):
            logger.error(f"Arquivo de análises não encontrado: {ANALISES_JSON_PATH}")
            return {'erro': 'Arquivo de análises não encontrado'}
            
        if not os.path.exists(HISTORICO_JSON_PATH):
            logger.warning(f"Arquivo de histórico não encontrado: {HISTORICO_JSON_PATH}")
            return {'erro': 'Arquivo de histórico não encontrado'}
        
        # Carregar dados dos arquivos JSON com tratamento de erro melhorado
        try:
            with open(ANALISES_JSON_PATH, 'r', encoding='utf-8') as f:
                analises_content = f.read().strip()
                if not analises_content:
                    logger.error(f"Arquivo de análises vazio: {ANALISES_JSON_PATH}")
                    return {'erro': 'Arquivo de análises vazio'}
                try:
                    analises = json.loads(analises_content)
                except json.JSONDecodeError as e:
                    logger.error(f"Erro ao decodificar JSON de análises: {e}")
                    return {'erro': 'Formato inválido no arquivo de análises'}
                
            with open(HISTORICO_JSON_PATH, 'r', encoding='utf-8') as f:
                historico_content = f.read().strip()
                if not historico_content:
                    logger.warning(f"Arquivo de histórico vazio: {HISTORICO_JSON_PATH}")
                    return {'erro': 'Arquivo de histórico vazio'}
                try:
                    historico = json.loads(historico_content)
                except json.JSONDecodeError as e:
                    logger.error(f"Erro ao decodificar JSON de histórico: {e}")
                    return {'erro': 'Formato inválido no arquivo de histórico'}
        except Exception as e:
            logger.error(f"Erro ao ler arquivos JSON: {e}")
            return {'erro': f'Erro ao ler arquivos: {str(e)}'}
        
        # Processar dados para formatos utilizáveis pelo dashboard
        logger.info(f"Processando dados: analises={type(analises).__name__}, historico={len(historico)} registros")
        dados_processados = processar_dados_para_dashboard(analises, historico)
        logger.info("Dados carregados com sucesso.")
        return dados_processados
        
    except Exception as e:
        logger.error(f"Erro ao carregar dados do dashboard: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        return {'erro': f'Erro inesperado: {str(e)}'}

def processar_dados_para_dashboard(analises, historico):
    """
    Processa os dados dos arquivos JSON para o formato usado pelo dashboard.
    IMPORTANTE: Não usar dados fictícios, apenas o que está nos arquivos JSON.
    """
    # Inicializar estrutura de dados processados
    dados_processados = {
        'estatisticas': {
            'total_feedbacks': 0,
            'positivos_pct': 0,
            'negativos_pct': 0,
            'neutros_pct': 0,
            'media_sentimento': 0
        },
        'dados_graficos': {
            'sentimentos': {'positivo': 0, 'negativo': 0, 'neutro': 0},
            'aspectos': {},
            'evolucao_tempo': []
        }
    }
    
    # Verificar se os dados são válidos
    if not isinstance(analises, dict):
        logger.warning(f"Dados de análises inválidos, tipo: {type(analises)}")
        return {'erro': 'Formato de análises inválido'}
        
    if not isinstance(historico, list):
        logger.warning(f"Dados de histórico inválidos, tipo: {type(historico)}")
        return {'erro': 'Formato de histórico inválido'}
    
    # Contar total de feedbacks no histórico
    total_feedbacks = len(historico)
    dados_processados['estatisticas']['total_feedbacks'] = total_feedbacks
    
    if total_feedbacks == 0:
        logger.warning("Nenhum feedback encontrado no histórico.")
        return {'erro': 'Não há feedbacks no histórico'}
    
    # Processar sentimentos
    sentimentos_count = {'positivo': 0, 'negativo': 0, 'neutro': 0}
    soma_sentimento = 0
    
    for item in historico:
        if not isinstance(item, dict):
            logger.warning(f"Item de histórico inválido, ignorando: {item}")
            continue
            
        analise = item.get('analise', {})
        if not isinstance(analise, dict):
            logger.warning(f"Análise inválida, ignorando: {analise}")
            continue
            
        sentimento = analise.get('sentimento', '').lower()
        if sentimento in sentimentos_count:
            sentimentos_count[sentimento] += 1
        
        # Calcular média do compound (se disponível)
        try:
            compound = analise.get('compound', 0)
            if isinstance(compound, (int, float)):
                soma_sentimento += float(compound)
        except (ValueError, TypeError) as e:
            logger.warning(f"Erro ao processar compound: {e}")
    
    # Calcular porcentagens com proteção contra divisão por zero
    if total_feedbacks > 0:
        dados_processados['dados_graficos']['sentimentos'] = sentimentos_count
        dados_processados['estatisticas']['positivos_pct'] = round((sentimentos_count['positivo'] / total_feedbacks) * 100, 1)
        dados_processados['estatisticas']['negativos_pct'] = round((sentimentos_count['negativo'] / total_feedbacks) * 100, 1)
        dados_processados['estatisticas']['neutros_pct'] = round((sentimentos_count['neutro'] / total_feedbacks) * 100, 1)
        dados_processados['estatisticas']['media_sentimento'] = round(soma_sentimento / total_feedbacks, 2)
    
    # Processar aspectos mencionados
    aspectos = {}
    for item in historico:
        if not isinstance(item, dict):
            continue
            
        analise = item.get('analise', {})
        if not isinstance(analise, dict):
            continue
            
        # Extrair aspectos da estrutura do JSON
        if 'aspectos' in analise:
            aspectos_data = analise['aspectos']
            
            # Verificar se temos a estrutura com detalhes ou summary
            if isinstance(aspectos_data, dict):
                if 'details' in aspectos_data:
                    # Formato com details e summary
                    details = aspectos_data.get('details', {})
                    for nome_aspecto, dados_aspecto in details.items():
                        nome_aspecto = nome_aspecto.lower()
                        if nome_aspecto not in aspectos:
                            aspectos[nome_aspecto] = 0
                        aspectos[nome_aspecto] += dados_aspecto.get('mencoes', 1)
                        
                elif 'summary' in aspectos_data:
                    # Formato com apenas summary
                    summary = aspectos_data.get('summary', {})
                    aspects_detected = summary.get('aspects_detected', [])
                    if isinstance(aspects_detected, list):
                        for aspecto in aspects_detected:
                            if aspecto:
                                aspecto = str(aspecto).lower()
                                if aspecto not in aspectos:
                                    aspectos[aspecto] = 0
                                aspectos[aspecto] += 1
    
    # Ordenar aspectos por contagem e limitar a 10 mais mencionados
    aspectos_ordenados = dict(sorted(aspectos.items(), key=lambda x: x[1], reverse=True)[:10])
    dados_processados['dados_graficos']['aspectos'] = aspectos_ordenados
    
    # Processar evolução temporal
    evolucao_tempo = []
    for item in historico:
        if not isinstance(item, dict):
            continue
            
        data = item.get('data', '')
        analise = item.get('analise', {})
        
        if not data or not isinstance(analise, dict):
            continue
            
        sentimento = analise.get('sentimento', 'neutro').lower()
        compound = 0
        try:
            compound = float(analise.get('compound', 0))
        except (ValueError, TypeError):
            compound = 0
            
        # Converter a string de data para formato padronizado
        try:
            # Tentar diferentes formatos de data ISO
            data_iso = None
            
            # Verificar se já temos uma data formatada adequadamente
            if isinstance(data, str):
                if 'T' in data:  # Formato padrão ISO com T separador
                    try:
                        if '.' in data:  # Com frações de segundo
                            data_iso = pd.to_datetime(data, format="%Y-%m-%dT%H:%M:%S.%f")
                        else:  # Sem frações de segundo
                            data_iso = pd.to_datetime(data, format="%Y-%m-%dT%H:%M:%S")
                    except:
                        # Tentar com formato 'mixed'
                        data_iso = pd.to_datetime(data, format='mixed')
                else:  # Outros formatos possíveis
                    data_iso = pd.to_datetime(data, format='mixed')
                    
            if data_iso is not None:
                evolucao_tempo.append({
                    'data': data_iso.strftime("%Y-%m-%d"),  # Formato padronizado
                    'sentimento': sentimento,
                    'compound': compound
                })
        except Exception as e:
            logger.warning(f"Erro ao processar data '{data}': {e}")
    
    # Ordenar por data
    try:
        evolucao_tempo.sort(key=lambda x: x['data'])
    except Exception as e:
        logger.warning(f"Erro ao ordenar evolução temporal: {e}")
    
    dados_processados['dados_graficos']['evolucao_tempo'] = evolucao_tempo
    
    # Adicionar dados de aspectos no dashboard
    dados_processados['estatisticas']['aspectos'] = {
        'aspecto_mais_mencionado': next(iter(aspectos_ordenados), None) if aspectos_ordenados else None,
        'total_analises_com_aspectos': sum(1 for item in historico if 'aspectos' in item.get('analise', {})),
        'total_aspectos_unicos': len(aspectos)
    }
    
    return dados_processados


def create_dashboard_graphs(data):
    """
    Cria gráficos para o dashboard com base nos dados carregados dos arquivos JSON.
    IMPORTANTE: Todos os gráficos devem ser baseados exclusivamente nos dados dos arquivos JSON.
    """
    try:
        # Verificar se há erros nos dados
        if 'erro' in data:
            logger.warning(f"Erro nos dados: {data['erro']}")
            return {
                'sentiment_pie': get_empty_figure(f"Erro: {data['erro']}"),
                'aspects_chart': get_empty_figure(f"Erro: {data['erro']}"),
                'time_sentiment_fig': get_empty_figure(f"Erro: {data['erro']}")
            }
        
        # Verificar se os dados estão na estrutura esperada
        if 'dados_graficos' not in data:
            logger.warning("Dados sem a estrutura esperada para gráficos")
            return {
                'sentiment_pie': get_empty_figure("Estrutura de dados inválida"),
                'aspects_chart': get_empty_figure("Estrutura de dados inválida"),
                'time_sentiment_fig': get_empty_figure("Estrutura de dados inválida")
            }
        
        dashboard_graphs = {}
        dados_graficos = data['dados_graficos']
        
        # 1. Gráfico de pizza para distribuição de sentimentos
        sentimentos = dados_graficos.get('sentimentos', {'positivo': 0, 'negativo': 0, 'neutro': 0})
        if sum(sentimentos.values()) > 0:
            sentiment_pie = go.Figure(data=[go.Pie(
                labels=list(sentimentos.keys()),
                values=list(sentimentos.values()),
                marker=dict(colors=[COLORS['positive'], COLORS['negative'], COLORS['neutral']]),
                textinfo='percent',
                hoverinfo='label+value',
                textfont=dict(size=14)
            )])
            
            sentiment_pie.update_layout(
                title='Distribuição de Sentimentos',
                height=350,
                margin=dict(l=40, r=40, t=40, b=40),
                legend=dict(
                    orientation="h",
                    yanchor="bottom",
                    y=-0.1,
                    xanchor="center",
                    x=0.5
                )
            )
            dashboard_graphs['sentiment_pie'] = sentiment_pie
        else:
            dashboard_graphs['sentiment_pie'] = get_empty_figure("Sem dados de sentimentos")
        
        # 2. Gráfico de barras para aspectos mencionados
        aspectos = dados_graficos.get('aspectos', {})
        if aspectos:
            aspects_df = pd.DataFrame(list(aspectos.items()), columns=['aspecto', 'contagem'])
            aspects_df = aspects_df.sort_values('contagem', ascending=True)  # Para mostrar do menor para o maior
            
            aspects_chart = go.Figure(data=[
                go.Bar(
                    x=aspects_df['contagem'],
                    y=aspects_df['aspecto'],
                    orientation='h',
                    marker=dict(color=COLORS['primary'])
                )
            ])
            
            aspects_chart.update_layout(
                title='Aspectos Mais Mencionados',
                xaxis_title='Número de Menções',
                height=350,
                margin=dict(l=120, r=40, t=40, b=40)
            )
            dashboard_graphs['aspects_chart'] = aspects_chart
        else:
            dashboard_graphs['aspects_chart'] = get_empty_figure("Sem dados de aspectos")
        
        # 3. Gráfico de linha para evolução temporal do sentimento
        evolucao_tempo = dados_graficos.get('evolucao_tempo', [])
        if evolucao_tempo:
            # Converter para DataFrame
            time_df = pd.DataFrame(evolucao_tempo)
            
            # Agrupar por data e calcular média de compound
            time_df['data'] = pd.to_datetime(time_df['data'])
            time_agg = time_df.groupby(time_df['data'].dt.date)['compound'].mean().reset_index()
            
            time_sentiment_fig = go.Figure(data=[
                go.Scatter(
                    x=time_agg['data'],
                    y=time_agg['compound'],
                    mode='lines+markers',
                    line=dict(color=COLORS['primary'], width=2),
                    marker=dict(size=8),
                    name='Média de Sentimento'
                )
            ])
            
            time_sentiment_fig.update_layout(
                title='Evolução do Sentimento ao Longo do Tempo',
                xaxis_title='Data',
                yaxis_title='Índice de Sentimento',
                height=350,
                margin=dict(l=40, r=40, t=40, b=40),
                yaxis=dict(
                    range=[-1, 1],
                    zeroline=True,
                    zerolinecolor='#aaaaaa',
                    zerolinewidth=1
                )
            )
            dashboard_graphs['time_sentiment_fig'] = time_sentiment_fig
        else:
            dashboard_graphs['time_sentiment_fig'] = get_empty_figure("Sem dados temporais")
        
        logger.info("Gráficos criados com sucesso para o dashboard")
        return dashboard_graphs
        
    except Exception as e:
        logger.error(f"Erro ao criar gráficos do dashboard: {str(e)}", exc_info=True)
        return {
            'sentiment_pie': get_empty_figure(f"Erro: {str(e)}"),
            'aspects_chart': get_empty_figure(f"Erro: {str(e)}"),
            'time_sentiment_fig': get_empty_figure(f"Erro: {str(e)}")
        }


def get_empty_figure(message="Sem dados disponíveis"):
    """Cria um gráfico vazio com mensagem"""
    fig = go.Figure()
    fig.update_layout(
        annotations=[dict(
            text=message,
            showarrow=False,
            font=dict(size=14)
        )]
    )
    return fig


def get_default_graphs():
    """Retorna gráficos vazios padronizados"""
    logger.info("Renderizando gráficos padrão para inicialização")
    return {
        'sentiment_pie': get_empty_figure("Sem dados de sentimentos"),
        'aspects_chart': get_empty_figure("Sem dados de aspectos"),
        'time_sentiment_fig': get_empty_figure("Sem dados temporais")
    }


# Inicialização do servidor Flask
server = Flask(__name__, static_folder='static')

# Inicialização da aplicação Dash
try:
    # Configurar ambiente para Dash
    os.environ['DASH_PRUNE_ERRORS'] = 'False'  # Mostrar erros completos
    os.environ['DASH_DEBUG'] = 'True'  # Ativar modo debug
    
    # Criar o app Dash com configurações mais seguras
    app = dash.Dash(
        __name__,
        server=server,
        routes_pathname_prefix='/dash/',
        requests_pathname_prefix='/dash/',
        assets_folder='static',
        serve_locally=True,  # Importante: servir localmente
        external_stylesheets=[
            dbc.themes.BOOTSTRAP,
            'https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0-beta3/css/all.min.css'
        ],
        suppress_callback_exceptions=True
    )
    logger.info("Aplicação Dash inicializada com sucesso")
    
    # Verificar se os componentes estão disponíveis
    # Suporta tanto Dash 1.x quanto 2.x
    try:
        # Dash 2.x
        from dash import dcc as dash_core_components
        from dash import html as dash_html_components
        logger.info("Componentes Dash 2.x importados com sucesso")
    except ImportError:
        # Fallback para Dash 1.x
        import dash_core_components
        import dash_html_components
        logger.info("Componentes Dash 1.x importados com sucesso")
except Exception as e:
    logger.error(f"Erro ao inicializar aplicação Dash: {str(e)}", exc_info=True)
    # Criar uma aplicação Dash mínima para evitar erros
    app = dash.Dash(__name__, server=server, 
                   routes_pathname_prefix='/dash/', 
                   requests_pathname_prefix='/dash/',
                   suppress_callback_exceptions=True,
                   serve_locally=True)

# Permitir que o server seja alterado externamente
def set_flask_server(flask_app):
    """
    Permite que o servidor Flask seja definido externamente.
    Isso é crucial para a integração do Dash com o Flask.
    """
    global server
    global app
    
    try:
        # Registrar informações no log
        flask_app.logger.info("Iniciando configuração da integração Dash com Flask...")
        
        # Configurar o servidor
        server = flask_app
        app.server = flask_app
        flask_app.logger.info("Servidor Flask atribuído ao app Dash")
        
        # Verificar configurações atuais do Dash
        flask_app.logger.info(f"Dash app URL pathname: {app.config.get('routes_pathname_prefix')}")
        
        # NÃO REGISTRAR NOVAS ROTAS AQUI - isso causa o erro
        # Em vez disso, apenas configure o servidor e retorne o app
        
        # NÃO configurar properties que são read-only após a inicialização
        # Somente configurar propriedades que podem ser modificadas após a criação
        app.config.update({
            'suppress_callback_exceptions': True,
            'serve_locally': True
        })
        flask_app.logger.info("Configurações seguras do Dash atualizadas")
        
        flask_app.logger.info("Dashboard Dash configurado e integrado ao Flask com sucesso!")
        flask_app.logger.info(f"Acesse o dashboard em: http://localhost:5000/dash/")
        
        return app
    except Exception as e:
        flask_app.logger.error(f"ERRO durante configuração Dash-Flask: {str(e)}")
        import traceback
        flask_app.logger.error(traceback.format_exc())
        
        # Retornar o app mesmo com erro para evitar falhas completas
        return app


# Carregar os dados e preparar os gráficos iniciais ao iniciar
try:
    # Carregar dados diretamente dos arquivos JSON
    logger.info("Carregando dados iniciais para o dashboard a partir dos arquivos JSON...")
    initial_data = load_dashboard_data()
    
    # Verificar se os dados foram carregados corretamente
    if not initial_data or 'erro' in initial_data:
        erro_msg = initial_data.get('erro', 'Dados não disponíveis') if initial_data else 'Dados não carregados'
        logger.warning(f"Dados iniciais com erro: {erro_msg}")
        initial_graphs = {
            'sentiment_pie': get_empty_figure(f"Erro nos dados: {erro_msg}"),
            'aspects_chart': get_empty_figure(f"Erro nos dados: {erro_msg}"),
            'time_sentiment_fig': get_empty_figure(f"Erro nos dados: {erro_msg}")
        }
    else:
        logger.info("Criando gráficos iniciais com os dados dos arquivos JSON...")
        initial_graphs = create_dashboard_graphs(initial_data)
        logger.info("Gráficos iniciais carregados com sucesso com dados dos arquivos JSON")
except Exception as e:
    logger.error(f"Erro ao carregar gráficos iniciais dos arquivos JSON: {str(e)}", exc_info=True)
    # Criar gráficos com mensagem de erro
    error_message = f"Erro ao carregar dados: {str(e)}"
    initial_graphs = {
        'sentiment_pie': get_empty_figure(error_message),
        'aspects_chart': get_empty_figure(error_message),
        'time_sentiment_fig': get_empty_figure(error_message)
    }

# Definir layout com componentes de filtragem e visualização
app.layout = dbc.Container([
    # Armazenamento para filtros
    dcc.Store(id="filter-store", data={}),
    
    # Cabeçalho do dashboard
    dbc.Row([
        dbc.Col([
            html.H2("ASCEND - Dashboard Estratégico", className="mb-3"),
            html.P("Análise detalhada de feedbacks e sentimentos dos clientes", className="text-muted")
        ], md=9),
        dbc.Col([
            dbc.Alert(
                [
                    html.I(className="fas fa-info-circle me-2"),
                    "Todos os dados são obtidos exclusivamente dos arquivos JSON. Nenhum dado fictício é utilizado."
                ],
                color="info",
                className="small p-2 text-center"
            )
        ], md=3)
    ], className="mb-4 mt-3"),
    
    # Abas para organizar as visualizações
    dbc.Tabs([
        # Aba de Visão Geral
        dbc.Tab(label="Visão Geral", children=[
            dbc.Row([
                # Gráfico de pizza para sentimentos
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader("Sentimentos"),
                        dbc.CardBody([
                            dcc.Graph(
                                id='sentiment-pie-chart',
                                figure=initial_graphs['sentiment_pie']
                            )
                        ])
                    ], className="h-100")
                ], md=6, className="mb-4"),
                
                # Gráfico de aspectos mencionados
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader("Aspectos Mencionados"),
                        dbc.CardBody([
                            dcc.Graph(
                                id='aspects-chart',
                                figure=initial_graphs['aspects_chart']
                            )
                        ])
                    ], className="h-100")
                ], md=6, className="mb-4")
            ]),
            
            # Gráfico de evolução temporal
            dbc.Row([
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader("Evolução do Sentimento"),
                        dbc.CardBody([
                            dcc.Graph(
                                id='sentiment-time-chart',
                                figure=initial_graphs['time_sentiment_fig']
                            )
                        ])
                    ])
                ], md=12, className="mb-4")
            ]),
            
            # Rodapé informativo
            dbc.Row([
                dbc.Col([
                    html.Hr(),
                    html.P(
                        "Todos os gráficos são gerados com dados exclusivamente de arquivos JSON. Se não houver dados suficientes, mensagens de erro serão exibidas.",
                        className="text-muted small text-center"
                    )
                ], md=12)
            ], className="mt-3")
        ], className="p-4"),
        
        # Aba de Filtros e Análises
        dbc.Tab(label="Filtros e Análises", children=[
            dbc.Row([
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader("Filtros"),
                        dbc.CardBody([
                            dbc.Row([
                                dbc.Col([
                                    html.Label("Período"),
                                    dcc.Dropdown(
                                        id="filtro-periodo",
                                        options=[
                                            {"label": "Último mês", "value": "mes"},
                                            {"label": "Últimos 3 meses", "value": "trimestre"},
                                            {"label": "Último ano", "value": "ano"},
                                            {"label": "Período específico", "value": "custom"}
                                        ],
                                        value="mes"
                                    )
                                ], md=4),
                                
                                dbc.Col([
                                    html.Label("Data inicial"),
                                    dcc.DatePickerSingle(
                                        id="filtro-data-inicial",
                                        date=None,
                                        disabled=True
                                    )
                                ], md=4),
                                
                                dbc.Col([
                                    html.Label("Data final"),
                                    dcc.DatePickerSingle(
                                        id="filtro-data-final",
                                        date=None,
                                        disabled=True
                                    )
                                ], md=4)
                            ], className="mb-3"),
                            
                            dbc.Row([
                                dbc.Col([
                                    html.Label("Canal"),
                                    dcc.Dropdown(
                                        id="filtro-canal",
                                        options=[
                                            {"label": "Todos", "value": "todos"},
                                            {"label": "E-mail", "value": "email"},
                                            {"label": "Telefone", "value": "telefone"},
                                            {"label": "Chat", "value": "chat"},
                                            {"label": "Formulário", "value": "formulario"}
                                        ],
                                        value="todos"
                                    )
                                ], md=4),
                                
                                dbc.Col([
                                    html.Label("Sentimento"),
                                    dcc.Dropdown(
                                        id="filtro-sentimento",
                                        options=[
                                            {"label": "Todos", "value": "todos"},
                                            {"label": "Positivo", "value": "positivo"},
                                            {"label": "Negativo", "value": "negativo"},
                                            {"label": "Neutro", "value": "neutro"}
                                        ],
                                        value="todos"
                                    )
                                ], md=4),
                                
                                dbc.Col([
                                    html.Label("Aspecto"),
                                    dcc.Dropdown(
                                        id="filtro-aspecto",
                                        options=[
                                            {"label": "Todos", "value": "todos"},
                                            {"label": "Atendimento", "value": "atendimento"},
                                            {"label": "Produto", "value": "produto"},
                                            {"label": "Preço", "value": "preco"},
                                            {"label": "Qualidade", "value": "qualidade"}
                                        ],
                                        value="todos"
                                    )
                                ], md=4)
                            ], className="mb-3"),
                            
                            dbc.Row([
                                dbc.Col([
                                    dbc.Button(
                                        "Aplicar Filtros", 
                                        id="btn-apply-filters", 
                                        color="primary", 
                                        className="me-2"
                                    ),
                                    dbc.Button(
                                        "Limpar Filtros", 
                                        id="btn-clear-filters", 
                                        color="secondary"
                                    )
                                ], className="d-flex")
                            ])
                        ])
                    ], className="mb-4")
                ], md=12)
            ]),
            
            # Área de visualização dos gráficos filtrados
            dbc.Row([
                dbc.Col([
                    dbc.Spinner(
                        dbc.Card([
                            dbc.CardHeader("Resultados da Análise"),
                            dbc.CardBody([
                                html.Div(id="filtered-results")
                            ])
                        ]),
                        color="primary"
                    )
                ], md=12)
            ]),
            
            # Rodapé informativo
            dbc.Row([
                dbc.Col([
                    html.Hr(),
                    html.P(
                        "Todos os filtros são aplicados aos dados carregados dos arquivos JSON. Se não houver dados disponíveis, não será possível aplicar os filtros.",
                        className="text-muted small text-center"
                    )
                ], md=12)
            ], className="mt-3")
        ], className="p-4")
    ])
], fluid=True, className="p-4", style={"backgroundColor": COLORS['background']})

# Callback para habilitar/desabilitar seleção de datas personalizadas
@callback(
    [Output("filtro-data-inicial", "disabled"),
     Output("filtro-data-final", "disabled")],
    [Input("filtro-periodo", "value")]
)
def toggle_custom_date(periodo):
    return periodo != "custom", periodo != "custom"

# Callback para limpar filtros
@callback(
    [Output("filtro-periodo", "value"),
     Output("filtro-data-inicial", "date"),
     Output("filtro-data-final", "date"),
     Output("filtro-canal", "value"),
     Output("filtro-sentimento", "value"),
     Output("filtro-aspecto", "value")],
    [Input("btn-clear-filters", "n_clicks")],
    prevent_initial_call=True
)
def clear_filters(n_clicks):
    return "mes", None, None, "todos", "todos", "todos"

# Callback para armazenar configurações de filtro
@callback(
    Output("filter-store", "data"),
    [Input("btn-apply-filters", "n_clicks")],
    [State("filtro-periodo", "value"),
     State("filtro-data-inicial", "date"),
     State("filtro-data-final", "date"),
     State("filtro-canal", "value"),
     State("filtro-sentimento", "value"),
     State("filtro-aspecto", "value"),
     State("filter-store", "data")],
    prevent_initial_call=True
)
def store_filter_settings(n_clicks, periodo, data_inicial, data_final, 
                        canal, sentimento, aspecto, current_data):
    if n_clicks is None:
        return current_data
    
    return {
        "periodo": periodo,
        "data_inicial": data_inicial,
        "data_final": data_final,
        "canal": canal,
        "sentimento": sentimento,
        "aspecto": aspecto,
        "timestamp": datetime.now().isoformat()
    }

# Callback para atualizar gráficos com base nos filtros
@callback(
    [Output("sentiment-pie-chart", "figure"),
     Output("aspects-chart", "figure"),
     Output("sentiment-time-chart", "figure")],
    [Input("filter-store", "data")]
)
def update_graphs_from_filter(filter_data):
    try:
        logger.info("Atualizando gráficos com base nos filtros")
        
        # Verificar se há dados de filtro e se os filtros foram aplicados
        if not filter_data or "timestamp" not in filter_data:
            logger.info("Sem filtros aplicados, carregando dados padrão")
            data = load_dashboard_data()
            
            # Verificar se há erro nos dados carregados
            if not data or 'erro' in data:
                erro_msg = data.get('erro', 'Dados não disponíveis') if data else 'Dados não carregados'
                logger.warning(f"Erro nos dados carregados: {erro_msg}")
                empty_fig = get_empty_figure(f"Erro: {erro_msg}")
                return empty_fig, empty_fig, empty_fig
            
            graphs = create_dashboard_graphs(data)
            return graphs['sentiment_pie'], graphs['aspects_chart'], graphs['time_sentiment_fig']
        
        # Carregar dados completos do dashboard
        logger.info("Carregando dados para aplicar filtros")
        data = load_dashboard_data()
        
        # Verificar se há erro nos dados carregados
        if not data or 'erro' in data:
            erro_msg = data.get('erro', 'Dados não disponíveis') if data else 'Dados não carregados'
            logger.warning(f"Erro nos dados carregados para filtros: {erro_msg}")
            empty_fig = get_empty_figure(f"Erro: {erro_msg}")
            return empty_fig, empty_fig, empty_fig
        
        # Aplicar filtros aos dados (esta parte seria implementada conforme necessário)
        # Por enquanto, apenas retornamos os gráficos normais
        # TODO: Implementar lógica de filtragem real
        logger.info("Criando gráficos com dados carregados")
        graphs = create_dashboard_graphs(data)
        
        # Verificar se os gráficos foram criados corretamente
        if not all(key in graphs for key in ['sentiment_pie', 'aspects_chart', 'time_sentiment_fig']):
            logger.warning("Gráficos não foram criados corretamente")
            empty_fig = get_empty_figure("Erro na criação dos gráficos")
            return empty_fig, empty_fig, empty_fig
        
        logger.info("Gráficos atualizados com sucesso")
        return graphs['sentiment_pie'], graphs['aspects_chart'], graphs['time_sentiment_fig']
    except Exception as e:
        logger.error(f"Erro ao atualizar gráficos com filtros: {str(e)}", exc_info=True)
        empty_fig = get_empty_figure(f"Erro ao carregar dados filtrados: {str(e)}")
        return empty_fig, empty_fig, empty_fig

# Callback de diagnóstico para verificar se o sistema está funcionando
@callback(
    Output("filtered-results", "children"),
    [Input("btn-apply-filters", "n_clicks")],
    prevent_initial_call=True
)
def diagnostic_callback(n_clicks):
    if n_clicks is None:
        return html.Div("Aguardando aplicação de filtros...")
    
    # Registrar atividade para diagnóstico
    logger.info(f"Callback de diagnóstico acionado! n_clicks={n_clicks}")
    
    # Verificar se os arquivos JSON existem
    analises_existe = os.path.exists(ANALISES_JSON_PATH)
    historico_existe = os.path.exists(HISTORICO_JSON_PATH)
    
    status_files = []
    if not analises_existe:
        status_files.append(html.Li(f"Arquivo de análises não encontrado: {ANALISES_JSON_PATH}"))
    if not historico_existe:
        status_files.append(html.Li(f"Arquivo de histórico não encontrado: {HISTORICO_JSON_PATH}"))
    
    # Verificar se os dados podem ser carregados
    try:
        if not (analises_existe and historico_existe):
            data_status = "Arquivos JSON necessários não encontrados"
            data_details = html.Ul(status_files)
        else:
            test_data = load_dashboard_data()
            if 'erro' in test_data:
                data_status = f"Erro ao processar dados: {test_data['erro']}"
                data_details = None
            else:
                data_status = "Dados carregados com sucesso dos arquivos JSON!"
                data_details = html.Div([
                    html.P("Informações dos dados:"),
                    html.Ul([
                        html.Li(f"Total de feedbacks: {test_data['dados_graficos'].get('total_feedbacks', 0)}"),
                        html.Li(f"Sentimentos: {', '.join([f'{k}: {v}' for k, v in test_data['dados_graficos'].get('sentimentos', {}).items()])}"),
                        html.Li(f"Aspectos: {len(test_data['dados_graficos'].get('aspectos', {}))} encontrados")
                    ])
                ])
    except Exception as e:
        logger.error(f"Erro ao carregar dados no callback de diagnóstico: {e}", exc_info=True)
        data_status = f"Erro ao carregar dados dos arquivos JSON: {str(e)}"
        data_details = None
    
    # Retornar informações de diagnóstico
    return html.Div([
        html.H4("Diagnóstico do Dashboard", className="text-primary"),
        html.P(f"Status: Sistema de callbacks funcionando (cliques: {n_clicks})"),
        html.P(f"Data/Hora: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"),
        html.H5("Status dos dados:"),
        html.P(data_status, className="font-weight-bold"),
        data_details if data_details else html.Div(),
        html.Hr(),
        html.P("Todos os dados deste dashboard são obtidos exclusivamente dos arquivos JSON. Nenhum dado fictício é utilizado.", 
              className="text-muted font-italic")
    ])

def check_dash_status():
    """
    Função para verificar o status do Dash e dos arquivos JSON.
    Pode ser chamada externamente para diagnóstico.
    """
    status = {
        "status": "ok",
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "mensagem": "Usando exclusivamente dados de arquivos JSON"
    }
    
    # Verificar a existência dos arquivos JSON
    status["data_files"] = {
        "analises_existe": os.path.exists(ANALISES_JSON_PATH),
        "historico_existe": os.path.exists(HISTORICO_JSON_PATH),
        "analises_path": ANALISES_JSON_PATH,
        "historico_path": HISTORICO_JSON_PATH
    }
    
    # Se algum arquivo não existir, definir status como erro
    if not all([status["data_files"]["analises_existe"], status["data_files"]["historico_existe"]]):
        status["status"] = "erro"
        status["error"] = "Arquivos JSON necessários não encontrados"
        logger.error("Arquivos JSON necessários para o dashboard não encontrados")
        return status
    
    # Tentar carregar dados para verificar integridade
    try:
        test_data = load_dashboard_data()
        if 'erro' in test_data:
            status["status"] = "erro"
            status["error"] = test_data['erro']
            logger.error(f"Erro ao carregar dados para o dashboard: {test_data['erro']}")
            return status
            
        status["data_status"] = "ok"
        status["data_info"] = {
            "total_feedbacks": test_data['dados_graficos'].get('total_feedbacks', 0),
            "total_sentimentos": sum(test_data['dados_graficos'].get('sentimentos', {}).values()),
            "total_aspectos": len(test_data['dados_graficos'].get('aspectos', {}))}
    except Exception as e:
        status["status"] = "erro"
        status["data_status"] = "erro"
        status["data_error"] = str(e)
        logger.error(f"Erro ao verificar dados JSON: {e}", exc_info=True)
        return status
        
    # Verificar o app Dash
    try:
        if not hasattr(app, 'layout') or app.layout is None:
            status["status"] = "erro"
            status["error"] = "Layout Dash não encontrado"
            logger.error("Dash layout não encontrado")
            return status
        
        # Verificar callbacks
        if not hasattr(app, '_callback_list') or not app._callback_list:
            status["status"] = "aviso"
            status["warning"] = "Callbacks não registrados"
            logger.warning("Dash callbacks não encontrados")
            return status
        
        # Verificar configurações
        status["config"] = {
            "routes_prefix": app.config.get('routes_pathname_prefix', 'não definido'),
            "requests_prefix": app.config.get('requests_pathname_prefix', 'não definido'),
            "assets_folder": app.config.get('assets_folder', 'não definido')
        }
        
        logger.info("Verificação do status do Dash concluída com sucesso")
        return status
    except Exception as e:
        logger.error(f"Erro ao verificar status do Dash: {e}", exc_info=True)
        status["status"] = "erro"
        status["error"] = str(e)
        return status

# Ponto de entrada para execução direta do arquivo
if __name__ == '__main__':
    # Se executado diretamente, inicia o servidor
    print("Servidor Dash iniciando...")
    app.run(debug=True, port=8050) 