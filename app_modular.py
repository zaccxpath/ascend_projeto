import os
import json
import sys
from datetime import datetime
import logging
from logging.handlers import RotatingFileHandler
import traceback
from typing import Any, Optional, Tuple, Union
import pandas as pd
import plotly.graph_objects as go

from flask import Flask, render_template, request, jsonify, redirect, url_for, Response, current_app, flash, send_from_directory

from models.sentiment import SentimentAnalyzer
from models.speech import SpeechHandler
from models.data_handler import DataHandler
from models.aspect_extractor import AspectExtractor
from models.estatisticas import AnaliseEstatistica
from models.sarcasm_factory import integrar_detector_sarcasmo_ao_sistema_melhorado
from utils.helpers import criar_diretorios_essenciais, formatar_sentimento, formatar_confianca

from models.sentiment import SentimentAnalyzer
from models.speech import SpeechHandler
from models.data_handler import DataHandler
from models.aspect_extractor import AspectExtractor
from models.estatisticas import AnaliseEstatistica
from models.sarcasm_integration import integrar_detector_sarcasmo_avancado, analisar_texto_com_sarcasmo
from utils.helpers import criar_diretorios_essenciais

# Importar novo módulo de integração com LLM local
from models.local_llm_integration import integrar_llm_local_ao_sentiment_analyzer
from models.local_llm_analyzer import LocalLLMAnalyzer, AnalyzerConfig as LLMConfig, ModelNotLoadedException

# Configurações de log e diretórios
LOG_FILENAME = 'app.log'
LOG_MAX_BYTES = 10 * 1024 * 1024
LOG_BACKUP_COUNT = 3


def configurar_logger() -> logging.Logger:
    logger = logging.getLogger('app')
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    # Console
    ch = logging.StreamHandler(sys.stdout)
    ch.setFormatter(formatter)
    logger.addHandler(ch)
    # Arquivo rotativo
    fh = RotatingFileHandler(
        LOG_FILENAME,
        maxBytes=LOG_MAX_BYTES,
        backupCount=LOG_BACKUP_COUNT
    )
    fh.setFormatter(formatter)
    logger.addHandler(fh)
    return logger


def inicializar_componentes(app: Flask, logger: logging.Logger) -> bool:
    """Inicializa componentes essenciais e retorna status"""
    try:
        logger.info("Inicializando componentes...")
        # DataHandler sem parâmetro de schema
        dh = DataHandler()
        app.data_handler = dh
        logger.info("DataHandler inicializado")

        # SentimentAnalyzer
        sa = SentimentAnalyzer()
        app.sentiment_analyzer = sa
        logger.info("SentimentAnalyzer inicializado")

        # SpeechHandler
        sh = SpeechHandler(app.sentiment_analyzer)
        app.speech_handler = sh
        logger.info("SpeechHandler inicializado")

        # AspectExtractor
        ae = AspectExtractor()
        app.aspect_extractor = ae
        logger.info("AspectExtractor inicializado")

        # Modificado: Integrar detector de sarcasmo avançado
        app = integrar_detector_sarcasmo_avancado(
            app,
            app.sentiment_analyzer,
            app.speech_handler,
            app.data_handler
        )
        app.config['SARCASM_DETECTION_ENABLED'] = True
        logger.info("Detector de sarcasmo integrado")
        
        app.config['SYSTEM_READY'] = True
        return True

    except Exception as e:
        logger.critical(f"Falha na inicialização: {e}")
        logger.critical(traceback.format_exc())
        app.config['SYSTEM_READY'] = False
        return False


def criar_app() -> Flask:
    app = Flask(__name__)
    app.config['SYSTEM_READY'] = False
    logger = configurar_logger()
    app.logger = logger

    criar_diretorios_essenciais()
    inicializar_componentes(app, logger)

    # Importar e configurar o Dash app - Abordagem simplificada
    try:
        app.logger.info("Iniciando integração do Dash...")
        
        # 1. Importar o Dash diretamente
        import dash
        from dash import dcc, html
        import dash_bootstrap_components as dbc
        
        # 2. Criar uma instância Dash vinculada ao Flask
        dash_app = dash.Dash(
            __name__,
            server=app,
            url_base_pathname='/dash/',
            suppress_callback_exceptions=True,
            external_stylesheets=[dbc.themes.BOOTSTRAP],
            serve_locally=True
        )
        app.logger.info("Instância Dash criada com sucesso")
        
        # 3. Importar layouts e callbacks
        try:
            # Tentar importar do arquivo dashboard.py com ênfase na importação de dados dos JSONs
            app.logger.info("Importando funções de processamento de dados do dashboard.py")
            from dashboard import load_dashboard_data, create_dashboard_graphs
            
            # Carregar dados diretamente dos JSON
            data = load_dashboard_data()
            
            # Verificar se temos dados válidos
            if 'erro' in data:
                app.logger.error(f"Erro ao carregar dados do dashboard: {data['erro']}")
                empty_graph = {
                    'data': [], 
                    'layout': {'title': f"Erro: {data['erro']}", 'height': 400}
                }
                initial_graphs = {
                    'sentiment_pie': empty_graph,
                    'aspects_chart': empty_graph,
                    'time_sentiment_fig': empty_graph
                }
            else:
                # Criar gráficos a partir dos dados dos JSON
                app.logger.info("Criando gráficos a partir dos dados JSON")
                graphs = create_dashboard_graphs(data)
                initial_graphs = {
                    'sentiment_pie': graphs['sentiment_pie'].to_dict() if hasattr(graphs['sentiment_pie'], 'to_dict') else graphs['sentiment_pie'],
                    'aspects_chart': graphs['aspects_chart'].to_dict() if hasattr(graphs['aspects_chart'], 'to_dict') else graphs['aspects_chart'],
                    'time_sentiment_fig': graphs['time_sentiment_fig'].to_dict() if hasattr(graphs['time_sentiment_fig'], 'to_dict') else graphs['time_sentiment_fig']
                }
            
            # Configurar layout básico usando apenas dados dos JSON
            dash_app.layout = dbc.Container([
                dbc.Row([
                    dbc.Col([
                        html.H2("ASCEND - Dashboard Estratégico", className="mb-3"),
                        html.P("Análise de feedbacks de clientes baseada exclusivamente nos dados JSON", className="text-muted")
                    ])
                ], className="mb-4 mt-3"),
                
                dbc.Row([
                    dbc.Col([
                        dbc.Card([
                            dbc.CardHeader("Distribuição de Sentimentos"),
                            dbc.CardBody([
                                dcc.Graph(
                                    id='sentiment-pie-chart',
                                    figure=initial_graphs['sentiment_pie']
                                )
                            ])
                        ])
                    ], md=6),
                    
                    dbc.Col([
                        dbc.Card([
                            dbc.CardHeader("Aspectos Mencionados"),
                            dbc.CardBody([
                                dcc.Graph(
                                    id='aspects-chart',
                                    figure=initial_graphs['aspects_chart']
                                )
                            ])
                        ])
                    ], md=6)
                ]),
                
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
                    ], md=12, className="mt-4")
                ]),
                
            ])
            
            # Callback para atualizar gráficos
            @dash_app.callback(
                [dash.Output("sentiment-pie-chart", "figure"),
                 dash.Output("aspects-chart", "figure"),
                 dash.Output("sentiment-time-chart", "figure")],
                [dash.Input("sentiment-pie-chart", "id")]  # Trigger dummy
            )
            def update_dashboard(dummy):
                app.logger.info("Callback Dash executado - atualizando dashboard com dados dos JSON")
                # Recarregar dados diretamente dos JSON
                data = load_dashboard_data()
                
                # Verificar se temos dados válidos
                if 'erro' in data:
                    app.logger.error(f"Erro ao atualizar dashboard: {data['erro']}")
                    empty_graph = {
                        'data': [], 
                        'layout': {'title': f"Erro: {data['erro']}", 'height': 400}
                    }
                    return empty_graph, empty_graph, empty_graph
                
                # Criar gráficos a partir dos dados dos JSON
                graphs = create_dashboard_graphs(data)
                return (
                    graphs['sentiment_pie'], 
                    graphs['aspects_chart'], 
                    graphs['time_sentiment_fig']
                )
            
            app.logger.info("Layout e callbacks Dash configurados com sucesso")
            
        except Exception as e:
            app.logger.error(f"Erro ao configurar layout do Dash: {e}")
            app.logger.error(traceback.format_exc())
            
            # Layout de fallback em caso de erro - ainda indicando que dados devem vir dos JSON
            dash_app.layout = html.Div([
                html.H1("Dashboard ASCEND", className="text-center mt-4"),
                html.Div([
                    html.H3("Erro ao carregar dashboard", className="text-danger"),
                    html.P(f"Erro: {str(e)}"),
                    html.P("Não foi possível carregar os dados dos arquivos JSON. Verifique se os arquivos existem e estão formatados corretamente."),
                    html.P("Por favor, verifique os logs do servidor para mais detalhes.")
                ], className="p-4 border rounded mt-4")
            ])
            
        app.logger.info("Integração com Dash concluída com sucesso!")
        
    except ImportError as e:
        app.logger.error(f"Dash não instalado: {e}")
    except Exception as e:
        app.logger.error(f"Erro na integração com Dash: {e}")
        app.logger.error(traceback.format_exc())

    # Handlers globais
    @app.errorhandler(Exception)
    def handle_exception(e: Exception) -> Union[Response, Tuple[Response,int]]:
        is_api = request.path.startswith('/api/')
        app.logger.error(str(e))
        app.logger.error(traceback.format_exc())
        if is_api:
            return jsonify({
                'sucesso': False,
                'erro': str(e),
                'tipo': type(e).__name__,
                'timestamp': datetime.now().isoformat()
            }), 500
        return render_template('error.html', erro=str(e)), 500

    @app.before_request
    def verificar_status():
        if request.path.startswith(('/diagnostico', '/static/', '/favicon.ico')):
            return None
        if not app.config.get('SYSTEM_READY'):
            msg = 'Sistema indisponível'
            if request.path.startswith('/api/'):
                return jsonify({'sucesso': False, 'erro': msg}), 503
            return render_template('error.html', erro=msg), 503
        return None

    registrar_rotas(app)
    return app


def registrar_rotas(app: Flask) -> None:
    @app.route('/')
    def index():
        # Verificar status do modelo XLM-RoBERTa para passar ao template
        xlm_roberta_ativo = False
        if hasattr(current_app, 'sentiment_analyzer'):
            # Verificar se o modelo está ativo
            if hasattr(current_app.sentiment_analyzer, 'using_xlm_roberta'):
                xlm_roberta_ativo = current_app.sentiment_analyzer.using_xlm_roberta
                
            # Se o modelo parece disponível mas a flag está falsa, corrigir
            if not xlm_roberta_ativo and hasattr(current_app.sentiment_analyzer, 'sentiment_model'):
                if current_app.sentiment_analyzer.sentiment_model is not None:
                    current_app.logger.info("Modelo encontrado mas flag estava incorreta. Corrigindo...")
                    current_app.sentiment_analyzer.using_xlm_roberta = True
                    xlm_roberta_ativo = True
                    
        return render_template('index.html', xlm_roberta_ativo=xlm_roberta_ativo)

    @app.route('/diagnostico')
    def diagnostico():
        try:
            # Verificação direta do status do modelo
            xlm_roberta_ativo = False
            if hasattr(current_app, 'sentiment_analyzer'):
                # Verificação mais detalhada do estado do modelo
                if hasattr(current_app.sentiment_analyzer, 'using_xlm_roberta'):
                    xlm_roberta_ativo = current_app.sentiment_analyzer.using_xlm_roberta
                    current_app.logger.info(f"Estado do XLM-RoBERTa reportado como: {xlm_roberta_ativo}")
                    
                # Se o modelo está reportando como não ativo, mas parece estar inicializado corretamente
                if not xlm_roberta_ativo and hasattr(current_app.sentiment_analyzer, 'sentiment_model'):
                    if current_app.sentiment_analyzer.sentiment_model is not None:
                        # Forçar a atualização do status
                        current_app.logger.info("Modelo encontrado mas flag using_xlm_roberta está falsa. Corrigindo...")
                        current_app.sentiment_analyzer.using_xlm_roberta = True
                        xlm_roberta_ativo = True
            
            info = {
                'status': 'online' if current_app.config.get('SYSTEM_READY') else 'degradado',
                'components': {
                    'data_handler': hasattr(current_app, 'data_handler'),
                    'sentiment_analyzer': hasattr(current_app, 'sentiment_analyzer'),
                    'speech_handler': hasattr(current_app, 'speech_handler'),
                    'aspect_extractor': hasattr(current_app, 'aspect_extractor'),
                    'sarcasm': current_app.config.get('SARCASM_DETECTION_ENABLED', False)
                },
                'timestamp': datetime.now().isoformat(),
                'xlm_roberta_ativo': xlm_roberta_ativo,
                'modelo_info': {
                    'tipo': 'xlm-roberta' if xlm_roberta_ativo else 'fallback',
                    'data_atualizacao': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                }
            }
            
            # Adicionar detalhes extras para diagnóstico
            if hasattr(current_app, 'sentiment_analyzer'):
                sa = current_app.sentiment_analyzer
                info['detalhe_modelo'] = {
                    'has_sentiment_model': hasattr(sa, 'sentiment_model') and sa.sentiment_model is not None,
                    'has_sentiment_task': hasattr(sa, 'sentiment_task') and sa.sentiment_task is not None,
                    'transformers_available': hasattr(sa, 'transformers_available') and sa.transformers_available
                }
                
            # Checar se a requisição espera HTML ou JSON
            if request.headers.get('accept', '').find('application/json') != -1 or request.args.get('format') == 'json':
                return jsonify(info)
            
            return render_template('diagnostico.html', info=info)
        except Exception as e:
            current_app.logger.error(f"Erro ao gerar diagnóstico: {e}")
            current_app.logger.error(traceback.format_exc())
            return jsonify({
                'erro': str(e),
                'status': 'erro',
                'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            })

    @app.route('/api/testar-modelo', methods=['POST'])
    def testar_modelo_api():
        data = request.get_json() or {}
        text = data.get('texto', '')
        if not text:
            return jsonify({'sucesso': False, 'erro': 'Texto vazio'}), 400
            
        # Verificar se o sistema de sarcasmo está habilitado
        sarcasmo_habilitado = current_app.config.get('SARCASM_DETECTION_ENABLED', False)
        
        if sarcasmo_habilitado:
            # Usar a função analisar_texto_com_sarcasmo para análise integrada
            result = analisar_texto_com_sarcasmo(
                text,
                current_app.sentiment_analyzer,
                incluir_detalhes=True
            )
            current_app.logger.info(f"Análise de sarcasmo integrada aplicada ao testar modelo")
        else:
            # Obter análise de sentimento com extração de tópicos
            try:
                # Se a função analisar_sentimento_final existe, usar ela
                if hasattr(current_app.sentiment_analyzer, 'analisar_sentimento_final'):
                    result = current_app.sentiment_analyzer.analisar_sentimento_final(text)
                else:
                    # Caso contrário, usar a função padrão
                    result = current_app.sentiment_analyzer.analisar_sentimento(text)
                    
                    # Garantir que tópicos existe no resultado
                    if 'topicos' not in result:
                        # Tentar extrair tópicos usando método separado
                        if hasattr(current_app.sentiment_analyzer, 'extrair_palavras_chave'):
                            result['topicos'] = current_app.sentiment_analyzer.extrair_palavras_chave(text)
                        else:
                            result['topicos'] = []
            except Exception as e:
                current_app.logger.error(f"Erro ao analisar sentimento: {e}")
                result = {
                    'sentimento': 'neutro',
                    'confianca': 0.5,
                    'topicos': []
                }
            
        aspectos = current_app.aspect_extractor.extrair_aspectos(text, result)
        resp = {
            'sucesso': True,
            'texto': text,
            'sentimento': result.get('sentimento'),
            'confianca': result.get('confianca'),
            'confianca_formatada': formatar_confianca(result.get('confianca')),
            'aspectos': aspectos,
            'sarcasmo': result.get('sarcasmo', {}),
            'topicos': result.get('topicos', [])
        }
        return jsonify(resp)

    @app.route('/historico')
    def historico():
        try:
            # Carregar histórico e análises
            historico = current_app.data_handler.carregar_historico()
            current_app.logger.info(f"Histórico carregado: {len(historico)} entradas")
            analises = current_app.data_handler.carregar_analises()
            current_app.logger.info(f"Análises carregadas: {len(analises)} registros")
            
            # Verificar o tipo de analises e extrair a lista de dados
            if not isinstance(analises, list):
                if isinstance(analises, dict) and 'dados' in analises:
                    analises_lista = analises['dados']
                    current_app.logger.info(f"Convertendo analises de dicionário para lista de dados ({len(analises_lista)} itens)")
                else:
                    analises_lista = []
                    current_app.logger.warning(f"Impossível extrair lista de analises, tipo: {type(analises).__name__}")
            else:
                analises_lista = analises
            
            # Criar um dicionário para relacionar os IDs das análises com seus respectivos dados
            analises_por_id = {item['id']: item for item in analises_lista}
            
            # Combinar dados de histórico e análises
            registros_combinados = []
            for item_historico in historico:
                id_registro = item_historico.get('id')
                if id_registro in analises_por_id:
                    # Combinar dados de análise e histórico
                    registro_combinado = analises_por_id[id_registro].copy()
                    # Adicionar campos específicos do histórico
                    registro_combinado['texto'] = item_historico.get('texto', 'Sem texto')
                    registro_combinado['timestamp'] = item_historico.get('data_formatada', item_historico.get('data', ''))
                    # Adicionar campo de data original para ordenação adequada
                    registro_combinado['data_original'] = item_historico.get('data', '')
                    
                    # Extrai informações de aspecto, se existirem
                    analise_original = item_historico.get('analise', {})
                    
                    # Garantir que campos de aspectos existam
                    if 'aspectos' in analise_original and isinstance(analise_original['aspectos'], dict) and 'summary' in analise_original['aspectos']:
                        summary = analise_original['aspectos']['summary']
                        registro_combinado['aspecto_principal'] = summary.get('primary_aspect', '')
                        registro_combinado['aspectos_detectados'] = ','.join(summary.get('aspects_detected', []))
                    elif 'aspecto_principal' in analise_original:
                        registro_combinado['aspecto_principal'] = analise_original.get('aspecto_principal', '')
                        registro_combinado['aspectos_detectados'] = analise_original.get('aspectos_detectados', '')
                    
                    # Adicionar tópicos, se existirem
                    registro_combinado['topicos'] = analise_original.get('topicos', '')
                    
                    registros_combinados.append(registro_combinado)
                else:
                    # Caso não encontre na análise, usar direto do histórico
                    registro = {
                        'id': id_registro,
                        'texto': item_historico.get('texto', 'Sem texto'),
                        'timestamp': item_historico.get('data_formatada', item_historico.get('data', '')),
                        'data_original': item_historico.get('data', ''),
                        'sentimento': item_historico.get('analise', {}).get('sentimento', 'neutro')
                    }
                    
                    # Extrair aspectos e tópicos se existirem
                    analise = item_historico.get('analise', {})
                    if 'aspectos' in analise and isinstance(analise['aspectos'], dict) and 'summary' in analise['aspectos']:
                        summary = analise['aspectos']['summary']
                        registro['aspecto_principal'] = summary.get('primary_aspect', '')
                        registro['aspectos_detectados'] = ','.join(summary.get('aspects_detected', []))
                    elif 'aspecto_principal' in analise:
                        registro['aspecto_principal'] = analise.get('aspecto_principal', '')
                        registro['aspectos_detectados'] = analise.get('aspectos_detectados', '')
                    
                    registro['topicos'] = analise.get('topicos', '')
                    
                    registros_combinados.append(registro)
            
            # Ordem decrescente por data (mais recente primeiro)
            # Usar data_original para ordenação (formato ISO) que garante ordem cronológica correta
            registros_combinados.sort(key=lambda x: x.get('data_original', ''), reverse=True)
            
            current_app.logger.info(f"Registros ordenados por data original em ordem decrescente (total: {len(registros_combinados)} registros)")
            if registros_combinados and len(registros_combinados) > 1:
                primeiro = registros_combinados[0].get('data_original', '')
                ultimo = registros_combinados[-1].get('data_original', '')
                current_app.logger.info(f"Primeiro registro (mais recente): {primeiro}")
                current_app.logger.info(f"Último registro (mais antigo): {ultimo}")
            
            # Garantir que todos os campos relevantes existam e sejam serializáveis
            for registro in registros_combinados:
                # Converter valores vazios para strings vazias
                for campo in ['texto', 'topicos', 'aspecto_principal', 'aspectos_detectados']:
                    if campo not in registro or registro[campo] is None:
                        registro[campo] = ''
                    elif not isinstance(registro[campo], str):
                        registro[campo] = str(registro[campo])
            
            # Log para debug
            for i in range(min(3, len(registros_combinados))):
                analise = registros_combinados[i]
                current_app.logger.info(f"Registro {i+1}: texto={analise.get('texto', '')[:30]}..., sentimento={analise.get('sentimento', '')}")
            
            return render_template('historico.html', historico=historico, analises=registros_combinados)
        except Exception as e:
            current_app.logger.error(f"Erro na rota /historico: {e}")
            import traceback
            traceback_str = traceback.format_exc()
            current_app.logger.error(f"Detalhes do erro: {traceback_str}")
            return render_template('historico.html', historico=["Erro ao carregar histórico."], analises=[])

    @app.route('/adicionar', methods=['GET', 'POST'])
    def adicionar_feedback():
        if request.method == 'POST':
            texto = request.form.get('texto', '')
            if texto:
                current_app.logger.info(f"Feedback manual recebido: {texto}")
                try:
                    # Verificar se o sistema de sarcasmo está habilitado
                    sarcasmo_habilitado = current_app.config.get('SARCASM_DETECTION_ENABLED', False)
                    
                    # Usar a função de análise melhorada se disponível
                    if hasattr(current_app.sentiment_analyzer, 'analisar_sentimento_final'):
                        analise = current_app.sentiment_analyzer.analisar_sentimento_final(texto)
                        current_app.logger.info("Análise completa com extração de tópicos aplicada ao feedback manual")
                    elif sarcasmo_habilitado:
                        # Usar a função analisar_texto_com_sarcasmo para análise integrada
                        from models.sarcasm_integration import analisar_texto_com_sarcasmo
                        analise = analisar_texto_com_sarcasmo(
                            texto,
                            current_app.sentiment_analyzer,
                            incluir_detalhes=True
                        )
                        current_app.logger.info(f"Análise de sarcasmo integrada aplicada ao feedback manual")
                    else:
                        # Obter análise de sentimento padrão
                        analise = current_app.sentiment_analyzer.analisar_sentimento(texto)
                    
                    # Verificar se temos tópicos e extrair se necessário
                    if 'topicos' not in analise and hasattr(current_app.sentiment_analyzer, 'extrair_palavras_chave'):
                        analise['topicos'] = current_app.sentiment_analyzer.extrair_palavras_chave(texto)
                        current_app.logger.info(f"Tópicos extraídos adicionalmente: {analise['topicos']}")
                    
                    # Extrair aspectos do texto e adicionar à análise
                    aspectos_resultado = current_app.aspect_extractor.extrair_aspectos(texto, analise)
                    current_app.logger.info(f"Aspectos extraídos: {aspectos_resultado}")
                    
                    # Criar objeto de aspectos formatado para armazenamento
                    if 'aspectos_encontrados' in aspectos_resultado and aspectos_resultado['aspectos_encontrados']:
                        # Determinar aspecto principal (com mais menções)
                        aspectos_ordenados = sorted(
                            aspectos_resultado['aspectos_encontrados'].items(),
                            key=lambda x: x[1]['mencoes'],
                            reverse=True
                        )
                        
                        aspecto_principal = aspectos_ordenados[0][0] if aspectos_ordenados else None
                        aspects_detected = list(aspectos_resultado['aspectos_encontrados'].keys())
                        
                        # Adicionar à análise
                        if 'aspectos' not in analise:
                            analise['aspectos'] = {}
                        
                        analise['aspectos']['summary'] = {
                            'primary_aspect': aspecto_principal,
                            'aspects_detected': aspects_detected,
                            'total_mentions': aspectos_resultado['mencoes_totais']
                        }
                        
                        # Adicionar detalhes completos
                        analise['aspectos']['details'] = aspectos_resultado['aspectos_encontrados']
                        
                        current_app.logger.info(f"Aspecto principal: {aspecto_principal}, Aspectos detectados: {aspects_detected}")
                    else:
                        current_app.logger.info("Nenhum aspecto relevante encontrado no texto")
                    
                    # Garantir que os dados estão no formato correto
                    if 'topicos' in analise and not isinstance(analise['topicos'], str):
                        if analise['topicos'] is None:
                            analise['topicos'] = []
                        elif isinstance(analise['topicos'], list):
                            # Já está no formato correto
                            pass
                        else:
                            try:
                                # Tentar converter para lista se for outro tipo
                                analise['topicos'] = [str(analise['topicos'])]
                            except Exception as e:
                                current_app.logger.error(f"Erro ao processar tópicos: {e}")
                                analise['topicos'] = []
                    elif 'topicos' not in analise:
                        analise['topicos'] = []
                    
                    # Validar aspectos
                    if 'aspectos' in analise and 'summary' in analise['aspectos']:
                        aspectos_detectados = analise['aspectos']['summary'].get('aspects_detected', [])
                        aspecto_principal = analise['aspectos']['summary'].get('primary_aspect', None)
                        
                        # Garantir que aspectos_detectados é uma lista de strings
                        if aspectos_detectados and isinstance(aspectos_detectados, list):
                            analise['aspectos_detectados'] = ','.join(str(a) for a in aspectos_detectados)
                        else:
                            analise['aspectos_detectados'] = ''
                            
                        # Garantir que aspecto_principal é uma string
                        analise['aspecto_principal'] = str(aspecto_principal) if aspecto_principal else ''
                    
                    # CORREÇÃO: Adicionar sub-aspectos e resumo à análise
                    if 'sub_aspectos' in aspectos_resultado:
                        analise['sub_aspectos'] = aspectos_resultado['sub_aspectos']
                        current_app.logger.info(f"Sub-aspectos adicionados à análise: {analise['sub_aspectos']}")
                    
                    if 'resumo_sub_aspectos' in aspectos_resultado:
                        analise['resumo_sub_aspectos'] = aspectos_resultado['resumo_sub_aspectos']
                        current_app.logger.info(f"Resumo de sub-aspectos adicionado à análise: {analise['resumo_sub_aspectos']}")
                    
                    # Salvar no banco de dados
                    current_app.data_handler.salvar_transcricao(texto, analise)
                    return redirect(url_for('historico'))
                except Exception as e:
                    current_app.logger.error(f"Erro ao processar feedback: {e}")
                    import traceback
                    traceback_str = traceback.format_exc()
                    current_app.logger.error(f"Detalhes do erro: {traceback_str}")
                    flash("Erro ao processar feedback. Por favor, tente novamente.")
            else:
                current_app.logger.warning("Tentativa de adicionar feedback vazio")
                flash("O texto do feedback não pode estar vazio.")
        return render_template('adicionar_feedback.html')

    @app.route('/dashboard')
    def dashboard():
        try:
            # Gerar gráficos e estatísticas para o dashboard
            current_app.logger.info("Iniciando geração de dashboard")
            dados_dashboard = current_app.data_handler.gerar_graficos()
            current_app.logger.info(f"Dados do dashboard gerados: {list(dados_dashboard.keys()) if isinstance(dados_dashboard, dict) else 'não é dicionário'}")
            
            # Verificar se ocorreu algum erro na geração dos gráficos
            if 'erro' in dados_dashboard:
                current_app.logger.error(f"Erro ao gerar dashboard: {dados_dashboard['erro']}")
                return render_template('dashboard.html', 
                                    dados_json=json.dumps({
                                        'graficos': {},
                                        'estatisticas': {},
                                        'erro': dados_dashboard['erro']
                                    }),
                                    graficos={}, 
                                    erro=dados_dashboard['erro'],
                                    estatisticas={})
            
            # Separar estatísticas para o dashboard
            estatisticas = dados_dashboard.pop('estatisticas_dashboard', {})
            current_app.logger.info(f"Estatísticas extraídas: {list(estatisticas.keys()) if isinstance(estatisticas, dict) else 'não é dicionário'}")
            
            # Verificar a presença e estrutura dos gráficos principais
            graficos_obrigatorios = ['sentimento_pie', 'sentimento_tempo', 'palavras_top']
            for chave in graficos_obrigatorios:
                if chave not in dados_dashboard:
                    current_app.logger.error(f"Gráfico obrigatório '{chave}' não encontrado nos dados")
                    return render_template('dashboard.html', 
                                        dados_json=json.dumps({
                                            'graficos': {},
                                            'estatisticas': {},
                                            'erro': f"Dados insuficientes para gráfico '{chave}'"
                                        }),
                                        graficos={}, 
                                        erro=f"Dados insuficientes para gráfico '{chave}'",
                                        estatisticas={})
                
                # Verificar estrutura do gráfico
                grafico = dados_dashboard[chave]
                if not isinstance(grafico, dict) or 'data' not in grafico or 'layout' not in grafico:
                    current_app.logger.error(f"Estrutura inválida para o gráfico '{chave}': {grafico}")
                    return render_template('dashboard.html', 
                                        dados_json=json.dumps({
                                            'graficos': {},
                                            'estatisticas': {},
                                            'erro': f"Estrutura inválida para o gráfico '{chave}'"
                                        }),
                                        graficos={}, 
                                        erro=f"Estrutura inválida para o gráfico '{chave}'",
                                        estatisticas={})
            
            # Log detalhado dos gráficos disponíveis para depuração
            current_app.logger.info(f"Gráficos disponíveis: {list(dados_dashboard.keys())}")
            
            # Preparar insights - pode ser personalizado com base nos dados
            insights = []
            
            # Verificar se temos dados de aspectos
            if 'aspectos' in estatisticas:
                current_app.logger.info(f"Aspectos encontrados: {estatisticas['aspectos']}")
                
                # Adicionar insight sobre aspecto mais mencionado
                if estatisticas['aspectos'].get('aspecto_mais_mencionado'):
                    insights.append({
                        'tipo': 'informacao',
                        'titulo': f'Aspecto mais mencionado: {estatisticas["aspectos"]["aspecto_mais_mencionado"].title()}',
                        'texto': f'O aspecto "{estatisticas["aspectos"]["aspecto_mais_mencionado"]}" é o mais mencionado nos feedbacks dos clientes.',
                        'classe': 'info'
                    })
            
            # Verificar distribuição de sentimentos
            if 'distribuicao_sentimentos' in estatisticas:
                current_app.logger.info(f"Distribuição de sentimentos: {estatisticas['distribuicao_sentimentos']}")
                
                # Calcular proporção de negativos
                perc_negativos = estatisticas['distribuicao_sentimentos'].get('negativo', {}).get('percentual', 0)
                if perc_negativos > 30:
                    insights.append({
                        'tipo': 'problema',
                        'titulo': 'Alta proporção de feedback negativo',
                        'texto': f'Há uma proporção elevada de feedbacks negativos ({perc_negativos}%). Recomendamos uma análise mais detalhada das causas.',
                        'classe': 'danger'
                    })
                elif perc_negativos < 15:
                    insights.append({
                        'tipo': 'destaque',
                        'titulo': 'Baixa proporção de feedback negativo',
                        'texto': f'A proporção de feedbacks negativos é baixa ({perc_negativos}%). Continue mantendo essa tendência positiva.',
                        'classe': 'success'
                    })
            
            # Informações do modelo
            info_modelo = {
                'usando_xlm_roberta': True,
                'data_atualizacao': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                'sarcasmo_habilitado': current_app.config.get('SARCASM_DETECTION_ENABLED', False)
            }
            
            # Gerar recomendações estratégicas baseadas nos dados reais
            recomendacoes = []
            
            # Verificar se temos dados suficientes para gerar recomendações significativas
            if not estatisticas or not isinstance(estatisticas, dict) or 'total_feedbacks' not in estatisticas or estatisticas['total_feedbacks'] < 5:
                current_app.logger.warning("Dados insuficientes para gerar recomendações estratégicas")
            else:
                current_app.logger.info("Gerando recomendações baseadas nos dados")
                
                # 1. Verificar distribuição de sentimentos e criar recomendações específicas
                if 'distribuicao_sentimentos' in estatisticas:
                    # Verificar sentimentos negativos
                    perc_negativos = estatisticas['distribuicao_sentimentos'].get('negativo', {}).get('percentual', 0)
                    qtd_negativos = estatisticas['distribuicao_sentimentos'].get('negativo', {}).get('count', 0)
                    
                    if perc_negativos > 25:
                        recomendacoes.append({
                            'titulo': 'Ação para reduzir feedback negativo',
                            'descricao': f'Implemente um plano de ação para investigar causas de insatisfação, focando nos {qtd_negativos} feedbacks negativos identificados.',
                            'prioridade_texto': 'Alta Prioridade',
                            'prioridade_classe': 'danger',
                            'status_texto': 'Pendente',
                            'status_classe': 'warning',
                            'icone': 'thumbs-down',
                            'mencoes': qtd_negativos,
                            'impacto': 'Médio a Alto',
                            'impacto_tipo': 'negativo'
                        })
                    
                    # Verificar sentimentos positivos
                    perc_positivos = estatisticas['distribuicao_sentimentos'].get('positivo', {}).get('percentual', 0)
                    qtd_positivos = estatisticas['distribuicao_sentimentos'].get('positivo', {}).get('count', 0)
                    
                    if perc_positivos > 50:
                        recomendacoes.append({
                            'titulo': 'Capitalizar pontos fortes',
                            'descricao': f'Analise os {qtd_positivos} feedbacks positivos para identificar pontos fortes e promover estes aspectos em campanhas de marketing.',
                            'prioridade_texto': 'Média',
                            'prioridade_classe': 'success',
                            'status_texto': 'Recomendado',
                            'status_classe': 'info',
                            'icone': 'star',
                            'mencoes': qtd_positivos,
                            'impacto': 'Positivo',
                            'impacto_tipo': 'comercial'
                        })
                
                # 2. Verificar aspectos específicos se disponíveis
                if 'aspectos' in estatisticas and estatisticas['aspectos'].get('aspecto_mais_mencionado'):
                    aspecto_principal = estatisticas['aspectos'].get('aspecto_mais_mencionado')
                    total_mencoes = estatisticas['aspectos'].get('total_analises_com_aspectos', 0)
                    
                    recomendacoes.append({
                        'titulo': f'Foco em {aspecto_principal.title()}',
                        'descricao': f'Elabore estratégia focada no aspecto "{aspecto_principal}" que foi identificado em {total_mencoes} feedbacks de clientes.',
                        'prioridade_texto': 'Alta',
                        'prioridade_classe': 'primary',
                        'status_texto': 'Proposto',
                        'status_classe': 'secondary',
                        'icone': 'bullseye',
                        'mencoes': total_mencoes,
                        'impacto': 'Alto',
                        'impacto_tipo': 'estratégico'
                    })
                
                # 3. Verificar tendências para recomendações de monitoramento
                if 'tendencia_feedbacks' in estatisticas:
                    direcao = estatisticas['tendencia_feedbacks'].get('direcao')
                    valor = estatisticas['tendencia_feedbacks'].get('valor', 0)
                    
                    if direcao == 'aumento' and valor > 15:
                        recomendacoes.append({
                            'titulo': 'Ajuste de capacidade de atendimento',
                            'descricao': f'Com aumento de {valor}% nos feedbacks, considere aumentar capacidade de resposta para manter qualidade do atendimento.',
                            'prioridade_texto': 'Média',
                            'prioridade_classe': 'info',
                            'status_texto': 'Em análise',
                            'status_classe': 'light',
                            'icone': 'chart-line',
                            'impacto_tipo': 'operacional'
                        })
                    elif direcao == 'queda' and valor < -15:
                        recomendacoes.append({
                            'titulo': 'Investigar diminuição de engajamento',
                            'descricao': f'Queda de {abs(valor)}% nos feedbacks pode indicar menor engajamento. Avalie canais de coleta e incentivos para feedback.',
                            'prioridade_texto': 'Média',
                            'prioridade_classe': 'warning',
                            'status_texto': 'Necessita atenção',
                            'status_classe': 'secondary',
                            'icone': 'arrow-down',
                            'impacto_tipo': 'engajamento'
                        })
                
                # 4. Recomendação sobre qualidade dos dados, se disponível
                if 'ponderacao' in estatisticas and 'concordancia_cliente_modelo' in estatisticas['ponderacao']:
                    concordancia = estatisticas['ponderacao']['concordancia_cliente_modelo']
                    
                    if concordancia < 70:
                        recomendacoes.append({
                            'titulo': 'Melhorar precisão da análise de sentimentos',
                            'descricao': f'Com taxa de concordância de apenas {concordancia}%, recomenda-se refinar o modelo de análise de sentimentos.',
                            'prioridade_texto': 'Média',
                            'prioridade_classe': 'warning',
                            'status_texto': 'Técnico',
                            'status_classe': 'dark',
                            'icone': 'tools',
                            'impacto_tipo': 'técnico'
                        })
            
            # Insights qualitativos baseados em padrões complexos nos dados
            insights_qualitativos = []
            if len(insights) >= 2:
                # Extrair insights qualitativos a partir dos insights regulares
                for i, insight in enumerate(insights):
                    if i < 3:  # Limitar a 3 insights qualitativos
                        icone = 'comment'
                        if insight['tipo'] == 'problema':
                            icone = 'exclamation-triangle'
                        elif insight['tipo'] == 'destaque':
                            icone = 'star'
                        elif insight['tipo'] == 'informacao':
                            icone = 'info-circle'
                            
                        insights_qualitativos.append({
                            'titulo': insight['titulo'],
                            'descricao': insight['texto'],
                            'classe': insight['classe'],
                            'icone': icone
                        })
            
            # Criar um objeto JSON com todos os dados para passar ao template
            dados_json = {
                'graficos': dados_dashboard,
                'estatisticas': estatisticas,
                'insights': insights,
                'recomendacoes': recomendacoes,
                'insights_qualitativos': insights_qualitativos,
                'modelo_info': info_modelo
            }
            
            # Converter o objeto dados_json para string JSON para debug
            try:
                json_string = json.dumps(dados_json)
                tamanho_json = len(json_string)
                current_app.logger.info(f"JSON de dados gerado com sucesso, tamanho: {tamanho_json} caracteres")
                
                # Verificar se o tamanho do JSON é razoável
                if tamanho_json > 5 * 1024 * 1024:  # Mais de 5MB
                    current_app.logger.warning(f"JSON gerado é muito grande ({tamanho_json/1024/1024:.2f} MB)")
            except Exception as json_err:
                current_app.logger.error(f"Erro ao converter dados para JSON: {json_err}")
                return render_template('dashboard.html', 
                                    dados_json=json.dumps({'erro': 'Erro ao serializar dados JSON'}),
                                    graficos={}, 
                                    erro="Erro ao serializar dados para JSON",
                                    estatisticas={})
                
            current_app.logger.info("Renderizando template do dashboard")
            
            # Renderizar o dashboard com todos os dados
            return render_template('dashboard.html', 
                                dados_json=json.dumps(dados_json),  # Passar como JSON string para o JavaScript
                                graficos=dados_dashboard, 
                                estatisticas=estatisticas,
                                insights=insights,
                                recomendacoes=recomendacoes,
                                insights_qualitativos=insights_qualitativos,
                                modelo_info=info_modelo)
        except Exception as e:
            current_app.logger.error(f"Erro na rota /dashboard: {e}")
            import traceback
            error_traceback = traceback.format_exc()
            current_app.logger.error(error_traceback)
            return render_template('dashboard.html', 
                                dados_json=json.dumps({'erro': str(e)}),
                                graficos={}, 
                                erro=f"Erro ao gerar dashboard: {str(e)}",
                                estatisticas={})

    @app.route('/dashboard-melhorado')
    def dashboard_melhorado():
        """
        Exibe o dashboard melhorado usando Dash
        """
        try:
            # Verificar se os arquivos JSON existem
            import os
            analises_path = os.path.join('data', 'analises.json')
            historico_path = os.path.join('data', 'historico.json')
            
            # Verificar se os arquivos existem
            if not os.path.exists(analises_path):
                current_app.logger.error(f"Arquivo de análises não encontrado: {analises_path}")
                return render_template('erro.html', 
                                    erro=f"Arquivo de análises não encontrado: {analises_path}. Por favor, adicione feedbacks primeiro.")
            
            if not os.path.exists(historico_path):
                current_app.logger.error(f"Arquivo de histórico não encontrado: {historico_path}")
                return render_template('erro.html', 
                                    erro=f"Arquivo de histórico não encontrado: {historico_path}. Por favor, adicione feedbacks primeiro.")
            
            # Verificar o conteúdo dos arquivos JSON
            try:
                import json
                with open(analises_path, 'r', encoding='utf-8') as f:
                    analises_content = f.read().strip()
                    if not analises_content:
                        current_app.logger.error(f"Arquivo de análises vazio: {analises_path}")
                        return render_template('erro.html', 
                                            erro=f"Arquivo de análises vazio. Por favor, adicione feedbacks primeiro.")
                
                with open(historico_path, 'r', encoding='utf-8') as f:
                    historico_content = f.read().strip()
                    if not historico_content:
                        current_app.logger.error(f"Arquivo de histórico vazio: {historico_path}")
                        return render_template('erro.html', 
                                            erro=f"Arquivo de histórico vazio. Por favor, adicione feedbacks primeiro.")
            except json.JSONDecodeError as e:
                current_app.logger.error(f"Erro ao decodificar JSON: {e}")
                return render_template('erro.html', 
                                    erro=f"Erro ao ler arquivos JSON: {str(e)}. Verifique se os arquivos estão formatados corretamente.")
            
            # Verificar se o Dash está disponível
            try:
                import dash
                current_app.logger.info("Dash encontrado, redirecionando para /dash/")
            except ImportError:
                current_app.logger.error("Dash não está instalado")
                return render_template('erro.html', 
                                    erro="Dashboard Dash não está disponível. Verifique a instalação do pacote dash.")
            
            # Redirecionar para o Dash app
            current_app.logger.info("Redirecionando para o dashboard Dash após validação dos arquivos JSON")
            return redirect('/dash/')
            
        except Exception as e:
            current_app.logger.error(f"Erro na rota /dashboard-melhorado: {e}")
            import traceback
            error_traceback = traceback.format_exc()
            current_app.logger.error(error_traceback)
            return render_template('erro.html', 
                                erro=f"Erro ao acessar dashboard melhorado: {str(e)}")

    # Rotas para componentes do Dash já foram definidas na inicialização
    # Basta manter a rota para assets genéricos
    @app.route('/_dash-<path:asset_path>')
    def dash_static(asset_path):
        try:
            # Rotas específicas de assets
            if asset_path.startswith('assets/'):
                # Corrigir problema com assets
                asset_filename = asset_path.replace('assets/', '')
                return send_from_directory('static', asset_filename)
            return send_from_directory('static', asset_path.split('/')[-1])
        except Exception as e:
            current_app.logger.error(f"Erro ao servir arquivo estático do Dash: {e}")
            return "", 404

    @app.route('/api/reavaliar-historico', methods=['POST'])
    def reavaliar_historico():
        if not current_app.config.get('SARCASM_DETECTION_ENABLED'):
            return jsonify({'sucesso':False,'erro':'Detector indisponível'}),503
        hist = current_app.data_handler.carregar_historico()
        atualizados = 0
        
        # Importar função de análise de sarcasmo
        from models.sarcasm_integration import analisar_texto_com_sarcasmo
        
        for reg in hist:
            texto = reg.get('texto','')
            if texto:
                # Analisar com detecção de sarcasmo
                analise = analisar_texto_com_sarcasmo(
                    texto,
                    current_app.sentiment_analyzer,
                    incluir_detalhes=True
                )
                # Atualizar análise no registro
                reg['analise'] = analise
                atualizados += 1
                
        # Salvar histórico atualizado
        with open(current_app.data_handler.arquivo_historico,'w',encoding='utf-8') as f:
            json.dump(hist,f,ensure_ascii=False,indent=2)
            
        current_app.logger.info(f"Reavaliação de histórico concluída: {atualizados} registros atualizados com sarcasmo")
        return jsonify({'sucesso':True,'atualizados':atualizados})

    @app.route('/ouvir', methods=['POST'])
    def ouvir():
        try:
            # Verificar se é uma solicitação de parada manual
            if request.is_json and request.get_json().get('manual_stop', False):
                current_app.logger.info("Recebida solicitação de parada manual da gravação")
                return jsonify({
                    'texto': "Gravação finalizada manualmente. A transcrição não está disponível.",
                    'sentimento': 'neutro',
                    'compound': 0,
                    'success': True,
                    'motivo_parada': 'parada_manual',
                    'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                })
                
            resultado = current_app.speech_handler.ouvir_microfone()
            
            # Verificar se houve sucesso na operação
            if not resultado.get('sucesso', False):
                # Retornar erro específico
                erro_mensagem = resultado.get('erro', 'Erro desconhecido na captura de áudio')
                return jsonify({
                    'texto': erro_mensagem,
                    'sentimento': 'neutro',
                    'success': False,
                    'motivo_parada': resultado.get('codigo_erro', 'erro'),
                    'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                })
            
            # Extrair texto e análise do resultado
            texto = resultado.get('texto', '')
            analise = resultado.get('analise', {})
            
            # Verificar se o texto é válido antes de processar
            mensagens_para_ignorar = [
                "Para usar a gravação de voz",
                "Não entendi o que você disse",
                "Erro ao inicializar microfone",
                "Erro ao solicitar resultados",
                "Erro inesperado",
                "Não detectei nenhuma fala"
            ]
            
            # Verificar se o sistema de sarcasmo está habilitado e se o texto é válido
            sarcasmo_habilitado = current_app.config.get('SARCASM_DETECTION_ENABLED', False)
            texto_valido = not any(texto.startswith(msg) for msg in mensagens_para_ignorar)
            
            if texto_valido:
                # Aplicar análise melhorada com tópicos
                try:
                    # Verificar se temos o método de análise completa disponível
                    if hasattr(current_app.sentiment_analyzer, 'analisar_sentimento_final'):
                        analise = current_app.sentiment_analyzer.analisar_sentimento_final(texto)
                        current_app.logger.info("Aplicada análise completa com extração de tópicos")
                    elif sarcasmo_habilitado:
                        # Usar análise com sarcasmo se disponível
                        from models.sarcasm_integration import analisar_texto_com_sarcasmo
                        analise = analisar_texto_com_sarcasmo(
                            texto,
                            current_app.sentiment_analyzer,
                            incluir_detalhes=True
                        )
                        current_app.logger.info("Análise de sarcasmo aplicada ao texto transcrito")
                        
                        # Adicionar tópicos se não existirem
                        if 'topicos' not in analise and hasattr(current_app.sentiment_analyzer, 'extrair_palavras_chave'):
                            analise['topicos'] = current_app.sentiment_analyzer.extrair_palavras_chave(texto)
                    
                    # Extrair aspectos do texto e adicionar à análise
                    aspectos_resultado = current_app.aspect_extractor.extrair_aspectos(texto, analise)
                    current_app.logger.info(f"Aspectos extraídos: {aspectos_resultado}")
                    
                    # Criar objeto de aspectos formatado para armazenamento
                    if 'aspectos_encontrados' in aspectos_resultado and aspectos_resultado['aspectos_encontrados']:
                        # Determinar aspecto principal (com mais menções)
                        aspectos_ordenados = sorted(
                            aspectos_resultado['aspectos_encontrados'].items(),
                            key=lambda x: x[1]['mencoes'],
                            reverse=True
                        )
                        
                        aspecto_principal = aspectos_ordenados[0][0] if aspectos_ordenados else None
                        aspects_detected = list(aspectos_resultado['aspectos_encontrados'].keys())
                        
                        # Adicionar à análise
                        if 'aspectos' not in analise:
                            analise['aspectos'] = {}
                        
                        analise['aspectos']['summary'] = {
                            'primary_aspect': aspecto_principal,
                            'aspects_detected': aspects_detected,
                            'total_mentions': aspectos_resultado['mencoes_totais']
                        }
                        
                        # Adicionar detalhes completos
                        analise['aspectos']['details'] = aspectos_resultado['aspectos_encontrados']
                        
                        current_app.logger.info(f"Aspecto principal: {aspecto_principal}, Aspectos detectados: {aspects_detected}")
                        
                        # Adicionar campos diretos para facilitar acesso no frontend
                        analise['aspecto_principal'] = aspecto_principal
                        analise['aspectos_detectados'] = aspects_detected
                    else:
                        current_app.logger.info("Nenhum aspecto relevante encontrado no texto")
                    
                    # CORREÇÃO: Adicionar sub-aspectos e resumo à análise
                    if 'sub_aspectos' in aspectos_resultado:
                        analise['sub_aspectos'] = aspectos_resultado['sub_aspectos']
                        current_app.logger.info(f"Sub-aspectos adicionados à análise: {analise['sub_aspectos']}")
                    
                    if 'resumo_sub_aspectos' in aspectos_resultado:
                        analise['resumo_sub_aspectos'] = aspectos_resultado['resumo_sub_aspectos']
                        current_app.logger.info(f"Resumo de sub-aspectos adicionado à análise: {analise['resumo_sub_aspectos']}")
                    
                    # Salvar no histórico
                    current_app.data_handler.salvar_transcricao(texto, analise)
                except Exception as e:
                    current_app.logger.error(f"Erro ao processar análise melhorada: {e}")
            
            # Preparar resposta com aspectos
            aspectos = {}
            if 'aspectos' in analise and 'summary' in analise['aspectos']:
                aspectos = {
                    'principal': analise['aspectos']['summary'].get('primary_aspect'),
                    'detectados': analise['aspectos']['summary'].get('aspects_detected', []),
                    'mencoes_totais': analise['aspectos']['summary'].get('total_mentions', 0)
                }
            
            return jsonify({
                'texto': texto, 
                'sentimento': analise.get('sentimento', 'neutro'),
                'compound': analise.get('compound', 0),
                'success': True,
                'motivo_parada': 'silencio_detectado' if texto != "Não detectei nenhuma fala. Por favor, tente novamente falando mais alto." else 'sem_fala',
                'sarcasmo': analise.get('sarcasmo', {}),
                'topicos': analise.get('topicos', []),
                'aspectos': aspectos,
                'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            })
        except Exception as e:
            current_app.logger.error(f"Erro na rota /ouvir: {e}")
            return jsonify({
                'texto': f'Erro ao processar áudio: {str(e)}', 
                'sentimento': 'neutro',
                'success': False,
                'motivo_parada': 'erro',
                'error_details': str(e),
                'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            })
            
    @app.route('/api/salvar_historico', methods=['POST'])
    def salvar_historico_api():
        """
        Endpoint para salvar feedback diretamente no histórico e CSV
        """
        try:
            data = request.get_json()
            if not data:
                return jsonify({"success": False, "message": "Dados não fornecidos"}), 400
            
            texto = data.get('feedback', {}).get('texto')
            analise = data.get('analise', {})
            
            if not texto or not analise:
                return jsonify({"success": False, "message": "Feedback ou análise ausente"}), 400
            
            # Verificar se o sistema de sarcasmo está habilitado e se análise deve ser refeita
            sarcasmo_habilitado = current_app.config.get('SARCASM_DETECTION_ENABLED', False)
            
            if sarcasmo_habilitado and data.get('analisar_sarcasmo', True):
                # Usar a função analisar_texto_com_sarcasmo para análise integrada
                from models.sarcasm_integration import analisar_texto_com_sarcasmo
                analise = analisar_texto_com_sarcasmo(
                    texto,
                    current_app.sentiment_analyzer,
                    incluir_detalhes=True
                )
                current_app.logger.info(f"Análise de sarcasmo integrada aplicada via API")
            
            # Salvar no histórico e CSV usando a função existente
            current_app.data_handler.salvar_transcricao(texto, analise)
            
            return jsonify({
                "success": True,
                "message": "Feedback salvo com sucesso",
                "analise": analise,
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            })
        except Exception as e:
            current_app.logger.error(f"Erro ao salvar feedback no histórico: {e}")
            return jsonify({"success": False, "message": str(e)}), 500

    @app.route('/api/sincronizar_feedbacks', methods=['POST'])
    def sincronizar_feedbacks():
        """
        Endpoint para sincronizar feedbacks salvos localmente
        """
        try:
            data = request.get_json()
            if not data or 'feedbacks' not in data:
                return jsonify({"success": False, "message": "Dados inválidos"}), 400
            
            feedbacks = data.get('feedbacks', [])
            
            if not feedbacks or not isinstance(feedbacks, list):
                return jsonify({"success": False, "message": "Lista de feedbacks inválida"}), 400
            
            # Verificar se o sistema de sarcasmo está habilitado
            sarcasmo_habilitado = current_app.config.get('SARCASM_DETECTION_ENABLED', False)
            analisar_com_sarcasmo = data.get('analisar_sarcasmo', True)
            
            if sarcasmo_habilitado and analisar_com_sarcasmo:
                from models.sarcasm_integration import analisar_texto_com_sarcasmo
            
            sucessos = 0
            falhas = 0
            
            for item in feedbacks:
                try:
                    feedback = item.get('feedback')
                    analise = item.get('analise')
                    
                    if feedback and isinstance(feedback, str):
                        # Se temos o texto do feedback mas não a análise, ou se devemos reanalisar com sarcasmo
                        if not analise or (sarcasmo_habilitado and analisar_com_sarcasmo):
                            analise = analisar_texto_com_sarcasmo(
                                feedback,
                                current_app.sentiment_analyzer,
                                incluir_detalhes=True
                            )
                            current_app.logger.info(f"Feedback sincronizado analisado com sarcasmo")
                        
                        # Salvar no histórico
                        current_app.data_handler.salvar_transcricao(feedback, analise)
                        sucessos += 1
                    else:
                        falhas += 1
                except Exception as e:
                    current_app.logger.error(f"Erro ao processar feedback na sincronização: {e}")
                    falhas += 1
            
            return jsonify({
                "success": True,
                "message": f"Sincronização concluída: {sucessos} sucessos, {falhas} falhas",
                "sucessos": sucessos,
                "falhas": falhas,
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            })
        except Exception as e:
            current_app.logger.error(f"Erro na sincronização de feedbacks: {e}")
            return jsonify({"success": False, "message": str(e)}), 500

    @app.route('/baixar-modelo', methods=['GET'])
    def baixar_modelo():
        try:
            current_app.logger.info("=" * 50)
            current_app.logger.info("Iniciando procedimento de download do modelo XLM-RoBERTa...")
            current_app.logger.info("=" * 50)
            
            # Importar as bibliotecas necessárias
            try:
                current_app.logger.info("Verificando disponibilidade das bibliotecas necessárias...")
                from transformers import AutoModelForSequenceClassification, AutoTokenizer
                from huggingface_hub import snapshot_download
                current_app.logger.info("Bibliotecas transformers e huggingface_hub importadas com sucesso")
            except ImportError as e:
                current_app.logger.error(f"Erro ao importar dependências: {e}")
                return jsonify({
                    "success": False,
                    "message": "Erro ao importar bibliotecas necessárias. Verifique se o transformers e huggingface_hub estão instalados."
                }), 500
            
            # Nome do modelo
            model_name = "cardiffnlp/twitter-xlm-roberta-base-sentiment"
            
            # Diretório para salvar o modelo
            model_dir = os.path.join(os.getcwd(), "models", "cardiffnlp-xlm-roberta")
            os.makedirs(model_dir, exist_ok=True)
            
            current_app.logger.info(f"Diretório do modelo: {model_dir}")
            current_app.logger.info(f"Verificando permissões de escrita...")
            
            # Verificar permissões de escrita
            try:
                test_file = os.path.join(model_dir, 'test_write.txt')
                with open(test_file, 'w') as f:
                    f.write('test')
                os.unlink(test_file)
                current_app.logger.info("Permissões de escrita OK no diretório do modelo")
            except Exception as perm_error:
                current_app.logger.error(f"Problema de permissão de escrita no diretório do modelo: {perm_error}")
                return jsonify({
                    "success": False,
                    "message": f"Erro de permissão: Não foi possível escrever no diretório {model_dir}. Verifique as permissões."
                }), 500
            
            current_app.logger.info(f"Iniciando download do modelo {model_name} para {model_dir}")
            
            # Verificar se há conexão com a internet
            try:
                current_app.logger.info("Verificando conexão com a internet...")
                import socket
                socket.create_connection(("huggingface.co", 443), timeout=5)
                current_app.logger.info("Conexão com huggingface.co estabelecida com sucesso")
            except Exception as e:
                current_app.logger.error(f"Erro ao verificar conexão com a internet: {e}")
                return jsonify({
                    "success": False,
                    "message": "Não foi possível conectar ao Hugging Face. Verifique sua conexão com a internet."
                }), 500
            
            # Primeiro método: download via Hugging Face Hub
            try:
                current_app.logger.info("MÉTODO 1: Download via snapshot_download")
                # Baixar arquivos do modelo
                snapshot_download(
                    repo_id=model_name,
                    local_dir=model_dir,
                    local_dir_use_symlinks=False
                )
                current_app.logger.info("Modelo baixado com sucesso via snapshot_download")
                
                # Carregar tokenizador e modelo para verificar se estão funcionando
                current_app.logger.info("Verificando tokenizador...")
                tokenizer = AutoTokenizer.from_pretrained(model_dir)
                current_app.logger.info("Tokenizador carregado com sucesso")
                
                current_app.logger.info("Verificando modelo...")
                model = AutoModelForSequenceClassification.from_pretrained(model_dir)
                current_app.logger.info("Modelo carregado com sucesso")
                
                # Reiniciar o analisador de sentimento para usar o novo modelo
                current_app.logger.info("Reiniciando analisador de sentimento com o novo modelo...")
                current_app.sentiment_analyzer = SentimentAnalyzer()
                current_app.speech_handler = SpeechHandler(current_app.sentiment_analyzer)
                
                current_app.logger.info("XLM-RoBERTa configurado e pronto para uso")
                return jsonify({
                    "success": True,
                    "message": "Modelo XLM-RoBERTa baixado e carregado com sucesso",
                    "using_xlm_roberta": True
                })
                
            except Exception as e:
                current_app.logger.error(f"Erro ao baixar/carregar modelo (método 1): {e}")
                current_app.logger.error(traceback.format_exc())
                
                # Tentar método alternativo
                try:
                    current_app.logger.info("=" * 50)
                    current_app.logger.info("MÉTODO 2: Tentando método alternativo - download direto via AutoTokenizer/AutoModel...")
                    
                    # Baixar e salvar tokenizador
                    current_app.logger.info("Baixando tokenizador...")
                    tokenizer = AutoTokenizer.from_pretrained(model_name)
                    tokenizer.save_pretrained(model_dir)
                    current_app.logger.info("Tokenizador baixado e salvo com sucesso")
                    
                    # Baixar e salvar modelo
                    current_app.logger.info("Baixando modelo...")
                    model = AutoModelForSequenceClassification.from_pretrained(model_name)
                    model.save_pretrained(model_dir)
                    current_app.logger.info("Modelo baixado e salvo com sucesso")
                    
                    # Reiniciar o analisador de sentimento
                    current_app.logger.info("Reiniciando analisador de sentimento com o novo modelo...")
                    current_app.sentiment_analyzer = SentimentAnalyzer()
                    current_app.speech_handler = SpeechHandler(current_app.sentiment_analyzer)
                    
                    current_app.logger.info("XLM-RoBERTa configurado e pronto para uso (método alternativo)")
                    return jsonify({
                        "success": True,
                        "message": "Modelo XLM-RoBERTa baixado e carregado com sucesso (método alternativo)",
                        "using_xlm_roberta": True
                    })
                    
                except Exception as e2:
                    current_app.logger.error(f"Erro no método alternativo: {e2}")
                    current_app.logger.error(traceback.format_exc())
                    return jsonify({
                        "success": False,
                        "message": f"Falha ao baixar modelo: {str(e2)}",
                        "error_details": traceback.format_exc()
                    }), 500
                
        except Exception as e:
            current_app.logger.error(f"Erro geral ao baixar modelo: {e}")
            current_app.logger.error(traceback.format_exc())
            return jsonify({
                "success": False, 
                "message": str(e),
                "error_details": traceback.format_exc()
            }), 500
            
    @app.route('/reiniciar-modelo', methods=['GET'])
    def reiniciar_modelo():
        try:
            current_app.logger.info("=" * 50)
            current_app.logger.info("Reiniciando modelo de análise de sentimento...")
            current_app.logger.info("=" * 50)
            
            # Salvar o estado anterior para comparação
            xlm_roberta_anterior = False
            if hasattr(current_app, 'sentiment_analyzer'):
                xlm_roberta_anterior = getattr(current_app.sentiment_analyzer, 'using_xlm_roberta', False)
                current_app.logger.info(f"Estado anterior do XLM-RoBERTa: {xlm_roberta_anterior}")
            
            # Reiniciar o analisador de sentimento
            current_app.sentiment_analyzer = SentimentAnalyzer()
            current_app.logger.info("SentimentAnalyzer reiniciado")
            
            # Verificar o estado atual
            xlm_roberta_atual = getattr(current_app.sentiment_analyzer, 'using_xlm_roberta', False)
            current_app.logger.info(f"Estado atual do XLM-RoBERTa: {xlm_roberta_atual}")
            
            # Forçar verificação explícita do status
            if hasattr(current_app.sentiment_analyzer, '_verificar_status_xlm_roberta'):
                current_app.logger.info("Realizando verificação explícita do status do XLM-RoBERTa...")
                status_xlm = current_app.sentiment_analyzer._verificar_status_xlm_roberta()
                current_app.logger.info(f"Resultado da verificação: {status_xlm}")
            
            # Obter detalhes de status se o método existir
            status_detalhado = {}
            if hasattr(current_app.sentiment_analyzer, 'status_xlm_roberta'):
                status_detalhado = current_app.sentiment_analyzer.status_xlm_roberta()
                current_app.logger.info(f"Status detalhado do XLM-RoBERTa: {status_detalhado}")
            
            # Atualizar outros componentes
            current_app.speech_handler = SpeechHandler(current_app.sentiment_analyzer)
            current_app.logger.info("SpeechHandler reiniciado")
            
            current_app.aspect_extractor = AspectExtractor()
            current_app.logger.info("AspectExtractor reiniciado")
            
            # Tratamento cuidadoso para a integração do detector de sarcasmo
            try:
                # Verificar se o detector de sarcasmo está disponível
                from models.sarcasm_factory import integrar_detector_sarcasmo_ao_sistema_melhorado
                integrar_detector_sarcasmo_ao_sistema_melhorado(
                    current_app,
                    current_app.sentiment_analyzer,
                    current_app.speech_handler,
                    current_app.data_handler
                )
                current_app.logger.info("Detector de sarcasmo reiniciado")
            except Exception as e:
                current_app.logger.warning(f"Não foi possível reintegrar o detector de sarcasmo: {e}")
            
            current_app.logger.info("Reinicialização completa!")
            
            # Preparar resposta com informações detalhadas
            resposta = {
                "success": True,
                "message": "Modelo reiniciado com sucesso",
                "xlm_roberta": {
                    "antes": xlm_roberta_anterior,
                    "depois": xlm_roberta_atual,
                    "status_detalhado": status_detalhado
                },
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            }
            
            return jsonify(resposta)
        except Exception as e:
            current_app.logger.error(f"Erro ao reiniciar modelo: {e}")
            current_app.logger.error(traceback.format_exc())
            return jsonify({
                "success": False, 
                "message": str(e),
                "error_details": traceback.format_exc()
            }), 500
            
    @app.route('/relatorio-ponderado', methods=['GET'])
    def gerar_relatorio_ponderado():
        """Endpoint para gerar e visualizar relatório de análise ponderada"""
        try:
            # Pesos padrão (pode ser alterado via parâmetros de URL)
            peso_cliente = int(request.args.get('peso_cliente', 30))
            peso_modelo = int(request.args.get('peso_modelo', 80))
            
            # Gerar relatório
            df_relatorio = current_app.data_handler.gerar_relatorio_analise_ponderada(
                peso_cliente=peso_cliente,
                peso_modelo=peso_modelo
            )
            
            if df_relatorio is None or df_relatorio.empty:
                return render_template('erro.html', erro="Não foi possível gerar o relatório ponderado. Verifique os logs."), 500
            
            # Verificar se arquivo de estatísticas foi gerado
            if not os.path.exists('data/estatisticas_ponderacao.json'):
                current_app.logger.warning("Arquivo de estatísticas da ponderação não encontrado")
                estatisticas = {}
            else:
                try:
                    with open('data/estatisticas_ponderacao.json', 'r', encoding='utf-8') as f:
                        estatisticas = json.load(f)
                except Exception as e:
                    current_app.logger.error(f"Erro ao carregar estatísticas de ponderação: {e}")
                    estatisticas = {}
            
            # Renderizar template com dados
            return render_template(
                'relatorio_ponderado.html',
                pesos={
                    'cliente': peso_cliente,
                    'modelo': peso_modelo
                },
                estatisticas=estatisticas,
                total_registros=len(df_relatorio),
                concordancias=estatisticas.get('concordancias', 0),
                discordancias=estatisticas.get('discordancias', 0)
            )
        except Exception as e:
            current_app.logger.error(f"Erro ao gerar relatório ponderado: {e}")
            import traceback
            current_app.logger.error(traceback.format_exc())
            return render_template('erro.html', erro=f"Erro ao processar relatório ponderado: {e}"), 500
    
    @app.route('/api/relatorio-ponderado', methods=['GET'])
    def api_relatorio_ponderado():
        """API para obter dados do relatório ponderado"""
        try:
            # Verificar se arquivo existe
            if not os.path.exists('data/relatorio_ponderado.csv'):
                return jsonify({"success": False, "message": "Relatório não encontrado"}), 404
                
            # Carregar dados
            df_relatorio = pd.read_csv('data/relatorio_ponderado.csv')
            
            # Carregar estatísticas
            if os.path.exists('data/estatisticas_ponderacao.json'):
                with open('data/estatisticas_ponderacao.json', 'r', encoding='utf-8') as f:
                    estatisticas = json.load(f)
            else:
                estatisticas = {}
                
            # Preparar dados para resposta
            registros = df_relatorio.to_dict('records')
            
            return jsonify({
                "success": True,
                "data": {
                    "registros": registros,
                    "estatisticas": estatisticas,
                    "total": len(registros)
                }
            })
        except Exception as e:
            current_app.logger.error(f"Erro na API de relatório ponderado: {e}")
            return jsonify({"success": False, "message": str(e)}), 500
            
    @app.route('/analise-aspectos', methods=['GET'])
    def visualizar_aspectos():
        """Endpoint para visualizar estatísticas de aspectos"""
        try:
            # Gerar estatísticas atualizadas
            estatisticas = current_app.data_handler.gerar_estatisticas_aspectos()
            
            if not estatisticas:
                return render_template('erro.html', erro="Não foi possível gerar estatísticas de aspectos. Verifique os logs."), 500
            
            # Preparar dados para gráficos
            aspectos = list(estatisticas["contagem_por_aspecto"].keys())
            contagens = list(estatisticas["contagem_por_aspecto"].values())
            
            # Gráfico de distribuição de aspectos
            fig_distribuicao = go.Figure(data=[go.Pie(
                labels=aspectos,
                values=contagens,
                hole=.3
            )])
            fig_distribuicao.update_layout(title_text='Distribuição de Aspectos Mencionados', height=400)
            
            # Gráfico de sentimentos por aspecto
            sentimentos_data = []
            for aspecto in aspectos:
                positivos = estatisticas["aspectos_positivos"].get(aspecto, 0)
                neutros = estatisticas["aspectos_neutros"].get(aspecto, 0)
                negativos = estatisticas["aspectos_negativos"].get(aspecto, 0)
                
                sentimentos_data.append({
                    "aspecto": aspecto,
                    "positivos": positivos,
                    "neutros": neutros,
                    "negativos": negativos
                })
            
            # Retornar template com dados
            return render_template(
                'analise_aspectos.html',
                estatisticas=estatisticas,
                grafico_distribuicao=json.dumps(fig_distribuicao.to_dict()),
                sentimentos_data=sentimentos_data
            )
        except Exception as e:
            current_app.logger.error(f"Erro ao gerar visualização de aspectos: {e}")
            current_app.logger.error(traceback.format_exc())
            return render_template('erro.html', erro=f"Erro ao processar aspectos: {e}"), 500

    @app.route('/feedback')
    def feedback():
        """Exibe o formulário de feedback detalhado"""
        try:
            # Verificar se o recurso de sarcasmo está habilitado para informar ao template
            sarcasmo_habilitado = current_app.config.get('SARCASM_DETECTION_ENABLED', False)
            return render_template('feedback.html', sarcasmo_habilitado=sarcasmo_habilitado)
        except Exception as e:
            current_app.logger.error(f"Erro na rota /feedback: {e}")
            return render_template('erro.html', erro=f"Erro ao carregar página de feedback: {e}"), 500

    @app.route('/api/feedback', methods=['POST'])
    def feedback_api():
        """API para registrar feedback do usuário com análise de texto"""
        try:
            data = request.get_json()
            texto = data.get('texto', '').strip()
            
            if not texto:
                current_app.logger.warning("Tentativa de enviar feedback com texto vazio")
                return jsonify({
                    "sucesso": False,
                    "erro": "Texto não pode ser vazio"
                }), 400
            
            # Verificar se o sistema de sarcasmo está habilitado
            sarcasmo_habilitado = current_app.config.get('SARCASM_DETECTION_ENABLED', False)
            
            # Realizar análise de sentimento com sarcasmo se disponível
            if sarcasmo_habilitado:
                from models.sarcasm_integration import analisar_texto_com_sarcasmo
                analise = analisar_texto_com_sarcasmo(
                    texto,
                    current_app.sentiment_analyzer,
                    incluir_detalhes=True
                )
                current_app.logger.info("Análise de sarcasmo aplicada ao feedback")
            else:
                # Verificar se temos o método de análise completa disponível
                if hasattr(current_app.sentiment_analyzer, 'analisar_sentimento_final'):
                    analise = current_app.sentiment_analyzer.analisar_sentimento_final(texto)
                    current_app.logger.info("Análise completa com extração de tópicos aplicada ao feedback")
                else:
                    analise = current_app.sentiment_analyzer.analisar_sentimento(texto)
            
            # Extrair tópicos se não existirem na análise
            if 'topicos' not in analise and hasattr(current_app.sentiment_analyzer, 'extrair_palavras_chave'):
                analise['topicos'] = current_app.sentiment_analyzer.extrair_palavras_chave(texto)
                current_app.logger.info(f"Tópicos extraídos: {analise['topicos']}")
            
            # Extrair aspectos do texto e adicionar à análise
            aspectos_resultado = current_app.aspect_extractor.extrair_aspectos(texto, analise)
            current_app.logger.info(f"Aspectos extraídos: {aspectos_resultado}")
            
            # Criar objeto de aspectos formatado para armazenamento
            if 'aspectos_encontrados' in aspectos_resultado and aspectos_resultado['aspectos_encontrados']:
                # Determinar aspecto principal (com mais menções)
                aspectos_ordenados = sorted(
                    aspectos_resultado['aspectos_encontrados'].items(),
                    key=lambda x: x[1]['mencoes'],
                    reverse=True
                )
                
                aspecto_principal = aspectos_ordenados[0][0] if aspectos_ordenados else None
                aspects_detected = list(aspectos_resultado['aspectos_encontrados'].keys())
                
                # Adicionar à análise
                if 'aspectos' not in analise:
                    analise['aspectos'] = {}
                
                analise['aspectos']['summary'] = {
                    'primary_aspect': aspecto_principal,
                    'aspects_detected': aspects_detected,
                    'total_mentions': aspectos_resultado['mencoes_totais']
                }
                
                # Adicionar detalhes completos
                analise['aspectos']['details'] = aspectos_resultado['aspectos_encontrados']
                
                # Adicionar campos diretos para facilitar acesso no frontend
                analise['aspecto_principal'] = aspecto_principal
                analise['aspectos_detectados'] = aspects_detected
                
                current_app.logger.info(f"Aspecto principal: {aspecto_principal}, Aspectos detectados: {aspects_detected}")
            else:
                current_app.logger.info("Nenhum aspecto relevante encontrado no texto")
            
            # CORREÇÃO: Adicionar sub-aspectos e resumo à análise
            if 'sub_aspectos' in aspectos_resultado:
                analise['sub_aspectos'] = aspectos_resultado['sub_aspectos']
                current_app.logger.info(f"Sub-aspectos adicionados à análise: {analise['sub_aspectos']}")
            
            if 'resumo_sub_aspectos' in aspectos_resultado:
                analise['resumo_sub_aspectos'] = aspectos_resultado['resumo_sub_aspectos']
                current_app.logger.info(f"Resumo de sub-aspectos adicionado à análise: {analise['resumo_sub_aspectos']}")
            
            # Capturar dados do usuário
            avaliacao_estrelas = data.get('rating', 0)
            emocao_usuario = data.get('emotion', '')
            tags_usuario = data.get('tags', [])
            origem = data.get('origem', 'feedback_form')  # Identificar origem da solicitação
            
            # Verificar se há inconsistência entre emoção selecionada e análise de texto
            sentimento_texto = analise.get('sentimento', 'neutro')
            sentimento_original = sentimento_texto
            confianca_original = analise.get('confianca', 0.5)
            
            # Mapear emoções para sentimentos
            mapa_emocao_sentimento = {
                'Muito satisfeito': 'positivo',
                'Satisfeito': 'positivo',
                'Neutro': 'neutro',
                'Insatisfeito': 'negativo',
                'Muito insatisfeito': 'negativo'
            }
            
            # Verificar inconsistência entre avaliação por estrelas e emoção
            sentimento_estrelas = 'neutro'
            if avaliacao_estrelas >= 4:
                sentimento_estrelas = 'positivo'
            elif avaliacao_estrelas <= 2:
                sentimento_estrelas = 'negativo'
                
            # Determinar sentimento pela emoção selecionada
            sentimento_emocao = mapa_emocao_sentimento.get(emocao_usuario, 'neutro')
            
            # Priorizar o sentimento do texto quando há inconsistência clara
            # Caso 1: Texto positivo + avaliação alta, mas emoção negativa
            if sentimento_texto == 'positivo' and sentimento_estrelas == 'positivo' and sentimento_emocao == 'negativo':
                current_app.logger.info(f"Inconsistência detectada: texto positivo ({confianca_original:.2f}) com emoção negativa.")
                # Manter o sentimento do texto, mas registrar a inconsistência
                analise['inconsistencia_detectada'] = True
                analise['detalhes_inconsistencia'] = {
                    'sentimento_texto': sentimento_texto,
                    'sentimento_emocao': sentimento_emocao,
                    'sentimento_estrelas': sentimento_estrelas
                }
                # Garantir que o sentimento final seja positivo (do texto)
                analise['sentimento'] = sentimento_texto
                analise['sentimento_final_ajustado'] = True
                analise['motivo_ajuste'] = 'inconsistência_emoção_vs_texto_estrelas'
                current_app.logger.info(f"Ajustando sentimento final para '{sentimento_texto}' devido à consistência entre texto e avaliação por estrelas")
                
            # Caso 2: Texto negativo + avaliação baixa, mas emoção positiva
            elif sentimento_texto == 'negativo' and sentimento_estrelas == 'negativo' and sentimento_emocao == 'positivo':
                current_app.logger.info(f"Inconsistência detectada: texto negativo ({confianca_original:.2f}) com emoção positiva.")
                # Manter o sentimento do texto, mas registrar a inconsistência
                analise['inconsistencia_detectada'] = True
                analise['detalhes_inconsistencia'] = {
                    'sentimento_texto': sentimento_texto,
                    'sentimento_emocao': sentimento_emocao,
                    'sentimento_estrelas': sentimento_estrelas
                }
                # Garantir que o sentimento final seja negativo (do texto)
                analise['sentimento'] = sentimento_texto
                analise['sentimento_final_ajustado'] = True
                analise['motivo_ajuste'] = 'inconsistência_emoção_vs_texto_estrelas'
                current_app.logger.info(f"Ajustando sentimento final para '{sentimento_texto}' devido à consistência entre texto e avaliação por estrelas")
                
            # Caso 3: Texto e estrelas concordam, emoção é diferente
            elif sentimento_texto == sentimento_estrelas and sentimento_texto != sentimento_emocao:
                current_app.logger.info(f"Inconsistência: texto e estrelas ({sentimento_texto}) concordam, mas emoção difere ({sentimento_emocao})")
                # Priorizar texto+estrelas sobre emoção
                analise['inconsistencia_detectada'] = True
                analise['detalhes_inconsistencia'] = {
                    'sentimento_texto': sentimento_texto,
                    'sentimento_emocao': sentimento_emocao,
                    'sentimento_estrelas': sentimento_estrelas,
                    'decisão': 'priorizar_texto_estrelas'
                }
                # Garantir que o sentimento final seja o do texto
                analise['sentimento'] = sentimento_texto
                analise['sentimento_final_ajustado'] = True
                analise['motivo_ajuste'] = 'texto_estrelas_vs_emoção'
                current_app.logger.info(f"Ajustando sentimento final para '{sentimento_texto}' devido à consistência entre texto e avaliação por estrelas")
                
            # Caso 4: Outros casos de inconsistência - usar uma ponderação mais balanceada
            elif sentimento_texto != sentimento_emocao:
                # Dar mais peso ao texto (90%) e menos à emoção selecionada (10%)
                peso_texto = 0.90
                peso_emocao = 0.10
                
                # Registrar detalhes para análise futura
                analise['ponderacao_aplicada'] = True
                analise['pesos_utilizados'] = {
                    'peso_texto': peso_texto,
                    'peso_emocao': peso_emocao
                }
                
                current_app.logger.info(f"Aplicando ponderação: texto={sentimento_texto} ({peso_texto}), emoção={sentimento_emocao} ({peso_emocao})")
                
                # Manter o sentimento original para referência
                analise['sentimento_original'] = sentimento_texto
                analise['confianca_original'] = confianca_original
                
                # Em caso de dúvida, manter o sentimento do texto
                analise['sentimento'] = sentimento_texto
                analise['sentimento_final_ajustado'] = True
                analise['motivo_ajuste'] = 'ponderação_prioriza_texto'
            
            # Adicionar informações do usuário à análise de forma separada
            analise['avaliacao_usuario'] = avaliacao_estrelas
            analise['emocao'] = emocao_usuario
            analise['tags_usuario'] = tags_usuario
            analise['sentimento_inferido_estrelas'] = sentimento_estrelas
            
            # Salvar o feedback com a análise - diretamente da página principal 
            # (previne duplicação com chamada posterior de API)
            if origem == 'feedback_form':
                # Quando chamado da página de feedback, salvar direto
                current_app.data_handler.salvar_transcricao(texto, analise)
                
                # Log para rastreabilidade
                current_app.logger.info(f"Feedback salvo diretamente da API. Origem: {origem}, texto: {texto[:30]}...")
            
            # Preparar resposta com aspectos para o frontend
            aspectos = {}
            if 'aspectos' in analise and 'summary' in analise['aspectos']:
                aspectos = {
                    'principal': analise['aspectos']['summary'].get('primary_aspect'),
                    'detectados': analise['aspectos']['summary'].get('aspects_detected', []),
                    'mencoes_totais': analise['aspectos']['summary'].get('total_mentions', 0)
                }
            
            return jsonify({
                "success": True,
                "message": "Feedback processado com sucesso",
                "analise": analise,
                "sentimento": analise.get('sentimento', 'neutro'),
                "sentimento_original": sentimento_original,
                "confianca": analise.get('confianca', 0.5),
                "confianca_formatada": formatar_confianca(analise.get('confianca', 0.5)),
                "sarcasmo": analise.get('sarcasmo', {}),
                "topicos": analise.get('topicos', []),
                "aspectos": aspectos
            })
        except Exception as e:
            current_app.logger.error(f"Erro na API de feedback: {e}")
            import traceback
            current_app.logger.error(traceback.format_exc())
            return jsonify({"success": False, "message": str(e)}), 500

    @app.route('/favicon.ico')
    def favicon():
        return redirect(url_for('static', filename='favicon.ico'))
        
    @app.route('/criar_template_diagnostico')
    def criar_template_diagnostico():
        """Cria um template HTML para a página de diagnóstico"""
        try:
            # Simulação de dados de diagnóstico
            dados = {
                "versao_sistema": "2.0.0 (Modular)",
                "data_atualizacao": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "status_geral": "Online",
                "componentes": [
                    {"nome": "DataHandler", "status": "OK", "versao": "1.0.0"},
                    {"nome": "SentimentAnalyzer", "status": "OK", "versao": "2.0.0"},
                    {"nome": "SpeechHandler", "status": "OK", "versao": "1.0.0"},
                    {"nome": "AspectExtractor", "status": "OK", "versao": "1.0.0"},
                    {"nome": "Detector de Sarcasmo", "status": "OK", "versao": "1.0.0"}
                ],
                "uso_memoria": {
                    "total": "100MB",
                    "usado": "45MB",
                    "disponivel": "55MB"
                },
                "uso_disco": {
                    "total": "5GB",
                    "usado": "1.2GB",
                    "disponivel": "3.8GB"
                },
                "informacoes_adicionais": [
                    "Sistema funcionando normalmente",
                    "Última atualização: hoje",
                    "Logs sendo gerados corretamente"
                ]
            }
            
            # Nome do arquivo a ser criado
            nome_arquivo = 'templates/diagnostico_template.html'
            
            # Conteúdo do template
            conteudo = """
            <!DOCTYPE html>
            <html lang="pt-br">
            <head>
                <meta charset="UTF-8">
                <meta name="viewport" content="width=device-width, initial-scale=1.0">
                <title>Diagnóstico do Sistema</title>
                <link rel="stylesheet" href="{{ url_for('static', filename='css/bootstrap.min.css') }}">
                <style>
                    .card-status-ok { border-left: 5px solid #28a745; }
                    .card-status-warning { border-left: 5px solid #ffc107; }
                    .card-status-error { border-left: 5px solid #dc3545; }
                    .status-badge-ok { background-color: #28a745; }
                    .status-badge-warning { background-color: #ffc107; }
                    .status-badge-error { background-color: #dc3545; }
                </style>
            </head>
            <body>
                <div class="container mt-4">
                    <h1>Diagnóstico do Sistema</h1>
                    <div class="alert alert-info">
                        <strong>Versão:</strong> {{ dados.versao_sistema }} | 
                        <strong>Atualizado em:</strong> {{ dados.data_atualizacao }}
                    </div>
                    
                    <div class="card mb-4">
                        <div class="card-header bg-primary text-white">
                            <h2>Status Geral: <span class="badge bg-success">{{ dados.status_geral }}</span></h2>
                        </div>
                        <div class="card-body">
                            <h3>Componentes do Sistema</h3>
                            <div class="row">
                                {% for componente in dados.componentes %}
                                <div class="col-md-4 mb-3">
                                    <div class="card card-status-{% if componente.status == 'OK' %}ok{% elif componente.status == 'Atenção' %}warning{% else %}error{% endif %}">
                                        <div class="card-body">
                                            <h5 class="card-title">{{ componente.nome }}</h5>
                                            <p class="card-text">
                                                <span class="badge status-badge-{% if componente.status == 'OK' %}ok{% elif componente.status == 'Atenção' %}warning{% else %}error{% endif %}">{{ componente.status }}</span>
                                                <br>
                                                <small>Versão: {{ componente.versao }}</small>
                                            </p>
                                        </div>
                                    </div>
                                </div>
                                {% endfor %}
                            </div>
                            
                            <h3 class="mt-4">Recursos do Sistema</h3>
                            <div class="row">
                                <div class="col-md-6">
                                    <div class="card">
                                        <div class="card-header">Uso de Memória</div>
                                        <div class="card-body">
                                            <div class="progress mb-2">
                                                <div class="progress-bar" role="progressbar" style="width: {{ (dados.uso_memoria.usado * 100 / dados.uso_memoria.total)|int }}%"></div>
                                            </div>
                                            <p>
                                                <strong>Total:</strong> {{ dados.uso_memoria.total }}<br>
                                                <strong>Usado:</strong> {{ dados.uso_memoria.usado }}<br>
                                                <strong>Disponível:</strong> {{ dados.uso_memoria.disponivel }}
                                            </p>
                                        </div>
                                    </div>
                                </div>
                                <div class="col-md-6">
                                    <div class="card">
                                        <div class="card-header">Uso de Disco</div>
                                        <div class="card-body">
                                            <div class="progress mb-2">
                                                <div class="progress-bar" role="progressbar" style="width: {{ (dados.uso_disco.usado * 100 / dados.uso_disco.total)|int }}%"></div>
                                            </div>
                                            <p>
                                                <strong>Total:</strong> {{ dados.uso_disco.total }}<br>
                                                <strong>Usado:</strong> {{ dados.uso_disco.usado }}<br>
                                                <strong>Disponível:</strong> {{ dados.uso_disco.disponivel }}
                                            </p>
                                        </div>
                                    </div>
                                </div>
                            </div>
                            
                            <h3 class="mt-4">Informações Adicionais</h3>
                            <ul class="list-group">
                                {% for info in dados.informacoes_adicionais %}
                                <li class="list-group-item">{{ info }}</li>
                                {% endfor %}
                            </ul>
                        </div>
                    </div>
                </div>
                
                <script src="{{ url_for('static', filename='js/bootstrap.bundle.min.js') }}"></script>
            </body>
            </html>
            """
            
            # Criar o diretório se não existir
            os.makedirs(os.path.dirname(nome_arquivo), exist_ok=True)
            
            # Escrever o arquivo
            with open(nome_arquivo, 'w', encoding='utf-8') as f:
                f.write(conteudo)
            
            return jsonify({
                "success": True,
                "message": f"Template criado em {nome_arquivo}",
                "dados_exemplo": dados
            })
        except Exception as e:
            current_app.logger.error(f"Erro ao criar template de diagnóstico: {e}")
            return jsonify({"success": False, "message": str(e)}), 500

    @app.route('/teste-sarcasmo')
    def teste_sarcasmo():
        """Exibe a página de teste de detecção de sarcasmo"""
        try:
            return render_template('teste_sarcasmo.html')
        except Exception as e:
            current_app.logger.error(f"Erro na rota /teste-sarcasmo: {e}")
            return render_template('erro.html', erro=f"Erro ao carregar página de teste de sarcasmo: {e}"), 500

  
    @app.route('/api/analisar-sarcasmo', methods=['POST'])
    def analisar_sarcasmo_api():
        """API para análise de sarcasmo em texto"""
        try:
            dados = request.get_json()
            texto = dados.get('texto', '')
            
            if not texto:
                return jsonify({"sucesso": False, "erro": "Texto não fornecido"}), 400
            
            # Realizar análise com detalhes
            resultado = analisar_texto_com_sarcasmo(texto, current_app.sentiment_analyzer, incluir_detalhes=True)
            
            # Verificar se há sentimento original e final
            sentimento_original = resultado.get('sentimento_original', resultado.get('sentimento', 'neutro'))
            sentimento_final = resultado.get('sentimento', 'neutro')
            confianca_original = resultado.get('confianca_original', resultado.get('confianca', 0.5))
            confianca_ajustada = resultado.get('confianca', 0.5)
            
            # Formatar resposta padronizada
            resposta = {
                "sucesso": True,
                "texto": texto,
                "analise": {  # Adicionando objeto analise para compatibilidade com o template
                    "texto": texto,
                    "sarcasmo": {
                        "detectado": resultado.get('tem_sarcasmo', False),
                        "nivel": resultado.get('nivel_sarcasmo', 'baixo'),
                        "probabilidade": resultado.get('probabilidade_sarcasmo', 0.0),
                        "componentes": resultado.get('componentes_sarcasmo', None),
                        "evidencias": resultado.get('evidencias_sarcasmo', []),
                        "detalhes": resultado.get('detalhes_sarcasmo', {})
                    }
                },
                "resultado": {
                    "sentimento_original": sentimento_original,
                    "sentimento_final": sentimento_final,
                    "confianca_original": confianca_original,
                    "confianca_ajustada": confianca_ajustada
                }
            }
            
            return jsonify(resposta)
        
        except Exception as e:
            current_app.logger.error(f"Erro ao analisar sarcasmo: {str(e)}")
            return jsonify({"sucesso": False, "erro": f"Erro ao processar solicitação: {str(e)}"}), 500
    
    @app.route('/varejo')
    def varejo():
        """Página para análise especializada de feedbacks de varejo"""
        return render_template('varejo.html')
    
    @app.route('/api/analisar-varejo', methods=['POST'])
    def analisar_varejo_api():
        """API para análise especializada de feedbacks de varejo"""
        try:
            dados = request.get_json() or {}
            texto = dados.get('texto', '')
            categoria = dados.get('categoria', None)
            
            if not texto:
                return jsonify({"sucesso": False, "erro": "Texto não fornecido"}), 400
            
            # Verificar se temos o analisador de varejo disponível
            if not hasattr(current_app.sentiment_analyzer, 'analisar_feedback_varejo'):
                current_app.logger.warning("RetailSentimentEnhancer não disponível, usando análise padrão")
                resultado = current_app.sentiment_analyzer.analisar_sentimento(texto)
            else:
                # Usar análise especializada para varejo
                resultado = current_app.sentiment_analyzer.analisar_feedback_varejo(texto, categoria)
            
            # Extrair aspectos do feedback
            aspectos_resultado = current_app.aspect_extractor.extrair_aspectos(texto, resultado)
            
            # CORREÇÃO: Adicionar sub-aspectos e resumo à análise
            if 'sub_aspectos' in aspectos_resultado:
                resultado['sub_aspectos'] = aspectos_resultado['sub_aspectos']
                current_app.logger.info(f"Sub-aspectos adicionados à análise de varejo: {resultado['sub_aspectos']}")
            
            if 'resumo_sub_aspectos' in aspectos_resultado:
                resultado['resumo_sub_aspectos'] = aspectos_resultado['resumo_sub_aspectos']
                current_app.logger.info(f"Resumo de sub-aspectos adicionado à análise de varejo: {resultado['resumo_sub_aspectos']}")
            
            # Analisar sarcasmo integrado com varejo
            if current_app.config.get('SARCASM_DETECTION_ENABLED', False):
                try:
                    resultado_integrado = analisar_texto_com_sarcasmo(
                        texto, 
                        current_app.sentiment_analyzer, 
                        incluir_detalhes=True,
                        is_retail=True
                    )
                    
                    # Usar o sentimento ajustado com sarcasmo
                    resultado['sentimento'] = resultado_integrado.get('sentimento', resultado['sentimento'])
                    resultado['sentimento_original'] = resultado_integrado.get('sentimento_original', resultado.get('sentimento'))
                    resultado['sarcasmo'] = resultado_integrado.get('sarcasmo', {})
                    
                    current_app.logger.info(f"Análise de sarcasmo para varejo: {resultado['sarcasmo']}")
                except Exception as e:
                    current_app.logger.error(f"Erro ao analisar sarcasmo em feedback de varejo: {e}")
            
            # Formatar resposta
            resposta = {
                "sucesso": True,
                "texto": texto,
                "sentimento": resultado.get('sentimento', 'neutro'),
                "confianca": resultado.get('confianca', 0.5),
                "confianca_formatada": formatar_confianca(resultado.get('confianca', 0.5)),
                "categoria_varejo": resultado.get('categoria_varejo', 'geral'),
                "aspectos": aspectos_resultado,
                "sarcasmo": resultado.get('sarcasmo', {})
            }
            
            # Salvar análise no histórico
            try:
                current_app.data_handler.salvar_transcricao(texto, resultado, origem="varejo")
                resposta["salvo"] = True
            except Exception as e:
                current_app.logger.error(f"Erro ao salvar feedback de varejo: {e}")
                resposta["salvo"] = False
            
            return jsonify(resposta)
        except Exception as e:
            current_app.logger.error(f"Erro ao analisar feedback de varejo: {str(e)}")
            import traceback
            current_app.logger.error(traceback.format_exc())
            return jsonify({"sucesso": False, "erro": f"Erro ao processar solicitação: {str(e)}"}), 500

    @app.route('/api/configurar-llm-local', methods=['POST'])
    def configurar_llm_local():
        """
        Endpoint para configurar e ativar/desativar a integração com modelos locais (DeepSeek ou LLaMA).
        """
        try:
            dados = request.get_json() or {}
            modelo = dados.get('modelo', 'deepseek-ai/deepseek-llm-7b-base')
            device = dados.get('device', 'auto')
            quantization = dados.get('quantization')
            ativar = dados.get('ativar', True)
            
            # Se já está ativado, desativar antes de reconfigurar
            if current_app.config.get('LLM_LOCAL_ENABLED', False):
                current_app.logger.info("Desativando integração LLM local existente antes de reconfigurar")
                # Remover enhancer se existir
                if hasattr(current_app.sentiment_analyzer, 'llm_enhancer'):
                    delattr(current_app.sentiment_analyzer, 'llm_enhancer')
            
            if ativar:
                # Configurar integração
                try:
                    current_app.logger.info(f"Configurando integração LLM local com modelo: {modelo}")
                    
                    # Integrar LLM local com o sentiment analyzer
                    sentiment_analyzer = integrar_llm_local_ao_sentiment_analyzer(
                        current_app.sentiment_analyzer, 
                        modelo=modelo,
                        device=device,
                        quantization=quantization
                    )
                    
                    # Atualizar no app
                    current_app.sentiment_analyzer = sentiment_analyzer
                    
                    # Atualizar SpeechHandler para usar o analyzer atualizado
                    current_app.speech_handler = SpeechHandler(current_app.sentiment_analyzer)
                    
                    # Marcar como ativado
                    current_app.config['LLM_LOCAL_ENABLED'] = True
                    current_app.config['LLM_LOCAL_MODEL'] = modelo
                    
                    current_app.logger.info("Integração LLM local ativada com sucesso")
                    
                    return jsonify({
                        "sucesso": True,
                        "mensagem": f"Integração LLM local ativada com modelo {modelo}",
                        "status": "ativado"
                    })
                except ModelNotLoadedException as e:
                    current_app.logger.error(f"Erro ao carregar modelo local: {e}")
                    
                    # Garantir que está desativado em caso de erro
                    current_app.config['LLM_LOCAL_ENABLED'] = False
                    
                    return jsonify({
                        "sucesso": False,
                        "mensagem": f"Erro ao carregar modelo: {str(e)}",
                        "status": "erro",
                        "tipo": "modelo_nao_encontrado"
                    }), 500
                except Exception as e:
                    current_app.logger.error(f"Erro ao configurar integração LLM local: {e}")
                    current_app.logger.error(traceback.format_exc())
                    
                    # Garantir que está desativado em caso de erro
                    current_app.config['LLM_LOCAL_ENABLED'] = False
                    
                    return jsonify({
                        "sucesso": False,
                        "mensagem": f"Erro ao configurar integração LLM local: {str(e)}",
                        "status": "erro"
                    }), 500
            else:
                # Desativar integração
                current_app.config['LLM_LOCAL_ENABLED'] = False
                
                # Remover enhancer se existir
                if hasattr(current_app.sentiment_analyzer, 'llm_enhancer'):
                    delattr(current_app.sentiment_analyzer, 'llm_enhancer')
                
                current_app.logger.info("Integração LLM local desativada com sucesso")
                
                return jsonify({
                    "sucesso": True,
                    "mensagem": "Integração LLM local desativada",
                    "status": "desativado"
                })
                
        except Exception as e:
            current_app.logger.error(f"Erro no endpoint de configuração LLM local: {e}")
            current_app.logger.error(traceback.format_exc())
            return jsonify({
                "sucesso": False,
                "mensagem": f"Erro inesperado: {str(e)}",
                "status": "erro"
            }), 500
    
    @app.route('/api/status-llm-local')
    def status_llm_local():
        """Retorna o status atual da integração com LLM local"""
        try:
            ativado = current_app.config.get('LLM_LOCAL_ENABLED', False)
            modelo = current_app.config.get('LLM_LOCAL_MODEL', 'não configurado')
            
            # Verificar se tem métricas disponíveis
            metricas = {}
            if ativado and hasattr(current_app.sentiment_analyzer, 'llm_enhancer'):
                try:
                    metricas = current_app.sentiment_analyzer.llm_enhancer.obter_metricas()
                except Exception as e:
                    current_app.logger.error(f"Erro ao obter métricas LLM local: {e}")
            
            return jsonify({
                "sucesso": True,
                "ativado": ativado,
                "modelo": modelo,
                "metricas": metricas
            })
        except Exception as e:
            current_app.logger.error(f"Erro ao verificar status LLM local: {e}")
            return jsonify({
                "sucesso": False,
                "erro": str(e)
            }), 500
    
    @app.route('/api/analisar-llm-local', methods=['POST'])
    def analisar_llm_local():
        """
        Endpoint para análise direta com LLM local, sem passar pelo pipeline normal.
        Útil para testar e comparar análises.
        """
        try:
            # Verificar se LLM local está ativado
            if not current_app.config.get('LLM_LOCAL_ENABLED', False):
                return jsonify({
                    "sucesso": False,
                    "mensagem": "Integração LLM local não está ativada. Configure primeiro."
                }), 400
            
            dados = request.get_json() or {}
            texto = dados.get('texto', '')
            tipo = dados.get('tipo', 'completo')
            
            if not texto:
                return jsonify({
                    "sucesso": False,
                    "mensagem": "Texto não fornecido para análise."
                }), 400
            
            # Verificar se o tipo é válido
            tipos_validos = ['sentimento', 'sarcasmo', 'completo', 'varejo']
            if tipo not in tipos_validos:
                return jsonify({
                    "sucesso": False,
                    "mensagem": f"Tipo de análise inválido. Use um dos seguintes: {', '.join(tipos_validos)}"
                }), 400
            
            # Verificar se temos o enhancer configurado
            if not hasattr(current_app.sentiment_analyzer, 'llm_enhancer'):
                return jsonify({
                    "sucesso": False,
                    "mensagem": "LLM Enhancer não está configurado corretamente no SentimentAnalyzer."
                }), 500
            
            # Realizar análise direta com LLM local
            try:
                if tipo == 'sentimento':
                    resultado = current_app.sentiment_analyzer.llm_enhancer.llm_analyzer.analisar_sentimento(texto)
                elif tipo == 'sarcasmo':
                    resultado = current_app.sentiment_analyzer.llm_enhancer.llm_analyzer.analisar_sarcasmo(texto)
                elif tipo == 'varejo':
                    resultado = current_app.sentiment_analyzer.llm_enhancer.llm_analyzer.analisar_varejo(texto)
                else:  # 'completo'
                    resultado = current_app.sentiment_analyzer.llm_enhancer.llm_analyzer.analisar_completo(texto)
                
                if not resultado:
                    return jsonify({
                        "sucesso": False,
                        "mensagem": "Falha ao analisar texto com LLM local. Verifique os logs para mais detalhes."
                    }), 500
            except ModelNotLoadedException:
                # Erro específico de modelo não carregado
                current_app.logger.error("Análise abortada devido a erro no carregamento do modelo")
                return jsonify({
                    "sucesso": False,
                    "erro_tipo": "modelo_nao_carregado",
                    "mensagem": "O modelo não foi carregado corretamente. Verifique se o caminho está correto e se há memória suficiente."
                }), 500
            except Exception as e:
                current_app.logger.error(f"Erro específico na análise LLM local: {e}")
                return jsonify({
                    "sucesso": False,
                    "mensagem": f"Erro ao analisar texto: {str(e)}"
                }), 500
            
            # Adicionar campos para compatibilidade com frontend
            if 'confianca' in resultado:
                resultado['confianca_formatada'] = formatar_confianca(resultado['confianca'])
            
            return jsonify({
                "sucesso": True,
                "resultado": resultado,
                "tipo_analise": tipo
            })
            
        except Exception as e:
            current_app.logger.error(f"Erro ao analisar com LLM local: {e}")
            current_app.logger.error(traceback.format_exc())
            return jsonify({
                "sucesso": False,
                "mensagem": f"Erro inesperado: {str(e)}"
            }), 500
    
    @app.route('/llm-config')
    def llm_config():
        """Página de configuração e teste da integração com LLM local"""
        # Verificar se há modelos disponíveis para usar
        modelos_disponiveis = [
            {"nome": "deepseek-ai/deepseek-llm-7b-base", "descricao": "DeepSeek 7B Base - Excelente para português"},
            {"nome": "deepseek-ai/deepseek-coder-6.7b-instruct", "descricao": "DeepSeek Coder 6.7B - Bom para análise estruturada"},
            {"nome": "meta-llama/Llama-3-8b", "descricao": "LLaMA 3 8B - Muito bom em contexto e análise"}
        ]
        
        # Verificar transformers instalado
        transformers_disponivel = True
        try:
            import transformers
        except ImportError:
            transformers_disponivel = False
        
        # Verificar llama.cpp instalado
        llama_cpp_disponivel = True
        try:
            import llama_cpp
        except ImportError:
            llama_cpp_disponivel = False
        
        return render_template(
            'llm_config.html', 
            modelos=modelos_disponiveis,
            transformers_disponivel=transformers_disponivel,
            llama_cpp_disponivel=llama_cpp_disponivel,
            status_atual=current_app.config.get('LLM_LOCAL_ENABLED', False)
        )

# Entrypoint
if __name__ == '__main__':
    app = criar_app()
    app.run(host='0.0.0.0', port=int(os.getenv('PORT', 5000)), debug=True)
