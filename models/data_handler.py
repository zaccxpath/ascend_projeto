import os
import json
import datetime
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from utils.config import logger, DATA_DIR, STATIC_IMAGES_DIR, CORES_SENTIMENTOS
import uuid

class DataHandler:
    """Classe para manipulação de dados, gráficos e estatísticas"""
    
    def __init__(self):
        # Garantir que diretórios existam
        os.makedirs(DATA_DIR, exist_ok=True)
        os.makedirs(STATIC_IMAGES_DIR, exist_ok=True)
        
        # Arquivos para armazenamento de dados
        self.arquivo_historico = os.path.join(DATA_DIR, 'historico.json')
        self.arquivo_analises = os.path.join(DATA_DIR, 'analises.json')
        self.historico_dir = os.path.join(DATA_DIR, 'analises_sarcasmo')
        os.makedirs(self.historico_dir, exist_ok=True)
        
        logger.info("DataHandler inicializado")
    
    def salvar_transcricao(self, texto, analise):
        """
        Salva uma transcrição e sua análise no histórico
        
        Parâmetros:
        - texto: Texto transcrito
        - analise: Dicionário com a análise de sentimento
        
        Retorna:
        - ID do registro salvo
        """
        try:
            # Carregar histórico existente
            historico = self.carregar_historico()
            
            # Gerar ID único
            timestamp = datetime.datetime.now().isoformat()
            id_registro = f"reg_{timestamp.replace(':', '-').replace('.', '-')}"
            
            # Processar a análise para garantir que todos os campos são serializáveis
            analise_processada = {}
            for chave, valor in analise.items():
                if chave == 'topicos':
                    # Processar tópicos - manter como lista para o JSON, mas garantir que são strings
                    if valor is None:
                        analise_processada[chave] = []
                    elif isinstance(valor, list):
                        # Converter todos os elementos para string
                        analise_processada[chave] = [str(item) for item in valor]
                    elif isinstance(valor, str):
                        # Se for string, dividir por vírgulas caso tenha, ou adicionar como item único
                        if ',' in valor:
                            analise_processada[chave] = [item.strip() for item in valor.split(',')]
                        else:
                            analise_processada[chave] = [valor.strip()]
                    else:
                        # Para outros tipos, converter para string e adicionar como item único
                        analise_processada[chave] = [str(valor)]
                    
                    # Logar os tópicos para depuração
                    logger.info(f"Tópicos processados: {analise_processada[chave]}")
                    
                elif chave == 'aspectos':
                    # Processar estrutura de aspectos
                    aspectos_estrutura = {}
                    if isinstance(valor, dict) and 'summary' in valor:
                        aspectos_estrutura['summary'] = {}
                        summary = valor['summary']
                        
                        # Processar aspects_detected
                        if 'aspects_detected' in summary:
                            aspects_detected = summary['aspects_detected']
                            if isinstance(aspects_detected, list):
                                aspectos_estrutura['summary']['aspects_detected'] = aspects_detected
                            else:
                                aspectos_estrutura['summary']['aspects_detected'] = []
                        
                        # Processar primary_aspect
                        if 'primary_aspect' in summary:
                            primary_aspect = summary['primary_aspect']
                            if primary_aspect is None:
                                aspectos_estrutura['summary']['primary_aspect'] = ''
                            else:
                                aspectos_estrutura['summary']['primary_aspect'] = str(primary_aspect)
                    
                    analise_processada[chave] = aspectos_estrutura
                elif chave == 'sub_aspectos':
                    # Processar sub-aspectos, garantindo a estrutura
                    if isinstance(valor, dict):
                        analise_processada[chave] = valor
                    else:
                        analise_processada[chave] = {}
                elif chave == 'resumo_sub_aspectos':
                    # Processar resumo de sub-aspectos
                    if isinstance(valor, dict):
                        analise_processada[chave] = valor
                    else:
                        analise_processada[chave] = {}
                else:
                    # Processar outros campos
                    if isinstance(valor, (str, int, float, bool, type(None))):
                        analise_processada[chave] = valor
                    elif isinstance(valor, list):
                        # Manter listas como estão para JSON, mas garantir que elementos são serializáveis
                        analise_processada[chave] = [
                            str(item) if not isinstance(item, (str, int, float, bool, type(None), dict, list)) else item
                            for item in valor
                        ]
                    elif isinstance(valor, dict):
                        # Manter dicionários como estão
                        analise_processada[chave] = valor
                    else:
                        # Converter outros tipos para string
                        analise_processada[chave] = str(valor)
            
            # Criar novo registro
            registro = {
                'id': id_registro,
                'data': timestamp,
                'texto': texto,
                'analise': analise_processada,
                'data_formatada': datetime.datetime.now().strftime('%d/%m/%Y %H:%M:%S')
            }
            
            # Adicionar ao histórico
            historico.append(registro)
            
            # Salvar histórico atualizado
            with open(self.arquivo_historico, 'w', encoding='utf-8') as arquivo:
                json.dump(historico, arquivo, ensure_ascii=False, indent=2)
                
            logger.info(f"Transcrição salva no histórico com ID: {id_registro}")
            
            # Atualizar também arquivo de análises para estatísticas
            self._atualizar_analises(registro)
            
            return id_registro
        except Exception as e:
            logger.error(f"Erro ao salvar transcrição: {e}")
            import traceback
            traceback_str = traceback.format_exc()
            logger.error(f"Detalhes do erro: {traceback_str}")
            return None
    
    def carregar_historico(self):
        """Carrega o histórico de transcrições"""
        try:
            if os.path.exists(self.arquivo_historico):
                with open(self.arquivo_historico, 'r', encoding='utf-8') as arquivo:
                    return json.load(arquivo)
            else:
                logger.info("Arquivo de histórico não encontrado, retornando lista vazia")
                return []
        except Exception as e:
            logger.error(f"Erro ao carregar histórico: {e}")
            return []
    
    def carregar_analises(self):
        """Carrega as análises estatísticas"""
        try:
            if os.path.exists(self.arquivo_analises):
                with open(self.arquivo_analises, 'r', encoding='utf-8') as arquivo:
                    analises = json.load(arquivo)
                    
                    # Verificar o formato dos dados carregados
                    if isinstance(analises, dict) and 'dados' in analises:
                        # O formato é como esperado
                        logger.info(f"Análises carregadas no formato dicionário: {len(analises.get('dados', []))} registros de dados")
                        
                        # Para evitar problemas no template, garantir que campos esperados existem
                        if 'dados' not in analises:
                            analises['dados'] = []
                            
                        return analises
                    elif isinstance(analises, list):
                        # Se já é uma lista, criar um dicionário para manter formato consistente
                        logger.info(f"Análises carregadas no formato lista: {len(analises)} registros")
                        return {
                            'total': len(analises),
                            'positivo': sum(1 for a in analises if a.get('sentimento') == 'positivo'),
                            'neutro': sum(1 for a in analises if a.get('sentimento') == 'neutro'),
                            'negativo': sum(1 for a in analises if a.get('sentimento') == 'negativo'),
                            'dados': analises
                        }
                    else:
                        # Formato inesperado, criar um dicionário vazio com estrutura padrão
                        logger.warning(f"Formato inesperado de analises: {type(analises).__name__}")
                        return {
                            'total': 0,
                            'positivo': 0,
                            'neutro': 0,
                            'negativo': 0,
                            'dados': []
                        }
            else:
                logger.info("Arquivo de análises não encontrado, retornando dicionário vazio")
                return {
                    'total': 0,
                    'positivo': 0,
                    'neutro': 0,
                    'negativo': 0,
                    'dados': []
                }
        except Exception as e:
            logger.error(f"Erro ao carregar análises: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return {
                'total': 0,
                'positivo': 0,
                'neutro': 0,
                'negativo': 0,
                'dados': []
            }
    
    def _atualizar_analises(self, registro):
        """Atualiza as estatísticas de análises com um novo registro"""
        try:
            # Carregar análises existentes
            analises = self.carregar_analises()
            
            # Incrementar contadores
            analises['total'] += 1
            
            # Extrair sentimento da análise
            sentimento = registro['analise'].get('sentimento', 'neutro')
            
            # Garantir que sentimento é um valor válido
            if sentimento not in ['positivo', 'neutro', 'negativo']:
                logger.warning(f"Sentimento inválido encontrado: {sentimento}, usando 'neutro' como padrão")
                sentimento = 'neutro'
            
            # Incrementar contador específico do sentimento
            analises[sentimento] += 1
            
            # Obter confiança com validação
            confianca = registro['analise'].get('confianca', 0.5)
            if not isinstance(confianca, (int, float)):
                try:
                    confianca = float(confianca)
                except (ValueError, TypeError):
                    confianca = 0.5
            
            # Adicionar aos dados para série temporal com campos seguros
            novo_dado = {
                'id': registro['id'],
                'data': registro['data'],
                'sentimento': sentimento,
                'confianca': confianca
            }
            
            # Processar tópicos
            if 'topicos' in registro['analise']:
                topicos = registro['analise']['topicos']
                
                # Tratar tópicos conforme o formato (lista ou string)
                if isinstance(topicos, list):
                    if topicos:  # Se a lista não estiver vazia
                        # Juntar tópicos em string separada por vírgulas
                        novo_dado['topicos'] = ','.join(str(t) for t in topicos)
                    else:
                        novo_dado['topicos'] = ''
                elif isinstance(topicos, str):
                    novo_dado['topicos'] = topicos
                else:
                    # Para outros tipos, converter para string
                    novo_dado['topicos'] = str(topicos) if topicos else ''
                    
                logger.info(f"Tópicos no registro: {novo_dado['topicos']}")
            else:
                novo_dado['topicos'] = ''
            
            # Verificar se há campos extras que gostaríamos de manter
            # Aspectos detectados
            if 'aspectos' in registro['analise'] and isinstance(registro['analise']['aspectos'], dict):
                aspectos = registro['analise']['aspectos']
                if 'summary' in aspectos and isinstance(aspectos['summary'], dict):
                    summary = aspectos['summary']
                    
                    # Processar aspectos_detectados
                    if 'aspects_detected' in summary:
                        aspects_detected = summary['aspects_detected']
                        if isinstance(aspects_detected, list):
                            novo_dado['aspectos_detectados'] = ','.join(str(a) for a in aspects_detected)
                        else:
                            novo_dado['aspectos_detectados'] = ''
                    
                    # Processar aspecto_principal
                    if 'primary_aspect' in summary:
                        primary_aspect = summary['primary_aspect']
                        novo_dado['aspecto_principal'] = str(primary_aspect) if primary_aspect else ''
            
            # Processar sub-aspectos se existirem
            if 'sub_aspectos' in registro['analise'] and isinstance(registro['analise']['sub_aspectos'], dict):
                novo_dado['sub_aspectos'] = registro['analise']['sub_aspectos']
            
            # Processar resumo de sub-aspectos se existirem
            if 'resumo_sub_aspectos' in registro['analise'] and isinstance(registro['analise']['resumo_sub_aspectos'], dict):
                novo_dado['resumo_sub_aspectos'] = registro['analise']['resumo_sub_aspectos']
            
            analises['dados'].append(novo_dado)
            
            # Salvar análises atualizadas
            with open(self.arquivo_analises, 'w', encoding='utf-8') as arquivo:
                json.dump(analises, arquivo, ensure_ascii=False, indent=2)
                
            logger.info(f"Estatísticas atualizadas: total={analises['total']}, {sentimento}={analises[sentimento]}")
        except Exception as e:
            logger.error(f"Erro ao atualizar análises: {e}")
            import traceback
            traceback_str = traceback.format_exc()
            logger.error(f"Detalhes do erro: {traceback_str}")
    
    def gerar_graficos(self):
        """
        Gera gráficos para o dashboard usando Plotly e retorna como dicionários
        
        Retorna:
        - Dicionário com objetos de gráfico e estatísticas
        """
        try:
            # Log para debug
            logger.info("Iniciando geração de gráficos")
            
            # Carregar dados
            analises = self.carregar_analises()
            historico = self.carregar_historico()
            
            # Verificar se temos dados no histórico
            if not historico:
                logger.error("Histórico vazio. Impossível gerar gráficos.")
                return {
                    'erro': 'Não há dados no histórico para gerar os gráficos.'
                }
            
            # Verificar se o arquivo analises.csv existe e tem dados
            arquivo_csv = os.path.join(DATA_DIR, 'analises.csv')
            if os.path.exists(arquivo_csv):
                try:
                    df = pd.read_csv(arquivo_csv)
                    logger.info(f"Arquivo CSV de análises encontrado com {len(df)} registros")
                    
                    # Verificar se temos registros suficientes
                    if len(df) < 3:  # Um mínimo de registros para gráficos significativos
                        logger.warning(f"Arquivo CSV com apenas {len(df)} registros. Mínimo recomendado: 3")
                except Exception as e:
                    logger.error(f"Erro ao ler arquivo CSV de análises: {e}")
                    df = None
            else:
                logger.warning(f"Arquivo CSV de análises não encontrado: {arquivo_csv}")
                df = None
            
            logger.info(f"Dados carregados: {analises['total'] if isinstance(analises, dict) else 0} análises, {len(historico)} registros no histórico")
            
            # Verificar se há dados suficientes
            if (isinstance(analises, dict) and analises.get('total', 0) == 0) or len(historico) == 0:
                logger.error("Sem dados suficientes para gerar gráficos")
                return {
                    'erro': 'Não há dados suficientes para gerar os gráficos. Adicione mais feedbacks.'
                }
            
            # Preparar resultados
            graficos = {}
            
            # Calcular estatísticas básicas
            total = analises['total'] if isinstance(analises, dict) and 'total' in analises else len(historico)
            
            # Contadores de sentimentos
            positivo = analises.get('positivo', 0) if isinstance(analises, dict) else 0
            neutro = analises.get('neutro', 0) if isinstance(analises, dict) else 0
            negativo = analises.get('negativo', 0) if isinstance(analises, dict) else 0
            
            # Se não temos dados de sentiment counts do analises.json, tentar extrair do histórico
            if positivo == 0 and neutro == 0 and negativo == 0 and historico:
                for item in historico:
                    sentimento = item.get('analise', {}).get('sentimento', 'neutro')
                    if sentimento == 'positivo':
                        positivo += 1
                    elif sentimento == 'negativo':
                        negativo += 1
                    else:
                        neutro += 1
                logger.info(f"Sentimentos extraídos do histórico: positivo={positivo}, neutro={neutro}, negativo={negativo}")
            
            # Distribuição de sentimentos para dashboard
            estatisticas_dashboard = {
                'total_feedbacks': total,
                'distribuicao_sentimentos': {
                    'positivo': {
                        'count': positivo,
                        'percentual': round(positivo / total * 100, 1) if total > 0 else 0
                    },
                    'neutro': {
                        'count': neutro,
                        'percentual': round(neutro / total * 100, 1) if total > 0 else 0
                    },
                    'negativo': {
                        'count': negativo,
                        'percentual': round(negativo / total * 100, 1) if total > 0 else 0
                    }
                },
                'tendencia_feedbacks': {
                    'direcao': 'aumento',
                    'valor': 12.5  # Valor fictício para demonstração
                }
            }
            
            # Gráfico de pizza de sentimentos
            logger.info("Gerando gráfico de pizza de sentimentos")
            labels = ['Positivo', 'Neutro', 'Negativo']
            values = [positivo, neutro, negativo]
            colors = [CORES_SENTIMENTOS['positivo'], CORES_SENTIMENTOS['neutro'], CORES_SENTIMENTOS['negativo']]
            
            fig = go.Figure(data=[go.Pie(
                labels=labels, 
                values=values,
                marker=dict(colors=colors),
                textinfo='label+percent',
                insidetextorientation='radial'
            )])
            
            fig.update_layout(
                title='Distribuição de Sentimentos',
                height=400,
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5)
            )
            
            graficos['sentimento_pie'] = fig.to_dict()
            
            # Gráfico de tendência temporal
            logger.info("Gerando gráfico de tendência temporal")
            
            # Extrair dados para tendência mensal
            meses = {}
            try:
                for item in historico:
                    try:
                        data = datetime.datetime.fromisoformat(item['data'].split('.')[0])
                        mes = data.strftime('%Y-%m')
                        sentimento = item['analise'].get('sentimento', 'neutro')
                        
                        if mes not in meses:
                            meses[mes] = {'positivo': 0, 'neutro': 0, 'negativo': 0}
                        
                        meses[mes][sentimento] += 1
                    except Exception as e:
                        logger.error(f"Erro ao processar item para tendência mensal: {e}")
                        continue
                        
                # Criar gráfico temporal
                if meses:
                    fig = go.Figure()
                    meses_ordenados = sorted(meses.keys())
                    
                    for sentimento in ['positivo', 'neutro', 'negativo']:
                        fig.add_trace(go.Scatter(
                            x=meses_ordenados,
                            y=[meses[mes][sentimento] for mes in meses_ordenados],
                            mode='lines+markers',
                            name=sentimento.capitalize(),
                            line=dict(color=CORES_SENTIMENTOS.get(sentimento, '#3498db'))
                        ))
                    
                    fig.update_layout(
                        title='Evolução de Sentimentos por Mês',
                        xaxis=dict(title='Mês'),
                        yaxis=dict(title='Quantidade'),
                        height=400,
                        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5)
                    )
                    
                    graficos['sentimento_tempo'] = fig.to_dict()
                else:
                    # Se não há dados temporais, criar um gráfico mostrando mensagem
                    fig = go.Figure()
                    # Adicionar um ponto com valor zero para cada sentimento para criar linhas vazias
                    for sentimento in ['positivo', 'neutro', 'negativo']:
                        fig.add_trace(go.Scatter(
                            x=['Sem dados temporais'],
                            y=[0],
                            mode='lines+markers',
                            name=sentimento.capitalize(),
                            line=dict(color=CORES_SENTIMENTOS.get(sentimento, '#3498db'))
                        ))
                    
                    fig.update_layout(
                        title='Evolução de Sentimentos por Mês',
                        xaxis=dict(title='Mês'),
                        yaxis=dict(title='Quantidade', range=[0, 1]),  # Fixar escala para mostrar zero
                        height=400,
                        annotations=[
                            dict(
                                text="Sem dados temporais suficientes",
                                showarrow=False,
                                xref="paper",
                                yref="paper",
                                x=0.5,
                                y=0.5
                            )
                        ]
                    )
                    graficos['sentimento_tempo'] = fig.to_dict()
            except Exception as e:
                logger.error(f"Erro ao gerar gráfico temporal: {e}")
                return {
                    'erro': f"Erro ao gerar gráfico temporal: {str(e)}"
                }
                
            # Gráfico de palavras mais frequentes
            logger.info("Gerando gráfico de palavras mais frequentes")
            
            # Extrair palavras-chave de todos os registros
            palavras = {}
            for item in historico:
                try:
                    # Obter tópicos
                    topicos = item.get('analise', {}).get('topicos', [])
                    
                    # Processar tópicos conforme formato
                    if isinstance(topicos, str):
                        # Dividir string por vírgulas se contiver
                        if ',' in topicos:
                            topicos_lista = [t.strip() for t in topicos.split(',')]
                        else:
                            topicos_lista = [topicos.strip()]
                    elif isinstance(topicos, list):
                        topicos_lista = topicos
                    else:
                        topicos_lista = []
                    
                    # Contabilizar ocorrências
                    for palavra in topicos_lista:
                        if not palavra:  # Ignorar strings vazias
                            continue
                            
                        palavra = palavra.lower()  # Normalizar para minúsculas
                        if palavra not in palavras:
                            palavras[palavra] = 0
                        palavras[palavra] += 1
                except Exception as e:
                    logger.error(f"Erro ao processar palavras-chave: {e}")
                    continue
            
            # Criar gráfico de barras horizontais para palavras frequentes
            if palavras:
                # Ordenar por frequência e pegar as top N
                top_palavras = sorted(palavras.items(), key=lambda x: x[1], reverse=True)[:10]
                
                # Gerar gráfico de barras
                fig = go.Figure(go.Bar(
                    x=[count for _, count in top_palavras],
                    y=[palavra for palavra, _ in top_palavras],
                    orientation='h',
                    marker_color='#3498db'
                ))
                
                fig.update_layout(
                    title='Palavras Mais Frequentes',
                    xaxis_title='Ocorrências',
                    yaxis_title='Palavras',
                    height=400
                )
                
                graficos['palavras_top'] = fig.to_dict()
            else:
                # Criar gráfico vazio para palavras
                fig = go.Figure(go.Bar(
                    x=[0, 0, 0],
                    y=['Sem', 'dados', 'disponíveis'],
                    orientation='h',
                    marker_color='#cccccc'
                ))
                
                fig.update_layout(
                    title='Palavras Mais Frequentes',
                    xaxis_title='Ocorrências',
                    yaxis_title='Palavras',
                    height=400,
                    annotations=[
                        dict(
                            text="Nenhum tópico encontrado nos dados",
                            showarrow=False,
                            xref="paper",
                            yref="paper",
                            x=0.5,
                            y=0.5
                        )
                    ]
                )
                
                graficos['palavras_top'] = fig.to_dict()
            
            # Verificar se os gráficos obrigatórios foram criados
            for grafico_chave in ['sentimento_pie', 'sentimento_tempo', 'palavras_top']:
                if grafico_chave not in graficos:
                    logger.error(f"Falha ao gerar gráfico obrigatório: {grafico_chave}")
                    return {
                        'erro': f"Falha ao gerar gráfico obrigatório: {grafico_chave}"
                    }
            
            # Adicionar estatísticas ao objeto de retorno
            graficos['estatisticas_dashboard'] = estatisticas_dashboard
            
            # Adicionar dados brutos para o dashboard
            graficos['sentimentos'] = {'positivo': positivo, 'neutro': neutro, 'negativo': negativo}
            graficos['tendencia_mensal'] = meses
            
            logger.info(f"Gráficos gerados com sucesso. Estrutura: {list(graficos.keys())}")
            return graficos
            
        except Exception as e:
            import traceback
            logger.error(f"Erro ao gerar gráficos: {e}")
            logger.error(traceback.format_exc())
            return {"erro": f"Erro ao gerar gráficos: {str(e)}"}

    def adicionar_feedback_manual(self, texto, sentimento, aspecto=None):
        """
        Adiciona um feedback manual ao histórico
        
        Parâmetros:
        - texto: Texto do feedback
        - sentimento: Sentimento ('positivo', 'neutro', 'negativo')
        - aspecto: Aspecto opcional ('produto', 'empresa', 'preço', 'entrega', 'atendimento')
        
        Retorna:
        - ID do registro adicionado
        """
        try:
            # Validar sentimento
            if sentimento not in ['positivo', 'neutro', 'negativo']:
                sentimento = 'neutro'
                
            # Criar análise manual
            analise = {
                'sentimento': sentimento,
                'confianca': 1.0,  # Alta confiança por ser manual
                'modelo': 'manual',
                'tokens_relevantes': {
                    'positivos': [],
                    'negativos': []
                }
            }
            
            # Adicionar informação de aspecto, se fornecido
            if aspecto:
                analise['aspecto'] = aspecto
                
            # Salvar no histórico
            return self.salvar_transcricao(texto, analise)
        except Exception as e:
            logger.error(f"Erro ao adicionar feedback manual: {e}")
            return None
            
    def registrar_analise_sarcasmo(self, texto, analise_sentimento, analise_sarcasmo):
        """
        Registra uma análise de sarcasmo no histórico
        
        Parâmetros:
        - texto: Texto analisado
        - analise_sentimento: Dicionário com resultado da análise de sentimento
        - analise_sarcasmo: Dicionário com resultado da análise de sarcasmo
        
        Retorna:
        - ID do registro adicionado ou None em caso de erro
        """
        try:
            # Verificação de entrada
            if not texto or not isinstance(analise_sentimento, dict) or not isinstance(analise_sarcasmo, dict):
                logger.warning("Dados inválidos para registro de análise de sarcasmo")
                return None
                
            # Criar registro com dados de sarcasmo
            data_atual = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            registro = {
                'id': str(uuid.uuid4()),
                'texto': texto,
                'data': data_atual,
                'tipo': 'analise_sarcasmo',
                'sentimento': analise_sentimento.get('sentimento', 'neutro'),
                'sentimento_final': analise_sentimento.get('sentimento', 'neutro'),  # Será alterado se houver inversão por sarcasmo
                'analise': analise_sentimento,
                'sarcasmo': {
                    'detectado': analise_sarcasmo.get('is_sarcastic', False),
                    'confianca': analise_sarcasmo.get('confidence', 0),
                    'marcadores': analise_sarcasmo.get('markers', []),
                    'metodo_deteccao': analise_sarcasmo.get('detection_method', 'regras')
                }
            }
            
            # Ajustar sentimento final conforme detecção de sarcasmo
            if analise_sarcasmo.get('is_sarcastic', False) and analise_sarcasmo.get('inverte_sentimento', False):
                sentimento_original = analise_sentimento.get('sentimento', 'neutro')
                if sentimento_original == 'positivo':
                    registro['sentimento_final'] = 'negativo'
                elif sentimento_original == 'negativo':
                    registro['sentimento_final'] = 'positivo'
                # neutro permanece neutro
            
            # Gravar em arquivo
            caminho_arquivo = os.path.join(self.historico_dir, f"{registro['id']}.json")
            with open(caminho_arquivo, 'w', encoding='utf-8') as f:
                json.dump(registro, f, ensure_ascii=False, indent=2)
                
            # Atualizar cache de análises
            self._atualizar_analises(registro)
            
            logger.info(f"Análise de sarcasmo registrada com ID: {registro['id']}")
            return registro['id']
            
        except Exception as e:
            logger.error(f"Erro ao registrar análise de sarcasmo: {e}")
            return None