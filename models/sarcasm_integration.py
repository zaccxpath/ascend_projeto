"""
Módulo de integração do detector de sarcasmo com o sistema principal.
Fornece funções para integrar o detector de sarcasmo modular com o analisador de sentimento.
"""

import logging
from typing import Dict, Any, Optional, Union
from flask import Flask, Blueprint, request, jsonify
import re

from models.sarcasm_factory import SarcasmDetectorManager, detectar_sarcasmo
from models.sarcasm_config import SARCASMO_CONFIG

# Verificar se o enhancer para varejo está disponível
try:
    from models.retail_sentiment_enhancer import RetailSentimentEnhancer, aplicar_melhorias_varejo
    retail_enhancer_available = True
except ImportError:
    retail_enhancer_available = False

# Configurar logger
logger = logging.getLogger(__name__)

def analisar_texto_com_sarcasmo(texto: str, 
                               sentiment_analyzer: Any,  
                               incluir_detalhes: bool = False,
                               is_retail: bool = False) -> Dict[str, Any]:
    """
    Analisa um texto detectando sentimento e sarcasmo de forma integrada.
    
    Args:
        texto: Texto a ser analisado
        sentiment_analyzer: Instância do analisador de sentimento
        incluir_detalhes: Se True, inclui detalhes completos da análise de sarcasmo
        is_retail: Se True, usa melhorias específicas para feedback de varejo
        
    Returns:
        Dicionário com resultados da análise incluindo sentimento e sarcasmo
    """
    # Checar sentimento aparente pela presença de palavras positivas
    # Estas são palavras que indicam um tom superficialmente positivo em frases sarcásticas
    palavras_aparentemente_positivas = [
        'excelente', 'ótimo', 'bom', 'incrível', 'maravilhoso', 'fantástico',
        'excepcional', 'impressionante', 'qualidade', 'perfeito'
    ]
    
    # Verificar a quantidade de palavras positivas no texto
    texto_lower = texto.lower()
    tem_palavra_positiva = any(palavra in texto_lower for palavra in palavras_aparentemente_positivas)
    qtd_palavras_positivas = sum(1 for palavra in palavras_aparentemente_positivas if palavra in texto_lower)
    
    # Determinar se é um provável feedback de varejo
    is_varejo = is_retail
    if not is_retail and hasattr(sentiment_analyzer, 'retail_enhancer') and sentiment_analyzer.retail_enhancer:
        # Palavras que indicam feedback de varejo
        palavras_varejo = [
            'entrega', 'loja', 'produto', 'compra', 'site', 'pedido', 'vendedor', 
            'atendimento', 'devolução', 'troca', 'reembolso', 'frete', 'pagamento'
        ]
        if any(palavra in texto_lower for palavra in palavras_varejo):
            is_varejo = True
            logger.info("Texto detectado automaticamente como feedback de varejo")
    
    # Analisar sentimento com o modelo, usando melhorias de varejo se aplicável
    if is_varejo and hasattr(sentiment_analyzer, 'analisar_feedback_varejo'):
        resultado_sentimento = sentiment_analyzer.analisar_feedback_varejo(texto)
        logger.info("Usando análise especializada de feedback de varejo")
    else:
        resultado_sentimento = sentiment_analyzer.analisar_sentimento(texto)
    
    # Extrair score de sentimento no formato -1 a 1
    sentimento = resultado_sentimento.get('score', 0)
    sentimento_original_label = resultado_sentimento.get('sentimento', 'neutro')
    
    # Verificar se o sentimento deve ser ajustado superficialmente antes da análise de sarcasmo
    # (Alguns textos sarcásticos são reconhecidos como negativos pelo modelo, mas deveriam ser positivos inicialmente)
    sentimento_aparente = sentimento
    sentimento_aparente_label = sentimento_original_label
    
    # Se o texto tem várias palavras positivas mas foi classificado como negativo,
    # pode ser um caso onde o modelo já detectou o sentimento final sem considerar o sarcasmo
    if sentimento_original_label == 'negativo' and qtd_palavras_positivas >= 1 and tem_palavra_positiva:
        # Criar uma cópia do resultado com sentimento ajustado para aparente positivo
        sentimento_aparente = 0.6  # Valor positivo arbitrário
        sentimento_aparente_label = 'positivo'
        
        logger.info(f"Texto com palavras positivas mas classificado como negativo, possível sarcasmo: '{texto}'")
        logger.info(f"Ajustando sentimento aparente para positivo para análise de sarcasmo")
    
    # Normalizar o sentimento de escala 0-1 para -1 a 1 (se necessário)
    if sentimento >= 0 and sentimento <= 1:
        sentimento = (sentimento * 2) - 1
    
    # Detectar sarcasmo passando o sentimento aparente (para casos de sarcasmo)
    resultado_sarcasmo = detectar_sarcasmo(texto, sentimento_aparente)
    
    # Se é um feedback de varejo, verificar se temos detecção específica de sarcasmo para varejo
    if is_varejo and retail_enhancer_available:
        try:
            enhancer = RetailSentimentEnhancer()
            resultado_sarcasmo_varejo = enhancer.detectar_sarcasmo_varejo(texto, resultado_sentimento)
            
            # Se o detector de varejo encontrou sarcasmo com maior probabilidade, usar ele
            if (resultado_sarcasmo_varejo['e_sarcastico'] and 
                resultado_sarcasmo_varejo['probabilidade'] > resultado_sarcasmo['probabilidade']):
                resultado_sarcasmo = resultado_sarcasmo_varejo
                logger.info(f"Detector de sarcasmo específico para varejo ativado: probabilidade={resultado_sarcasmo['probabilidade']:.2f}")
        except Exception as e:
            logger.error(f"Erro ao usar detector de sarcasmo para varejo: {e}")
    
    # Ajustar o sentimento com base no sarcasmo detectado
    score_ajustado = ajustar_sentimento_por_sarcasmo(
        sentimento_aparente,
        resultado_sarcasmo['probabilidade'],
        resultado_sarcasmo['nivel']
    )
    
    # Verificar palavras-chave específicas de tempo de espera
    palavras_tempo = ['hora', 'horas', 'minuto', 'minutos', 'dia', 'dias', 'fila', 'espera', 'esperei', 'aguardei']
    tem_palavra_tempo = any(palavra in texto.lower() for palavra in palavras_tempo)
    
    # Verificar palavras positivas
    palavras_positivas = ['incrível', 'excelente', 'ótimo', 'maravilhoso', 'perfeito', 'bom', 'excepcional', 'impressionante', 'qualidade']
    tem_palavra_positiva = any(palavra in texto.lower() for palavra in palavras_positivas)
    
    # Verificar palavras que indicam problemas
    palavras_problema = ['quebrou', 'quebrado', 'parou', 'estragou', 'falhou', 'defeito', 'estrago', 'pane', 'danificado', 'piorou', 'ruim', 'péssimo']
    tem_palavra_problema = any(palavra in texto.lower() for palavra in palavras_problema)
    
    # Verificar palavras que indicam problemas de funcionamento
    palavras_funcionamento = ['travou', 'trava', 'travado', 'lento', 'lentidão', 'congela', 'congelou', 'bug', 'bugs', 'falha', 'pifou']
    tem_problema_funcionamento = any(palavra in texto.lower() for palavra in palavras_funcionamento)
    
    # Verificar padrão de número de vezes
    padrao_numero_vezes = re.search(r'(\d+)\s+vezes', texto.lower())
    tem_muitas_vezes = False
    if padrao_numero_vezes:
        try:
            numero = int(padrao_numero_vezes.group(1))
            if numero >= 3:
                tem_muitas_vezes = True
        except (ValueError, AttributeError):
            pass
    
    # Verificar contradição entre problemas e qualidade
    if tem_palavra_problema and tem_palavra_positiva:
        resultado_sarcasmo['probabilidade'] = max(resultado_sarcasmo['probabilidade'], 0.7)
        resultado_sarcasmo['nivel'] = 'high'
        resultado_sarcasmo['e_sarcastico'] = True
        score_ajustado = -sentimento_aparente  # Inverter o sentimento
        sentimento_aparente_label = 'positivo'  # Forçar o sentimento aparente para positivo
    
    # Verificar contradição entre problemas de funcionamento e qualidade
    if tem_problema_funcionamento and tem_palavra_positiva:
        resultado_sarcasmo['probabilidade'] = max(resultado_sarcasmo['probabilidade'], 0.65)
        resultado_sarcasmo['nivel'] = 'high' if resultado_sarcasmo['probabilidade'] >= 0.7 else 'medium'
        resultado_sarcasmo['e_sarcastico'] = True
        score_ajustado = -sentimento_aparente  # Inverter o sentimento
        sentimento_aparente_label = 'positivo'  # Forçar o sentimento aparente para positivo
    
    # Verificar caso especial de "só travou X vezes"
    if tem_problema_funcionamento and tem_muitas_vezes:
        resultado_sarcasmo['probabilidade'] = max(resultado_sarcasmo['probabilidade'], 0.7)
        resultado_sarcasmo['nivel'] = 'high'
        resultado_sarcasmo['e_sarcastico'] = True
        score_ajustado = -sentimento_aparente  # Inverter o sentimento
        sentimento_aparente_label = 'positivo'  # Forçar o sentimento aparente para positivo
    
    # Se menciona tempo e tem sentimento positivo, verificar com mais atenção
    if tem_palavra_tempo and tem_palavra_positiva and sentimento_aparente > 0.3:
        # Verificar padrões suspeitos
        verifica_numero = any(str(i) in texto for i in range(2, 10)) or any(f"{i}0" in texto for i in range(2, 10))
        tem_apenas = 'apenas' in texto.lower() or 'só' in texto.lower()
        
        # Verificar números especificamente altos para tempos de espera
        padrao_tempo_alto = re.search(r'(\d+)\s*(horas?|minutos?|dias?)', texto.lower())
        if padrao_tempo_alto:
            try:
                numero = int(padrao_tempo_alto.group(1))
                unidade = padrao_tempo_alto.group(2)
                
                # Thresholds para tempo excessivo
                if (unidade.startswith('hora') and numero >= 1) or \
                   (unidade.startswith('minuto') and numero >= 30) or \
                   (unidade.startswith('dia') and numero >= 2):
                    resultado_sarcasmo['probabilidade'] = max(resultado_sarcasmo['probabilidade'], 0.45)
                    
                # Tempos claramente excessivos
                if (unidade.startswith('hora') and numero >= 2) or \
                   (unidade.startswith('minuto') and numero >= 45) or \
                   (unidade.startswith('dia') and numero >= 4):
                    resultado_sarcasmo['probabilidade'] = max(resultado_sarcasmo['probabilidade'], 0.7)
                    resultado_sarcasmo['nivel'] = 'high'
                    resultado_sarcasmo['e_sarcastico'] = True
                    score_ajustado = -sentimento_aparente  # Inverter o sentimento
            except (ValueError, AttributeError, IndexError):
                pass
        
        # Se tem número e palavras como "apenas", aumentar a probabilidade de sarcasmo
        if verifica_numero and tem_apenas:
            resultado_sarcasmo['probabilidade'] = max(resultado_sarcasmo['probabilidade'], 0.65)
            resultado_sarcasmo['nivel'] = 'medium' if resultado_sarcasmo['probabilidade'] < 0.7 else 'high'
            resultado_sarcasmo['e_sarcastico'] = True
            score_ajustado = -sentimento_aparente  # Inverter o sentimento
    
    # Caso especial: detecção para padrões de "esperei X tempo + adjetivo positivo"
    padrao_esperado = re.search(r'esperei\s+(\d+)\s*(horas?|minutos?|dias?)', texto.lower())
    if padrao_esperado:
        try:
            numero = int(padrao_esperado.group(1))
            unidade = padrao_esperado.group(2)
            
            # Verificar se menciona adjetivo positivo após o tempo de espera
            pos_unidade = texto.lower().find(unidade) + len(unidade)
            resto_texto = texto.lower()[pos_unidade:]
            
            if any(palavra in resto_texto for palavra in palavras_positivas):
                # Tempo alto de espera seguido de adjetivo positivo é quase certamente sarcasmo
                if (unidade.startswith('hora') and numero >= 1) or \
                   (unidade.startswith('minuto') and numero >= 40) or \
                   (unidade.startswith('dia') and numero >= 1):
                    resultado_sarcasmo['probabilidade'] = max(resultado_sarcasmo['probabilidade'], 0.7)
                    resultado_sarcasmo['nivel'] = 'high'
                    resultado_sarcasmo['e_sarcastico'] = True
                    score_ajustado = -sentimento_aparente  # Inverter o sentimento
        except (ValueError, AttributeError, IndexError):
            pass
    
    # Determinar classificação final de sentimento
    sentimento_final = 'neutro'
    if score_ajustado >= 0.3:
        sentimento_final = 'positivo'
    elif score_ajustado <= -0.3:
        sentimento_final = 'negativo'
    
    # Se detectou sarcasmo com alta probabilidade, inverter o sentimento aparente
    if resultado_sarcasmo['e_sarcastico'] and resultado_sarcasmo['probabilidade'] >= 0.65:
        # PROTEÇÃO CONTRA FALSOS POSITIVOS DE SARCASMO
        # Verificar palavras claramente negativas no texto
        palavras_forte_negacao = ['horrível', 'péssimo', 'terrível', 'detestável', 'odiei', 'odeio', 'ruim', 'não presta', 
                                'não funciona', 'não serve', 'merda', 'inútil', 'lixo', 'nojento', 'decepção', 'decepcionado',
                                'vergonha', 'vergonhoso', 'porcaria', 'droga', 'falso', 'falsificado', 'fraude']
        
        # Contar quantas palavras de forte negação existem no texto
        texto_lower = texto.lower()
        qtd_palavras_negacao = sum(1 for palavra in palavras_forte_negacao if palavra in texto_lower)
        
        # Se houver várias palavras de forte negação, este texto provavelmente é genuinamente negativo, não sarcástico
        if qtd_palavras_negacao >= 2:
            logger.info(f"Ignorando detecção de sarcasmo em feedback claramente negativo: '{texto}'")
            logger.info(f"Encontradas {qtd_palavras_negacao} palavras de forte negação")
            # Manter o sentimento original (negativo)
            sentimento_final = sentimento_original_label
        else:
            # Inverter classificação de sentimento quando detecta sarcasmo
            if sentimento_aparente_label == 'positivo':
                sentimento_final = 'negativo'
                logger.info(f"Sentimento ajustado de positivo para negativo devido a sarcasmo {resultado_sarcasmo['nivel']}")
            elif sentimento_aparente_label == 'negativo':
                sentimento_final = 'positivo'
                logger.info(f"Sentimento ajustado de negativo para positivo devido a sarcasmo {resultado_sarcasmo['nivel']}")
    else:
        # Se não tem sarcasmo, manter o sentimento original
        if not resultado_sarcasmo['e_sarcastico'] or resultado_sarcasmo['probabilidade'] < 0.4:
            sentimento_final = sentimento_original_label
    
    # Formatar resultados
    resultado = {
        'sentimento': sentimento_final,
        'sentimento_original': sentimento_aparente_label,  # Usar o sentimento aparente como original
        'confianca': abs(score_ajustado),
        'score_original': sentimento,
        'score_ajustado': score_ajustado,
        'sarcasmo': {
            'detectado': resultado_sarcasmo['e_sarcastico'],
            'probabilidade': resultado_sarcasmo['probabilidade'],
            'nivel': resultado_sarcasmo['nivel']
        }
    }
    
    # Incluir metadados de varejo
    if is_varejo and 'categoria_varejo' in resultado_sentimento:
        resultado['categoria_varejo'] = resultado_sentimento['categoria_varejo']
        resultado['confianca_categoria'] = resultado_sentimento.get('confianca_categoria', 0.5)
    
    # Incluir detalhes adicionais se solicitado
    if incluir_detalhes:
        resultado['detalhes_sarcasmo'] = resultado_sarcasmo.get('detalhes', {})
        resultado['evidencias_sarcasmo'] = resultado_sarcasmo.get('evidencias', [])
        resultado['tem_sarcasmo'] = resultado_sarcasmo['e_sarcastico']
        resultado['nivel_sarcasmo'] = resultado_sarcasmo['nivel']
        resultado['probabilidade_sarcasmo'] = resultado_sarcasmo['probabilidade']
        if is_varejo:
            resultado['tipo_feedback'] = 'varejo'
    
    return resultado

def ajustar_sentimento_por_sarcasmo(sentimento: float, prob_sarcasmo: float, nivel_sarcasmo: str) -> float:
    """
    Ajusta o score de sentimento com base na probabilidade de sarcasmo.
    
    Args:
        sentimento: Score de sentimento (-1 a 1)
        prob_sarcasmo: Probabilidade de presença de sarcasmo (0 a 1)
        nivel_sarcasmo: Classificação do nível de sarcasmo ('low', 'medium', 'high')
        
    Returns:
        Score de sentimento ajustado
    """
    # Sem ajuste para baixa probabilidade de sarcasmo
    if prob_sarcasmo < 0.4:
        return sentimento
    
    # Ajuste para probabilidade média de sarcasmo
    if prob_sarcasmo < 0.7:
        # Reduzir a intensidade do sentimento
        if sentimento > 0:
            return sentimento * (1 - (prob_sarcasmo * 0.5))
        elif sentimento < 0:
            return sentimento * (1 - (prob_sarcasmo * 0.5))
        else:
            return 0
    
    # Ajuste para alta probabilidade de sarcasmo
    # Inverter completamente o sentimento
    if nivel_sarcasmo == 'high':
        return -sentimento
    else:
        # Reduzir drasticamente e possivelmente inverter
        factor = 1 - (2 * prob_sarcasmo)  # De 1 a -1
        return sentimento * factor

def integrar_detector_sarcasmo_avancado(app: Flask, 
                                      sentiment_analyzer: Any, 
                                      speech_handler: Optional[Any] = None, 
                                      data_handler: Optional[Any] = None) -> Flask:
    """
    Integra o detector de sarcasmo avançado ao sistema Flask.
    
    Args:
        app: Aplicação Flask
        sentiment_analyzer: Instância do analisador de sentimento
        speech_handler: Manipulador de fala (opcional)
        data_handler: Manipulador de dados (opcional)
        
    Returns:
        Aplicação Flask com o detector de sarcasmo integrado
    """
    logger.info("Integrando detector de sarcasmo avançado ao sistema...")
    
    # Criar um detector de sarcasmo e armazená-lo como atributo do analisador
    try:
        detector = SarcasmDetectorManager()
        sentiment_analyzer.detector_sarcasmo = detector
        logger.info("Detector de sarcasmo criado e anexado ao analisador de sentimento")
    except Exception as e:
        logger.error(f"Erro ao criar detector de sarcasmo: {e}")
        sentiment_analyzer.detector_sarcasmo = None
    
    # Armazenar a função original do analisador de sentimento
    if hasattr(sentiment_analyzer, 'analisar'):
        funcao_original = sentiment_analyzer.analisar
    else:
        funcao_original = sentiment_analyzer.analisar_sentimento
        sentiment_analyzer.analisar = sentiment_analyzer.analisar_sentimento
    
    # Substituir a função de análise por uma versão que inclui detecção de sarcasmo
    def analisar_com_sarcasmo(texto: str) -> Dict[str, Any]:
        # Chamar a função de análise integrada
        resultado = analisar_texto_com_sarcasmo(texto, sentiment_analyzer)
        return resultado
    
    # Substituir o método no analisador de sentimento
    sentiment_analyzer.analisar = analisar_com_sarcasmo
    sentiment_analyzer.analisar_com_sarcasmo = True
    
    # Adicionar configuração ao contexto da aplicação
    app.config['SARCASM_DETECTION_ENABLED'] = True
    app.config['SARCASM_CONFIG'] = SARCASMO_CONFIG
    
    # Criar um Blueprint para as rotas de sarcasmo
    sarcasmo_bp = Blueprint('sarcasmo', __name__, url_prefix='/api/sarcasmo')
    
    @sarcasmo_bp.route('/analisar', methods=['POST'])
    def analisar_sarcasmo():
        """Endpoint para analisar sarcasmo em um texto."""
        try:
            logger.info("Analisando sarcasmo...")
            dados = request.get_json()
            texto = dados.get('texto', '')
            
            if not texto:
                return jsonify({"sucesso": False, "erro": "Texto não fornecido"}), 400
            
            # Realizar análise com detalhes
            resultado = analisar_texto_com_sarcasmo(
                texto, 
                sentiment_analyzer, 
                incluir_detalhes=True
            )
            
            # Formatar resposta padronizada
            resposta = {
                "sucesso": True,
                "texto": texto,
                "sentimento": resultado.get('sentimento', 'neutro'),
                "sarcasmo": {
                    "detectado": resultado.get('tem_sarcasmo', False),
                    "nivel": resultado.get('nivel_sarcasmo', 'baixo'),
                    "probabilidade": resultado.get('probabilidade_sarcasmo', 0.0),
                    "detalhes": resultado.get('detalhes_sarcasmo', {})
                }
            }
            
            return jsonify(resposta)
        
        except Exception as e:
            logger.error(f"Erro ao analisar sarcasmo: {str(e)}")
            return jsonify({"sucesso": False, "erro": f"Erro ao processar solicitação: {str(e)}"}), 500
    
    # Registrar o blueprint
    app.register_blueprint(sarcasmo_bp)
    logger.info("Blueprint de sarcasmo registrado")
    
    logger.info("Detector de sarcasmo avançado integrado com sucesso!")
    return app 

class SentimentSarcasmAnalyzer:
    """
    Classe que combina análise de sentimento com detecção de sarcasmo
    """
    
    def __init__(self, sentimento_analyzer, sarcasmo_detector=None):
        """
        Inicializa o analisador combinado
        
        Args:
            sentimento_analyzer: Analisador de sentimentos
            sarcasmo_detector: Detector de sarcasmo (opcional)
        """
        self.sentimento = sentimento_analyzer
        self.sarcasmo = sarcasmo_detector or SarcasmDetectorManager()
    
    def analisar_texto_com_sarcasmo(self, texto, incluir_detalhes=False, is_retail=False):
        """
        Analisa um texto em busca de sarcasmo e ajusta o sentimento conforme necessário
        
        Args:
            texto (str): O texto a ser analisado
            incluir_detalhes (bool): Se deve incluir detalhes adicionais na resposta
            is_retail (bool): Se o texto é de varejo
            
        Returns:
            dict: Resultado da análise com detecção de sarcasmo
        """
        # Reutilizar a função global para evitar duplicação de código
        return analisar_texto_com_sarcasmo(
            texto=texto,
            sentiment_analyzer=self.sentimento,
            incluir_detalhes=incluir_detalhes,
            is_retail=is_retail
        )
        
    def analisar_texto(self, texto):
        """
        Analisa um texto para detectar sentimento e sarcasmo
        
        Args:
            texto (str): O texto a ser analisado
            
        Returns:
            dict: Resultado da análise completa
        """
        # Usa o método comum para analisar o sarcasmo
        return self.analisar_texto_com_sarcasmo(texto, incluir_detalhes=True)
        
    # Função legada mantida para compatibilidade
    def analisar(self, texto):
        """
        Método padrão de análise, mantido para compatibilidade
        
        Args:
            texto (str): O texto a ser analisado
            
        Returns:
            dict: Resultado da análise com sarcasmo
        """
        return self.analisar_texto(texto) 