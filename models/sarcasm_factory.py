"""
Fábrica de detectores de sarcasmo.

Este módulo fornece uma interface unificada para criar e obter
instâncias de detectores de sarcasmo para análise de sentimento.
"""

import logging
from typing import Dict, Any, Optional
import re

# Configurar logger
logger = logging.getLogger(__name__)

# Simplificação do Detector de Sarcasmo para contornar as importações problemáticas
class SarcasmDetectorManager:
    """
    Detector de sarcasmo simplificado que implementa regras específicas
    para identificar sarcasmo em textos em português.
    """
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.threshold_high = 0.7
        self.threshold_medium = 0.4
        
        # Frases sarcasticamente positivas
        self.frases_sarcasticas = [
            "que incrível", "que maravilha", "que bom", "que ótimo", "claro",
            "certamente", "obviamente", "com certeza", "exatamente", 
            "fantástico", "esperei apenas", "só esperei", "melhor impossível",
            "nossa", "uau", "caramba", "puxa", "genial", "brilhante"
        ]
        
        # Expressões de afirmação irônica frequentemente usadas em sarcasmo
        self.expressoes_afirmacao_ironica = [
            "claro", "com certeza", "certamente", "obviamente", "sem dúvida",
            "naturalmente", "evidentemente", "definitivamente", "absolutamente",
            "sim, sim", "aham", "ta bom", "sei"
        ]
        
        # Palavras de contexto negativo
        self.palavras_negativas = [
            "horas", "minutos", "fila", "demora", "lentidão", "péssimo",
            "ruim", "horrível", "terrível", "irritante", "cansativo",
            "problema", "erro", "falho", "dias", "semanas", 
            "quebrou", "quebrado", "parou", "estragou", "falhou", 
            "defeito", "estrago", "pane", "danificado", "piorou",
            "travou", "travado", "trava", "tela azul", "congela", "congelou",
            "lento", "lentidão", "bug", "bugs", "falha", "pifou",
            "atrasa", "atrasou", "atraso", "sempre", "nunca", "jamais"
        ]
        
        # Palavras que indicam alta qualidade (para detectar contradições)
        self.palavras_qualidade = [
            "qualidade", "excelente", "excepcional", "ótima", "perfeita", 
            "incrível", "impressionante", "durável", "resistente",
            "bom", "muito bom", "ótimo", "funciona bem", "recomendo"
        ]
        
        # Padrões de expressão regular para detecção de sarcasmo
        self.padroes_sarcasmo = [
            # Padrão específico para tempo de espera
            re.compile(r'(?i)(incrível|excelente|ótimo|maravilhoso|perfeito|bom).+(apenas|só)?\s+\d+\s+(horas?|minutos?|dias?|semanas?)'),
            re.compile(r'(?i)(apenas|só)\s+\d+\s+(horas?|minutos?|dias?|semanas?)'),
            re.compile(r'(?i)(nossa|uau|puxa|caramba).+(atendimento|serviço).+(esperei|aguardei)'),
            # Combinação de positivo com longo tempo de espera
            re.compile(r'(?i)(atendimento|serviço)\s+(incrível|excelente|ótimo|maravilhoso|perfeito).+(esperei|aguardei)'),
            re.compile(r'(?i)(esperei|aguardei).+(apenas|só)?\s+\d+\s+(horas?|minutos?|dias?|semanas?)'),
            # Padrão para capturar problemas de qualidade
            re.compile(r'(?i)(quebrou|quebrado|parou|estragou|falhou|defeito|estrago|pane|danificado).+\d+.+(dias?|semanas?|meses?|horas?)'),
            re.compile(r'(?i)(quebrou|quebrado|parou|estragou|falhou|defeito|estrago|pane|danificado).+(qualidade|excelente|excepcional|ótima|perfeita|incrível)'),
            re.compile(r'(?i)(qualidade|excelente|excepcional|ótima|perfeita|incrível).+(quebrou|quebrado|parou|estragou|falhou|defeito|estrago|pane|danificado)'),
            # Padrões para problemas de funcionamento
            re.compile(r'(?i)(travou|trava|lento|congela|congelou|bug|bugs).+\d+.+vezes'),
            re.compile(r'(?i)(bom|ótimo|excelente).+(só|apenas)?.+(travou|trava|lento|congela|congelou|bug|bugs)'),
            re.compile(r'(?i)(só|apenas).+travou.+\d+.+vezes'),
            # Padrões para afirmações irônicas seguidas de contexto negativo
            re.compile(r'(?i)(claro|com certeza|certamente|obviamente|sem dúvida).+(sempre|nunca|jamais|atrasa|problema)'),
            re.compile(r'(?i)(claro|com certeza|obviamente).+vou.+(comprar|usar|recomendar).+(loja|produto|serviço).+(sempre|nunca|atrasa|problema)'),
            re.compile(r'(?i)(claro|com certeza|obviamente).+(de novo|novamente|outra vez)')
        ]
    
    def detect(self, text: str, sentiment: Optional[float] = None) -> Dict[str, Any]:
        """
        Detecta sarcasmo com base em regras linguísticas específicas.
        
        Args:
            text: Texto a ser analisado
            sentiment: Valor de sentimento (-1 a 1)
            
        Returns:
            Dicionário com resultados da detecção
        """
        text_lower = text.lower()
        
        # Verificar indícios de sarcasmo no texto
        tem_indicio_sarcasmo = False
        marcadores_encontrados = []
        palavras_negativas_encontradas = []
        padroes_encontrados = []
        expressoes_ironicas_encontradas = []
        
        # Procurar frases sarcásticas
        for frase in self.frases_sarcasticas:
            if frase in text_lower:
                tem_indicio_sarcasmo = True
                marcadores_encontrados.append(frase)
                
        # Procurar expressões de afirmação irônica
        for expressao in self.expressoes_afirmacao_ironica:
            if expressao in text_lower:
                expressoes_ironicas_encontradas.append(expressao)
        
        # Procurar palavras negativas
        for palavra in self.palavras_negativas:
            if palavra in text_lower:
                palavras_negativas_encontradas.append(palavra)
        
        # Verificar padrões de expressão regular
        for padrao in self.padroes_sarcasmo:
            match = padrao.search(text_lower)
            if match:
                padroes_encontrados.append(match.group(0))
                tem_indicio_sarcasmo = True
        
        # Calcular score de sarcasmo
        score_sarcasmo = 0.0
        
        # Adicionar pontuação por marcadores encontrados
        if marcadores_encontrados:
            score_sarcasmo += min(0.5, len(marcadores_encontrados) * 0.1)
        
        # Adicionar pontuação por palavras negativas encontradas
        if palavras_negativas_encontradas:
            score_sarcasmo += min(0.3, len(palavras_negativas_encontradas) * 0.05)
            
        # Verificar combinação de expressões irônicas com palavras negativas
        if expressoes_ironicas_encontradas and palavras_negativas_encontradas:
            score_sarcasmo += 0.4
            tem_indicio_sarcasmo = True
        
        # Verificar contradição entre problema de qualidade e avaliação positiva
        tem_problema_qualidade = any(p in text_lower for p in ["quebrou", "quebrado", "parou", "estragou", "falhou", 
                                                             "defeito", "estrago", "pane", "danificado"])
        tem_problema_funcionamento = any(p in text_lower for p in ["travou", "trava", "lento", "lentidão", "congela", "bug", "bugs", "falha"])
        tem_palavras_qualidade = any(p in text_lower for p in self.palavras_qualidade)
        
        # Verificar se menciona um número elevado de problemas
        padrao_numero_vezes = re.search(r'(\d+)\s+vezes', text_lower)
        tem_muitas_vezes = False
        if padrao_numero_vezes:
            try:
                numero = int(padrao_numero_vezes.group(1))
                if numero >= 3:  # Consideramos 3 ou mais como um número excessivo
                    tem_muitas_vezes = True
                    score_sarcasmo += min(0.5, numero * 0.05)  # Pontuação proporcional ao número
            except (ValueError, AttributeError):
                pass
        
        # Se encontrar problema de qualidade junto com palavras positivas, é provável sarcasmo
        if tem_problema_qualidade and tem_palavras_qualidade:
            score_sarcasmo += 0.6
            tem_indicio_sarcasmo = True
            
        # Se encontrar problema de funcionamento junto com palavras positivas, é provável sarcasmo
        if tem_problema_funcionamento and tem_palavras_qualidade:
            score_sarcasmo += 0.5
            tem_indicio_sarcasmo = True
            
        # Se mencionar muitas vezes e tiver problema de funcionamento, é provável sarcasmo
        if tem_muitas_vezes and tem_problema_funcionamento:
            score_sarcasmo += 0.6
            tem_indicio_sarcasmo = True
            
        # Se encontrar problema de qualidade e tiver sentimento positivo, é provável sarcasmo
        if (tem_problema_qualidade or tem_problema_funcionamento) and sentiment is not None and sentiment > 0.3:
            score_sarcasmo += 0.5
            tem_indicio_sarcasmo = True
        
        # Adicionar pontuação por padrões encontrados (maior peso)
        if padroes_encontrados:
            score_sarcasmo += min(0.6, len(padroes_encontrados) * 0.2)
        
        # Bonus para combinação de marcadores positivos com palavras negativas
        if marcadores_encontrados and palavras_negativas_encontradas:
            score_sarcasmo += 0.3
        
        # Análise especial para tempo de espera
        tempo_espera_pattern = any("hora" in p or "minuto" in p or "dia" in p or "semana" in p for p in padroes_encontrados)
        sentimento_positivo = sentiment is not None and sentiment > 0.3
        
        # Verificar se há palavras ou padrões específicos de sarcasmo relacionado a tempo
        menciona_numero = re.search(r'(\d+)\s+(horas?|minutos?|dias?)', text_lower)
        tem_apenas_so = 'apenas' in text_lower or 'só' in text_lower
        
        # Verificar magnitude do número referente a tempo - números altos em contexto positivo são mais provavelmente sarcasmo
        if menciona_numero:
            try:
                numero = int(menciona_numero.group(1))
                unidade = menciona_numero.group(2)
                
                # Valores que geralmente indicam sarcasmo quando combinados com sentimento positivo
                if unidade.startswith('hora') and numero >= 2:
                    if sentimento_positivo:
                        score_sarcasmo += 0.3
                elif unidade.startswith('minuto') and numero >= 40:
                    if sentimento_positivo:
                        score_sarcasmo += 0.35  # Aumentar pontuação para minutos
                elif unidade.startswith('dia') and numero >= 1:
                    if sentimento_positivo:
                        score_sarcasmo += 0.35
                
                # Detectar palavras positivas próximas a menções de tempo elevado
                palavras_positivas_proximas = ['excelente', 'fantástico', 'maravilhoso', 'ótimo', 'incrível', 'bom']
                for palavra in palavras_positivas_proximas:
                    if palavra in text_lower and (
                        (unidade.startswith('hora') and numero >= 1) or
                        (unidade.startswith('minuto') and numero >= 30) or
                        (unidade.startswith('dia') and numero >= 1)
                    ):
                        score_sarcasmo += 0.3  # Forte indício de sarcasmo
                        tem_indicio_sarcasmo = True
                        break
            except (ValueError, AttributeError):
                pass  # Ignora erros na conversão
        
        # Bônus especial para padrões de tempo de espera com sentimento positivo
        if tempo_espera_pattern and sentimento_positivo:
            if tem_apenas_so:
                score_sarcasmo += 0.4
            else:
                score_sarcasmo += 0.2
        
        # Pontuação por pontuação excessiva
        exclamacoes = text.count('!')
        interrogacoes = text.count('?')
        if exclamacoes + interrogacoes > 1:
            score_sarcasmo += min(0.2, (exclamacoes + interrogacoes) * 0.05)
        
        # Classificar nível de sarcasmo
        if score_sarcasmo >= self.threshold_high:
            nivel = "high"
            e_sarcastico = True
        elif score_sarcasmo >= self.threshold_medium:
            nivel = "medium"
            e_sarcastico = True
        else:
            nivel = "low"
            e_sarcastico = False
        
        # Calcular sentimento ajustado
        sentimento_atual = 0.0 if sentiment is None else sentiment
        sentimento_ajustado = -sentimento_atual if e_sarcastico else sentimento_atual
        
        return {
            "probabilidade": score_sarcasmo,
            "nivel": nivel,
            "e_sarcastico": e_sarcastico,
            "sentimento_ajustado": sentimento_ajustado,
            "evidencias": marcadores_encontrados + palavras_negativas_encontradas + padroes_encontrados + expressoes_ironicas_encontradas,
            "detalhes": {
                "rule": {
                    "score": score_sarcasmo,
                    "details": {
                        "markers": marcadores_encontrados,
                        "negative_context": palavras_negativas_encontradas,
                        "patterns": padroes_encontrados,
                        "ironic_expressions": expressoes_ironicas_encontradas
                    },
                    "reasons": ["Detecção baseada em regras"]
                },
                "contradiction": {
                    "score": 0.0
                },
                "ml": {
                    "score": 0.0
                }
            }
        }


class SarcasmDetectorFactory:
    """
    Fábrica para criação e gerenciamento de detectores de sarcasmo.
    
    Esta classe implementa o padrão Singleton para garantir apenas
    uma instância do detector de sarcasmo em toda a aplicação.
    """
    
    _instance = None
    _detector = None
    
    def __new__(cls):
        """Implementação do padrão Singleton."""
        if cls._instance is None:
            cls._instance = super(SarcasmDetectorFactory, cls).__new__(cls)
        return cls._instance
    
    def __init__(self):
        """Inicializa a fábrica se ainda não tiver sido inicializada."""
        if self._detector is None:
            self._inicializar_detector()
    
    def _inicializar_detector(self):
        """Inicializa o detector de sarcasmo principal."""
        try:
            logger.info("Inicializando detector de sarcasmo via fábrica")
            self._detector = SarcasmDetectorManager()
            logger.info("Detector de sarcasmo inicializado com sucesso")
        except Exception as e:
            logger.error(f"Erro ao inicializar detector de sarcasmo: {e}")
            raise
    
    def obter_detector(self) -> SarcasmDetectorManager:
        """
        Retorna a instância existente do detector de sarcasmo.
        
        Returns:
            SarcasmDetectorManager: Instância do detector de sarcasmo
        """
        if self._detector is None:
            self._inicializar_detector()
        return self._detector
    
    def reinicializar_detector(self) -> SarcasmDetectorManager:
        """
        Reinicializa o detector de sarcasmo.
        
        Útil quando as configurações são alteradas e precisam ser
        aplicadas sem reiniciar a aplicação.
        
        Returns:
            SarcasmDetectorManager: Nova instância do detector de sarcasmo
        """
        logger.info("Reinicializando detector de sarcasmo")
        self._detector = None
        self._inicializar_detector()
        return self._detector

def obter_detector_sarcasmo() -> SarcasmDetectorManager:
    """
    Função de conveniência para obter o detector de sarcasmo.
    
    Returns:
        SarcasmDetectorManager: Instância do detector de sarcasmo
    """
    factory = SarcasmDetectorFactory()
    return factory.obter_detector()

def detectar_sarcasmo(texto: str, sentimento: float = 0.0) -> Dict[str, Any]:
    """
    Função de conveniência para detectar sarcasmo em um texto.
    
    Args:
        texto (str): Texto a ser analisado
        sentimento (float): Valor de sentimento (-1 a 1)
        
    Returns:
        Dict[str, Any]: Resultado da detecção de sarcasmo
    """
    detector = obter_detector_sarcasmo()
    return detector.detect(texto, sentimento)

def integrar_detector_sarcasmo_ao_sistema_melhorado(app, sentiment_analyzer, speech_handler, data_handler):
    """
    Integra o detector de sarcasmo melhorado ao sistema Flask.
    
    Args:
        app: Aplicação Flask
        sentiment_analyzer: Analisador de sentimento
        speech_handler: Manipulador de fala
        data_handler: Manipulador de dados
        
    Returns:
        app: Aplicação Flask com detector de sarcasmo integrado
    """
    try:
        # Obter detector de sarcasmo da fábrica
        detector = obter_detector_sarcasmo()
        
        # Verificar se o detector foi inicializado corretamente
        if detector is None:
            logger.error("Detector de sarcasmo não foi inicializado corretamente")
            app.config['SARCASM_DETECTION_ENABLED'] = False
            return app
        
        # Modificar o analisador de sentimento para incluir detecção de sarcasmo
        original_analyze = sentiment_analyzer.analisar_sentimento
        
        def analyze_with_sarcasm(text, *args, **kwargs):
            # Chamar o método original de análise
            result = original_analyze(text, *args, **kwargs)
            
            # Se o resultado for None ou não for um dicionário, retornar como está
            if result is None or not isinstance(result, dict):
                logger.warning("Resultado da análise de sentimento não é válido")
                return result
            
            try:
                # Adicionar análise de sarcasmo
                sentiment_value = result.get('sentimento', 0)
                if isinstance(sentiment_value, str):
                    # Converter string de sentimento para valor numérico para o detector
                    sentiment_numeric = {
                        'positivo': 0.6,
                        'neutro': 0.0,
                        'negativo': -0.6
                    }.get(sentiment_value, 0.0)
                else:
                    sentiment_numeric = sentiment_value
                
                sarcasm_result = detector.detect(text, sentiment_numeric)
                
                # Adicionar informações de sarcasmo ao resultado
                result['sarcasmo'] = {
                    'is_sarcastic': sarcasm_result.get('e_sarcastico', False),
                    'probability': sarcasm_result.get('probabilidade', 0.0),
                    'level': sarcasm_result.get('nivel', 'baixo'),
                    'evidence': sarcasm_result.get('evidencias', [])
                }
                
                # Ajustar o sentimento se necessário
                if sarcasm_result.get('e_sarcastico', False):
                    # Manter o sentimento original
                    result['sentimento_original'] = result['sentimento']
                    
                    # Se for string, inverter categoricamente
                    if isinstance(sentiment_value, str):
                        inverse_map = {
                            'positivo': 'negativo',
                            'negativo': 'positivo',
                            'neutro': 'neutro'
                        }
                        result['sentimento'] = inverse_map.get(sentiment_value, sentiment_value)
                    else:
                        # Se for numérico, usar o valor ajustado do detector
                        result['sentimento_ajustado'] = sarcasm_result.get('sentimento_ajustado', -sentiment_numeric)
                        
                    logger.info(f"Sarcasmo detectado ({sarcasm_result.get('nivel', 'baixo')}). "
                              f"Sentimento ajustado de {result.get('sentimento_original', 'desconhecido')} "
                              f"para {result.get('sentimento', 'desconhecido')}")
            except Exception as e:
                logger.error(f"Erro ao processar sarcasmo: {e}")
                # Adicionar informação mínima de sarcasmo para não quebrar o fluxo
                result['sarcasmo'] = {
                    'is_sarcastic': False,
                    'probability': 0.0,
                    'level': 'baixo',
                    'evidence': [],
                    'error': str(e)
                }
            
            return result
        
        # Substituir o método original pelo modificado
        sentiment_analyzer.analisar_sentimento = analyze_with_sarcasm
        # Também adicionar o método analisar para compatibilidade
        sentiment_analyzer.analisar = analyze_with_sarcasm
        
        # Adicionar o detector ao contexto da aplicação
        app.config['SARCASM_DETECTOR'] = detector
        app.config['SARCASM_DETECTION_ENABLED'] = True
        
        logger.info("Detector de sarcasmo melhorado integrado ao sistema com sucesso")
        
        return app
    
    except Exception as e:
        logger.error(f"Erro ao integrar detector de sarcasmo ao sistema: {e}")
        # Configurar a aplicação para funcionar sem detector de sarcasmo
        app.config['SARCASM_DETECTION_ENABLED'] = False
        # Garantir que sentiment_analyzer tenha o método analisar
        if hasattr(sentiment_analyzer, 'analisar_sentimento') and not hasattr(sentiment_analyzer, 'analisar'):
            sentiment_analyzer.analisar = sentiment_analyzer.analisar_sentimento
        # Retornar a aplicação sem modificações em caso de erro
        return app