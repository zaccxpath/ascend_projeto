import os
import re
import json
import logging
import traceback
from typing import Dict, Any, List, Tuple, Union, Optional
from utils.config import (
    STOPWORDS_PT, PALAVRAS_POSITIVAS, PALAVRAS_NEGATIVAS, 
    logger
)

# Configurar logger
logger = logging.getLogger(__name__)

# Verificar disponibilidade do NLTK
nltk_available = False
try:
    import nltk
    from nltk.corpus import stopwords
    from nltk.tokenize import word_tokenize
    nltk_available = True
except ImportError:
    logger.warning("NLTK não disponível. Usando recursos básicos.")

# Verificar disponibilidade do transformers
transformers_available = False
try:
    from transformers import pipeline, AutoModelForSequenceClassification, AutoTokenizer
    from huggingface_hub import snapshot_download
    transformers_available = True
except ImportError:
    logger.warning("Transformers não disponível. Usando modelo de fallback simples.")

# Importar o aprimorador de sentimentos para varejo, com tratamento de erro
try:
    from models.retail_sentiment_enhancer import RetailSentimentEnhancer, aplicar_melhorias_varejo
    retail_enhancer_available = True
    logger.info("RetailSentimentEnhancer importado com sucesso.")
except ImportError:
    retail_enhancer_available = False
    logger.warning("RetailSentimentEnhancer não disponível. Feedbacks de varejo não terão tratamento especializado.")

# Modelo de fallback simplificado baseado em regras
class ModeloFallbackSimplificado:
    """Implementação simplificada do modelo para uso quando o XLM-RoBERTa não está disponível"""
    
    def __init__(self):
        self.palavras_positivas = set(PALAVRAS_POSITIVAS)
        self.palavras_negativas = set(PALAVRAS_NEGATIVAS)
        logger.info("Modelo de fallback simplificado inicializado")
    
    def __call__(self, texto):
        """Simula a interface do pipeline de análise de sentimento do Transformers"""
        texto = texto.lower()
        
        # Tokenização simples
        tokens = re.findall(r'\b\w+\b', texto)
        
        # Contagem de palavras positivas e negativas
        count_pos = sum(1 for token in tokens if token in self.palavras_positivas)
        count_neg = sum(1 for token in tokens if token in self.palavras_negativas)
        
        # Verificar negações
        negacoes = ['não', 'nao', 'nunca', 'jamais', 'nem']
        for i in range(len(tokens) - 1):
            if tokens[i] in negacoes and i + 1 < len(tokens):
                if tokens[i + 1] in self.palavras_positivas:
                    count_pos -= 1
                    count_neg += 1
        
        # Expressões específicas
        if 'não gostei' in texto or 'nao gostei' in texto:
            count_neg += 2
        if 'adorei' in texto and not any(n in texto for n in negacoes):
            count_pos += 2
        
        # Determinar o sentimento
        if count_pos > count_neg:
            label = 'positive'
            score = min(0.5 + (count_pos - count_neg) * 0.1, 0.99)
        elif count_neg > count_pos:
            label = 'negative'
            score = min(0.5 + (count_neg - count_pos) * 0.1, 0.99)
        else:
            label = 'neutral'
            score = 0.7  # Confiança moderada para neutro
        
        logger.info(f"Modelo de fallback: texto='{texto[:20]}...', pos={count_pos}, neg={count_neg}, label={label}, score={score:.2f}")
        
        # Retornar no formato esperado pelo código existente
        return [{'label': label, 'score': score}]

class SentimentAnalyzer:
    """Classe para analisar sentimentos de textos usando XLM-RoBERTa"""
    
    def __init__(self):
        """Inicializa o analisador de sentimentos"""
        self.stopwords = set(stopwords.words('portuguese')) if nltk_available else set()
        self.max_length = 512  # Limite padrão para modelos transformer
        self.using_xlm_roberta = False
        self.sentiment_model = None
        self.sentiment_task = None
        self._in_sarcasm_analysis = False
        
        # Inicializar o enhancer para varejo se disponível
        self.retail_enhancer = RetailSentimentEnhancer() if retail_enhancer_available else None
        self.retail_mode_enabled = retail_enhancer_available
        
        try:
            self._inicializar_modelo()
        except Exception as e:
            logger.error(f"Erro ao inicializar modelo de sentimento: {e}")
            logger.warning("Usando modelo de fallback baseado em regras simples")
        
    def _inicializar_modelo(self):
        """Inicializa o modelo XLM-RoBERTa para análise de sentimento"""
        try:
            logger.info("Inicializando modelo de análise de sentimentos XLM-RoBERTa...")
            
            # Carregar modelo pré-treinado do Hugging Face
            self.sentiment_model = pipeline(
                "sentiment-analysis",
                model="cardiffnlp/twitter-xlm-roberta-base-sentiment",
                tokenizer="cardiffnlp/twitter-xlm-roberta-base-sentiment"
            )
            
            # Configurar a tarefa principal
            self.sentiment_task = self.sentiment_model
            
            # Definir explicitamente que estamos usando XLM-RoBERTa
            self.using_xlm_roberta = True
            logger.info("Modelo XLM-RoBERTa carregado com sucesso!")
            logger.info("Flag using_xlm_roberta definida como: True")
            
            # Testar modelo com algumas frases simples
            self._testar_modelo()
            
            # Realizar uma verificação final para confirmar que o modelo está funcionando
            self._verificar_status_xlm_roberta()
            
        except Exception as e:
            logger.error(f"Falha ao carregar modelo XLM-RoBERTa: {e}")
            self.using_xlm_roberta = False
            self.sentiment_model = None
            self.sentiment_task = None
            raise
    
    def _testar_modelo(self):
        """Testa o modelo com algumas frases simples para verificar se está funcionando"""
        try:
            logger.info("Testando modelo XLM-RoBERTa com frases simples...")
            
            frases_teste = [
                'Eu gostei muito desse produto, recomendo!',
                'Não estou satisfeito com a qualidade do serviço.',
                'O produto chegou no prazo, é resistente e funciona bem.'
            ]
            
            for frase in frases_teste:
                if self.sentiment_task:
                    resultado = self.sentiment_task(frase)[0]
                    logger.info(f"Teste: '{frase[:30]}...' => Label: {resultado['label']}, Score: {resultado['score']:.4f}")
                else:
                    logger.warning("Modelo de sentimento não está disponível para teste")
                    break
            
            logger.info("Teste do modelo concluído com sucesso!")
        except Exception as e:
            logger.error(f"Erro ao testar modelo: {e}")
            logger.error(traceback.format_exc())
        
    def inicializar_nltk(self):
        """Inicializa os recursos do NLTK necessários"""
        try:
            logger.info("Inicializando recursos NLTK...")
            nltk.download('punkt', quiet=True)
            nltk.download('stopwords', quiet=True)
            logger.info("Inicialização do NLTK concluída.")
        except Exception as e:
            logger.error(f"Erro ao inicializar NLTK: {e}")
            
    def _carregar_stopwords(self):
        """Carrega stopwords em português ou usa fallback"""
        try:
            from nltk.corpus import stopwords
            stop_words = set(stopwords.words('portuguese'))
            logger.info("Stopwords em português carregadas com sucesso.")
            return stop_words
        except Exception as e:
            logger.warning(f"Erro ao carregar stopwords: {e}. Usando lista predefinida.")
            return STOPWORDS_PT
            
    def tokenizar_texto(self, texto):
        """Tokeniza o texto em palavras, removendo stopwords"""
        try:
            # Tokenização do NLTK com fallback
            try:
                tokens = nltk.word_tokenize(texto.lower(), language='portuguese')
            except:
                # Fallback para tokenização simples com regex
                tokens = re.findall(r'\b\w+\b', texto.lower())
            
            # Remover stopwords
            tokens_sem_stopwords = [w for w in tokens if w not in self.stopwords]
            return tokens_sem_stopwords
        except Exception as e:
            logger.error(f"Erro ao tokenizar texto: {e}")
            # Fallback mais simples em caso de erro
            return texto.lower().split()
    
    def analisar_sentimento_basico(self, texto):
        """Análise de sentimento básica baseada em regras"""
        try:
            tokens = self.tokenizar_texto(texto)
            
            # Contagem de palavras positivas e negativas
            count_pos = sum(1 for token in tokens if token in PALAVRAS_POSITIVAS)
            count_neg = sum(1 for token in tokens if token in PALAVRAS_NEGATIVAS)
            
            # Verificar negações antes de palavras positivas no texto original
            negacoes = ['não', 'nao', 'nunca', 'jamais', 'nem']
            texto_lower = texto.lower()
            
            # Expressões específicas
            if 'não gostei' in texto_lower or 'nao gostei' in texto_lower:
                count_neg += 2
            if 'adorei' in texto_lower and not any(n + ' ' in texto_lower for n in negacoes):
                count_pos += 2
            
            # Determinar o sentimento
            if count_pos > count_neg + 1:  # Favorecimento leve a positivo
                sentimento = 'positivo'
                confianca = min(0.5 + (count_pos - count_neg) * 0.1, 0.95)
            elif count_neg > count_pos:
                sentimento = 'negativo'
                confianca = min(0.5 + (count_neg - count_pos) * 0.1, 0.95)
            else:
                sentimento = 'neutro'
                confianca = 0.6  # Confiança moderada
            
            # Ajustar intensidade baseada em palavras fortes
            palavras_fortes_pos = ['excelente', 'maravilhoso', 'perfeito', 'espetacular', 'incrível', 'amei', 'maravilha']
            palavras_fortes_neg = ['péssimo', 'horrível', 'terrível', 'detestei', 'odiei', 'lixo', 'nojento']
            
            if sentimento == 'positivo' and any(p in texto_lower for p in palavras_fortes_pos):
                confianca = min(confianca + 0.15, 0.98)
            elif sentimento == 'negativo' and any(p in texto_lower for p in palavras_fortes_neg):
                confianca = min(confianca + 0.15, 0.98)
            
            return {
                'sentimento': sentimento, 
                'confianca': confianca, 
                'tokens_relevantes': {
                    'positivos': [t for t in tokens if t in PALAVRAS_POSITIVAS],
                    'negativos': [t for t in tokens if t in PALAVRAS_NEGATIVAS]
                }
            }
        except Exception as e:
            logger.error(f"Erro na análise básica de sentimento: {e}")
            return {'sentimento': 'neutro', 'confianca': 0.5, 'tokens_relevantes': {'positivos': [], 'negativos': []}}
            
    def _mapear_sentimento(self, label, score):
        """Mapeia o label do XLM-RoBERTa para o formato esperado pelo sistema"""
        if label == 'positive':
            return 'positivo'
        elif label == 'negative':
            return 'negativo'
        else:
            return 'neutro'
            
    def _mapear_sentimento_xlm_roberta(self, resultado):
        """Mapeia o resultado do XLM-RoBERTa para o formato esperado pelo sistema"""
        label = resultado['label']
        score = resultado['score']
        
        logger.info(f"Mapeamento XLM-RoBERTa: {label} ({score:.4f}) -> {self._mapear_sentimento(label, score)}")
        
        return {
            'sentimento': self._mapear_sentimento(label, score),
            'confianca': score,
            'score': score if label == 'positive' else (-score if label == 'negative' else 0)
        }
    
    def analisar_sentimento(self, texto, debug=False, is_retail=False):
        """
        Analisa o sentimento de um texto usando modelo XLM-RoBERTa.
        
        Args:
            texto (str): Texto a ser analisado
            debug (bool): Se True, imprime informações de debug
            is_retail (bool): Se True, aplica melhorias específicas para varejo
            
        Returns:
            dict: Dicionário com resultado da análise
        """
        try:
            # Remover espaços extras e verificar se o texto é válido
            if not texto or not isinstance(texto, str):
                logger.warning("Texto inválido para análise de sentimento")
                return {
                    'sentimento': 'neutro',
                    'confianca': 0.5,
                    'score': 0.0,
                    'topicos': []
                }
            
            texto = texto.strip()
            if len(texto) == 0:
                return {
                    'sentimento': 'neutro',
                    'confianca': 0.5,
                    'score': 0.0,
                    'topicos': []
                }
                
            # Verificar se já temos o classificador inicializado
            if not hasattr(self, 'sentiment_task') or self.sentiment_task is None:
                logger.warning("Classificador não inicializado, inicializando agora...")
                self._inicializar_modelo()
                
            # Verificar limite de comprimento (para evitar problemas com textos muito longos)
            if len(texto) > self.max_length:
                texto = texto[:self.max_length]
                logger.info(f"Texto truncado para {self.max_length} caracteres")
            
            # Executar inferência
            result = self.sentiment_task(texto)
            
            # Extrair sentimento e score
            if len(result) > 0:
                sentiment_dict = result[0]
                
                # Obter label e score
                label = sentiment_dict['label']
                score = sentiment_dict['score']
                
                # Mapeamento para valores em português
                sentimento = self._mapear_sentimento(label, score)
                confianca = score
                
                # Normalizar a confiança para valor entre 0 e 1
                if confianca < 0:
                    confianca = 0
                elif confianca > 1:
                    confianca = 1
                    
                resultado = {
                    'sentimento': sentimento,
                    'confianca': confianca,
                    'score': confianca if sentimento == 'positivo' else (-confianca if sentimento == 'negativo' else 0)
                }
                
                # Aplicar melhorias específicas para varejo se solicitado
                if is_retail and self.retail_enhancer:
                    logger.info(f"Aplicando melhorias específicas para varejo ao texto: '{texto[:50]}...'")
                    resultado = aplicar_melhorias_varejo(texto, resultado)
                    logger.info(f"Análise de varejo: sentimento={resultado['sentimento']}, confiança={resultado['confianca']:.2f}")
                
                # Detectar sarcasmo - apenas se for sentiment task (para evitar recursão)
                if not hasattr(self, '_in_sarcasm_analysis') or not self._in_sarcasm_analysis:
                    # Verificar se temos um detector de sarcasmo disponível
                    try:
                        self._in_sarcasm_analysis = True
                        
                        # Se temos um detector direto, usar ele
                        if hasattr(self, 'detector_sarcasmo') and self.detector_sarcasmo is not None:
                            sarcasm_result = self.detector_sarcasmo.detect(texto, resultado['score'])
                            
                            # Adicionar resultado do sarcasmo
                            resultado['sarcasmo'] = {
                                'is_sarcastic': sarcasm_result.get('e_sarcastico', False),
                                'probability': sarcasm_result.get('probabilidade', 0.0),
                                'level': sarcasm_result.get('nivel', 'low'),
                                'evidence': sarcasm_result.get('evidencias', [])
                            }
                            
                            # Ajustar sentimento se necessário
                            if sarcasm_result.get('e_sarcastico', False):
                                # Guardar sentimento original
                                resultado['sentimento_original'] = resultado['sentimento']
                                
                                # PROTEÇÃO CONTRA FALSOS POSITIVOS DE SARCASMO
                                # Verificar palavras claramente negativas no texto, independente do nível de sarcasmo
                                palavras_forte_negacao = ['horrível', 'péssimo', 'terrível', 'detestável', 'odiei', 'odeio', 'ruim', 'não presta', 
                                                      'não funciona', 'não serve', 'merda', 'inútil', 'lixo', 'nojento', 'decepção', 'decepcionado',
                                                      'vergonha', 'vergonhoso', 'porcaria', 'droga', 'falso', 'falsificado', 'fraude']
                                
                                # Contar quantas palavras de forte negação existem no texto
                                texto_lower = texto.lower()
                                qtd_palavras_negacao = sum(1 for palavra in palavras_forte_negacao if palavra in texto_lower)
                                
                                # Ajustar sentimento conforme a regra
                                if resultado['sentimento'] == 'positivo':
                                    resultado['sentimento'] = 'negativo'
                                    logger.info("Sentimento ajustado de positivo para negativo devido ao sarcasmo")
                                elif resultado['sentimento'] == 'negativo':
                                    # Se houver várias palavras de forte negação, este texto provavelmente é genuinamente negativo, não sarcástico
                                    if qtd_palavras_negacao >= 2:
                                        logger.info(f"Ignorando detecção de sarcasmo em feedback claramente negativo: '{texto}'")
                                        logger.info(f"Encontradas {qtd_palavras_negacao} palavras de forte negação")
                                        # Manter o sentimento original (negativo)
                                        resultado['sentimento_final_ajustado'] = False
                                        resultado['protecao_falso_positivo'] = True
                                    else:
                                        # Em casos raros, sarcasmo forte em texto negativo pode ser positivo
                                        resultado['sentimento'] = 'positivo'
                                        logger.info("Sentimento ajustado de negativo para positivo devido a sarcasmo forte")
                    except Exception as e:
                        logger.error(f"Erro ao analisar sarcasmo: {e}")
                    finally:
                        # Limpar flag para próximas análises
                        self._in_sarcasm_analysis = False
                
                # Adicionar mais campos para compatibilidade com formato anterior (VADER)
                resultado['compound'] = resultado['score']
                resultado['pos'] = max(0, resultado['score'])
                resultado['neg'] = max(0, -resultado['score'])
                resultado['neu'] = 1.0 - (resultado['pos'] + resultado['neg'])
                
                # Extrair tópicos/palavras-chave do texto
                palavras_chave = self.extrair_palavras_chave(texto)
                resultado['topicos'] = palavras_chave
                
                logger.info(f"Tópicos extraídos: {palavras_chave}")
                
                return resultado
            else:
                # Se não conseguir análise pelo modelo, usar método básico
                logger.warning("Não foi possível obter resultado do modelo. Usando análise básica.")
                resultado = self.analisar_sentimento_basico(texto)
                
                # Normalizar resultado para formato esperado
                score = 0.5 if resultado['sentimento'] == 'positivo' else (-0.5 if resultado['sentimento'] == 'negativo' else 0.0)
                
                # Extrair tópicos/palavras-chave do texto
                palavras_chave = self.extrair_palavras_chave(texto)
                
                return {
                    'sentimento': resultado['sentimento'],
                    'confianca': resultado['confianca'],
                    'score': score,
                    'compound': score,
                    'pos': max(0, score),
                    'neg': max(0, -score),
                    'neu': 1.0 - (max(0, score) + max(0, -score)),
                    'topicos': palavras_chave
                }
        except Exception as e:
            logger.error(f"Erro na análise de sentimento: {e}")
            logger.error(traceback.format_exc())
            
            # Extrair palavras-chave mesmo em caso de erro
            try:
                palavras_chave = self.extrair_palavras_chave(texto)
            except:
                palavras_chave = []
                
            return {
                'sentimento': 'neutro',
                'confianca': 0.5,
                'score': 0.0,
                'compound': 0.0,
                'pos': 0.0,
                'neg': 0.0,
                'neu': 1.0,
                'topicos': palavras_chave
            }
            
    def analisar_feedback_varejo(self, texto, categoria=None, debug=False):
        """
        Método especializado para análise de feedbacks de varejo com maior precisão
        
        Args:
            texto (str): Texto do feedback
            categoria (str, optional): Categoria do feedback (se conhecida)
            debug (bool): Ativar logs de debug
            
        Returns:
            dict: Resultado da análise especializada
        """
        if not self.retail_enhancer:
            logger.warning("RetailSentimentEnhancer não disponível. Usando análise padrão.")
            return self.analisar_sentimento(texto, debug)
        
        # Realizar análise base
        resultado_base = self.analisar_sentimento(texto, debug, is_retail=False)
        
        # Aplicar melhorias específicas para varejo
        resultado_melhorado = aplicar_melhorias_varejo(texto, resultado_base)
        
        # Se categoria foi fornecida, adicionar ao resultado
        if categoria:
            resultado_melhorado['categoria_varejo'] = categoria
            
        # Log detalhado se debug estiver ativado
        if debug:
            logger.info(f"Análise original: {resultado_base['sentimento']} ({resultado_base['confianca']:.2f})")
            logger.info(f"Análise melhorada: {resultado_melhorado['sentimento']} ({resultado_melhorado['confianca']:.2f})")
            
            # Se houve mudança de sentimento
            if resultado_base['sentimento'] != resultado_melhorado['sentimento']:
                logger.info(f"Sentimento alterado: {resultado_base['sentimento']} -> {resultado_melhorado['sentimento']}")
        
        return resultado_melhorado

    def _verificar_status_xlm_roberta(self):
        """Verifica se o modelo XLM-RoBERTa está realmente disponível e funcional"""
        try:
            # Tentar fazer uma inferência simples
            if self.sentiment_model is not None and self.sentiment_task is not None:
                resultado = self.sentiment_task("Este é um teste.")
                
                if resultado and len(resultado) > 0:
                    # O modelo está funcionando
                    self.using_xlm_roberta = True
                    logger.info("Verificação de status do XLM-RoBERTa: ATIVO")
                    return True
                else:
                    logger.warning("Verificação de status do XLM-RoBERTa: Resultado vazio na inferência")
            else:
                logger.warning("Verificação de status do XLM-RoBERTa: Modelo não inicializado")
                
            # Se chegamos aqui, há algum problema
            self.using_xlm_roberta = False
            return False
        except Exception as e:
            logger.error(f"Erro ao verificar status do XLM-RoBERTa: {e}")
            self.using_xlm_roberta = False
            return False
            
    def status_xlm_roberta(self):
        """Retorna informações sobre o status do modelo XLM-RoBERTa"""
        status = {
            'ativo': self.using_xlm_roberta,
            'modelo_inicializado': self.sentiment_model is not None,
            'tarefa_inicializada': self.sentiment_task is not None
        }
        
        # Se não estiver ativo, mas o modelo parecer estar inicializado, 
        # tenta fazer uma verificação e correção
        if not status['ativo'] and status['modelo_inicializado']:
            logger.info("Disparidade detectada: modelo inicializado mas flag indica inativo. Corrigindo...")
            status['ativo'] = self._verificar_status_xlm_roberta()
            
        return status 

    def extrair_palavras_chave(self, texto):
        """
        Extrai palavras-chave significativas do texto para usar como tópicos
        
        Args:
            texto (str): Texto a ser analisado
            
        Returns:
            list: Lista de palavras-chave extraídas
        """
        try:
            if not texto:
                return []
                
            # Tokenizar o texto
            tokens = self.tokenizar_texto(texto)
            
            # Filtrar palavras relevantes (remover palavras curtas, stopwords, etc)
            palavras_relevantes = []
            for palavra in tokens:
                # Ignorar palavras muito curtas
                if len(palavra) < 3:
                    continue
                    
                # Ignorar stopwords
                if palavra in self.stopwords:
                    continue
                    
                # Ignorar números
                if palavra.isdigit():
                    continue
                    
                palavras_relevantes.append(palavra)
            
            # Obter as principais palavras-chave
            palavras_chave = []
            
            # Primeiro, extrair aspectos específicos do domínio
            aspectos_encontrados = []
            aspectos_mapeamento = {
                "produto": ["produto", "qualidade", "item", "design", "material"],
                "entrega": ["entrega", "prazo", "chegou", "frete", "transportadora", "enviado"],
                "atendimento": ["atendimento", "suporte", "contato", "vendedor", "ajuda"],
                "preço": ["preço", "valor", "custo", "caro", "barato", "desconto", "promoção"],
                "empresa": ["empresa", "loja", "marca", "site", "vendedor"]
            }
            
            for categoria, termos in aspectos_mapeamento.items():
                for termo in termos:
                    if termo in palavras_relevantes and categoria not in aspectos_encontrados:
                        aspectos_encontrados.append(categoria)
                        palavras_chave.append(termo)
            
            # Adicionar palavras relevantes que não são aspectos
            for palavra in palavras_relevantes:
                if palavra not in palavras_chave and len(palavras_chave) < 5:
                    palavras_chave.append(palavra)
            
            # Limitar a 3 palavras-chave
            return palavras_chave[:3]
            
        except Exception as e:
            logger.error(f"Erro ao extrair palavras-chave: {e}")
            return []

    def analisar_sentimento_final(self, texto, debug=False, is_retail=False):
        """
        Análise completa de sentimento com extração de tópicos
        
        Args:
            texto (str): Texto a analisar
            debug (bool): Se True, imprime informações adicionais
            is_retail (bool): Se True, aplica melhorias para varejo
            
        Returns:
            dict: Resultado completo da análise
        """
        try:
            # Obter análise de sentimento
            resultado = self.analisar_sentimento(texto, debug, is_retail)
            
            # Extrair palavras-chave como tópicos
            palavras_chave = self.extrair_palavras_chave(texto)
            
            # Adicionar tópicos ao resultado
            resultado['topicos'] = palavras_chave
            
            return resultado
        except Exception as e:
            logger.error(f"Erro na análise final: {e}")
            return {
                'sentimento': 'neutro',
                'confianca': 0.5,
                'score': 0.0,
                'topicos': []
            } 