"""
Módulo de detecção de sarcasmo aprimorado.
Sistema avançado para detecção de sarcasmo em textos em português usando múltiplas estratégias:
1. Análise baseada em regras e padrões linguísticos
2. Modelos de machine learning específicos para sarcasmo
3. Detecção de contradições semânticas
4. Ensemble adaptativo que combina as abordagens acima
"""
import re
import logging
import numpy as np
from typing import Dict, List, Tuple, Any, Optional, Union, Set
from abc import ABC, abstractmethod

# Tentativa de importar dependências opcionais
try:
    from transformers import AutoModelForSequenceClassification, AutoTokenizer, pipeline
    import torch
    from torch.nn import functional as F
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False

# Configurar logger
logger = logging.getLogger(__name__)

class SarcasmDetectorInterface(ABC):
    """
    Interface abstrata para detectores de sarcasmo.
    Garante que todos os detectores implementem o mesmo contrato.
    """
    @abstractmethod
    def detect(self, text: str, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Método abstrato para detecção de sarcasmo.
        
        Args:
            text: Texto a ser analisado
            context: Informações contextuais (sentimento, etc)
        
        Returns:
            Dicionário com resultados da detecção
        """
        pass

class ContextualFeatureExtractor:
    """
    Extrai características contextuais e linguísticas de um texto
    que podem ser indicativas de sarcasmo.
    """
    def __init__(self):
        # Características positivas em contextos sarcásticos
        self.positive_features = {
            'lexical': {
                'marcadores': [
                    'nossa', 'uau', 'incrível', 'maravilhoso', 'excelente', 
                    'perfeito', 'ótimo', 'fantástico', 'brilhante', 'sensacional',
                    'genial', 'extraordinário', 'espetacular', 'magnífico'
                ],
                'intensificadores': [
                    'muito', 'super', 'hiper', 'ultra', 'mega', 'extremamente', 
                    'absolutamente', 'completamente', 'totalmente'
                ],
                'concordância': [
                    'claro', 'certamente', 'obviamente', 'evidentemente', 
                    'sem dúvida', 'com certeza', 'exatamente', 'definitivamente'
                ]
            },
            'patterns': [
                r'que (incrível|maravilh|fantástic|espetacular|genial)',
                r'(nossa|puxa|caramba|uau|meu deus)',
                r'(melhor|maior|mais) (.*?) (de todos|do mundo|da história|que já vi)'
            ]
        }
        
        # Características negativas em contextos sarcásticos
        self.negative_context = {
            'lexical': {
                'temporal': [
                    'só', 'apenas', 'somente', 'esperei', 'esperar', 'demorou',
                    'minutos', 'horas', 'dias', 'semanas', 'meses'
                ],
                'problemas': [
                    'fila', 'demora', 'problema', 'erro', 'falha', 'bug', 'defeito',
                    'quebrado', 'travado', 'lento', 'péssimo', 'terrível'
                ],
                'negações': [
                    'não', 'nunca', 'jamais', 'nada', 'nenhum', 'nem'
                ]
            },
            'patterns': [
                r'(apenas|só) (\d+) (minutos|horas|dias)',
                r'(depois|mais) (de|\d+) (tentativas|ligações|emails)',
                r'(não|sem) (funciona|resolver|atender|responder)'
            ]
        }
        
        # Compilar os padrões de regex
        self.compiled_patterns = {
            'positive': [re.compile(p, re.IGNORECASE) for p in self.positive_features['patterns']],
            'negative': [re.compile(p, re.IGNORECASE) for p in self.negative_context['patterns']]
        }
    
    def extract(self, text: str) -> Dict[str, Any]:
        """
        Extrai características contextuais do texto.
        
        Args:
            text: Texto para análise
            
        Returns:
            Dicionário com características extraídas
        """
        text_lower = text.lower()
        words = set(re.findall(r'\b\w+\b', text_lower))
        
        # Extrair marcadores lexicais
        pos_markers = self._find_lexical_items(words, text_lower, self.positive_features['lexical'])
        neg_markers = self._find_lexical_items(words, text_lower, self.negative_context['lexical'])
        
        # Extrair padrões de regex
        pos_patterns = self._find_patterns(text, self.compiled_patterns['positive'])
        neg_patterns = self._find_patterns(text, self.compiled_patterns['negative'])
        
        # Análise de contraste entre positivo e negativo
        has_contrast = (len(pos_markers) > 0 and len(neg_markers) > 0) or \
                       (len(pos_patterns) > 0 and len(neg_patterns) > 0)
        
        # Análise de sinais de pontuação
        exclamations = text.count('!')
        questions = text.count('?')
        
        return {
            'lexical': {
                'positive': pos_markers,
                'negative': neg_markers
            },
            'patterns': {
                'positive': pos_patterns,
                'negative': neg_patterns
            },
            'punctuation': {
                'exclamations': exclamations,
                'questions': questions,
                'excessive': (exclamations + questions) > 2
            },
            'contrast': has_contrast,
            'sarcasm_score': self._calculate_sarcasm_score(
                pos_markers, neg_markers, pos_patterns, neg_patterns,
                exclamations, questions, has_contrast
            )
        }
    
    def _find_lexical_items(self, words: Set[str], text: str, categories: Dict[str, List[str]]) -> List[Dict[str, str]]:
        """
        Encontra itens lexicais no texto.
        
        Args:
            words: Conjunto de palavras no texto
            text: Texto completo
            categories: Categorias de itens lexicais
            
        Returns:
            Lista de itens encontrados com suas categorias
        """
        found_items = []
        
        for category, items in categories.items():
            for item in items:
                if ' ' in item:  # Frases
                    if item in text:
                        found_items.append({'item': item, 'category': category})
                else:  # Palavras individuais
                    if item in words:
                        found_items.append({'item': item, 'category': category})
        
        return found_items
    
    def _find_patterns(self, text: str, patterns: List[re.Pattern]) -> List[Dict[str, str]]:
        """
        Encontra padrões de regex no texto.
        
        Args:
            text: Texto para análise
            patterns: Padrões compilados
            
        Returns:
            Lista de padrões encontrados
        """
        found_patterns = []
        
        for pattern in patterns:
            matches = pattern.findall(text)
            if matches:
                found_patterns.append({
                    'pattern': pattern.pattern,
                    'matches': matches
                })
        
        return found_patterns
    
    def _calculate_sarcasm_score(
        self, 
        pos_markers: List[Dict[str, str]], 
        neg_markers: List[Dict[str, str]],
        pos_patterns: List[Dict[str, str]], 
        neg_patterns: List[Dict[str, str]],
        exclamations: int, 
        questions: int,
        has_contrast: bool
    ) -> float:
        """
        Calcula uma pontuação preliminar de sarcasmo com base nas características.
        
        Args:
            Vários indicadores extraídos do texto
            
        Returns:
            Pontuação de sarcasmo entre 0.0 e 1.0
        """
        score = 0.0
        
        # Pontuação baseada em marcadores lexicais
        score += min(1.0, len(pos_markers) * 0.15)
        score += min(1.0, len(neg_markers) * 0.1)
        
        # Pontuação baseada em padrões
        score += min(1.0, len(pos_patterns) * 0.2)
        score += min(1.0, len(neg_patterns) * 0.2)
        
        # Pontuação baseada em contraste
        if has_contrast:
            score += 0.3
        
        # Pontuação baseada em pontuação
        score += min(0.2, (exclamations + questions) * 0.05)
        
        # Normalizar para 0-1
        return min(1.0, score)


class RuleBasedDetector(SarcasmDetectorInterface):
    """
    Detector de sarcasmo baseado em regras linguísticas e padrões.
    """
    def __init__(self, config: Dict[str, Any] = None):
        """
        Inicializa o detector baseado em regras.
        
        Args:
            config: Configurações opcionais
        """
        self.config = config or {}
        self.feature_extractor = ContextualFeatureExtractor()
        
        # Thresholds de classificação
        self.threshold_high = self.config.get('thresholds', {}).get('high', 0.7)
        self.threshold_medium = self.config.get('thresholds', {}).get('medium', 0.4)

    def detect(self, text: str, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Detecta sarcasmo usando regras e padrões linguísticos.
        
        Args:
            text: Texto para análise
            context: Contexto adicional (sentimento, etc)
            
        Returns:
            Resultados da detecção
        """
        # Extrair características textuais
        features = self.feature_extractor.extract(text)
        
        # Calcular pontuação base
        base_score = features['sarcasm_score']
        
        # Ajustar com base no sentimento, se disponível
        sentiment_score = context.get('sentiment', 0.0) if context else 0.0
        
        # Adicionar bônus para contradições entre sentimento e conteúdo
        if sentiment_score > 0.3 and len(features['lexical']['negative']) > 0:
            # Texto com sentimento positivo, mas com marcadores negativos
            contradiction_bonus = 0.25
            base_score = min(1.0, base_score + contradiction_bonus)
        
        elif sentiment_score < -0.3 and len(features['lexical']['positive']) > 0:
            # Texto com sentimento negativo, mas com marcadores positivos
            contradiction_bonus = 0.25
            base_score = min(1.0, base_score + contradiction_bonus)
        
        # Classificar o nível de sarcasmo
        if base_score >= self.threshold_high:
            level = "high"
            is_sarcastic = True
        elif base_score >= self.threshold_medium:
            level = "medium"
            is_sarcastic = True
        else:
            level = "low"
            is_sarcastic = False
        
        # Construir evidências encontradas
        evidence = []
        if features['lexical']['positive']:
            evidence.append(f"Marcadores positivos: {', '.join([m['item'] for m in features['lexical']['positive']])}")
        if features['lexical']['negative']:
            evidence.append(f"Contexto negativo: {', '.join([m['item'] for m in features['lexical']['negative']])}")
        if features['contrast']:
            evidence.append("Contraste entre elementos positivos e negativos")
        if features['punctuation']['excessive']:
            evidence.append("Uso excessivo de pontuação")
        
        return {
            'score': base_score,
            'level': level,
            'is_sarcastic': is_sarcastic,
            'evidences': evidence,
            'features': features
        }


class MLBasedDetector(SarcasmDetectorInterface):
    """
    Detector de sarcasmo baseado em modelos de machine learning.
    """
    def __init__(self, config: Dict[str, Any] = None):
        """
        Inicializa o detector baseado em ML.
        
        Args:
            config: Configurações opcionais
        """
        self.config = config or {}
        
        # Thresholds de classificação
        self.threshold_high = self.config.get('thresholds', {}).get('high', 0.7)
        self.threshold_medium = self.config.get('thresholds', {}).get('medium', 0.4)
        
        # Configurações do modelo
        self.model_name = self.config.get('model_name', 'mrm8488/bertweet-base-sarcasm-detection')
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        # Inicialização de componentes
        self.model = None
        self.tokenizer = None
        self.pipeline = None
        self.model_loaded = False
        
        # Carregar modelo se disponível
        if TRANSFORMERS_AVAILABLE:
            self._load_model()
    
    def _load_model(self) -> None:
        """
        Carrega o modelo de ML para detecção de sarcasmo.
        """
        try:
            logger.info(f"Carregando modelo ML para sarcasmo: {self.model_name}")
            
            # Carregar modelo diretamente com pipeline
            self.pipeline = pipeline(
                "text-classification", 
                model=self.model_name,
                device=0 if self.device == 'cuda' else -1
            )
            
            self.model_loaded = True
            logger.info("Modelo ML para sarcasmo carregado com sucesso")
        except Exception as e:
            logger.warning(f"Falha ao carregar modelo ML para sarcasmo: {e}")
            logger.info("Usando fallback para detecção de sarcasmo")
            self.model_loaded = False
    
    def detect(self, text: str, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Detecta sarcasmo usando modelo de ML.
        
        Args:
            text: Texto para análise
            context: Contexto adicional (sentimento, etc)
            
        Returns:
            Resultados da detecção
        """
        # Verificar disponibilidade do modelo
        if not TRANSFORMERS_AVAILABLE or not self.model_loaded:
            # Fallback para sarcasmo leve
            return {
                'score': 0.1,
                'level': 'low',
                'is_sarcastic': False,
                'evidences': ["Modelo ML não disponível"],
                'model_available': False
            }
        
        try:
            # Executar detecção com o pipeline
            result = self.pipeline(text)
            
            # Extrair resultado principal
            prediction = result[0]
            label = prediction['label']
            score = prediction['score']
            
            # Normalizar score conforme o label
            if 'SARCASM' in label or 'IRONY' in label or 'NOT_SINCERE' in label or label == 'LABEL_1':
                # Label positivo para sarcasmo, mantém o score
                sarcasm_score = score
            else:
                # Label negativo para sarcasmo, inverte o score
                sarcasm_score = 1.0 - score
            
            # Classificar o nível de sarcasmo
            if sarcasm_score >= self.threshold_high:
                level = "high"
                is_sarcastic = True
            elif sarcasm_score >= self.threshold_medium:
                level = "medium"
                is_sarcastic = True
            else:
                level = "low"
                is_sarcastic = False
            
            # Construir evidências
            evidences = [
                f"Modelo ML ({self.model_name}): {label}, confidence: {score:.4f}"
            ]
            
            return {
                'score': sarcasm_score,
                'level': level,
                'is_sarcastic': is_sarcastic,
                'evidences': evidences,
                'model_details': {
                    'name': self.model_name,
                    'prediction': prediction
                }
            }
        
        except Exception as e:
            logger.error(f"Erro ao usar modelo ML para sarcasmo: {e}")
            
            # Fallback para resultado neutro em caso de erro
            return {
                'score': 0.0,
                'level': 'low',
                'is_sarcastic': False,
                'evidences': [f"Erro no modelo ML: {str(e)}"],
                'error': str(e)
            }


class SemanticContradictionDetector(SarcasmDetectorInterface):
    """
    Detector de sarcasmo baseado em contradições semânticas.
    Analisa discrepâncias entre sentimento expresso e contexto.
    """
    def __init__(self, config: Dict[str, Any] = None):
        """
        Inicializa o detector de contradições semânticas.
        
        Args:
            config: Configurações opcionais
        """
        self.config = config or {}
        
        # Thresholds de classificação
        self.threshold_high = self.config.get('thresholds', {}).get('high', 0.7)
        self.threshold_medium = self.config.get('thresholds', {}).get('medium', 0.4)
        
        # Carregar recursos linguísticos
        self._load_linguistic_resources()
    
    def _load_linguistic_resources(self) -> None:
        """
        Carrega recursos linguísticos para análise de contradições.
        """
        # Palavras de sentimento positivo
        self.positive_words = set([
            'bom', 'bem', 'ótimo', 'excelente', 'maravilhoso', 'incrível', 
            'fantástico', 'sensacional', 'adorável', 'impressionante', 'perfeito',
            'feliz', 'alegre', 'satisfeito', 'contente', 'agradável', 'positivo',
            'fabuloso', 'extraordinário', 'magnífico', 'esplêndido', 'divino',
            'surpreendente', 'grato', 'encantado', 'animado', 'esperançoso',
            'adorei', 'adoro', 'amo', 'gosto', 'confio', 'acredito', 'recomendo'
        ])
        
        # Palavras de sentimento negativo
        self.negative_words = set([
            'ruim', 'mal', 'péssimo', 'terrível', 'horrível', 'detestável', 'odioso',
            'desagradável', 'defeituoso', 'falho', 'triste', 'decepcionante',
            'deplorável', 'desapontador', 'insatisfatório', 'negativo', 'indesejável',
            'inadequado', 'lamentável', 'medíocre', 'problemático', 'caótico',
            'confuso', 'instável', 'pobre', 'fraco', 'odeio', 'detesto', 'lamento',
            'não gosto', 'não confio', 'não acredito', 'não recomendo'
        ])
        
        # Intensificadores para análise de ênfase
        self.intensifiers = set([
            'muito', 'super', 'extremamente', 'incrivelmente', 'absolutamente',
            'completamente', 'totalmente', 'demais', 'enormemente'
        ])
        
        # Contextos negativos comuns em sarcasmo
        self.negative_contexts = set([
            'horas', 'espera', 'fila', 'demora', 'atraso', 'problema', 'erro',
            'defeito', 'bug', 'falha', 'quebra', 'dificuldade'
        ])
    
    def detect(self, text: str, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Detecta sarcasmo usando contradições semânticas.
        
        Args:
            text: Texto para análise
            context: Contexto adicional (sentimento, etc)
            
        Returns:
            Resultados da detecção
        """
        # Normalizar e tokenizar texto
        text_lower = text.lower()
        words = set(re.findall(r'\b\w+\b', text_lower))
        
        # Contar ocorrências
        pos_count = sum(1 for w in words if w in self.positive_words)
        neg_count = sum(1 for w in words if w in self.negative_words)
        intensifier_count = sum(1 for w in words if w in self.intensifiers)
        neg_context_count = sum(1 for w in words if w in self.negative_contexts)
        
        # Obter sentimento, se disponível
        sentiment_score = context.get('sentiment', 0.0) if context else 0.0
        
        # Calcular possíveis contradições
        contradictions = []
        contradiction_score = 0.0
        
        # Contradição 1: Sentimento positivo com muitas palavras negativas
        if sentiment_score > 0.3 and neg_count > 0:
            contradiction_weight = min(1.0, 0.2 * neg_count)
            contradiction_score += contradiction_weight
            contradictions.append(f"Sentimento positivo ({sentiment_score:.2f}) mas {neg_count} palavras negativas")
        
        # Contradição 2: Sentimento positivo com contexto negativo
        if sentiment_score > 0.3 and neg_context_count > 0:
            contradiction_weight = min(1.0, 0.3 * neg_context_count)
            contradiction_score += contradiction_weight
            contradictions.append(f"Sentimento positivo com {neg_context_count} indicadores de contexto negativo")
        
        # Contradição 3: Sentimento negativo com muitas palavras positivas
        if sentiment_score < -0.3 and pos_count > 1:
            contradiction_weight = min(1.0, 0.2 * pos_count)
            contradiction_score += contradiction_weight
            contradictions.append(f"Sentimento negativo ({sentiment_score:.2f}) mas {pos_count} palavras positivas")
        
        # Bônus para intensificadores
        if intensifier_count > 0 and (
            (sentiment_score > 0.3 and neg_context_count > 0) or
            (sentiment_score < -0.3 and pos_count > 1)
        ):
            intensifier_bonus = min(0.3, 0.1 * intensifier_count)
            contradiction_score += intensifier_bonus
            contradictions.append(f"Uso de {intensifier_count} intensificadores em contexto contraditório")
        
        # Limitar score máximo
        contradiction_score = min(1.0, contradiction_score)
        
        # Classificar o nível de sarcasmo
        if contradiction_score >= self.threshold_high:
            level = "high"
            is_sarcastic = True
        elif contradiction_score >= self.threshold_medium:
            level = "medium"
            is_sarcastic = True
        else:
            level = "low"
            is_sarcastic = False
        
        return {
            'score': contradiction_score,
            'level': level,
            'is_sarcastic': is_sarcastic,
            'evidences': contradictions,
            'details': {
                'positive_words': pos_count,
                'negative_words': neg_count,
                'intensifiers': intensifier_count,
                'negative_context': neg_context_count,
                'sentiment': sentiment_score
            }
        }


class AdaptiveSarcasmDetector:
    """
    Detector de sarcasmo adaptativo que combina múltiplas abordagens.
    Implementa estratégias de ensemble para melhorar a precisão.
    """
    def __init__(self, config: Dict[str, Any] = None):
        """
        Inicializa o detector adaptativo.
        
        Args:
            config: Configurações opcionais
        """
        self.config = config or {}
        
        # Thresholds de classificação
        self.threshold_high = self.config.get('thresholds', {}).get('high', 0.7)
        self.threshold_medium = self.config.get('thresholds', {}).get('medium', 0.4)
        
        # Pesos para combinação de detectores
        self.weights = self.config.get('weights', {
            'rule_based': 0.4,
            'ml_based': 0.3,
            'contradiction': 0.3
        })
        
        # Inicializar detectores
        self.rule_detector = RuleBasedDetector(self.config)
        self.ml_detector = MLBasedDetector(self.config)
        self.contradiction_detector = SemanticContradictionDetector(self.config)
        
        logger.info("AdaptiveSarcasmDetector inicializado com sucesso")
    
    def detect(self, text: str, sentiment: Optional[float] = None) -> Dict[str, Any]:
        """
        Detecta sarcasmo combinando múltiplas abordagens.
        
        Args:
            text: Texto para análise
            sentiment: Valor de sentimento (-1 a 1)
            
        Returns:
            Resultado completo da detecção
        """
        # Criar contexto para os detectores
        context = {'sentiment': sentiment} if sentiment is not None else {}
        
        # Obter resultados dos detectores individuais
        try:
            rule_result = self.rule_detector.detect(text, context)
        except Exception as e:
            logger.warning(f"Erro no detector baseado em regras: {e}")
            rule_result = {'score': 0.0, 'level': 'low', 'is_sarcastic': False, 'error': str(e)}
        
        try:
            ml_result = self.ml_detector.detect(text, context)
        except Exception as e:
            logger.warning(f"Erro no detector baseado em ML: {e}")
            ml_result = {'score': 0.0, 'level': 'low', 'is_sarcastic': False, 'error': str(e)}
        
        try:
            contradiction_result = self.contradiction_detector.detect(text, context)
        except Exception as e:
            logger.warning(f"Erro no detector de contradições: {e}")
            contradiction_result = {'score': 0.0, 'level': 'low', 'is_sarcastic': False, 'error': str(e)}
        
        # Verificar se há indicadores fortes em algum detector individual
        # Isso permite que um detector confiante "vete" os outros
        strong_indicators = []
        
        if rule_result.get('score', 0.0) > 0.8:
            strong_indicators.append(('rule_based', rule_result.get('score', 0.0)))
        
        if ml_result.get('score', 0.0) > 0.8:
            strong_indicators.append(('ml_based', ml_result.get('score', 0.0)))
        
        if contradiction_result.get('score', 0.0) > 0.8:
            strong_indicators.append(('contradiction', contradiction_result.get('score', 0.0)))
        
        # Se houver indicadores fortes, usar o mais forte
        if strong_indicators:
            # Ordenar por score descendente
            strong_indicators.sort(key=lambda x: x[1], reverse=True)
            strongest = strong_indicators[0]
            
            logger.info(f"Usando detector {strongest[0]} com forte indicação ({strongest[1]:.2f})")
            
            # Ajustar pesos para favorecer o detector mais confiante
            adjusted_weights = {
                'rule_based': 0.1,
                'ml_based': 0.1,
                'contradiction': 0.1
            }
            adjusted_weights[strongest[0]] = 0.8
        else:
            # Usar pesos padrão
            adjusted_weights = self.weights
        
        # Combinar pontuações ponderadas
        total_weight = sum(adjusted_weights.values())
        if total_weight > 0:
            combined_score = (
                rule_result.get('score', 0.0) * adjusted_weights['rule_based'] +
                ml_result.get('score', 0.0) * adjusted_weights['ml_based'] +
                contradiction_result.get('score', 0.0) * adjusted_weights['contradiction']
            ) / total_weight
        else:
            combined_score = 0.0
            logger.warning("Soma dos pesos é zero, usando score zero para sarcasmo")
        
        # Classificar nível de sarcasmo
        if combined_score >= self.threshold_high:
            level = "high"
            is_sarcastic = True
        elif combined_score >= self.threshold_medium:
            level = "medium"
            is_sarcastic = True
        else:
            level = "low"
            is_sarcastic = False
        
        # Ajustar sentimento se for sarcástico
        adjusted_sentiment = -sentiment if is_sarcastic and sentiment is not None else sentiment
        
        # Combinar evidências dos detectores
        evidences = []
        if 'evidences' in rule_result and rule_result['evidences']:
            evidences.extend([f"Regras: {e}" for e in rule_result['evidences']])
        if 'evidences' in ml_result and ml_result['evidences']:
            evidences.extend([f"ML: {e}" for e in ml_result['evidences']])
        if 'evidences' in contradiction_result and contradiction_result['evidences']:
            evidences.extend([f"Contradição: {e}" for e in contradiction_result['evidences']])
        
        # Construir resultado final
        return {
            "probabilidade": combined_score,
            "nivel": level,
            "e_sarcastico": is_sarcastic,
            "sentimento_ajustado": adjusted_sentiment,
            "evidencias": evidences,
            "detalhes": {
                "rule": rule_result,
                "contradiction": contradiction_result,
                "ml": ml_result
            }
        }