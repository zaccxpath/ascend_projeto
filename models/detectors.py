"""
Detectores especializados para análise de sarcasmo em textos 
usando diferentes técnicas de detecção.
"""
import re
import logging
from typing import Dict, List, Optional, Any, Tuple, Union, Set
import os
import traceback

from utils.config import logger

class BaseDetector:
    """Classe base para detectores de sarcasmo"""
    
    def __init__(self, config: Dict[str, Any]) -> None:
        """
        Inicializa o detector base
        
        Args:
            config: Configuração para o detector
        """
        self.config = config
        self.name = "base"
    
    def detect(self, texto: str, contexto: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Método base para detecção
        
        Args:
            texto: Texto a ser analisado
            contexto: Informações de contexto para ajudar na análise
            
        Returns:
            Resultado da detecção com probabilidade e detalhes
        """
        return {
            "probabilidade": 0.0,
            "detalhes": {"detector": self.name},
            "debug_info": {}
        }


class RuleBasedDetector(BaseDetector):
    """Detector de sarcasmo baseado em regras linguísticas"""
    
    def __init__(self, config: Dict[str, Any]) -> None:
        """
        Inicializa o detector baseado em regras
        
        Args:
            config: Configuração contendo marcadores, padrões e pesos
        """
        super().__init__(config)
        self.name = "regras"
        
        # Carregar marcadores de sarcasmo
        self.marcadores_sarcasmo = config.get('marcadores_sarcasmo', [])
        
        # Carregar padrões de sarcasmo
        self.padroes_sarcasmo = config.get('padroes_sarcasmo', [])
        
        # Pré-compilar padrões para melhor performance
        try:
            self.padroes_compilados = [re.compile(padrao) for padrao in self.padroes_sarcasmo]
            logger.info(f"Padrões de sarcasmo pré-compilados: {len(self.padroes_compilados)} padrões")
        except Exception as e:
            logger.error(f"Erro ao pré-compilar expressões regulares: {e}")
            self.padroes_compilados = []
        
        # Contextos de sarcasmo (combinações de palavras)
        self.contextos_sarcasmo = config.get('contextos_sarcasmo', [])
        
        # Pesos para cada tipo de verificação
        self.pesos = config.get('pesos', {
            'marcadores': 1.0,
            'padroes': 1.5,
            'contextos': 2.0,
            'pontuacao': 1.0,
            'contraste': 2.0,
            'verbos_condicional': 0.5,
            'expectativa_realidade': 1.5,
            'tempo_espera': 2.0  # Peso maior para padrões de tempo de espera
        })
        
        # Threshold para early-exit
        self.threshold_early_exit = config.get('threshold_early_exit', 10)
    
    def detect(self, texto: str, contexto: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Detecta sarcasmo usando regras linguísticas
        
        Args:
            texto: Texto a ser analisado
            contexto: Informações de contexto (não utilizado neste detector)
            
        Returns:
            Resultado da detecção com probabilidade e detalhes
        """
        texto_lower = texto.lower()
        pontos_sarcasmo = 0
        total_checks = 0
        
        # Armazenar informações de debug
        marcadores_encontrados = []
        padroes_encontrados = []
        contextos_encontrados = []
        
        # 1. Verificar marcadores de sarcasmo (mais rápido, fazer primeiro)
        for marcador in self.marcadores_sarcasmo:
            if marcador in texto_lower:
                # Dar peso maior para marcadores especiais, se definidos
                peso = self.pesos.get('marcadores', 1.0)
                if 'marcadores_tempo_espera' in self.config and marcador in self.config['marcadores_tempo_espera']:
                    peso = self.pesos.get('tempo_espera', 2.0)
                    
                pontos_sarcasmo += peso
                marcadores_encontrados.append(marcador)
                
                # Early-exit: se já temos pontos suficientes, podemos parar cedo
                if pontos_sarcasmo >= self.threshold_early_exit:
                    logger.debug(f"Early-exit na verificação de marcadores: {pontos_sarcasmo} pontos")
                    # Calcular probabilidade baseada apenas em marcadores
                    prob_sarcasmo = min(pontos_sarcasmo / len(self.marcadores_sarcasmo) * 2, 1.0)
                    return {
                        "probabilidade": prob_sarcasmo,
                        "detalhes": {
                            "detector": self.name,
                            "marcadores": marcadores_encontrados
                        },
                        "debug_info": {
                            "pontos": pontos_sarcasmo,
                            "total_checks": len(self.marcadores_sarcasmo),
                            "early_exit": True
                        }
                    }
        
        total_checks += len(self.marcadores_sarcasmo) * self.pesos.get('marcadores', 1.0)
        
        # 2. Verificar padrões de expressões regulares (usar padrões compilados)
        if hasattr(self, 'padroes_compilados') and self.padroes_compilados:
            for i, padrao_compilado in enumerate(self.padroes_compilados):
                if padrao_compilado.search(texto_lower):
                    # Verificar se é padrão relacionado a tempo
                    peso = self.pesos.get('padroes', 1.5)
                    padrao_texto = self.padroes_sarcasmo[i] if i < len(self.padroes_sarcasmo) else str(padrao_compilado.pattern)
                    
                    # Dar peso maior para padrões de tempo
                    if any(palavra in padrao_texto for palavra in ['tempo', 'espera', 'fila', 'hora', 'minuto', 'segundo', 'atraso']):
                        peso = self.pesos.get('tempo_espera', 2.0)
                    
                    pontos_sarcasmo += peso
                    padroes_encontrados.append(padrao_texto)
                    
                    # Early-exit: se já temos pontos suficientes, podemos parar
                    if pontos_sarcasmo >= self.threshold_early_exit:
                        logger.debug(f"Early-exit na verificação de padrões: {pontos_sarcasmo} pontos")
                        # Calcular probabilidade baseada em marcadores e padrões
                        total_verificado = (len(self.marcadores_sarcasmo) * self.pesos.get('marcadores', 1.0) + 
                                           (i+1) * self.pesos.get('padroes', 1.5))
                        prob_sarcasmo = min(pontos_sarcasmo / total_verificado * 2, 1.0)
                        return {
                            "probabilidade": prob_sarcasmo,
                            "detalhes": {
                                "detector": self.name,
                                "marcadores": marcadores_encontrados,
                                "padroes": padroes_encontrados
                            },
                            "debug_info": {
                                "pontos": pontos_sarcasmo,
                                "total_checks": total_verificado,
                                "early_exit": True
                            }
                        }
        else:
            # Fallback para verificação sem compilação
            for padrao in self.padroes_sarcasmo:
                if re.search(padrao, texto_lower):
                    pontos_sarcasmo += self.pesos.get('padroes', 1.5)
                    padroes_encontrados.append(padrao)
        
        total_checks += len(self.padroes_sarcasmo) * self.pesos.get('padroes', 1.5)
        
        # 3. Verificar contextos de sarcasmo (pares de palavras contraditórias)
        for palavra1, palavra2 in self.contextos_sarcasmo:
            if palavra1 in texto_lower and palavra2 in texto_lower:
                # Verificar se é contexto relacionado a tempo
                peso = self.pesos.get('contextos', 2.0)
                
                # Dar peso maior para contextos de tempo
                if palavra1 in ['rápido', 'veloz', 'ágil', 'instantâneo', 'eficiente'] or palavra2 in ['espera', 'demora', 'atraso', 'lentidão']:
                    peso = self.pesos.get('tempo_espera', 2.0)
                
                pontos_sarcasmo += peso
                contextos_encontrados.append(f"{palavra1} + {palavra2}")
                
                # Early-exit após cada contexto encontrado
                if pontos_sarcasmo >= self.threshold_early_exit:
                    logger.debug(f"Early-exit na verificação de contextos: {pontos_sarcasmo} pontos")
                    # Calcular probabilidade com base no que já foi verificado
                    total_verificado = (len(self.marcadores_sarcasmo) * self.pesos.get('marcadores', 1.0) + 
                                      len(self.padroes_sarcasmo) * self.pesos.get('padroes', 1.5) +
                                      len(contextos_encontrados) * self.pesos.get('contextos', 2.0))
                    
                    prob_sarcasmo = min(pontos_sarcasmo / total_verificado * 1.5, 1.0)  # Multiplicador ajustado
                    return {
                        "probabilidade": prob_sarcasmo,
                        "detalhes": {
                            "detector": self.name,
                            "marcadores": marcadores_encontrados,
                            "padroes": padroes_encontrados,
                            "contextos": contextos_encontrados
                        },
                        "debug_info": {
                            "pontos": pontos_sarcasmo,
                            "total_checks": total_verificado,
                            "early_exit": True
                        }
                    }
        
        total_checks += len(self.contextos_sarcasmo) * self.pesos.get('contextos', 2.0)
        
        # 4. Verificar pontuação excessiva (!!!, ???)
        pontuacao_excessiva = len(re.findall(r'(!{3,}|\?{3,})', texto))
        if pontuacao_excessiva > 0:
            pontos_sarcasmo += min(pontuacao_excessiva, 3) * self.pesos.get('pontuacao', 1.0)
        total_checks += 3 * self.pesos.get('pontuacao', 1.0)
        
        # 5. Verificar contraste entre início e fim da frase (só para textos curtos/médios)
        if len(texto_lower) < 500:  # Otimização: não processar textos muito longos
            frases = texto_lower.split('.')
            for frase in frases:
                tokens = frase.split()
                if len(tokens) >= 6:  # Frase com tamanho mínimo
                    inicio = ' '.join(tokens[:3])
                    fim = ' '.join(tokens[-3:])
                    
                    # Verificar se o sentimento no início e fim são opostos
                    for pos in ['bom', 'ótimo', 'excelente', 'perfeito', 'maravilhoso', 'incrível']:
                        for neg in ['mas', 'porém', 'contudo', 'entretanto', 'no entanto', 'problema', 'ruim', 'péssimo']:
                            if (pos in inicio and neg in fim) or (neg in inicio and pos in fim):
                                pontos_sarcasmo += self.pesos.get('contraste', 2.0)
        total_checks += 5 * self.pesos.get('contraste', 2.0)  # Peso para esta verificação
        
        # 6. Verificar uso de verbos no futuro do pretérito (apenas se pontuação insuficiente)
        if pontos_sarcasmo < self.threshold_early_exit:
            verbos_condicional = ['seria', 'estaria', 'teria', 'faria', 'poderia', 'deveria', 'gostaria']
            for verbo in verbos_condicional:
                if verbo in texto_lower:
                    pontos_sarcasmo += self.pesos.get('verbos_condicional', 0.5)
            total_checks += len(verbos_condicional) * self.pesos.get('verbos_condicional', 0.5)
        
        # 7. Verificar expressões de expectativa vs. realidade (se pontuação insuficiente)
        if pontos_sarcasmo < self.threshold_early_exit:
            exp_vs_realidade = [
                ('esperava', 'realidade'),
                ('pensei que', 'mas'),
                ('achei que', 'mas'),
                ('imaginei', 'porém'),
                ('era para ser', 'mas')
            ]
            for exp, realidade in exp_vs_realidade:
                if exp in texto_lower and realidade in texto_lower:
                    pontos_sarcasmo += self.pesos.get('expectativa_realidade', 1.5)
            total_checks += len(exp_vs_realidade) * self.pesos.get('expectativa_realidade', 1.5)
        
        # Normalizar para escala 0-1
        if total_checks > 0:
            prob_sarcasmo = min(pontos_sarcasmo / total_checks * 2, 1.0)  # Multiplicador para ajustar escala
        else:
            prob_sarcasmo = 0.0
            
        logger.info(f"Detecção baseada em regras: {pontos_sarcasmo}/{total_checks} pontos de sarcasmo")
        
        return {
            "probabilidade": prob_sarcasmo,
            "detalhes": {
                "detector": self.name,
                "marcadores": marcadores_encontrados,
                "padroes": padroes_encontrados,
                "contextos": contextos_encontrados
            },
            "debug_info": {
                "pontos": pontos_sarcasmo,
                "total_checks": total_checks
            }
        }


class ContradictionDetector(BaseDetector):
    """Detector de sarcasmo baseado em contradições entre sentimento e palavras"""
    
    def __init__(self, config: Dict[str, Any], sentiment_analyzer: Any) -> None:
        """
        Inicializa o detector de contradição
        
        Args:
            config: Configuração para o detector
            sentiment_analyzer: Analisador de sentimento para obter listas de palavras
        """
        super().__init__(config)
        self.name = "contradicao"
        self.sentiment_analyzer = sentiment_analyzer
        
        # Marcadores de sarcasmo para detecção em sentimentos extremos
        self.marcadores_sarcasmo = config.get('marcadores_sarcasmo', [])
    
    def detect(self, texto: str, contexto: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Detecta sarcasmo por contradição entre sentimento expresso e palavras usadas
        
        Args:
            texto: Texto a ser analisado
            contexto: Resultado da análise de sentimento (obrigatório)
            
        Returns:
            Resultado da detecção com probabilidade e detalhes
        """
        if not contexto or 'sentimento' not in contexto:
            logger.warning("Análise de sentimento não fornecida para detector de contradição")
            return {
                "probabilidade": 0.0,
                "detalhes": {"detector": self.name, "erro": "sem_analise_sentimento"},
                "debug_info": {}
            }
        
        try:
            # Obter sentimento geral do texto
            sentimento = contexto.get('sentimento', 'neutro')
            compound = contexto.get('compound', 0)
            
            # Se sentimento for positivo, verificar palavras negativas e vice-versa
            texto_lower = texto.lower()
            
            # Contagens
            count_palavras_positivas = 0
            count_palavras_negativas = 0
            
            # Verificar contradição usando listas de palavras do analisador
            if hasattr(self.sentiment_analyzer, 'PALAVRAS_POSITIVAS') and hasattr(self.sentiment_analyzer, 'PALAVRAS_NEGATIVAS'):
                for palavra in texto_lower.split():
                    if palavra in self.sentiment_analyzer.PALAVRAS_POSITIVAS:
                        count_palavras_positivas += 1
                    if palavra in self.sentiment_analyzer.PALAVRAS_NEGATIVAS:
                        count_palavras_negativas += 1
            
            nivel_contradicao = 0.0
            tipo_contradicao = "nenhuma"
            
            # Verificar contradição entre sentimento e palavras
            if sentimento == 'positivo' and count_palavras_negativas > count_palavras_positivas:
                # Sentimento geral positivo mas predominância de palavras negativas
                nivel_contradicao = min(1.0, (count_palavras_negativas - count_palavras_positivas) / 5)
                tipo_contradicao = "positivo_com_palavras_negativas"
                logger.debug(f"Contradição detectada: sentimento positivo com {count_palavras_negativas} palavras negativas")
                prob_contradicao = 0.3 + (nivel_contradicao * 0.7)  # Escala de 0.3 a 1.0
            elif sentimento == 'negativo' and count_palavras_positivas > count_palavras_negativas:
                # Sentimento geral negativo mas predominância de palavras positivas
                nivel_contradicao = min(1.0, (count_palavras_positivas - count_palavras_negativas) / 5)
                tipo_contradicao = "negativo_com_palavras_positivas"
                logger.debug(f"Contradição detectada: sentimento negativo com {count_palavras_positivas} palavras positivas")
                prob_contradicao = 0.3 + (nivel_contradicao * 0.7)  # Escala de 0.3 a 1.0
            else:
                # Verificar intensidade extrema como possível sinal de sarcasmo
                if abs(compound) > 0.8:
                    # Verificar marcadores de sarcasmo em mensagens extremamente positivas/negativas
                    count_marcadores = sum(1 for marcador in self.marcadores_sarcasmo if marcador in texto_lower)
                    if count_marcadores >= 2:
                        logger.debug(f"Sentimento extremo ({compound}) com {count_marcadores} marcadores de sarcasmo")
                        tipo_contradicao = "sentimento_extremo_com_marcadores"
                        prob_contradicao = 0.7
                    else:
                        prob_contradicao = 0.1  # Baixa probabilidade
                else:
                    prob_contradicao = 0.1  # Baixa probabilidade
            
            return {
                "probabilidade": prob_contradicao,
                "detalhes": {
                    "detector": self.name,
                    "tipo_contradicao": tipo_contradicao,
                    "palavras_positivas": count_palavras_positivas,
                    "palavras_negativas": count_palavras_negativas,
                    "sentimento": sentimento,
                    "compound": compound
                },
                "debug_info": {
                    "nivel_contradicao": nivel_contradicao
                }
            }
        except Exception as e:
            logger.error(f"Erro na detecção de sarcasmo por contradição: {e}")
            logger.error(traceback.format_exc())
            return {
                "probabilidade": 0.0,
                "detalhes": {"detector": self.name, "erro": str(e)},
                "debug_info": {"exception": traceback.format_exc()}
            }


class MLBasedDetector(BaseDetector):
    """Detector de sarcasmo baseado em modelos de machine learning"""
    
    def __init__(self, config: Dict[str, Any]) -> None:
        """
        Inicializa o detector baseado em modelos
        
        Args:
            config: Configuração para o detector (modelo, threshold, etc)
        """
        super().__init__(config)
        self.name = "modelo"
        self.model = None
        self.using_ml_model = False
        
        # Tentar inicializar o modelo
        try:
            from transformers import pipeline, AutoModelForSequenceClassification, AutoTokenizer
            
            # Verificar conexão com HuggingFace
            import socket
            try:
                socket.create_connection(("huggingface.co", 443), timeout=5)
                internet_disponivel = True
            except Exception as e:
                logger.error(f"Erro ao verificar conexão com a internet: {e}")
                internet_disponivel = False
            
            if not internet_disponivel:
                logger.warning("Sem conexão com a Internet. Usando apenas regras para detectar sarcasmo.")
                return
            
            # Carregar configuração do modelo
            model_name = config.get('model_name', "cardiffnlp/twitter-xlm-roberta-base-sentiment")
            model_dir = config.get('model_dir', os.path.join(os.getcwd(), "models", "sarcasm-detector"))
            device = config.get('device', -1)  # CPU por padrão
            
            logger.info(f"Tentando carregar modelo para detecção de sarcasmo: {model_name}")
            
            # Verificar se o modelo existe localmente primeiro
            os.makedirs(model_dir, exist_ok=True)
            
            # Tentativa de carregar localmente
            try:
                if os.path.exists(os.path.join(model_dir, "config.json")):
                    tokenizer = AutoTokenizer.from_pretrained(model_dir, local_files_only=True)
                    model = AutoModelForSequenceClassification.from_pretrained(model_dir, local_files_only=True)
                    
                    # Criar pipeline
                    self.model = pipeline(
                        "text-classification", 
                        model=model, 
                        tokenizer=tokenizer,
                        device=device
                    )
                    logger.info("Modelo de sarcasmo carregado com sucesso do cache local.")
                    self.using_ml_model = True
                else:
                    raise ValueError("Modelo não encontrado localmente")
            except Exception as e:
                logger.warning(f"Não foi possível carregar o modelo localmente: {e}")
                
                # Tentar download se estiver online
                try:
                    logger.info("Tentando baixar modelo...")
                    tokenizer = AutoTokenizer.from_pretrained(model_name)
                    model = AutoModelForSequenceClassification.from_pretrained(model_name)
                    
                    # Criar pipeline
                    self.model = pipeline(
                        "text-classification", 
                        model=model, 
                        tokenizer=tokenizer,
                        device=device
                    )
                    
                    # Salvar modelo localmente para uso futuro
                    logger.info(f"Salvando modelo em {model_dir}")
                    tokenizer.save_pretrained(model_dir)
                    model.save_pretrained(model_dir)
                    
                    logger.info("Modelo de sarcasmo baixado e salvo localmente.")
                    self.using_ml_model = True
                except Exception as e:
                    logger.error(f"Erro ao baixar modelo de sarcasmo: {e}")
                    logger.warning("Usando apenas detecção baseada em regras.")
        except Exception as e:
            logger.error(f"Erro na inicialização do detector de sarcasmo ML: {e}")
            logger.error(traceback.format_exc())
    
    def detect(self, texto: str, contexto: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Detecta sarcasmo usando um modelo pré-treinado
        
        Args:
            texto: Texto a ser analisado
            contexto: Informações de contexto (não utilizado neste detector)
            
        Returns:
            Resultado da detecção com probabilidade e detalhes
        """
        if not self.using_ml_model or not self.model:
            return {
                "probabilidade": 0.0,
                "detalhes": {"detector": self.name, "erro": "modelo_indisponivel"},
                "debug_info": {}
            }
            
        try:
            # Limitar tamanho do texto para o modelo
            texto_limitado = texto[:512]
            
            # Classificar texto com o modelo
            resultado = self.model(texto_limitado)[0]
            
            # Extrair informações do resultado
            label = resultado.get('label', '')
            score = resultado.get('score', 0.0)
            
            # Lógica para converter resultado do modelo em probabilidade de sarcasmo
            # Nota: Esta é uma lógica simplificada que deve ser adaptada ao seu modelo específico
            if label == 'positive' and 0.4 < score < 0.8:
                prob_sarcasmo = 0.7  # Retorna probabilidade de sarcasmo
            elif label == 'negative' and score > 0.9:
                prob_sarcasmo = 0.3  # Baixa probabilidade de sarcasmo
            else:
                prob_sarcasmo = 0.1
            
            return {
                "probabilidade": prob_sarcasmo,
                "detalhes": {
                    "detector": self.name,
                    "modelo": self.config.get('model_name', "desconhecido"),
                    "label": label,
                    "score": score
                },
                "debug_info": {
                    "resultado_modelo": resultado
                }
            }
        except Exception as e:
            logger.error(f"Erro na detecção de sarcasmo com modelo: {e}")
            logger.error(traceback.format_exc())
            return {
                "probabilidade": 0.0,
                "detalhes": {"detector": self.name, "erro": str(e)},
                "debug_info": {"exception": traceback.format_exc()}
            } 