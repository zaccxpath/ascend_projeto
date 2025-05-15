import os
import logging
from typing import Dict, Any, Optional, Union

from .local_llm_analyzer import LocalLLMAnalyzer, AnalyzerConfig

logger = logging.getLogger(__name__)

class LocalLLMSentimentEnhancer:
    """
    Classe que integra o LocalLLMAnalyzer com o SentimentAnalyzer existente.
    Usa modelos locais como DeepSeek ou LLaMA para análise de sentimentos e sarcasmo.
    """
    
    def __init__(self, 
                 model_name: str = "deepseek-ai/deepseek-llm-7b-base", 
                 limiar_confianca: float = 0.7,
                 device: str = "auto",
                 quantization: Optional[str] = None,
                 llm_analyzer = None):
        """
        Inicializa o enhancer com configurações personalizáveis.
        
        Args:
            model_name: Nome ou caminho do modelo a ser utilizado
            limiar_confianca: Limiar abaixo do qual o LLM será usado (padrão: 0.7)
            device: Dispositivo para inferência ('cpu', 'cuda', 'auto')
            quantization: Tipo de quantização a ser usada (4bit, 8bit, None)
            llm_analyzer: Instância opcional de LocalLLMAnalyzer.
        """
        self.model_name = model_name
        self.limiar_confianca = limiar_confianca
        self.logger = logger
        self.ultimo_erro = None
        
        # Configurar o analisador LLM
        if llm_analyzer:
            self.logger.info("Usando LocalLLMAnalyzer fornecido externamente")
            self.llm_analyzer = llm_analyzer
        else:
            config = AnalyzerConfig(
                model_name=model_name,
                device=device,
                quantization=quantization
            )
            self.llm_analyzer = LocalLLMAnalyzer(config)
            self.logger.info(f"LocalLLMAnalyzer inicializado com modelo: {model_name}")
        
        self.logger.info(f"LocalLLMSentimentEnhancer inicializado")
    
    def melhorar_analise(self, 
                        texto: str, 
                        analise_atual: Dict[str, Any],
                        forcar_uso_llm: bool = False) -> Dict[str, Any]:
        """
        Melhora uma análise existente usando o modelo local quando necessário.
        
        Args:
            texto: Texto original sendo analisado
            analise_atual: Resultado da análise atual dos modelos existentes
            forcar_uso_llm: Se True, usa o modelo mesmo com alta confiança
            
        Returns:
            Análise melhorada ou a original se não for necessário melhorar
        """
        # Extrair confiança atual
        confianca = analise_atual.get('confianca', 1.0)
        
        # Decidir se vamos usar o modelo local
        usar_llm = forcar_uso_llm or confianca < self.limiar_confianca
        
        if not usar_llm:
            self.logger.info(f"Confiança suficiente ({confianca:.2f}), mantendo análise original")
            return analise_atual
        
        self.logger.info(f"Confiança baixa ({confianca:.2f}) ou forçado, aplicando análise com LLM local")
        
        # Usar o LocalLLMAnalyzer para refinar a análise
        try:
            analise_refinada = self.llm_analyzer.ajustar_analise_existente(texto, analise_atual)
            
            # Garantir que a estrutura esperada esteja presente
            if 'sentimento' not in analise_refinada:
                self.logger.warning("Análise refinada não contém sentimento, usando original")
                return analise_atual
            
            # Marcar que foi processado pelo enhancer
            analise_refinada['processado_por_llm_local'] = True
            
            # Limpar qualquer erro anterior
            self.ultimo_erro = None
            
            return analise_refinada
            
        except Exception as e:
            erro_str = str(e)
            self.logger.error(f"Erro ao usar enhancer LLM local: {e}")
            
            # Registrar erro
            self.ultimo_erro = f"Erro no modelo local: {erro_str}"
            
            # Retornar a análise original em caso de erro
            return analise_atual
    
    def analisar_caso_complexo(self, texto: str, tipo: str = 'completo') -> Optional[Dict[str, Any]]:
        """
        Analisa um caso especialmente complexo diretamente com o modelo local.
        Útil para casos onde os modelos convencionais têm dificuldade.
        
        Args:
            texto: Texto a ser analisado
            tipo: Tipo de análise ('sentimento', 'sarcasmo', 'completo', 'varejo')
            
        Returns:
            Resultado da análise ou None se falhar
        """
        self.logger.info(f"Realizando análise direta com modelo local para caso complexo (tipo: {tipo})")
        
        try:
            result = None
            if tipo == 'sentimento':
                result = self.llm_analyzer.analisar_sentimento(texto)
            elif tipo == 'sarcasmo':
                result = self.llm_analyzer.analisar_sarcasmo(texto)
            elif tipo == 'varejo':
                result = self.llm_analyzer.analisar_varejo(texto)
            else:  # 'completo' é o padrão
                result = self.llm_analyzer.analisar_completo(texto)
                
            # Limpar último erro se a análise for bem-sucedida
            if result:
                self.ultimo_erro = None
                
            return result
            
        except Exception as e:
            erro_str = str(e)
            self.logger.error(f"Erro ao realizar análise direta com LLM local: {e}")
            self.ultimo_erro = f"Erro no modelo local: {erro_str}"
            
            # Propagar a exceção para ser tratada no nível superior
            raise
    
    def obter_metricas(self) -> Dict[str, Any]:
        """
        Retorna métricas de uso do modelo local.
        
        Returns:
            Dicionário com métricas coletadas
        """
        return self.llm_analyzer.obter_metricas()

def integrar_llm_local_ao_sentiment_analyzer(sentiment_analyzer, modelo="deepseek-ai/deepseek-llm-7b-base", device="auto", quantization=None):
    """
    Integra o LocalLLMSentimentEnhancer a um SentimentAnalyzer existente.
    
    Args:
        sentiment_analyzer: A instância de SentimentAnalyzer a ser aprimorada
        modelo: Nome do modelo a ser utilizado (padrão: deepseek-ai/deepseek-llm-7b-base)
        device: Dispositivo para inferência ('cpu', 'cuda', 'auto')
        quantization: Tipo de quantização a ser usada (4bit, 8bit, None)
        
    Returns:
        A mesma instância, agora com capacidade LLM local
    """
    # Verificar se já tem um enhancer
    if hasattr(sentiment_analyzer, 'llm_enhancer'):
        logger.info("SentimentAnalyzer já possui LLM enhancer, atualizando")
    
    # Criar e anexar o enhancer
    enhancer = LocalLLMSentimentEnhancer(
        model_name=modelo,
        device=device,
        quantization=quantization
    )
    sentiment_analyzer.llm_enhancer = enhancer
    
    # Sobrescrever o método analisar_sentimento_final para usar LLM quando necessário
    original_analisar = getattr(sentiment_analyzer, 'analisar_sentimento_final', 
                               sentiment_analyzer.analisar_sentimento)
    
    def analisar_sentimento_melhorado(texto, *args, **kwargs):
        # Chamar o método original primeiro
        resultado = original_analisar(texto, *args, **kwargs)
        
        # Verificar se devemos melhorar com LLM local
        usar_llm = kwargs.get('usar_llm', True)
        
        if not usar_llm:
            return resultado
        
        # Tentar melhorar com LLM local
        try:
            resultado_melhorado = enhancer.melhorar_analise(texto, resultado)
            return resultado_melhorado
        except Exception as e:
            logger.error(f"Erro ao usar enhancer LLM local: {e}")
            return resultado
    
    # Substituir o método
    sentiment_analyzer.analisar_sentimento_final = analisar_sentimento_melhorado
    
    # Adicionar método para análise de casos complexos
    sentiment_analyzer.analisar_caso_complexo = lambda texto, tipo='completo': enhancer.analisar_caso_complexo(texto, tipo)
    
    logger.info("LocalLLMSentimentEnhancer integrado com sucesso ao SentimentAnalyzer")
    return sentiment_analyzer 