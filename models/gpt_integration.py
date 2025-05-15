import os
import logging
from typing import Dict, Any, Optional, Union

from .gpt_analyzer import GPTAnalyzer, AnalyzerConfig

logger = logging.getLogger(__name__)

class GPTSentimentEnhancer:
    """
    Classe que integra o GPTAnalyzer com o SentimentAnalyzer existente.
    Usa o GPT para melhorar análises de baixa confiança ou casos complexos.
    """
    
    def __init__(self, 
                 api_key: Optional[str] = None, 
                 modelo: str = "gpt-3.5-turbo",
                 limiar_confianca: float = 0.7,
                 gpt_analyzer = None):
        """
        Inicializa o enhancer com configurações personalizáveis.
        
        Args:
            api_key: Chave da API OpenAI (opcional, pode usar variável de ambiente)
            modelo: Modelo GPT a ser usado (padrão: gpt-3.5-turbo)
            limiar_confianca: Limiar abaixo do qual o GPT será usado (padrão: 0.7)
            gpt_analyzer: Instância opcional de GPTAnalyzer. Se fornecido, será usado em vez de criar um novo.
        """
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.modelo = modelo
        self.limiar_confianca = limiar_confianca
        self.logger = logger
        self.ultimo_erro = None  # Para armazenar informações sobre o último erro ocorrido
        
        # Configurar o analisador GPT
        if gpt_analyzer:
            self.logger.info("Usando GPTAnalyzer fornecido externamente")
            self.gpt_analyzer = gpt_analyzer
        else:
            config = AnalyzerConfig(
                api_key=self.api_key,
                modelo=self.modelo,
                temperature=0.1,
                max_tokens=500
            )
            self.gpt_analyzer = GPTAnalyzer(config)
            self.logger.info(f"GPTAnalyzer inicializado com modelo: {modelo}")
        
        self.logger.info(f"GPTSentimentEnhancer inicializado")
    
    def melhorar_analise(self, 
                        texto: str, 
                        analise_atual: Dict[str, Any],
                        forcar_uso_gpt: bool = False) -> Dict[str, Any]:
        """
        Melhora uma análise existente usando GPT quando necessário.
        
        Args:
            texto: Texto original sendo analisado
            analise_atual: Resultado da análise atual dos modelos existentes
            forcar_uso_gpt: Se True, usa GPT mesmo com alta confiança
            
        Returns:
            Análise melhorada ou a original se não for necessário melhorar
        """
        # Extrair confiança atual
        confianca = analise_atual.get('confianca', 1.0)
        
        # Decidir se vamos usar GPT
        usar_gpt = forcar_uso_gpt or confianca < self.limiar_confianca
        
        if not usar_gpt:
            self.logger.info(f"Confiança suficiente ({confianca:.2f}), mantendo análise original")
            return analise_atual
        
        self.logger.info(f"Confiança baixa ({confianca:.2f}) ou forçado, aplicando análise GPT")
        
        # Usar o GPTAnalyzer para refinar a análise
        try:
            analise_refinada = self.gpt_analyzer.ajustar_analise_existente(texto, analise_atual)
            
            # Garantir que a estrutura esperada esteja presente
            if 'sentimento' not in analise_refinada:
                self.logger.warning("Análise refinada não contém sentimento, usando original")
                return analise_atual
            
            # Marcar que foi processado pelo enhancer
            analise_refinada['processado_por_gpt'] = True
            
            # Limpar qualquer erro anterior
            self.ultimo_erro = None
            
            return analise_refinada
            
        except Exception as e:
            erro_str = str(e)
            self.logger.error(f"Erro ao usar enhancer GPT: {e}")
            
            # Registrar erro se for relacionado a cota
            if "insufficient_quota" in erro_str or "quota" in erro_str or "exceeded" in erro_str:
                self.ultimo_erro = f"Erro de cota da API: {erro_str}"
                self.logger.error(f"Erro de cota da API OpenAI: {erro_str}")
            
            # Retornar a análise original em caso de erro
            return analise_atual
    
    def analisar_caso_complexo(self, texto: str, tipo: str = 'completo') -> Optional[Dict[str, Any]]:
        """
        Analisa um caso especialmente complexo diretamente com GPT.
        Útil para casos onde os modelos convencionais têm dificuldade.
        
        Args:
            texto: Texto a ser analisado
            tipo: Tipo de análise ('sentimento', 'sarcasmo', 'completo', 'varejo')
            
        Returns:
            Resultado da análise ou None se falhar
        """
        self.logger.info(f"Realizando análise direta com GPT para caso complexo (tipo: {tipo})")
        
        try:
            result = None
            if tipo == 'sentimento':
                result = self.gpt_analyzer.analisar_sentimento(texto)
            elif tipo == 'sarcasmo':
                result = self.gpt_analyzer.analisar_sarcasmo(texto)
            elif tipo == 'varejo':
                result = self.gpt_analyzer.analisar_varejo(texto)
            else:  # 'completo' é o padrão
                result = self.gpt_analyzer.analisar_completo(texto)
                
            # Limpar último erro se a análise for bem-sucedida
            if result:
                self.ultimo_erro = None
                
            return result
            
        except Exception as e:
            erro_str = str(e)
            self.logger.error(f"Erro ao realizar análise direta com GPT: {e}")
            
            # Registrar erro se for relacionado a cota
            if "insufficient_quota" in erro_str or "quota" in erro_str or "exceeded" in erro_str:
                self.ultimo_erro = f"Erro de cota da API: {erro_str}"
                self.logger.error(f"Erro de cota da API OpenAI: {erro_str}")
                
            # Propagar a exceção para ser tratada no nível superior
            raise
    
    def obter_metricas(self) -> Dict[str, Any]:
        """
        Retorna métricas de uso do GPT.
        
        Returns:
            Dicionário com métricas coletadas
        """
        return self.gpt_analyzer.obter_metricas()

def integrar_gpt_ao_sentiment_analyzer(sentiment_analyzer, gpt_analyzer=None):
    """
    Integra o GPTSentimentEnhancer a um SentimentAnalyzer existente.
    
    Args:
        sentiment_analyzer: A instância de SentimentAnalyzer a ser aprimorada
        gpt_analyzer: Instância opcional de GPTAnalyzer. Se não fornecido, será criado um novo.
        
    Returns:
        A mesma instância, agora com capacidade GPT
    """
    # Verificar se já tem um enhancer
    if hasattr(sentiment_analyzer, 'gpt_enhancer'):
        logger.info("SentimentAnalyzer já possui GPT enhancer, atualizando")
    
    # Criar e anexar o enhancer com o GPTAnalyzer existente, se fornecido
    enhancer = GPTSentimentEnhancer(gpt_analyzer=gpt_analyzer)
    sentiment_analyzer.gpt_enhancer = enhancer
    
    # Sobrescrever o método analisar_sentimento_final para usar GPT quando necessário
    original_analisar = getattr(sentiment_analyzer, 'analisar_sentimento_final', 
                               sentiment_analyzer.analisar_sentimento)
    
    def analisar_sentimento_melhorado(texto, *args, **kwargs):
        # Chamar o método original primeiro
        resultado = original_analisar(texto, *args, **kwargs)
        
        # Verificar se devemos melhorar com GPT
        usar_gpt = kwargs.get('usar_gpt', True)
        
        if not usar_gpt:
            return resultado
        
        # Tentar melhorar com GPT
        try:
            resultado_melhorado = enhancer.melhorar_analise(texto, resultado)
            return resultado_melhorado
        except Exception as e:
            logger.error(f"Erro ao usar enhancer GPT: {e}")
            return resultado
    
    # Substituir o método
    sentiment_analyzer.analisar_sentimento_final = analisar_sentimento_melhorado
    
    # Adicionar método para análise de casos complexos
    sentiment_analyzer.analisar_caso_complexo = lambda texto, tipo='completo': enhancer.analisar_caso_complexo(texto, tipo)
    
    logger.info("GPTSentimentEnhancer integrado com sucesso ao SentimentAnalyzer")
    return sentiment_analyzer 