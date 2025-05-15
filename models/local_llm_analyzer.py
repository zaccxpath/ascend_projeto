import os
import logging
import time
import json
from typing import Dict, Any, List, Optional, Union, Tuple

from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification

logger = logging.getLogger(__name__)

class AnalyzerConfig:
    """Configurações para o analisador local de LLM"""
    
    def __init__(self, 
                 model_name: str = "deepseek-ai/deepseek-llm-7b-base",
                 device: str = "auto",
                 quantization: Optional[str] = None,
                 cache_dir: Optional[str] = None,
                 gpu_layers: int = -1):
        """
        Inicializa as configurações para o analisador local
        
        Args:
            model_name: Nome do modelo a ser utilizado (de HuggingFace ou caminho local)
            device: Dispositivo de inferência ('cpu', 'cuda', 'auto')
            quantization: Tipo de quantização a ser usada (4bit, 8bit, None)
            cache_dir: Diretório para cache de modelos
            gpu_layers: Quantidade de camadas para executar na GPU (-1 para todas)
        """
        self.model_name = model_name
        self.device = device
        self.quantization = quantization
        self.cache_dir = cache_dir
        self.gpu_layers = gpu_layers

class ModelNotLoadedException(Exception):
    """Exceção para quando o modelo não pode ser carregado"""
    pass

class LocalLLMAnalyzer:
    """
    Implementa análise de sentimento e sarcasmo usando modelos locais como DeepSeek ou LLaMA
    """
    
    def __init__(self, config: AnalyzerConfig = None):
        """
        Inicializa o analisador local
        
        Args:
            config: Configurações para o analisador. Se None, usa configurações padrão.
        """
        self.config = config or AnalyzerConfig()
        self.tokenizer = None
        self.model = None
        self.pipeline = None
        self.model_loaded = False
        self.metricas = {
            "chamadas_totais": 0,
            "sucessos": 0,
            "falhas": 0,
            "tempo_total": 0,
            "tempo_medio": 0,
            "ultimo_erro": None,
            "sentimentos": {"positivo": 0, "neutro": 0, "negativo": 0}
        }
        self.logger = logger
        
        # Tentar carregar o modelo na inicialização
        try:
            self._inicializar_modelo()
        except Exception as e:
            self.logger.error(f"Erro ao inicializar modelo local: {e}")
            self.model_loaded = False
            self.metricas["ultimo_erro"] = str(e)
    
    def _inicializar_modelo(self):
        """Inicializa o modelo e tokenizador"""
        try:
            self.logger.info(f"Inicializando modelo: {self.config.model_name}")
            inicio = time.time()
            
            # Verificar se é para usar LLaMA CPP Python (modelos GGUF)
            if self.config.model_name.endswith(".gguf") or "llama" in self.config.model_name.lower():
                try:
                    from llama_cpp import Llama
                    # Carregar modelo LLaMA usando llama.cpp
                    self._carregar_modelo_llama_cpp()
                except ImportError:
                    self.logger.warning("llama_cpp não disponível, tentando carregar com transformers")
                    self._carregar_modelo_transformers()
            else:
                # Carregar modelo usando transformers
                self._carregar_modelo_transformers()
            
            fim = time.time()
            self.logger.info(f"Modelo inicializado em {fim - inicio:.2f} segundos")
            self.model_loaded = True
            
        except Exception as e:
            self.logger.error(f"Falha ao inicializar modelo: {e}")
            self.model_loaded = False
            raise
    
    def _carregar_modelo_transformers(self):
        """Carrega modelo usando a biblioteca transformers"""
        kwargs = {}
        
        # Aplicar quantização se configurada
        if self.config.quantization:
            if self.config.quantization == "4bit":
                kwargs["load_in_4bit"] = True
            elif self.config.quantization == "8bit":
                kwargs["load_in_8bit"] = True
        
        # Adicionar cache_dir se configurado
        if self.config.cache_dir:
            kwargs["cache_dir"] = self.config.cache_dir
        
        # Carregar o tokenizador
        self.tokenizer = AutoTokenizer.from_pretrained(self.config.model_name)
            
        # Carregar o modelo de classificação
        self.model = AutoModelForSequenceClassification.from_pretrained(
            self.config.model_name, 
            **kwargs
        )
            
        # Criar pipeline de classificação
        self.pipeline = pipeline(
            "text-classification", 
            model=self.model, 
            tokenizer=self.tokenizer,
            device=self.config.device
        )
        
        self.logger.info(f"Modelo e tokenizador carregados via transformers")
    
    def _carregar_modelo_llama_cpp(self):
        """Carrega modelo usando llama-cpp-python para modelos GGUF"""
        from llama_cpp import Llama
        
        # Configurar o caminho do modelo
        model_path = self.config.model_name
        if not os.path.exists(model_path):
            model_path = os.path.join("models", self.config.model_name)
        
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Arquivo do modelo não encontrado: {model_path}")
        
        # Carregar o modelo LLaMA
        self.llm = Llama(
            model_path=model_path,
            n_gpu_layers=self.config.gpu_layers,
            n_ctx=2048
        )
        
        self.logger.info(f"Modelo LLaMA carregado via llama-cpp-python: {model_path}")
    
    def _verificar_modelo_carregado(self):
        """Verifica se o modelo está carregado, levanta exceção se não estiver"""
        if not self.model_loaded:
            raise ModelNotLoadedException("Modelo não carregado corretamente")
    
    def analisar_sentimento(self, texto: str) -> Dict[str, Any]:
        """
        Analisa o sentimento do texto usando o modelo local
        
        Args:
            texto: Texto a ser analisado
            
        Returns:
            Dicionário com resultado da análise
        """
        inicio = time.time()
        self.metricas["chamadas_totais"] += 1
        
        try:
            self._verificar_modelo_carregado()
            
            # Verificar se estamos usando transformers ou llama-cpp
            if hasattr(self, "pipeline"):
                # Analisar com pipeline do transformers
                resultado = self.pipeline(texto, truncation=True)
                
                # Converter resultado para o formato esperado
                sentimento, confianca = self._mapear_sentimento(resultado[0]["label"], resultado[0]["score"])
            else:
                # Analisar com LLaMA-CPP
                prompt = f"Analise o sentimento do seguinte texto, respondendo apenas 'positivo', 'neutro' ou 'negativo': {texto}"
                resposta = self.llm(prompt=prompt, max_tokens=10, temperature=0.1)
                texto_resposta = resposta["choices"][0]["text"].strip().lower()
                
                # Extrair sentimento da resposta
                if "positivo" in texto_resposta:
                    sentimento = "positivo"
                    confianca = 0.85
                elif "negativo" in texto_resposta:
                    sentimento = "negativo"
                    confianca = 0.85
                else:
                    sentimento = "neutro"
                    confianca = 0.7
            
            # Atualizar métricas
            self.metricas["sucessos"] += 1
            self.metricas["sentimentos"][sentimento] += 1
            
            # Preparar resultado
            resultado = {
                "sentimento": sentimento,
                "confianca": confianca,
                "processado_por_llm_local": True,
                "modelo": self.config.model_name
            }
            
        except Exception as e:
            self.logger.error(f"Erro ao analisar sentimento: {e}")
            self.metricas["falhas"] += 1
            self.metricas["ultimo_erro"] = str(e)
            
            # Fornecer resultado padrão em caso de erro
            resultado = {
                "sentimento": "neutro",
                "confianca": 0.5,
                "erro": str(e)
            }
        
        # Calcular e atualizar métricas de tempo
        fim = time.time()
        tempo_execucao = fim - inicio
        self.metricas["tempo_total"] += tempo_execucao
        self.metricas["tempo_medio"] = self.metricas["tempo_total"] / self.metricas["chamadas_totais"]
        
        return resultado
    
    def analisar_sarcasmo(self, texto: str) -> Dict[str, Any]:
        """
        Analisa se o texto contém sarcasmo
        
        Args:
            texto: Texto a ser analisado
            
        Returns:
            Dicionário com resultado da análise de sarcasmo
        """
        inicio = time.time()
        self.metricas["chamadas_totais"] += 1
        
        try:
            self._verificar_modelo_carregado()
            
            # Verificar se estamos usando transformers ou llama-cpp
            if hasattr(self, "pipeline"):
                # Para pipeline de transformers, precisamos de um modelo específico para sarcasmo
                # Aqui usamos o mesmo modelo e interpretamos
                resultado = self.pipeline(texto, truncation=True)
                
                # Inferir sarcasmo baseado no sentimento (simplificação, não é ideal)
                tem_sarcasmo = False
                probabilidade = 0.1
                
                # Um texto muito negativo com alta confiança pode indicar sarcasmo
                if resultado[0]["label"] == "negative" and resultado[0]["score"] > 0.9:
                    tem_sarcasmo = True
                    probabilidade = 0.6
            else:
                # Análise com LLaMA-CPP
                prompt = f"""
                Analise o texto a seguir e determine se contém sarcasmo ou ironia.
                Responda apenas com 'sim' ou 'não' seguido de um número entre 0 e 1 indicando a probabilidade.
                
                Texto: {texto}
                """
                resposta = self.llm(prompt=prompt, max_tokens=20, temperature=0.1)
                texto_resposta = resposta["choices"][0]["text"].strip().lower()
                
                # Extrair resultado
                tem_sarcasmo = "sim" in texto_resposta
                
                # Tentar extrair probabilidade
                try:
                    import re
                    match = re.search(r'(\d+\.\d+|\d+)', texto_resposta)
                    if match:
                        probabilidade = float(match.group(1))
                        if probabilidade > 1.0:
                            probabilidade = probabilidade / 100.0  # Caso retorne como porcentagem
                    else:
                        probabilidade = 0.7 if tem_sarcasmo else 0.1
                except:
                    probabilidade = 0.7 if tem_sarcasmo else 0.1
            
            # Atualizar métricas
            self.metricas["sucessos"] += 1
            
            # Categorizar nível de sarcasmo
            nivel_sarcasmo = "nenhum"
            if tem_sarcasmo:
                if probabilidade > 0.8:
                    nivel_sarcasmo = "alto"
                elif probabilidade > 0.5:
                    nivel_sarcasmo = "medio"
                else:
                    nivel_sarcasmo = "baixo"
            
            # Preparar resultado
            resultado = {
                "tem_sarcasmo": tem_sarcasmo,
                "probabilidade_sarcasmo": probabilidade,
                "nivel_sarcasmo": nivel_sarcasmo,
                "processado_por_llm_local": True,
                "modelo": self.config.model_name
            }
            
        except Exception as e:
            self.logger.error(f"Erro ao analisar sarcasmo: {e}")
            self.metricas["falhas"] += 1
            self.metricas["ultimo_erro"] = str(e)
            
            # Fornecer resultado padrão em caso de erro
            resultado = {
                "tem_sarcasmo": False,
                "probabilidade_sarcasmo": 0.1,
                "nivel_sarcasmo": "nenhum",
                "erro": str(e)
            }
        
        # Calcular e atualizar métricas de tempo
        fim = time.time()
        tempo_execucao = fim - inicio
        self.metricas["tempo_total"] += tempo_execucao
        self.metricas["tempo_medio"] = self.metricas["tempo_total"] / self.metricas["chamadas_totais"]
        
        return resultado
    
    def analisar_completo(self, texto: str) -> Dict[str, Any]:
        """
        Realiza análise completa do texto (sentimento e sarcasmo)
        
        Args:
            texto: Texto a ser analisado
            
        Returns:
            Dicionário com resultado completo da análise
        """
        # Analisar sentimento
        resultado_sentimento = self.analisar_sentimento(texto)
        
        # Analisar sarcasmo
        resultado_sarcasmo = self.analisar_sarcasmo(texto)
        
        # Ajustar sentimento baseado em sarcasmo
        sentimento_ajustado = resultado_sentimento["sentimento"]
        confianca_ajustada = resultado_sentimento["confianca"]
        
        # Se detectou sarcasmo com alta probabilidade, pode inverter o sentimento
        if resultado_sarcasmo["tem_sarcasmo"] and resultado_sarcasmo["probabilidade_sarcasmo"] > 0.7:
            # Preservar dados originais
            sentimento_original = sentimento_ajustado
            confianca_original = confianca_ajustada
            
            # Inverter sentimento se for muito positivo ou negativo
            if sentimento_ajustado == "positivo":
                sentimento_ajustado = "negativo"
                confianca_ajustada = confianca_ajustada * 0.8  # Reduzir confiança na inversão
            elif sentimento_ajustado == "negativo":
                sentimento_ajustado = "neutro"  # Sarcasmo negativo geralmente é mais complexo
                confianca_ajustada = confianca_ajustada * 0.7
            
            # Registrar ajuste baseado em sarcasmo
            resultado_sentimento["sentimento_original"] = sentimento_original
            resultado_sentimento["confianca_original"] = confianca_original
            resultado_sentimento["ajustado_por_sarcasmo"] = True
        
        # Atualizar sentimento após ajuste por sarcasmo
        resultado_sentimento["sentimento"] = sentimento_ajustado
        resultado_sentimento["confianca"] = confianca_ajustada
        
        # Combinar resultados
        resultado = {
            **resultado_sentimento,
            "sarcasmo": resultado_sarcasmo
        }
        
        return resultado
    
    def analisar_varejo(self, texto: str, categoria: str = None) -> Dict[str, Any]:
        """
        Análise especializada para feedback de varejo
        
        Args:
            texto: Texto a ser analisado
            categoria: Categoria de varejo (opcional)
            
        Returns:
            Dicionário com resultado da análise
        """
        resultado = self.analisar_completo(texto)
        
        # Adicionar campos específicos para varejo
        resultado["categoria_varejo"] = categoria or "geral"
        resultado["tipo_analise"] = "varejo"
        
        # Se estamos usando LLaMA-CPP, podemos extrair tópicos específicos de varejo
        if hasattr(self, "llm"):
            try:
                prompt = f"""
                Identifique até 3 aspectos principais mencionados no seguinte feedback de varejo:
                
                Texto: {texto}
                
                Liste apenas os aspectos, separados por vírgula.
                """
                resposta = self.llm(prompt=prompt, max_tokens=50, temperature=0.1)
                topicos = [t.strip() for t in resposta["choices"][0]["text"].split(",")]
                resultado["topicos"] = topicos
            except Exception as e:
                self.logger.error(f"Erro ao extrair tópicos de varejo: {e}")
                resultado["topicos"] = []
        
        return resultado
    
    def ajustar_analise_existente(self, texto: str, analise_atual: Dict[str, Any]) -> Dict[str, Any]:
        """
        Ajusta uma análise existente usando o modelo local
        
        Args:
            texto: Texto original sendo analisado
            analise_atual: Resultado da análise atual dos modelos existentes
            
        Returns:
            Análise melhorada ou a original se não for possível melhorar
        """
        if not self.model_loaded:
            return analise_atual
        
        # Se a confiança atual é baixa, realizar análise completa
        confianca_atual = analise_atual.get('confianca', 1.0)
        
        if confianca_atual < 0.7:
            try:
                # Realizar análise completa
                nova_analise = self.analisar_completo(texto)
                
                # Manter alguns campos da análise original
                for campo in ['topicos', 'aspectos']:
                    if campo in analise_atual:
                        nova_analise[campo] = analise_atual[campo]
                
                # Marcar como processado pelo LLM local
                nova_analise['processado_por_llm_local'] = True
                
                return nova_analise
            except Exception as e:
                self.logger.error(f"Erro ao ajustar análise existente: {e}")
                return analise_atual
        
        # Se sarcasmo, ajustar especificamente para isso
        tem_sarcasmo = analise_atual.get('tem_sarcasmo', False) or \
                       analise_atual.get('sarcasmo', {}).get('tem_sarcasmo', False)
                       
        if tem_sarcasmo:
            try:
                # Obter análise de sarcasmo
                resultado_sarcasmo = self.analisar_sarcasmo(texto)
                
                # Se nosso modelo detecta sarcasmo, ajustar o sentimento
                if resultado_sarcasmo['tem_sarcasmo'] and resultado_sarcasmo['probabilidade_sarcasmo'] > 0.6:
                    analise_atual['sarcasmo'] = resultado_sarcasmo
                    
                    # Preservar sentimento original
                    analise_atual['sentimento_original'] = analise_atual.get('sentimento', 'neutro')
                    analise_atual['confianca_original'] = analise_atual.get('confianca', 0.5)
                    
                    # Ajustar sentimento baseado no sarcasmo
                    if analise_atual['sentimento'] == 'positivo':
                        analise_atual['sentimento'] = 'negativo'
                        analise_atual['confianca'] = analise_atual['confianca'] * 0.8
                    elif analise_atual['sentimento'] == 'negativo':
                        analise_atual['sentimento'] = 'neutro'
                        analise_atual['confianca'] = analise_atual['confianca'] * 0.7
                    
                    analise_atual['ajustado_por_sarcasmo'] = True
                    analise_atual['processado_por_llm_local'] = True
            except Exception as e:
                self.logger.error(f"Erro ao ajustar análise de sarcasmo: {e}")
        
        return analise_atual
    
    def obter_metricas(self) -> Dict[str, Any]:
        """
        Retorna métricas de uso do analisador local
        
        Returns:
            Dicionário com métricas coletadas
        """
        return self.metricas
    
    def _mapear_sentimento(self, label: str, score: float) -> Tuple[str, float]:
        """
        Mapeia labels do modelo para o formato usado pelo sistema
        
        Args:
            label: Label retornado pelo modelo
            score: Score de confiança
            
        Returns:
            Tupla (sentimento, confianca)
        """
        # Mapear sentimento
        sentimento = "neutro"
        
        # Mapeamento para modelos tipicos de sentiment analysis
        if label == "positive" or label == "POSITIVE" or label == "1":
            sentimento = "positivo"
        elif label == "negative" or label == "NEGATIVE" or label == "-1":
            sentimento = "negativo"
        elif label == "neutral" or label == "NEUTRAL" or label == "0":
            sentimento = "neutro"
        
        return sentimento, score 