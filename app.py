from flask import Flask, render_template, request, jsonify, redirect, url_for
import speech_recognition as sr
import os
import sys
import datetime
import nltk
import pandas as pd
import plotly.graph_objects as go
import json
import logging
import re
from transformers import AutoModelForSequenceClassification, AutoTokenizer, pipeline
from huggingface_hub import snapshot_download
import numpy as np

# Configuração de logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Inicialização do Flask
app = Flask(__name__)

# Garantir que diretórios existam
os.makedirs('transcricoes', exist_ok=True)
os.makedirs('static/images', exist_ok=True)
os.makedirs('data', exist_ok=True)
os.makedirs('models/cardiffnlp-xlm-roberta', exist_ok=True)

# Verificar versão do Python sendo utilizada
logger.info(f"Python executando de: {sys.executable}")

# Stopwords em português (fallback)
STOPWORDS_PT = set(['a', 'ao', 'aos', 'aquela', 'aquelas', 'aquele', 'aqueles', 'aquilo', 'as', 'até', 
                'com', 'como', 'da', 'das', 'de', 'dela', 'delas', 'dele', 'deles', 'depois', 'do', 
                'dos', 'e', 'ela', 'elas', 'ele', 'eles', 'em', 'entre', 'era', 'eram', 'éramos', 
                'essa', 'essas', 'esse', 'esses', 'esta', 'estas', 'este', 'estes', 'eu', 'foi', 
                'fomos', 'for', 'foram', 'havia', 'isso', 'isto', 'já', 'lhe', 'lhes', 'mais', 'mas', 
                'me', 'mesmo', 'meu', 'meus', 'minha', 'minhas', 'muito', 'muitos', 'na', 'não', 
                'nas', 'nem', 'no', 'nos', 'nós', 'nossa', 'nossas', 'nosso', 'nossos', 'num', 'numa', 
                'o', 'os', 'ou', 'para', 'pela', 'pelas', 'pelo', 'pelos', 'por', 'qual', 'quando', 
                'que', 'quem', 'são', 'se', 'seja', 'sejam', 'sejamos', 'sem', 'será', 'serão', 'seu', 
                'seus', 'só', 'somos', 'sua', 'suas', 'também', 'te', 'tem', 'temos', 'tenho', 'teu', 
                'teus', 'tu', 'tua', 'tuas', 'um', 'uma', 'você', 'vocês'])

# Dicionário para análise de sentimentos em português (fallback)
PALAVRAS_POSITIVAS = set(['bom', 'boa', 'ótimo', 'ótima', 'excelente', 'maravilhoso', 'maravilhosa', 
                        'incrível', 'adorei', 'gostei', 'recomendo', 'top', 'perfeito', 'perfeita',
                        'melhor', 'satisfeito', 'satisfeita', 'feliz', 'adorou', 'gostou', 'rápido',
                        'qualidade', 'eficiente', 'legal', 'bacana', 'fantástico', 'fantástica',
                        'sensacional', 'surpreendente', 'superei', 'lindo', 'linda', 'eficaz', 
                        'amei', 'confiável', 'confortável', 'conveniente', 'durável', 'funcional', 
                        'fácil', 'prático', 'ideal', 'infalível', 'inovador', 'útil', 'vale a pena',
                        'vale o preço', 'vale o investimento', 'vantajoso', 'vantajosa', 'aprovado',
                        'aprovada', 'excepcional', 'impressionante', 'extraordinário', 'veloz', 'rápida',
                        'gostoso', 'gostosa', 'cheiroso', 'cheirosa', 'fragrância', 'aroma', 'marcante',
                        'elegante', 'sofisticado', 'sofisticada', 'luxuoso', 'luxuosa', 'delicioso',
                        'deliciosa', 'abismada', 'abismado', 'encantado', 'encantada', 'fixação', 
                        'projeção', 'duradouro', 'duradoura', 'potente', 'único', 'única', 'especial',
                        'adoro', 'adorado', 'admirado', 'admirada'])

PALAVRAS_NEGATIVAS = set(['ruim', 'péssimo', 'péssima', 'horrível', 'terrível', 'detestei', 'odiei',
                         'não gostei', 'nao gostei', 'não recomendo', 'nao recomendo', 'lixo', 'decepcionante', 'decepção', 
                         'insatisfeito', 'insatisfeita', 'problema', 'defeito', 'demorou', 'demora', 
                         'atraso', 'atrasou', 'piorar', 'falha', 'falhou', 'quebrou', 'quebrado', 
                         'nunca', 'duvidosa', 'duvidoso', 'triste', 'fraco', 'fraca', 'horrendo', 
                         'horroroso', 'caro', 'cara', 'desperdício', 'desapontado', 'desapontada',
                         'desapontamento', 'arrependido', 'arrependida', 'inutilizável', 'inútil',
                         'desperdício', 'dinheiro jogado fora', 'fraude', 'enganação', 'enganoso',
                         'enganosa', 'furado', 'furada', 'porcaria', 'pessimo', 'pessima', 'abaixo da média',
                         'não vale', 'nao vale', 'não funciona', 'nao funciona', 'complicado', 'difícil',
                         'incapaz', 'prejudicial', 'danoso', 'danosa', 'evitar', 'fuja', 'desisti',
                         'fedorento', 'fedorenta', 'fedor', 'enjoativo', 'enjoativa', 'sem fixação',
                         'fraca fixação', 'não fixa', 'nao fixa', 'evapora', 'evaporou', 'sumiu',
                         'aguado', 'sem graça', 'irritante', 'alérgico', 'alergia', 'sem projeção',
                         'barato demais', 'vagabundo', 'de quinta', 'de segunda', 'artificial',
                         'falsificado', 'falsificada', 'imitação', 'clone', 'genérico', 'superficial',
                         'infelizmente', 'trocar', 'não fiquei', 'nao fiquei', 'devolver', 'devolvi',
                         'troca', 'devolução'])

# Cores para os sentimentos nos gráficos
CORES_SENTIMENTOS = {
    'positivo': '#2ecc71',  # verde
    'neutro': '#f39c12',    # laranja
    'negativo': '#e74c3c'   # vermelho
}

# Adicionar após as importações e antes de declarar as variáveis globais

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
        self.inicializar_nltk()
        self.stop_words = self._carregar_stopwords()
        self.sentiment_model, self.using_xlm_roberta = self._inicializar_sentiment_model()
        self._testar_modelo()
        
    def _testar_modelo(self):
        """Testa o modelo após inicialização para garantir funcionamento correto"""
        try:
            if self.using_xlm_roberta and self.sentiment_model is not None:
                try:
                    # Frases simples para teste em português
                    textos_teste = [
                        "Eu gostei muito desse produto, é excelente!",
                        "Não estou satisfeito com a qualidade, não recomendo.",
                        "O produto chegou no prazo, é razoável pelo preço."
                    ]
                    
                    logger.info("Testando modelo XLM-RoBERTa com frases simples...")
                    for texto in textos_teste:
                        try:
                            resultado = self.sentiment_model(texto[:100])[0]
                            logger.info(f"Teste: '{texto[:30]}...' => Label: {resultado['label']}, Score: {resultado['score']:.4f}")
                        except Exception as e:
                            logger.error(f"Erro ao testar modelo com texto '{texto[:30]}...': {e}")
                    
                    logger.info("Teste do modelo concluído com sucesso!")
                except Exception as e:
                    logger.error(f"Erro ao testar modelo: {e}")
                    logger.warning("Desativando XLM-RoBERTa e usando análise básica como fallback")
                    self.using_xlm_roberta = False
            else:
                logger.warning("XLM-RoBERTa não disponível para teste, usando análise básica como fallback")
        except Exception as e:
            logger.error(f"Erro fatal no teste do modelo: {e}")
            logger.warning("Continuando com análise básica para evitar falha da aplicação")
        
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
            
    def _inicializar_sentiment_model(self):
        """Inicializa modelo XLM-RoBERTa para análise de sentimentos"""
        try:
            # Configuração mais detalhada de logging
            logger.info("=" * 50)
            logger.info("Iniciando carregamento do modelo XLM-RoBERTa")
            logger.info("=" * 50)
            
            # Nome do modelo
            model_name = "cardiffnlp/twitter-xlm-roberta-base-sentiment"
            
            # Diretório local onde o modelo será salvo
            model_dir = os.path.join(os.getcwd(), "models", "cardiffnlp-xlm-roberta")
            os.makedirs(model_dir, exist_ok=True)
            
            # Verificar se temos conexão com a internet
            try:
                import socket
                socket.create_connection(("huggingface.co", 443), timeout=5)
                logger.info("Conexão com huggingface.co estabelecida com sucesso")
                internet_disponivel = True
            except Exception as e:
                logger.error(f"Erro ao verificar conexão com a internet: {e}")
                internet_disponivel = False
                logger.warning("Tentando carregamento do modelo apenas do cache local")

            # Verificar se o modelo já existe localmente
            config_path = os.path.join(model_dir, "config.json")
            model_existe_localmente = os.path.exists(config_path)
            
            if model_existe_localmente:
                logger.info(f"Modelo encontrado localmente em: {model_dir}")
                logger.info("Tentando carregar modelo do diretório local...")
                
                try:
                    # Tentar carregar tokenizador e modelo localmente primeiro
                    logger.info("Carregando tokenizador local...")
                    tokenizer = AutoTokenizer.from_pretrained(model_dir, local_files_only=True)
                    logger.info("Tokenizador local carregado com sucesso!")
                    
                    logger.info("Carregando modelo local...")
                    model = AutoModelForSequenceClassification.from_pretrained(model_dir, local_files_only=True)
                    logger.info("Modelo local carregado com sucesso!")
                    
                    # Criar pipeline para análise de sentimentos
                    nlp_sentiment = pipeline("sentiment-analysis", 
                                            model=model, 
                                            tokenizer=tokenizer,
                                            max_length=512,
                                            truncation=True)
                    
                    logger.info("Pipeline criado com arquivos locais. Modelo XLM-RoBERTa carregado com sucesso.")
                    return nlp_sentiment, True
                    
                except Exception as e:
                    logger.error(f"Erro ao carregar modelo local: {e}")
                    logger.warning("Modelo local parece estar corrompido. Tentando download novamente.")
                    # Se o modelo local falhar, continuamos para tentar baixar
            
            # Se não existir localmente ou falhou ao carregar, tente baixar
            if not model_existe_localmente or (model_existe_localmente and 'nlp_sentiment' not in locals()):
                if not internet_disponivel:
                    logger.error("Impossível baixar modelo sem conexão com a internet")
                    logger.info("Usando modelo de fallback simplificado baseado em regras")
                    return ModeloFallbackSimplificado(), False
                
                logger.info("Tentando baixar modelo diretamente do Hugging Face Hub...")
                
                try:
                    # Primeiro método: via AutoTokenizer e AutoModel
                    logger.info("Método 1: Usando AutoTokenizer.from_pretrained")
                    tokenizer = AutoTokenizer.from_pretrained(model_name)
                    logger.info("Tokenizador baixado com sucesso!")
                    
                    logger.info("Usando AutoModelForSequenceClassification.from_pretrained")
                    model = AutoModelForSequenceClassification.from_pretrained(model_name)
                    logger.info("Modelo baixado com sucesso!")
                
                    # Criar pipeline
                    nlp_sentiment = pipeline("sentiment-analysis", 
                                            model=model, 
                                            tokenizer=tokenizer,
                                            max_length=512,
                                            truncation=True)
                
                    # Salvar modelo localmente para uso futuro
                    logger.info(f"Salvando modelo em {model_dir} para uso futuro")
                    tokenizer.save_pretrained(model_dir)
                    model.save_pretrained(model_dir)
                    
                    logger.info("Modelo XLM-RoBERTa carregado e salvo localmente com sucesso.")
                    return nlp_sentiment, True
                    
                except Exception as e:
                    logger.error(f"Método 1 falhou: {e}")
                    logger.info("Tentando método alternativo de download...")
                    
                    try:
                        # Segundo método: via snapshot_download
                        logger.info("Método 2: Usando snapshot_download")
                        from huggingface_hub import snapshot_download
                        
                        # Limpar diretório caso exista arquivos corrompidos
                        import shutil
                        if os.path.exists(model_dir):
                            logger.info(f"Removendo arquivos potencialmente corrompidos de {model_dir}")
                            for item in os.listdir(model_dir):
                                item_path = os.path.join(model_dir, item)
                                try:
                                    if os.path.isfile(item_path):
                                        os.unlink(item_path)
                                    elif os.path.isdir(item_path):
                                        shutil.rmtree(item_path)
                                except Exception as e:
                                    logger.error(f"Erro ao remover {item_path}: {e}")
                        
                        logger.info(f"Baixando modelo {model_name} para {model_dir}...")
                        snapshot_path = snapshot_download(
                            repo_id=model_name,
                            local_dir=model_dir,
                            local_dir_use_symlinks=False  # Evita symlinks que podem causar problemas no Windows
                        )
                        logger.info(f"Download do modelo concluído em: {snapshot_path}")
                        
                        # Carregar do diretório local após o download
                        logger.info("Carregando tokenizador do diretório baixado...")
                        tokenizer = AutoTokenizer.from_pretrained(model_dir)
                        logger.info("Carregando modelo do diretório baixado...")
                        model = AutoModelForSequenceClassification.from_pretrained(model_dir)
                    
                        # Criar pipeline
                        nlp_sentiment = pipeline("sentiment-analysis", 
                                                model=model, 
                                                tokenizer=tokenizer,
                                                max_length=512,
                                                truncation=True)
                    
                        logger.info("Modelo XLM-RoBERTa carregado com sucesso via método alternativo.")
                        return nlp_sentiment, True
                        
                    except Exception as e2:
                        logger.error(f"Método 2 também falhou: {e2}")
                        logger.error(f"Detalhes completos do erro: {str(e2)}")
                        logger.critical("Todos os métodos de download do modelo falharam.")
                        
                        # Verificar se há problemas de permissão de arquivo
                        try:
                            test_file = os.path.join(model_dir, 'test_write.txt')
                            with open(test_file, 'w') as f:
                                f.write('test')
                            os.unlink(test_file)
                            logger.info("Permissões de escrita OK no diretório do modelo")
                        except Exception as perm_error:
                            logger.error(f"Problema de permissão de escrita no diretório do modelo: {perm_error}")
                        
                        logger.info("Usando modelo de fallback simplificado baseado em regras")
                        return ModeloFallbackSimplificado(), False
            
            # Se chegamos aqui, algo deu errado
            logger.error("Não foi possível carregar o modelo por nenhum método")
            logger.info("Usando modelo de fallback simplificado baseado em regras")
            return ModeloFallbackSimplificado(), False
            
        except Exception as e:
            logger.error(f"Erro fatal ao inicializar modelo XLM-RoBERTa: {e}")
            logger.critical(f"Detalhes completos do erro: {str(e)}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
            logger.info("Usando modelo de fallback simplificado baseado em regras")
            return ModeloFallbackSimplificado(), False
    
    def tokenizar_texto(self, texto):
        """Tokeniza o texto usando regex para maior confiabilidade"""
        try:
            tokens = re.findall(r'\b\w+\b', texto.lower())
            return tokens
        except Exception as e:
            logger.warning(f"Erro ao tokenizar texto: {e}")
            return texto.lower().split()
            
    def analisar_sentimento_basico(self, texto):
        """Análise básica de sentimento usando contagem de palavras positivas/negativas"""
        tokens = self.tokenizar_texto(texto)
        
        # Log para debug
        logger.info(f"Análise básica - texto: {texto}")
        logger.info(f"Tokens detectados: {tokens}")
        
        # Verificar negações de palavras positivas (ex: "não fiquei satisfeita")
        # Inverte palavras positivas que são precedidas por negação
        tokens_negativos_invertidos = []
        negacoes = ['não', 'nao', 'nunca', 'jamais', 'nem']
        
        for i in range(len(tokens) - 1):
            if tokens[i] in negacoes and i + 1 < len(tokens):
                proximo_token = tokens[i + 1]
                if proximo_token in PALAVRAS_POSITIVAS:
                    tokens_negativos_invertidos.append(f"não_{proximo_token}")
                    logger.info(f"Palavra positiva negada: {tokens[i]} {proximo_token}")
        
        # Criar uma cópia da lista tokens para análise
        tokens_analise = tokens.copy()
        
        # Adicionar expressões compostas específicas que indicam negatividade
        texto_lower = texto.lower()
        if 'não fiquei satisfeita' in texto_lower or 'nao fiquei satisfeita' in texto_lower:
            tokens_negativos_invertidos.append('não_satisfeita')
            logger.info("Expressão negativa detectada: 'não fiquei satisfeita'")
        
        if 'quero trocar' in texto_lower:
            tokens_negativos_invertidos.append('quero_trocar')
            logger.info("Expressão de insatisfação detectada: 'quero trocar'")
        
        if 'infelizmente' in texto_lower:
            tokens_negativos_invertidos.append('infelizmente')
            logger.info("Expressão negativa detectada: 'infelizmente'")
            
        # Expressões específicas para fixação de perfumes
        if 'fixação' in texto_lower:
            if 'fraca' in texto_lower and any(t in tokens for t in ['fixação', 'projeção']):
                tokens_negativos_invertidos.append('fixação_fraca')
                logger.info("Expressão negativa detectada: 'fixação fraca'")
                
            if 'diferença' in texto_lower and 'fixação' in texto_lower:
                # Considerar isso como uma observação neutra/negativa
                tokens_negativos_invertidos.append('diferença_fixação')
                logger.info("Expressão comparativa potencialmente negativa: 'diferença fixação'")
                
            if any(adj in texto_lower for adj in ['menor', 'pior', 'baixa', 'pouca', 'fraca']):
                tokens_negativos_invertidos.append('fixação_inferior')
                logger.info("Expressão negativa sobre fixação detectada")
                
            # Compensar a contagem dupla de fixação quando mencionada em contexto negativo
            if any(termo in tokens_negativos_invertidos for termo in ['fixação_fraca', 'diferença_fixação', 'fixação_inferior']):
                for i in range(tokens.count('fixação')):
                    if 'fixação' in PALAVRAS_POSITIVAS:
                        # Remove o efeito positivo da palavra "fixação" quando em contexto negativo
                        logger.info("Removendo 'fixação' como palavra positiva devido ao contexto negativo")
        
        # Contar palavras positivas e negativas
        palavras_positivas_encontradas = [t for t in tokens if t in PALAVRAS_POSITIVAS]
        palavras_negativas_encontradas = [t for t in tokens if t in PALAVRAS_NEGATIVAS]
        
        logger.info(f"Palavras positivas: {palavras_positivas_encontradas}")
        logger.info(f"Palavras negativas: {palavras_negativas_encontradas}")
        logger.info(f"Expressões negativas invertidas: {tokens_negativos_invertidos}")
        
        # Contar palavras positivas, excluindo as que foram negadas
        count_pos = 0
        for t in tokens:
            if t in PALAVRAS_POSITIVAS:
                # Verificar se esta palavra foi precedida por negação
                if any(f"não_{t}" in neg for neg in tokens_negativos_invertidos):
                    continue
                    
                # Se for "fixação" e temos algum contexto negativo de fixação, não conte como positivo
                if t == 'fixação' and any(termo in tokens_negativos_invertidos for termo in ['fixação_fraca', 'diferença_fixação', 'fixação_inferior']):
                    continue
                    
                count_pos += 1
        
        # Contar palavras negativas + palavras positivas negadas
        count_neg = len(palavras_negativas_encontradas) + len(tokens_negativos_invertidos)
            
        # Verificações específicas para expressões negativas comuns em português
        if 'não gostei' in texto_lower or 'nao gostei' in texto_lower:
            count_neg += 3
            logger.info("Expressão negativa 'não gostei' detectada")
        if 'péssimo' in texto_lower or 'pessimo' in texto_lower:
            count_neg += 3
            logger.info("Palavra fortemente negativa 'péssimo' detectada")
        if 'ruim' in texto_lower:
            count_neg += 2
            logger.info("Palavra negativa 'ruim' detectada")
            
        # Verificações específicas para perfumes e produtos de beleza
        if 'perfume' in texto_lower or 'fragrância' in texto_lower:
            # Frases positivas específicas de perfumes
            if 'gostoso' in texto_lower or 'cheiroso' in texto_lower:
                if not any(n in texto_lower for n in negacoes):  # Se não tem negação
                    count_pos += 2
                    logger.info("Termos positivos de perfume: gostoso/cheiroso")
            if 'lindo' in texto_lower or 'linda' in texto_lower:
                if not any(n in texto_lower for n in negacoes):  # Se não tem negação
                    count_pos += 2
                    logger.info("Termo positivo: lindo/linda")
            if 'abismada' in texto_lower and 'quanto' in texto_lower and 'bom' in texto_lower:
                count_pos += 3
                logger.info("Frase de impacto positivo detectada: abismada o quanto é bom")
                
            # Verificação especial para trocas e fixação
            if ('trocar' in texto_lower or 'quero' in texto_lower) and 'fixação' in texto_lower:
                count_neg += 3
                logger.info("Detectada reclamação sobre fixação - cliente quer trocar")
                
            # Consideração especial para presentes
            if 'presente' in texto_lower and ('diferença' in texto_lower or 'fraca' in texto_lower):
                # Provavelmente comparando com outro produto de forma neutra/negativa
                count_neg += 1
                count_pos -= 1  # Reduz o peso positivo
                logger.info("Contexto de presente com comparação negativa detectado")
        
        # Lógica de decisão melhorada
        logger.info(f"Contagem inicial: positivas={count_pos}, negativas={count_neg}")
        
        # Se contém tanto 'não' quanto 'satisfeita', força para negativo
        if 'não' in texto_lower and 'satisfeita' in texto_lower:
            count_neg += 3
            count_pos -= 1 # Anula parcialmente o efeito de 'satisfeita'
            logger.info("Detectada negação de satisfação - forçando sentimento negativo")
            
        # Contexto neutro de análise/comparação
        if (('comparação' in texto_lower or 'diferença' in texto_lower or 'notou' in texto_lower) 
               and not any(termo in texto_lower for termo in ['adorei', 'gostei', 'detestei', 'odiei'])):
            # Em comentários comparativos sem emoção forte, forçar para neutro
            logger.info("Contexto de comparação sem emoção forte detectado - forçando neutro")
            
            # Se for um comentário de comparação de características técnicas, torne-o neutro
            if ('fixação' in texto_lower or 'aroma' in texto_lower or 'intensidade' in texto_lower) and 'diferença' in texto_lower:
                # Este é mais uma observação técnica do que uma crítica emocional
                sentimento_forcado = 'neutro'
                compound_forcado = 0.0
                logger.info("Comentário técnico comparativo detectado - forçando neutro")
                # Mas preservamos count_pos e count_neg para referência
            
            # Reduzir a diferença entre positivo e negativo para tornar mais neutro
            if count_pos > count_neg:
                count_pos = min(count_pos, count_neg + 1)
            elif count_neg > count_pos:
                count_neg = min(count_neg, count_pos + 1)
        
        # Decisão final baseada na contagem
        if 'sentimento_forcado' in locals():
            # Se já decidimos forçar um sentimento específico, use-o
            sentimento = sentimento_forcado
            compound = compound_forcado
            logger.info(f"Usando sentimento forçado: {sentimento}")
        elif count_pos > count_neg + 1:  # Precisamos de mais vantagem para positivo
            sentimento = 'positivo'
            compound = 0.7
        elif count_neg > count_pos + 1:  # Dando uma margem similar para negativo  
            sentimento = 'negativo'
            compound = -0.7
        else:
            # Empate ou diferença pequena -> neutro
            sentimento = 'neutro'
            compound = 0.0
            logger.info("Equilíbrio entre positivo e negativo - classificando como neutro")
        
        # Casos especiais de detecção
        if 'não fiquei satisfeita' in texto_lower or 'nao fiquei satisfeita' in texto_lower:
            sentimento = 'negativo'
            compound = -0.8
            logger.info("Expressão forte de insatisfação detectada - forçando sentimento negativo")
            
        if count_pos == count_neg:
            if any(p in texto.lower() for p in ['adorei', 'perfeito', 'excelente', 'maravilhoso']):
                if not any(f"{neg} {p}" in texto_lower for neg in negacoes for p in ['adorei', 'perfeito', 'excelente', 'maravilhoso']):
                    sentimento = 'positivo'
                    compound = 0.8
            elif any(p in texto.lower() for p in ['péssimo', 'horrível', 'detestei', 'nunca mais', 'infelizmente']):
                sentimento = 'negativo'
                compound = -0.8
        
        logger.info(f"Análise básica final: pos={count_pos}, neg={count_neg}, sentimento={sentimento}, compound={compound}")
        return {
            'compound': compound,
            'neg': 0.0 if sentimento != 'negativo' else abs(compound),
            'neu': 1.0 if sentimento == 'neutro' else 0.0,
            'pos': 0.0 if sentimento != 'positivo' else compound,
            'sentimento': sentimento
        }
    
    def _mapear_sentimento_xlm_roberta(self, resultado):
        """Mapeia a saída do modelo XLM-RoBERTa para o formato utilizado pelo sistema"""
        # O modelo XLM-RoBERTa retorna rótulos como "positive", "neutral" ou "negative"
        # Precisamos mapear para os rótulos em português usados no sistema
        label = resultado['label'].lower()
        score = resultado['score']
        
        # Mapeamento direto para o modelo CardiffNLP
        mapping = {
            'positive': ('positivo', score),
            'neutral': ('neutro', 0.0),  # Neutro sempre tem compound 0
            'negative': ('negativo', -score)  # Negativo tem compound negativo
        }
        
        # Se não for um dos labels esperados, verificamos por prefixos ou tentamos inferir
        if label not in mapping:
            logger.warning(f"Label desconhecido recebido do modelo: {label}")
            # Tentar fazer um mapeamento por substring
            if 'pos' in label:
                sentimento, compound = 'positivo', score
            elif 'neg' in label:
                sentimento, compound = 'negativo', -score
            elif 'neu' in label:
                sentimento, compound = 'neutro', 0.0
            else:
                # Fallback - assumir neutro
                sentimento, compound = 'neutro', 0.0
        else:
            sentimento, compound = mapping[label]
            
        logger.info(f"XLM-RoBERTa label mapeado: {label} -> {sentimento} (score: {score}, compound: {compound})")
            
        return {
            'compound': compound,
            'neg': score if sentimento == 'negativo' else 0.0,
            'neu': score if sentimento == 'neutro' else 0.0, 
            'pos': score if sentimento == 'positivo' else 0.0,
            'sentimento': sentimento,
            'score': score
        }
            
    def analisar_sentimento(self, texto):
        """Analisa o sentimento do texto usando XLM-RoBERTa ou método fallback"""
        try:
            # Verificar se o texto parece estar em português ou está vazio
            if not texto or len(texto.strip()) == 0:
                logger.warning("Texto vazio recebido para análise")
                return {
                    'compound': 0.0, 
                    'neg': 0.0, 
                    'neu': 1.0, 
                    'pos': 0.0, 
                    'sentimento': 'neutro',
                    'topicos': []
                }
            
            # Verificar caracteres específicos do português
            texto_lower = texto.lower()
            palavras_pt = ['não', 'muito', 'bom', 'ruim', 'gostei', 'ótimo', 'péssimo', 'é', 'para', 
                         'com', 'e', 'o', 'a', 'os', 'as', 'um', 'uma', 'eu', 'você', 'ele', 'ela', 
                         'nós', 'vocês', 'eles', 'elas', 'mais', 'menos', 'que', 'porque', 'pois',
                         'então', 'entrega', 'produto', 'comprei', 'gostoso', 'frasco', 'perfume']
            
            # Se várias palavras portuguesas forem encontradas, considere como texto em português
            palavras_pt_encontradas = sum(1 for palavra in palavras_pt if palavra in texto_lower)
            eh_portugues = palavras_pt_encontradas >= 3 or any(c in texto for c in 'áàâãéêíóôõúüçÁÀÂÃÉÊÍÓÔÕÚÜÇ')
            
            logger.info(f"Texto analisado: '{texto[:50]}...' - Parece ser português? {eh_portugues}")
            
            # Usar o modelo XLM-RoBERTa se disponível
            if self.using_xlm_roberta and self.sentiment_model is not None:
                try:
                    logger.info("Analisando sentimento com XLM-RoBERTa")
                    
                    # Limitar o tamanho do texto para o modelo (evitar truncamento excessivo)
                    texto_limitado = texto[:512]  # A maioria dos transformers tem limite de 512 tokens
                    
                    # Realizar análise com o modelo
                    resultado = self.sentiment_model(texto_limitado)[0]
                    logger.info(f"Resultado XLM-RoBERTa: {resultado}")
                    
                    # Mapear resultado para o formato usado pelo sistema
                    scores = self._mapear_sentimento_xlm_roberta(resultado)
                    
                    # Para textos em português, fazer ajustes específicos com base em expressões idiomáticas
                    if eh_portugues:
                        # Combinar com análise básica para casos específicos em português
                        # Especialmente para expressões idiomáticas e contexto de produtos
                        if 'não fiquei satisfeita' in texto_lower or 'nao fiquei satisfeita' in texto_lower:
                            logger.info("Expressão forte de insatisfação detectada - ajustando sentimento")
                            scores['sentimento'] = 'negativo'
                            scores['compound'] = -0.8
                            scores['neg'] = 0.8
                            scores['pos'] = 0.0
                            scores['neu'] = 0.0
                        
                        if 'perfume' in texto_lower and 'fixação' in texto_lower and 'fraca' in texto_lower:
                            # Contexto específico de produto
                            logger.info("Contexto específico de produto (perfume com fixação fraca) - ajustando sentimento")
                            scores['sentimento'] = 'negativo'
                            scores['compound'] = -0.7
                            scores['neg'] = 0.7
                            scores['pos'] = 0.0
                            scores['neu'] = 0.3
                        
                        # Para produtos com expressões fortes de satisfação ou insatisfação
                        if any(expr in texto_lower for expr in ['adorei', 'maravilhoso', 'perfeito', 'excelente']):
                            if not any(neg in texto_lower for neg in ['não', 'nao', 'nunca']):
                                logger.info("Expressão forte de satisfação detectada - reforçando sentimento positivo")
                                scores['sentimento'] = 'positivo'
                                scores['compound'] = 0.9
                                scores['pos'] = 0.9
                                scores['neg'] = 0.0
                                scores['neu'] = 0.1
                                
                        if any(expr in texto_lower for expr in ['horrível', 'péssimo', 'detestei', 'odiei']):
                            logger.info("Expressão forte de insatisfação detectada - reforçando sentimento negativo")
                            scores['sentimento'] = 'negativo'
                            scores['compound'] = -0.9
                            scores['neg'] = 0.9
                            scores['pos'] = 0.0
                            scores['neu'] = 0.1
                    
                    logger.info(f"XLM-RoBERTa (final): sentimento={scores['sentimento']}, score={scores.get('score', 0)}, compound={scores['compound']}")
                except Exception as e:
                    logger.error(f"Erro na análise com XLM-RoBERTa: {e}")
                    logger.warning("Fallback para análise básica devido a erro no modelo")
                    scores = self.analisar_sentimento_basico(texto)
            else:
                logger.warning("Modelo XLM-RoBERTa não disponível. Usando análise básica.")
                scores = self.analisar_sentimento_basico(texto)
                
            # Extrair tópicos/palavras-chave do texto
            tokens = self.tokenizar_texto(texto)
            palavras_filtradas = [palavra for palavra in tokens if palavra.isalnum() and palavra not in self.stop_words]
            scores['topicos'] = palavras_filtradas[:3] if len(palavras_filtradas) >= 3 else palavras_filtradas
            
            # Adicionar análise de aspectos
            if not hasattr(self, 'aspect_extractor'):
                self.aspect_extractor = AspectExtractor()
                logger.info("Inicializando o extrator de aspectos")
            
            # Extrair aspectos e associar aos sentimentos
            logger.info(f"Extraindo aspectos do texto: '{texto[:50]}...'")
            analise_aspectos = self.aspect_extractor.extrair_aspectos(texto, scores)
            
            # Adicionar logs detalhados sobre os aspectos extraídos
            if 'summary' in analise_aspectos:
                aspectos_detectados = analise_aspectos['summary'].get('aspects_detected', [])
                aspecto_principal = analise_aspectos['summary'].get('primary_aspect', None)
                logger.info(f"Aspectos detectados: {aspectos_detectados}")
                logger.info(f"Aspecto principal: {aspecto_principal}")
                
                # Detalhar relevância de cada aspecto
                if aspectos_detectados and 'aspectos' in analise_aspectos:
                    for aspecto in aspectos_detectados:
                        if aspecto in analise_aspectos['aspectos']:
                            relevancia = analise_aspectos['aspectos'][aspecto].get('relevance', 0)
                            sentimento = analise_aspectos['aspectos'][aspecto].get('sentiment', 0)
                            logger.info(f"Aspecto '{aspecto}': relevância={relevancia}, sentimento={sentimento}")
            else:
                logger.warning("Nenhum aspecto encontrado na análise")
            
            # Adicionar resultados de aspectos ao resultado original
            scores['aspectos'] = analise_aspectos
            
            return scores
        except Exception as e:
            logger.error(f"Erro na análise de sentimento: {e}")
            return self.analisar_sentimento_basico(texto)

class SpeechHandler:
    """Classe para lidar com reconhecimento de voz"""
    
    def __init__(self, sentiment_analyzer):
        self.sentiment_analyzer = sentiment_analyzer
        
    def ouvir_microfone(self):
        """Captura áudio do microfone e transcreve para texto"""
        try:
            try:
                import pyaudio
                p = pyaudio.PyAudio()
                p.terminate()
                logger.info("PyAudio importado e inicializado com sucesso!")
            except ImportError as e:
                logger.error(f"PyAudio não está instalado: {e}")
                return "Para usar a gravação de voz, o PyAudio precisa ser instalado. Por favor, use a opção 'Adicionar Feedback' para inserir texto manualmente.", {"compound": 0, "neg": 0, "neu": 1, "pos": 0, "sentimento": "neutro", "topicos": []}
            try:
                microfone_teste = sr.Microphone()
                logger.info("Microfone inicializado com sucesso!")
            except Exception as e:
                logger.error(f"Erro ao inicializar microfone: {e}")
                return f"Erro ao inicializar microfone: {e}. Por favor, use a entrada de texto manual.", {"compound": 0, "neg": 0, "neu": 1, "pos": 0, "sentimento": "neutro", "topicos": []}
            reconhecedor = sr.Recognizer()
            with sr.Microphone() as source:
                logger.info("Ajustando para ruído ambiente...")
                # Aumentar a sensibilidade para captar melhor o áudio
                reconhecedor.dynamic_energy_threshold = True
                reconhecedor.energy_threshold = 300
                reconhecedor.adjust_for_ambient_noise(source, duration=1.0)
                logger.info("Ouvindo...")
                
                # Configurações para detecção de silêncio e tempo máximo
                # Aumentar o tempo de pausa para 2.5 segundos, evitando cortes prematuros
                reconhecedor.pause_threshold = 2.5
                # Tempo máximo de gravação: 30 segundos
                max_duration = 30
                # Tempo para avisar o usuário que está próximo do limite
                warning_threshold = 25
                
                # Aumentar o phrase_time_limit para 15 segundos para permitir frases mais longas
                audio = reconhecedor.listen(source, timeout=max_duration, phrase_time_limit=15)
                logger.info("Processando áudio...")
            texto = reconhecedor.recognize_google(audio, language='pt-BR')
            logger.info(f"Texto transcrito: {texto}")
            
            # Verificar se o texto termina abruptamente com palavras cortadas
            palavras = texto.split()
            if len(palavras) > 0 and len(palavras[-1]) <= 2:
                # Se a última palavra for muito curta, pode estar cortada
                texto = " ".join(palavras[:-1])
                logger.info(f"Texto ajustado (removida possível palavra cortada): {texto}")
            
            analise = self.sentiment_analyzer.analisar_sentimento(texto)
            return texto, analise
        except sr.WaitTimeoutError:
            logger.warning("Timeout de espera: nenhuma fala detectada")
            return "Não detectei nenhuma fala. Por favor, tente novamente falando mais alto.", {"compound": 0, "neg": 0, "neu": 1, "pos": 0, "sentimento": "neutro", "topicos": []}
        except sr.UnknownValueError:
            logger.warning("Fala não reconhecida")
            return "Não entendi o que você disse.", {"compound": 0, "neg": 0, "neu": 1, "pos": 0, "sentimento": "neutro", "topicos": []}
        except sr.RequestError as e:
            logger.error(f"Erro ao solicitar resultados da API: {e}")
            return f"Erro ao solicitar resultados do serviço de reconhecimento de fala: {e}", {"compound": 0, "neg": 0, "neu": 1, "pos": 0, "sentimento": "neutro", "topicos": []}
        except Exception as e:
            logger.error(f"Erro inesperado: {e}")
            return f"Erro inesperado: {e}", {"compound": 0, "neg": 0, "neu": 1, "pos": 0, "sentimento": "neutro", "topicos": []}

class DataHandler:
    """Classe para manipulação de dados"""
    
    def __init__(self):
        pass
        
    def salvar_transcricao(self, texto, analise):
        """Salva a transcrição e análise no arquivo de histórico"""
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        with open('transcricoes/historico.txt', 'a', encoding='utf-8') as arquivo:
            arquivo.write(f"[{timestamp}] {texto}\n")
        dados = {
            'timestamp': timestamp,
            'texto': texto,
            'compound': analise['compound'],
            'negativo': analise.get('neg', 0),
            'neutro': analise.get('neu', 0),
            'positivo': analise.get('pos', 0),
            'sentimento': analise['sentimento'],
            'topicos': ','.join(analise['topicos']) if analise.get('topicos') else ''
        }
        
        # Extrair informações de aspectos se existirem
        aspectos_detectados = []
        aspecto_principal = None
        
        if 'aspectos' in analise:
            if 'summary' in analise['aspectos']:
                aspectos_detectados = analise['aspectos']['summary'].get('aspects_detected', [])
                aspecto_principal = analise['aspectos']['summary'].get('primary_aspect', None)
                logger.info(f"Salvando análise com aspectos: detectados={aspectos_detectados}, principal={aspecto_principal}")
            else:
                logger.warning("Objeto de aspectos não contém summary")
        else:
            logger.warning("Análise não contém informações de aspectos")
        
        # Adicionar aos dados a serem salvos
        dados['aspectos_detectados'] = ','.join(aspectos_detectados) if aspectos_detectados else ''
        dados['aspecto_principal'] = aspecto_principal if aspecto_principal else ''
        
        # Log dos dados que serão salvos
        logger.info(f"Salvando dados no CSV: aspectos_detectados='{dados['aspectos_detectados']}', aspecto_principal='{dados['aspecto_principal']}'")
        
        df_novo = pd.DataFrame([dados])
        csv_path = 'data/analises.csv'
        if os.path.exists(csv_path):
            df_existente = pd.read_csv(csv_path)
            df_atualizado = pd.concat([df_existente, df_novo], ignore_index=True)
        else:
            df_atualizado = df_novo
        df_atualizado.to_csv(csv_path, index=False)
        logger.info(f"Transcrição salva: {texto}")
        
    def carregar_historico(self):
        """Carrega o histórico de transcrições"""
        try:
            with open('transcricoes/historico.txt', 'r', encoding='utf-8') as arquivo:
                return arquivo.readlines()
        except FileNotFoundError:
            logger.warning("Arquivo de histórico não encontrado")
            return ["Nenhuma transcrição encontrada."]
        
    def carregar_analises(self):
        """Carrega as análises do CSV"""
        try:
            csv_path = 'data/analises.csv'
            if os.path.exists(csv_path):
                try:
                    analises = pd.read_csv(csv_path)
                    total_registros = len(analises)
                    logger.info(f"CSV carregado com sucesso: {total_registros} registros encontrados")
                    
                    # Preencher valores NaN com strings vazias
                    analises['aspectos_detectados'].fillna('', inplace=True)
                    analises['aspecto_principal'].fillna('', inplace=True)
                    
                    # Converter colunas numéricas para valores padrão para evitar problemas com NaN
                    for col in ['compound', 'negativo', 'neutro', 'positivo']:
                        if col in analises.columns:
                            analises[col].fillna(0.0, inplace=True)
                    
                    # Converter para dicionário
                    analises_dict = analises.to_dict('records')
                    logger.info(f"Primeiros registros: {analises_dict[:2] if analises_dict else []}")
                    
                    # Verificação extra para colunas importantes
                    for analise in analises_dict:
                        # Garantir que topicos seja uma string
                        if 'topicos' in analise and not isinstance(analise['topicos'], str):
                            analise['topicos'] = ''
                        
                        # Garantir que aspectos_detectados seja uma string
                        if 'aspectos_detectados' in analise and not isinstance(analise['aspectos_detectados'], str):
                            analise['aspectos_detectados'] = ''
                        
                        # Garantir que aspecto_principal seja uma string
                        if 'aspecto_principal' in analise and not isinstance(analise['aspecto_principal'], str):
                            analise['aspecto_principal'] = ''
                    
                    return analises_dict
                except Exception as e:
                    logger.error(f"Erro ao processar CSV: {e}")
                    return []
            else:
                logger.warning(f"Arquivo de análises CSV não encontrado: {csv_path}")
                return []
        except Exception as e:
            logger.error(f"Erro inesperado ao carregar análises: {e}")
            return []
            
    def gerar_graficos(self):
        """Gera gráficos para o dashboard"""
        graficos = {}
        estatisticas_dashboard = {}
        try:
            csv_path = 'data/analises.csv'
            if not os.path.exists(csv_path):
                logger.warning("Arquivo de análises não encontrado.")
                return {"erro": "Sem dados suficientes para gerar gráficos"}
            
            # Carregar o dataset principal
            df = pd.read_csv(csv_path)
            if len(df) < 2:
                logger.warning("Menos de 2 registros no arquivo de análises.")
                return {"erro": "É necessário pelo menos 2 registros para gerar gráficos comparativos"}
            
            # Converter timestamp para datetime
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            
            # --- ESTATÍSTICAS BÁSICAS ---
            # Total de feedbacks analisados
            estatisticas_dashboard['total_feedbacks'] = len(df)
            
            # Últimos 7 dias vs 7 dias anteriores (para tendências)
            hoje = pd.Timestamp.now().date()
            df['data'] = df['timestamp'].dt.date
            
            # Últimos 7 dias
            ultimos_7dias = df[df['data'] >= (hoje - pd.Timedelta(days=7))]
            # 7 dias anteriores
            anteriores_7dias = df[(df['data'] < (hoje - pd.Timedelta(days=7))) & 
                                 (df['data'] >= (hoje - pd.Timedelta(days=14)))]
            
            # Calcular tendências
            qtd_ultimos_7dias = len(ultimos_7dias)
            qtd_anteriores_7dias = len(anteriores_7dias)
            
            if qtd_anteriores_7dias > 0:
                variacao_percentual = ((qtd_ultimos_7dias - qtd_anteriores_7dias) / qtd_anteriores_7dias) * 100
                estatisticas_dashboard['tendencia_feedbacks'] = {
                    'valor': round(variacao_percentual, 1),
                    'direcao': 'aumento' if variacao_percentual >= 0 else 'queda'
                }
            else:
                estatisticas_dashboard['tendencia_feedbacks'] = {
                    'valor': 100,
                    'direcao': 'aumento'
                }
            
            # Métricas de sentimento
            contagem_sentimentos = df['sentimento'].value_counts()
            # Garantir todas as categorias
            for sentimento in ['positivo', 'neutro', 'negativo']:
                if sentimento not in contagem_sentimentos:
                    contagem_sentimentos[sentimento] = 0
            
            # Percentuais de sentimento
            total = contagem_sentimentos.sum()
            estatisticas_dashboard['distribuicao_sentimentos'] = {
                'positivo': {
                    'count': int(contagem_sentimentos.get('positivo', 0)),
                    'percentual': round((contagem_sentimentos.get('positivo', 0) / total) * 100, 1)
                },
                'neutro': {
                    'count': int(contagem_sentimentos.get('neutro', 0)),
                    'percentual': round((contagem_sentimentos.get('neutro', 0) / total) * 100, 1)
                },
                'negativo': {
                    'count': int(contagem_sentimentos.get('negativo', 0)),
                    'percentual': round((contagem_sentimentos.get('negativo', 0) / total) * 100, 1)
                }
            }
            
            # --- GRÁFICOS BÁSICOS (JÁ EXISTENTES) ---
            # Gráfico de pizza de sentimentos
            cores = [CORES_SENTIMENTOS.get(idx, '#3498db') for idx in contagem_sentimentos.index]
            fig = go.Figure(data=[go.Pie(
                labels=contagem_sentimentos.index.tolist(), 
                values=contagem_sentimentos.values.tolist(),
                hole=.3,
                marker_colors=cores
            )])
            fig.update_layout(
                title_text='Distribuição de Sentimentos',
                height=400,
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5)
            )
            graficos['sentimento_pie'] = fig.to_dict()
            
            # Gráfico de evolução temporal (linhas)
            df_agrupado = df.groupby([pd.Grouper(key='timestamp', freq='D'), 'sentimento']).size().reset_index(name='contagem')
            fig = go.Figure()
            if len(df_agrupado) == 0:
                fig.add_annotation(
                    x=0.5, y=0.5,
                    xref="paper", yref="paper",
                    text="Não há dados suficientes para agrupar por dia",
                    showarrow=False,
                    font=dict(size=14)
                )
            else:
                df_pivot = df_agrupado.pivot(index='timestamp', columns='sentimento', values='contagem').fillna(0)
                for sentimento in ['positivo', 'neutro', 'negativo']:
                    if sentimento not in df_pivot.columns:
                        df_pivot[sentimento] = 0
                datas_str = [str(d) for d in df_pivot.index]
                for coluna in df_pivot.columns:
                    fig.add_trace(go.Scatter(
                        x=datas_str, 
                        y=df_pivot[coluna].tolist(),
                        mode='lines+markers',
                        name=coluna,
                        line=dict(color=CORES_SENTIMENTOS.get(coluna, '#3498db'))
                    ))
            fig.update_layout(
                title_text='Evolução de Sentimentos ao Longo do Tempo',
                height=400,
                xaxis=dict(title='Data'),
                yaxis=dict(title='Contagem'),
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5)
            )
            graficos['sentimento_tempo'] = fig.to_dict()
            
            # Gráfico de barras de palavras mais frequentes
            fig = go.Figure()
            todas_palavras = []
            for topicos in df['topicos'].dropna():
                if isinstance(topicos, str):
                    todas_palavras.extend([t.strip() for t in topicos.split(',') if t.strip()])
            if len(todas_palavras) == 0:
                fig.add_annotation(
                    x=0.5, y=0.5,
                    xref="paper", yref="paper",
                    text="Nenhuma palavra-chave detectada ainda",
                    showarrow=False,
                    font=dict(size=14)
                )
            else:
                palavra_contagem = pd.Series(todas_palavras).value_counts().head(10)
                fig.add_trace(go.Bar(
                    x=palavra_contagem.index.tolist(),
                    y=palavra_contagem.values.tolist(),
                    marker_color='#3498db'
                ))
            fig.update_layout(
                title_text='Top 10 Palavras Mencionadas',
                height=400,
                xaxis=dict(title='Palavras'),
                yaxis=dict(title='Frequência')
            )
            graficos['palavras_top'] = fig.to_dict()
            
            # --- GRÁFICOS NOVOS BASEADOS EM ASPECTOS ---
            # Carregar estatísticas de aspectos (se existirem)
            aspectos_path = 'data/estatisticas_aspectos.json'
            try:
                if os.path.exists(aspectos_path):
                    with open(aspectos_path, 'r', encoding='utf-8') as f:
                        estatisticas_aspectos = json.load(f)
                    
                    # Adicionar dados de aspectos às estatísticas do dashboard
                    estatisticas_dashboard['aspectos'] = {
                        'total_analises_com_aspectos': estatisticas_aspectos.get('analises_com_aspectos', 0),
                        'percentual_com_aspectos': round((estatisticas_aspectos.get('analises_com_aspectos', 0) / 
                                                      estatisticas_aspectos.get('total_analises', 1)) * 100, 1),
                        'aspecto_mais_mencionado': estatisticas_aspectos.get('aspecto_mais_mencionado', 'Nenhum')
                    }
                    
                    # Gráfico de radar para distribuição de aspectos
                    aspectos = list(estatisticas_aspectos.get("contagem_por_aspecto", {}).keys())
                    if aspectos:
                        # Gráfico de radar para visualização de aspectos
                        fig = go.Figure()
                        valores = list(estatisticas_aspectos.get("contagem_por_aspecto", {}).values())
                        fig.add_trace(go.Scatterpolar(
                            r=valores,
                            theta=aspectos,
                            fill='toself',
                            name='Menções por Aspecto',
                            line_color='#3498db'
                        ))
                        fig.update_layout(
                            polar=dict(
                                radialaxis=dict(
                                    visible=True,
                                    range=[0, max(valores) if valores else 1]
                                )),
                            showlegend=False,
                            title='Distribuição de Aspectos (Radar)',
                            height=450
                        )
                        graficos['aspectos_radar'] = fig.to_dict()
                        
                        # Gráfico de barras empilhadas para sentimentos por aspecto
                        aspectos_positivos = estatisticas_aspectos.get("aspectos_positivos", {})
                        aspectos_neutros = estatisticas_aspectos.get("aspectos_neutros", {})
                        aspectos_negativos = estatisticas_aspectos.get("aspectos_negativos", {})
                        
                        fig = go.Figure()
                        fig.add_trace(go.Bar(
                            x=aspectos,
                            y=[aspectos_positivos.get(aspecto, 0) for aspecto in aspectos],
                            name='Positivo',
                            marker_color=CORES_SENTIMENTOS.get('positivo', '#2ecc71')
                        ))
                        fig.add_trace(go.Bar(
                            x=aspectos,
                            y=[aspectos_neutros.get(aspecto, 0) for aspecto in aspectos],
                            name='Neutro',
                            marker_color=CORES_SENTIMENTOS.get('neutro', '#f39c12')
                        ))
                        fig.add_trace(go.Bar(
                            x=aspectos,
                            y=[aspectos_negativos.get(aspecto, 0) for aspecto in aspectos],
                            name='Negativo',
                            marker_color=CORES_SENTIMENTOS.get('negativo', '#e74c3c')
                        ))
                        
                        fig.update_layout(
                            barmode='stack',
                            title='Sentimento por Aspecto',
                            xaxis_title='Aspectos',
                            yaxis_title='Quantidade de Menções',
                            height=450,
                            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5)
                        )
                        graficos['sentimento_por_aspecto'] = fig.to_dict()
            except Exception as e:
                logger.error(f"Erro ao processar estatísticas de aspectos: {e}")
            
            # --- GRÁFICOS BASEADOS NA ANÁLISE PONDERADA ---
            # Carregar estatísticas de ponderação (se existirem)
            ponderacao_path = 'data/estatisticas_ponderacao.json'
            try:
                if os.path.exists(ponderacao_path):
                    with open(ponderacao_path, 'r', encoding='utf-8') as f:
                        estatisticas_ponderacao = json.load(f)
                    
                    # Adicionar dados de ponderação às estatísticas do dashboard
                    estatisticas_dashboard['ponderacao'] = {
                        'concordancia_cliente_modelo': round(estatisticas_ponderacao.get('concordancia_cliente_modelo', 0), 1),
                        'peso_cliente': round(estatisticas_ponderacao.get('peso_cliente_normalizado', 0), 1),
                        'peso_modelo': round(estatisticas_ponderacao.get('peso_modelo_normalizado', 0), 1),
                        'data_geracao': estatisticas_ponderacao.get('data_geracao', 'N/A')
                    }
                    
                    # Gráfico de barras comparando distribuição de sentimentos original vs ponderada
                    fig = go.Figure()
                    
                    # Obter dados de distribuição
                    dist_original = estatisticas_ponderacao.get('distribuicao_sentimento_original', {})
                    dist_ponderada = estatisticas_ponderacao.get('distribuicao_sentimento_ponderado', {})
                    
                    # Garantir que todas as categorias existam
                    categorias = ['positivo', 'neutro', 'negativo']
                    for cat in categorias:
                        if cat not in dist_original:
                            dist_original[cat] = 0
                        if cat not in dist_ponderada:
                            dist_ponderada[cat] = 0
                    
                    # Criar gráfico de barras agrupadas
                    fig.add_trace(go.Bar(
                        x=categorias,
                        y=[dist_original.get(cat, 0) for cat in categorias],
                        name='Análise Original',
                        marker_color='#3498db'
                    ))
                    fig.add_trace(go.Bar(
                        x=categorias,
                        y=[dist_ponderada.get(cat, 0) for cat in categorias],
                        name='Análise Ponderada',
                        marker_color='#9b59b6'
                    ))
                    
                    fig.update_layout(
                        barmode='group',
                        title='Comparação: Sentimento Original vs Ponderado',
                        xaxis_title='Sentimento',
                        yaxis_title='Quantidade',
                        height=450,
                        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5)
                    )
                    graficos['comparacao_ponderacao'] = fig.to_dict()
                    
                    # Gráfico de pizza para taxa de concordância
                    concordancia = estatisticas_ponderacao.get('concordancia_cliente_modelo', 0)
                    fig = go.Figure(data=[go.Pie(
                        labels=['Concordância', 'Discordância'],
                        values=[concordancia, 100 - concordancia],
                        hole=.4,
                        marker_colors=['#2ecc71', '#e74c3c']
                    )])
                    fig.update_layout(
                        title_text='Taxa de Concordância Cliente-Modelo',
                        height=400,
                        annotations=[dict(text=f'{concordancia:.1f}%', x=0.5, y=0.5, font_size=20, showarrow=False)]
                    )
                    graficos['concordancia_pie'] = fig.to_dict()
            except Exception as e:
                logger.error(f"Erro ao processar estatísticas de ponderação: {e}")
            
            # Adicionar estatísticas ao objeto de retorno
            graficos['estatisticas_dashboard'] = estatisticas_dashboard
            
            logger.info(f"Gráficos gerados com sucesso. Estrutura: {list(graficos.keys())}")
            return graficos
        except Exception as e:
            logger.error(f"Erro ao gerar gráficos: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return {"erro": f"Erro ao gerar gráficos: {str(e)}"}

    def gerar_relatorio_analise_ponderada(self, peso_cliente=30, peso_modelo=80):
        """
        Gera relatório de análise ponderada entre feedback do cliente e modelo
        
        Args:
            peso_cliente (int): Peso para atribuir ao feedback do cliente (0-100)
            peso_modelo (int): Peso para atribuir à análise do modelo (0-100)
            
        Returns:
            DataFrame: Dataframe com análise ponderada
        """
        try:
            csv_path = 'data/analises.csv'
            if not os.path.exists(csv_path):
                logger.warning("Arquivo de análises não encontrado.")
                return pd.DataFrame()
                
            # Carregar dados
            df = pd.read_csv(csv_path)
            if len(df) == 0:
                logger.warning("Arquivo de análises está vazio.")
                return pd.DataFrame()
                
            # Configurar pesos
            pesos = {'cliente': peso_cliente / 100, 'modelo': peso_modelo / 100}
            pesos_normalizados = AnaliseEstatistica.normalizar_pesos(pesos)
            
            # Criar DataFrame de resultado
            df_resultado = pd.DataFrame()
            df_resultado['timestamp'] = df['timestamp']
            df_resultado['texto'] = df['texto']
            df_resultado['sentimento_original'] = df['sentimento']
            
            # Converter sentimentos para valores numéricos (1-3)
            df_resultado['valor_modelo'] = df['sentimento'].apply(
                lambda s: AnaliseEstatistica.mapear_sentimento_para_valor(s)
            )
            
            # Simular valor do cliente (poderia ser de uma fonte real)
            # Neste exemplo, estamos simulando uma avaliação do cliente baseada no sentimento do modelo,
            # mas com alguma variação para demonstrar a funcionalidade
            np.random.seed(42)  # Para reprodutibilidade
            
            def simular_valor_cliente(sentimento):
                valor_base = AnaliseEstatistica.mapear_sentimento_para_valor(sentimento)
                # Adicionar alguma variação (-1, 0, ou +1) com maior probabilidade de concordância
                variacao = np.random.choice([-1, 0, 1], p=[0.2, 0.6, 0.2])
                valor = max(1, min(3, valor_base + variacao))
                return valor
                
            df_resultado['valor_cliente'] = df['sentimento'].apply(simular_valor_cliente)
            
            # Calcular média ponderada para cada registro
            resultados = []
            for _, row in df_resultado.iterrows():
                valores = {'cliente': row['valor_cliente'], 'modelo': row['valor_modelo']}
                media, pesos_usados = AnaliseEstatistica.calcular_media_ponderada(valores, pesos)
                resultados.append({
                    'media_ponderada': media,
                    'peso_cliente_original': peso_cliente,
                    'peso_modelo_original': peso_modelo,
                    'peso_cliente_normalizado': pesos_usados['cliente'] * 100,
                    'peso_modelo_normalizado': pesos_usados['modelo'] * 100
                })
                
            # Adicionar resultados ao DataFrame
            resultados_df = pd.DataFrame(resultados)
            df_resultado = pd.concat([df_resultado, resultados_df], axis=1)
            
            # Mapear o resultado numérico de volta para categorias de sentimento
            def mapear_valor_para_sentimento(valor):
                if valor >= 2.5:
                    return 'positivo'
                elif valor >= 1.5:
                    return 'neutro'
                else:
                    return 'negativo'
                    
            df_resultado['sentimento_ponderado'] = df_resultado['media_ponderada'].apply(mapear_valor_para_sentimento)
            
            # Adicionar informações sobre concordância
            df_resultado['concordancia_cliente_modelo'] = df_resultado.apply(
                lambda row: row['valor_cliente'] == row['valor_modelo'], axis=1
            )
            
            # Salvar o relatório em CSV
            relatorio_path = 'data/relatorio_ponderado.csv'
            df_resultado.to_csv(relatorio_path, index=False)
            logger.info(f"Relatório de análise ponderada gerado e salvo em {relatorio_path}")
            
            # Gerar estatísticas adicionais
            estatisticas = {
                'total_registros': len(df_resultado),
                'concordancia_cliente_modelo': df_resultado['concordancia_cliente_modelo'].mean() * 100,
                'distribuicao_sentimento_original': df_resultado['sentimento_original'].value_counts().to_dict(),
                'distribuicao_sentimento_ponderado': df_resultado['sentimento_ponderado'].value_counts().to_dict(),
                'peso_cliente_normalizado': pesos_normalizados['cliente'] * 100,
                'peso_modelo_normalizado': pesos_normalizados['modelo'] * 100,
                'soma_pesos_original': peso_cliente + peso_modelo,
                'data_geracao': datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            }
            
            # Salvar estatísticas em JSON
            import json
            with open('data/estatisticas_ponderacao.json', 'w', encoding='utf-8') as f:
                json.dump(estatisticas, f, ensure_ascii=False, indent=4)
                
            return df_resultado
        except Exception as e:
            logger.error(f"Erro ao gerar relatório de análise ponderada: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return pd.DataFrame()

    def gerar_estatisticas_aspectos(self):
        """Gera estatísticas detalhadas sobre os aspectos mencionados nos feedbacks"""
        csv_path = 'data/analises.csv'
        if not os.path.exists(csv_path):
            logger.warning("Arquivo de análises não encontrado.")
            return {}
        
        df = pd.read_csv(csv_path)
        if len(df) == 0:
            logger.warning("Arquivo de análises está vazio.")
            return {}
        
        # Processamento básico
        tem_aspectos = df['aspectos_detectados'].notna() & (df['aspectos_detectados'] != '')
        
        # Estrutura de estatísticas
        estatisticas = {
            "total_analises": len(df),
            "analises_com_aspectos": tem_aspectos.sum(),
            "contagem_por_aspecto": {},
            "aspectos_positivos": {},
            "aspectos_neutros": {},
            "aspectos_negativos": {},
            "relevancia_media": {},
            "aspecto_mais_mencionado": "",
            "problemas_frequentes": {}
        }
        
        # Apenas exemplos básicos de processamento
        # Uma implementação completa exigiria processamento mais sofisticado
        
        # Contar menções por aspecto
        for index, row in df[tem_aspectos].iterrows():
            aspectos = row['aspectos_detectados'].split(',')
            for aspecto in aspectos:
                if aspecto:
                    # Contagem geral
                    estatisticas["contagem_por_aspecto"][aspecto] = estatisticas["contagem_por_aspecto"].get(aspecto, 0) + 1
                    
                    # Contagem por sentimento
                    sentimento = row['sentimento']
                    if sentimento == 'positivo':
                        estatisticas["aspectos_positivos"][aspecto] = estatisticas["aspectos_positivos"].get(aspecto, 0) + 1
                    elif sentimento == 'neutro':
                        estatisticas["aspectos_neutros"][aspecto] = estatisticas["aspectos_neutros"].get(aspecto, 0) + 1
                    elif sentimento == 'negativo':
                        estatisticas["aspectos_negativos"][aspecto] = estatisticas["aspectos_negativos"].get(aspecto, 0) + 1
        
        # Identificar aspecto mais mencionado
        if estatisticas["contagem_por_aspecto"]:
            estatisticas["aspecto_mais_mencionado"] = max(
                estatisticas["contagem_por_aspecto"].items(), 
                key=lambda x: x[1]
            )[0]
        
        # Salvar estatísticas
        with open('data/estatisticas_aspectos.json', 'w', encoding='utf-8') as f:
            json.dump(estatisticas, f, ensure_ascii=False, indent=4)
        
        return estatisticas

def criar_diretorios_essenciais():
    """Cria os diretórios essenciais para o funcionamento da aplicação"""
    diretorios = [
        'transcricoes',
        'static/images',
        'data',
        'models',
        'models/cardiffnlp-xlm-roberta'
    ]
    
    for diretorio in diretorios:
        try:
            os.makedirs(diretorio, exist_ok=True)
            logger.info(f"Diretório verificado/criado: {diretorio}")
        except Exception as e:
            logger.error(f"Erro ao criar diretório {diretorio}: {e}")

# Inicialização de diretórios
criar_diretorios_essenciais()

# Inicialização dos componentes
sentiment_analyzer = SentimentAnalyzer()
speech_handler = SpeechHandler(sentiment_analyzer)
data_handler = DataHandler()

# Rotas da aplicação
@app.route('/')
def index():
    return render_template('gravador_voz.html', modelo_xlm_roberta=sentiment_analyzer.using_xlm_roberta)

@app.route('/diagnostico')
def diagnostico():
    """Página de diagnóstico para verificar o status do sistema"""
    try:
        # Verificar disponibilidade do modelo XLM-RoBERTa
        modelo_status = {
            'xlm_roberta_disponivel': sentiment_analyzer.using_xlm_roberta,
            'modelo_carregado': sentiment_analyzer.sentiment_model is not None,
            'python_version': sys.version,
            'nltk_inicializado': True,  # Assumimos que o NLTK já foi inicializado
            'transformers_version': getattr(AutoTokenizer, '__version__', 'Desconhecido'),
        }
        
        # Testar o modelo com uma frase em português
        texto_teste = "Este é um teste para verificar o funcionamento do modelo de análise de sentimentos."
        
        if sentiment_analyzer.using_xlm_roberta and sentiment_analyzer.sentiment_model is not None:
            try:
                resultado = sentiment_analyzer.sentiment_model(texto_teste[:100])[0]
                modelo_status['teste_resultado'] = {
                    'label': resultado['label'],
                    'score': float(resultado['score']),
                    'tempo': datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                }
                modelo_status['teste_sucesso'] = True
            except Exception as e:
                modelo_status['teste_erro'] = str(e)
                modelo_status['teste_sucesso'] = False
        else:
            modelo_status['teste_sucesso'] = False
            modelo_status['teste_erro'] = "Modelo XLM-RoBERTa não disponível"
        
        # Obter informações do sistema
        import platform
        sistema_info = {
            'sistema': platform.system(),
            'release': platform.release(),
            'versao': platform.version(),
            'arquitetura': platform.architecture(),
            'processador': platform.processor(),
            'memoria': "N/A",  # Precisaria de psutil para isso
            'gpu': "N/A"  # Precisaria de bibliotecas específicas para detectar GPU
        }
        
        # Verificar requisitos
        requisitos_status = {
            'pytorch': 'torch' in sys.modules,
            'transformers': 'transformers' in sys.modules,
            'flask': 'flask' in sys.modules,
            'pandas': 'pandas' in sys.modules,
            'nltk': 'nltk' in sys.modules,
            'huggingface_hub': 'huggingface_hub' in sys.modules
        }
        
        # Verificar diretórios necessários
        diretorios = {
            'transcricoes': os.path.exists('transcricoes'),
            'static/images': os.path.exists('static/images'),
            'data': os.path.exists('data'),
            'models': os.path.exists('models'),
            'models/cardiffnlp-xlm-roberta': os.path.exists('models/cardiffnlp-xlm-roberta')
        }
        
        return render_template(
            'diagnostico.html', 
            modelo=modelo_status, 
            sistema=sistema_info,
            requisitos=requisitos_status,
            diretorios=diretorios
        )
    except Exception as e:
        return render_template('erro.html', erro=f"Erro ao gerar diagnóstico: {str(e)}"), 500

@app.route('/api/testar-modelo', methods=['POST'])
def testar_modelo_api():
    """Endpoint para testar o modelo de análise de sentimentos"""
    try:
        data = request.get_json()
        texto = data.get('texto', '')
        
        if not texto:
            return jsonify({
                'sucesso': False,
                'erro': 'Texto não fornecido'
            }), 400
            
        # Limitar o tamanho do texto para evitar problemas
        texto = texto[:500]
        
        # Realizar análise de sentimento
        analise = sentiment_analyzer.analisar_sentimento(texto)
        
        # Adicionar informações sobre o modelo usado
        resultado = {
            'sucesso': True,
            'analise': analise,
            'modelo': {
                'usando_xlm_roberta': sentiment_analyzer.using_xlm_roberta,
                'timestamp': datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            }
        }
        
        return jsonify(resultado)
    except Exception as e:
        logger.error(f"Erro ao testar modelo: {e}")
        return jsonify({
            'sucesso': False,
            'erro': str(e)
        }), 500

@app.route('/ouvir', methods=['POST'])
def ouvir():
    try:
        # Verificar se é uma solicitação de parada manual
        if request.is_json and request.get_json().get('manual_stop', False):
            logger.info("Recebida solicitação de parada manual da gravação")
            return jsonify({
                'texto': "Gravação finalizada manualmente. A transcrição não está disponível.",
                'sentimento': 'neutro',
                'compound': 0,
                'success': True,
                'motivo_parada': 'parada_manual',
                'timestamp': datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            })
            
        texto, analise = speech_handler.ouvir_microfone()
        # Não salve mensagens de erro ou instruções
        mensagens_para_ignorar = [
            "Para usar a gravação de voz",
            "Não entendi o que você disse",
            "Erro ao inicializar microfone",
            "Erro ao solicitar resultados",
            "Erro inesperado",
            "Não detectei nenhuma fala"
        ]
        
        # Verifique se o texto não começa com nenhuma das mensagens para ignorar
        if not any(texto.startswith(msg) for msg in mensagens_para_ignorar):
            data_handler.salvar_transcricao(texto, analise)
            
        return jsonify({
            'texto': texto, 
            'sentimento': analise['sentimento'],
            'compound': analise['compound'],
            'success': True,
            'motivo_parada': 'silencio_detectado' if texto != "Não detectei nenhuma fala. Por favor, tente novamente falando mais alto." else 'sem_fala',
            'timestamp': datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        })
    except Exception as e:
        logger.error(f"Erro na rota /ouvir: {e}")
        return jsonify({
            'texto': f'Erro ao processar áudio: {str(e)}', 
            'sentimento': 'neutro',
            'success': False,
            'motivo_parada': 'erro',
            'error_details': str(e),
            'timestamp': datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        })

@app.route('/historico')
def historico():
    try:
        historico = data_handler.carregar_historico()
        logger.info(f"Histórico carregado: {len(historico)} entradas")
        analises = data_handler.carregar_analises()
        logger.info(f"Análises carregadas: {len(analises)} registros")
        
        # Adicionar logs para debug dos aspectos
        for i, analise in enumerate(analises[:3]):  # Verificar apenas os primeiros 3 para não sobrecarregar o log
            aspectos_detectados = analise.get('aspectos_detectados', '')
            aspecto_principal = analise.get('aspecto_principal', '')
            logger.info(f"Análise {i+1}: aspectos_detectados={aspectos_detectados}, aspecto_principal={aspecto_principal}")
            
        for analise in analises:
            if 'topicos' in analise:
                if not isinstance(analise['topicos'], str):
                    analise['topicos'] = ''
        return render_template('historico.html', historico=historico, analises=analises)
    except Exception as e:
        logger.error(f"Erro na rota /historico: {e}")
        return render_template('historico.html', historico=["Erro ao carregar histórico."], analises=[])

@app.route('/adicionar', methods=['GET', 'POST'])
def adicionar_feedback():
    if request.method == 'POST':
        texto = request.form.get('texto', '')
        if texto:
            logger.info(f"Feedback manual recebido: {texto}")
            analise = sentiment_analyzer.analisar_sentimento(texto)
            data_handler.salvar_transcricao(texto, analise)
            return redirect(url_for('historico'))
        else:
            logger.warning("Tentativa de adicionar feedback vazio")
    return render_template('adicionar_feedback.html')

@app.route('/dashboard')
def dashboard():
    try:
        # Gerar gráficos e estatísticas para o dashboard
        dados_dashboard = data_handler.gerar_graficos()
        
        # Verificar se ocorreu algum erro na geração dos gráficos
        if 'erro' in dados_dashboard:
            return render_template('dashboard.html', 
                                  graficos={}, 
                                  erro=dados_dashboard['erro'],
                                  estatisticas={})
        
        # Separar gráficos e estatísticas
        estatisticas = dados_dashboard.pop('estatisticas_dashboard', {})
        
        # Verificar arquivos adicionais para insights
        insights = []
        
        # Verificar se temos dados de relatório ponderado
        if os.path.exists('data/relatorio_ponderado.csv'):
            try:
                df_relatorio = pd.read_csv('data/relatorio_ponderado.csv')
                # Encontrar registros onde o cliente e modelo discordam
                discordancias = df_relatorio[~df_relatorio['concordancia_cliente_modelo']]
                if len(discordancias) > 0:
                    insights.append({
                        'tipo': 'analise',
                        'titulo': 'Discordâncias na Análise de Sentimentos',
                        'texto': f'Há {len(discordancias)} casos onde a análise do cliente diverge da análise do modelo. Considere revisar estes feedbacks para melhorar a precisão.',
                        'classe': 'warning'
                    })
            except Exception as e:
                logger.error(f"Erro ao processar relatório ponderado para insights: {e}")
        
        # Verificar dados de aspectos para insights
        if os.path.exists('data/estatisticas_aspectos.json'):
            try:
                with open('data/estatisticas_aspectos.json', 'r', encoding='utf-8') as f:
                    dados_aspectos = json.load(f)
                
                # Identificar aspecto com mais reclamações
                if dados_aspectos.get('aspectos_negativos'):
                    aspecto_problematico = max(dados_aspectos['aspectos_negativos'].items(), 
                                             key=lambda x: x[1], 
                                             default=(None, 0))
                    if aspecto_problematico[0]:
                        insights.append({
                            'tipo': 'problema',
                            'titulo': f'Problemas com {aspecto_problematico[0].title()}',
                            'texto': f'O aspecto "{aspecto_problematico[0]}" recebeu {aspecto_problematico[1]} menções negativas. Recomendamos analisar este ponto com prioridade.',
                            'classe': 'danger'
                        })
                
                # Destacar aspecto mais elogiado
                if dados_aspectos.get('aspectos_positivos'):
                    aspecto_positivo = max(dados_aspectos['aspectos_positivos'].items(), 
                                        key=lambda x: x[1], 
                                        default=(None, 0))
                    if aspecto_positivo[0]:
                        insights.append({
                            'tipo': 'destaque',
                            'titulo': f'Ponto Forte: {aspecto_positivo[0].title()}',
                            'texto': f'O aspecto "{aspecto_positivo[0]}" recebeu {aspecto_positivo[1]} menções positivas. Este é um ponto forte a ser mantido.',
                            'classe': 'success'
                        })
            except Exception as e:
                logger.error(f"Erro ao processar estatísticas de aspectos para insights: {e}")
        
        # Adicionar recomendações gerais baseadas nas estatísticas
        if estatisticas:
            dist_sent = estatisticas.get('distribuicao_sentimentos', {})
            if dist_sent and dist_sent.get('negativo', {}).get('percentual', 0) > 30:
                insights.append({
                    'tipo': 'alerta',
                    'titulo': 'Alta Taxa de Sentimentos Negativos',
                    'texto': f'A taxa de feedbacks negativos está em {dist_sent["negativo"]["percentual"]}%. Recomendamos uma análise aprofundada das causas.',
                    'classe': 'danger'
                })
            
            # Verificar tendência dos feedbacks
            tendencia = estatisticas.get('tendencia_feedbacks', {})
            if tendencia and tendencia.get('direcao') == 'aumento' and tendencia.get('valor', 0) > 20:
                insights.append({
                    'tipo': 'informacao',
                    'titulo': 'Aumento Significativo de Feedbacks',
                    'texto': f'Houve um aumento de {tendencia["valor"]}% nos feedbacks nos últimos 7 dias. Verifique se houve alguma mudança recente que possa explicar este comportamento.',
                    'classe': 'info'
                })
        
        # Obter informações do modelo
        info_modelo = {
            'usando_xlm_roberta': sentiment_analyzer.using_xlm_roberta,
            'data_atualizacao': datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        
        # Renderizar o dashboard com todos os dados
        return render_template('dashboard.html', 
                             graficos=dados_dashboard, 
                             estatisticas=estatisticas,
                             insights=insights,
                             modelo_info=info_modelo)
    except Exception as e:
        logger.error(f"Erro na rota /dashboard: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return render_template('dashboard.html', 
                              graficos={}, 
                              erro=f"Erro ao gerar dashboard: {str(e)}",
                              estatisticas={})

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
        
        # Salvar no histórico e CSV usando a função existente
        data_handler.salvar_transcricao(texto, analise)
        
        return jsonify({
            "success": True,
            "message": "Feedback salvo com sucesso",
            "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        })
    except Exception as e:
        logger.error(f"Erro ao salvar feedback no histórico: {e}")
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
        
        sucessos = 0
        falhas = 0
        
        for item in feedbacks:
            try:
                feedback = item.get('feedback')
                analise = item.get('analise')
                
                if feedback and analise:
                    data_handler.salvar_transcricao(feedback, analise)
                    sucessos += 1
                else:
                    falhas += 1
            except Exception as e:
                logger.error(f"Erro ao sincronizar feedback: {e}")
                falhas += 1
        
        return jsonify({
            "success": True,
            "message": f"Sincronização concluída: {sucessos} feedbacks sincronizados, {falhas} falhas",
            "sucessos": sucessos,
            "falhas": falhas
        })
    except Exception as e:
        logger.error(f"Erro na sincronização de feedbacks: {e}")
        return jsonify({"success": False, "message": str(e)}), 500

@app.route('/baixar-modelo', methods=['GET'])
def baixar_modelo():
    """Endpoint para forçar o download do modelo XLM-RoBERTa"""
    try:
        logger.info("Iniciando download manual do modelo CardiffNLP XLM-RoBERTa...")
        model_name = "cardiffnlp/twitter-xlm-roberta-base-sentiment"
        model_dir = os.path.join(os.getcwd(), "models", "cardiffnlp-xlm-roberta")
        
        # Limpar diretório se já existir
        import shutil
        if os.path.exists(model_dir):
            for arquivo in os.listdir(model_dir):
                caminho = os.path.join(model_dir, arquivo)
                try:
                    if os.path.isfile(caminho):
                        os.unlink(caminho)
                    elif os.path.isdir(caminho):
                        shutil.rmtree(caminho)
                except Exception as e:
                    logger.error(f"Erro ao limpar arquivo {caminho}: {e}")
        
        # Criar diretório
        os.makedirs(model_dir, exist_ok=True)
        
        # Testar permissões
        try:
            with open(os.path.join(model_dir, 'test.txt'), 'w') as f:
                f.write('teste')
            logger.info("Permissões de escrita OK")
        except Exception as e:
            logger.error(f"Problema de permissões: {e}")
            return render_template('erro.html', erro=f"Erro de permissão ao escrever no diretório de modelos: {e}")
        
        try:
            from huggingface_hub import snapshot_download
            logger.info(f"Baixando modelo {model_name} para {model_dir}...")
            
            # Usar download com retry
            snapshot_path = snapshot_download(
                repo_id=model_name,
                local_dir=model_dir,
                local_dir_use_symlinks=False,
                retry_count=3,
                tqdm_class=None
            )
            
            logger.info(f"Download concluído em: {snapshot_path}")
            
            # Verificar se os arquivos essenciais foram baixados
            arquivos_essenciais = ['config.json', 'pytorch_model.bin', 'tokenizer.json']
            arquivos_faltantes = [f for f in arquivos_essenciais if not os.path.exists(os.path.join(model_dir, f))]
            
            if arquivos_faltantes:
                logger.error(f"Arquivos essenciais faltando após download: {arquivos_faltantes}")
                return render_template('erro.html', erro=f"Download incompleto. Arquivos faltantes: {arquivos_faltantes}")
            
            # Testar carregamento do modelo
            try:
                from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
                tokenizer = AutoTokenizer.from_pretrained(model_dir)
                model = AutoModelForSequenceClassification.from_pretrained(model_dir)
                
                # Criar e testar pipeline
                nlp = pipeline("sentiment-analysis", model=model, tokenizer=tokenizer)
                resultado = nlp("Este é um teste do modelo de sentimentos.")[0]
                
                logger.info(f"Modelo testado com sucesso. Resultado: {resultado}")
                
                return render_template('diagnostico.html', 
                    mensagem=f"Modelo baixado e testado com sucesso! Resultado do teste: {resultado['label']} ({resultado['score']:.4f})")
                
            except Exception as e:
                logger.error(f"Erro ao carregar modelo após download: {e}")
                return render_template('erro.html', erro=f"Falha ao carregar modelo após download: {e}")
                
        except Exception as e:
            logger.error(f"Erro ao baixar modelo: {e}")
            return render_template('erro.html', erro=f"Falha ao baixar modelo: {e}")
            
    except Exception as e:
        logger.error(f"Erro geral no processo de download: {e}")
        return render_template('erro.html', erro=f"Erro no processo de download: {e}")

@app.route('/reiniciar-modelo', methods=['GET'])
def reiniciar_modelo():
    """Endpoint para reiniciar o analisador de sentimentos"""
    global sentiment_analyzer
    try:
        logger.info("Reinicializando analisador de sentimentos...")
        # Recriar objeto SentimentAnalyzer
        sentiment_analyzer = SentimentAnalyzer()
        
        if sentiment_analyzer.using_xlm_roberta and sentiment_analyzer.sentiment_model is not None:
            # Testar modelo com frase simples
            texto_teste = "Este é um teste de reinicialização do modelo."
            resultado = sentiment_analyzer.sentiment_model(texto_teste)[0]
            
            return render_template('diagnostico.html', 
                mensagem=f"Modelo reiniciado com sucesso! Resultado do teste: {resultado['label']} ({resultado['score']:.4f})")
        else:
            return render_template('erro.html', 
                erro="Modelo XLM-RoBERTa não pôde ser inicializado. Usando análise básica de sentimentos como fallback.")
    except Exception as e:
        logger.error(f"Erro ao reiniciar analisador de sentimentos: {e}")
        return render_template('erro.html', erro=f"Erro ao reiniciar modelo: {e}")

# Tratamento de erros
@app.errorhandler(404)
def pagina_nao_encontrada(e):
    logger.warning(f"Página não encontrada: {request.path}")
    return render_template('erro.html', erro="Página não encontrada"), 404

@app.errorhandler(500)
def erro_servidor(e):
    logger.error(f"Erro do servidor: {str(e)}")
    return render_template('erro.html', erro="Erro interno do servidor"), 500

@app.route('/favicon.ico')
def favicon():
    return app.send_static_file('favicon.ico')

@app.route('/criar_template_diagnostico')
def criar_template_diagnostico():
    """Cria o template HTML para a página de diagnóstico se não existir"""
    try:
        template_dir = os.path.join(os.getcwd(), 'templates')
        os.makedirs(template_dir, exist_ok=True)
        
        template_path = os.path.join(template_dir, 'diagnostico.html')
        
        # Só cria se não existir
        if not os.path.exists(template_path):
            html_content = """
<!DOCTYPE html>
<html lang="pt-br">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Diagnóstico do Sistema</title>
    <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/bootstrap@4.6.0/dist/css/bootstrap.min.css">
    <style>
        .card {
            margin-bottom: 20px;
        }
        .status-success {
            color: #28a745;
        }
        .status-warning {
            color: #ffc107;
        }
        .status-danger {
            color: #dc3545;
        }
    </style>
</head>
<body>
    <div class="container mt-4">
        <h1 class="mb-4">Diagnóstico do Sistema</h1>
        
        <div class="card">
            <div class="card-header bg-primary text-white">
                Status do Modelo XLM-RoBERTa
            </div>
            <div class="card-body">
                <h5 class="card-title">Informações do Modelo</h5>
                <ul class="list-group list-group-flush">
                    <li class="list-group-item">
                        XLM-RoBERTa Disponível: 
                        <span class="{% if modelo.xlm_roberta_disponivel %}status-success{% else %}status-danger{% endif %}">
                            {{ "Sim" if modelo.xlm_roberta_disponivel else "Não" }}
                        </span>
                    </li>
                    <li class="list-group-item">
                        Modelo Carregado: 
                        <span class="{% if modelo.modelo_carregado %}status-success{% else %}status-danger{% endif %}">
                            {{ "Sim" if modelo.modelo_carregado else "Não" }}
                        </span>
                    </li>
                    <li class="list-group-item">
                        Versão Python: {{ modelo.python_version }}
                    </li>
                    <li class="list-group-item">
                        Versão Transformers: {{ modelo.transformers_version }}
                    </li>
                    <li class="list-group-item">
                        Teste do Modelo: 
                        <span class="{% if modelo.teste_sucesso %}status-success{% else %}status-danger{% endif %}">
                            {{ "Sucesso" if modelo.teste_sucesso else "Falha" }}
                        </span>
                        {% if modelo.teste_sucesso %}
                            <div class="mt-2">
                                <strong>Label:</strong> {{ modelo.teste_resultado.label }}<br>
                                <strong>Score:</strong> {{ "%.4f"|format(modelo.teste_resultado.score) }}<br>
                                <strong>Tempo:</strong> {{ modelo.teste_resultado.tempo }}
                            </div>
                        {% else %}
                            <div class="mt-2 status-danger">
                                <strong>Erro:</strong> {{ modelo.teste_erro }}
                            </div>
                        {% endif %}
                    </li>
                </ul>
            </div>
        </div>
        
        <div class="card">
            <div class="card-header bg-info text-white">
                Informações do Sistema
            </div>
            <div class="card-body">
                <h5 class="card-title">Detalhes do Sistema</h5>
                <ul class="list-group list-group-flush">
                    <li class="list-group-item"><strong>Sistema:</strong> {{ sistema.sistema }}</li>
                    <li class="list-group-item"><strong>Release:</strong> {{ sistema.release }}</li>
                    <li class="list-group-item"><strong>Versão:</strong> {{ sistema.versao }}</li>
                    <li class="list-group-item"><strong>Arquitetura:</strong> {{ sistema.arquitetura[0] }} ({{ sistema.arquitetura[1] }})</li>
                    <li class="list-group-item"><strong>Processador:</strong> {{ sistema.processador }}</li>
                </ul>
            </div>
        </div>
        
        <div class="card">
            <div class="card-header bg-success text-white">
                Status dos Requisitos
            </div>
            <div class="card-body">
                <h5 class="card-title">Bibliotecas Instaladas</h5>
                <ul class="list-group list-group-flush">
                    {% for lib, status in requisitos.items() %}
                    <li class="list-group-item">
                        {{ lib }}: 
                        <span class="{% if status %}status-success{% else %}status-danger{% endif %}">
                            {{ "Instalado" if status else "Não Instalado" }}
                        </span>
                    </li>
                    {% endfor %}
                </ul>
            </div>
        </div>
        
        <div class="card">
            <div class="card-header bg-warning text-dark">
                Status dos Diretórios
            </div>
            <div class="card-body">
                <h5 class="card-title">Diretórios Necessários</h5>
                <ul class="list-group list-group-flush">
                    {% for dir, exists in diretorios.items() %}
                    <li class="list-group-item">
                        {{ dir }}: 
                        <span class="{% if exists %}status-success{% else %}status-danger{% endif %}">
                            {{ "Existe" if exists else "Não Existe" }}
                        </span>
                    </li>
                    {% endfor %}
                </ul>
            </div>
        </div>
        
        <div class="card">
            <div class="card-header bg-secondary text-white">
                Testes Adicionais
            </div>
            <div class="card-body">
                <h5 class="card-title">Testar Modelo Manualmente</h5>
                <div class="form-group">
                    <label for="testText">Texto para testar:</label>
                    <textarea class="form-control" id="testText" rows="3" placeholder="Digite um texto para analisar..."></textarea>
                </div>
                <button id="testButton" class="btn btn-primary">Analisar Sentimento</button>
                
                <div id="testResult" class="mt-3" style="display: none;">
                    <h6>Resultado da Análise:</h6>
                    <pre id="resultJson" class="bg-light p-3"></pre>
                </div>
            </div>
        </div>
        
        <div class="mt-4 mb-4">
            <a href="/" class="btn btn-secondary">Voltar para Home</a>
        </div>
    </div>
    
    <script src="https://code.jquery.com/jquery-3.5.1.min.js"></script>
    <script src="https://cdn.jsdelivr.net/npm/bootstrap@4.6.0/dist/js/bootstrap.bundle.min.js"></script>
    <script>
        $(document).ready(function() {
            $("#testButton").click(function() {
                const texto = $("#testText").val();
                if (!texto) {
                    alert("Por favor, digite um texto para analisar.");
                    return;
                }
                
                $("#testButton").prop("disabled", true).html('<span class="spinner-border spinner-border-sm" role="status" aria-hidden="true"></span> Analisando...');
                
                $.ajax({
                    url: "/api/testar-modelo",
                    type: "POST",
                    contentType: "application/json",
                    data: JSON.stringify({texto: texto}),
                    success: function(response) {
                        $("#resultJson").text(JSON.stringify(response, null, 2));
                        $("#testResult").show();
                    },
                    error: function(xhr, status, error) {
                        $("#resultJson").text("Erro: " + error);
                        $("#testResult").show();
                    },
                    complete: function() {
                        $("#testButton").prop("disabled", false).text("Analisar Sentimento");
                    }
                });
            });
        });
    </script>
</body>
</html>
            """
            
            with open(template_path, 'w', encoding='utf-8') as f:
                f.write(html_content)
                
            return jsonify({
                'sucesso': True,
                'mensagem': f'Template de diagnóstico criado em {template_path}'
            })
        else:
            return jsonify({
                'sucesso': True,
                'mensagem': f'Template de diagnóstico já existe em {template_path}'
            })
    except Exception as e:
        logger.error(f"Erro ao criar template de diagnóstico: {e}")
        return jsonify({
            'sucesso': False,
            'erro': str(e)
        }), 500

class AnaliseEstatistica:
    """Classe para análises estatísticas avançadas com normalização de pesos"""
    
    @staticmethod
    def normalizar_pesos(pesos):
        """Normaliza pesos para que somem 100%"""
        soma_pesos = sum(pesos.values())
        if soma_pesos == 0:
            return {k: 0 for k in pesos}
        return {k: (v / soma_pesos) for k, v in pesos.items()}
    
    @staticmethod
    def calcular_media_ponderada(valores, pesos, normalizar=True):
        """
        Calcula média ponderada com normalização opcional de pesos
        
        Args:
            valores (dict): Dicionário com valores para cada fonte
            pesos (dict): Dicionário com pesos para cada fonte
            normalizar (bool): Se True, normaliza os pesos para somarem 100%
            
        Returns:
            float: Média ponderada calculada
            dict: Pesos normalizados utilizados
        """
        # Verificar se temos as mesmas chaves em valores e pesos
        if set(valores.keys()) != set(pesos.keys()):
            raise ValueError("As chaves de valores e pesos devem ser idênticas")
            
        # Normalizar pesos se necessário
        pesos_usados = AnaliseEstatistica.normalizar_pesos(pesos) if normalizar else pesos
        
        # Calcular média ponderada
        soma_produtos = sum(valores[k] * pesos_usados[k] for k in valores)
        
        return soma_produtos, pesos_usados
    
    @staticmethod
    def mapear_sentimento_para_valor(sentimento):
        """Mapeia sentimento textual para valor numérico"""
        mapeamento = {
            'positivo': 3,
            'neutro': 2,
            'negativo': 1
        }
        return mapeamento.get(sentimento, 2)  # Default para neutro

# Adicionar estas rotas após as rotas existentes
@app.route('/relatorio-ponderado', methods=['GET'])
def gerar_relatorio_ponderado():
    """Endpoint para gerar e visualizar relatório com pesos normalizados"""
    try:
        # Obter parâmetros de peso da query string, ou usar defaults
        peso_cliente = request.args.get('peso_cliente', type=int, default=30)
        peso_modelo = request.args.get('peso_modelo', type=int, default=80)
        
        # Validar parâmetros
        if peso_cliente < 0 or peso_modelo < 0:
            return render_template('erro.html', erro="Os pesos não podem ser negativos"), 400
            
        # Gerar relatório
        df_relatorio = data_handler.gerar_relatorio_analise_ponderada(
            peso_cliente=peso_cliente, 
            peso_modelo=peso_modelo
        )
        
        if df_relatorio.empty:
            return render_template('erro.html', erro="Não foi possível gerar o relatório. Verifique os logs."), 500
            
        # Estatísticas para exibição na página
        estatisticas = {
            'total_registros': len(df_relatorio),
            'peso_cliente_original': peso_cliente,
            'peso_modelo_original': peso_modelo,
            'soma_pesos_original': peso_cliente + peso_modelo,
            'peso_cliente_normalizado': round(df_relatorio['peso_cliente_normalizado'].iloc[0], 2),
            'peso_modelo_normalizado': round(df_relatorio['peso_modelo_normalizado'].iloc[0], 2),
            'distribuicao_original': df_relatorio['sentimento_original'].value_counts().to_dict(),
            'distribuicao_ponderada': df_relatorio['sentimento_ponderado'].value_counts().to_dict(),
            'concordancia': round(df_relatorio['concordancia_cliente_modelo'].mean() * 100, 2)
        }
        
        # Carregar os últimos 10 registros do relatório para exibição
        registros = df_relatorio.tail(10).to_dict('records')
        
        # Preparar dados para o gráfico
        # Criar duas distribuições para comparação antes/depois da normalização
        original_counts = df_relatorio['sentimento_original'].value_counts()
        ponderado_counts = df_relatorio['sentimento_ponderado'].value_counts()
        
        # Garantir que todas as categorias estão presentes
        categorias = ['positivo', 'neutro', 'negativo']
        for cat in categorias:
            if cat not in original_counts:
                original_counts[cat] = 0
            if cat not in ponderado_counts:
                ponderado_counts[cat] = 0
                
        # Criar gráfico de distribuição
        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=categorias,
            y=[original_counts.get(cat, 0) for cat in categorias],
            name='Sentimento Original',
            marker_color='#3498db'
        ))
        fig.add_trace(go.Bar(
            x=categorias,
            y=[ponderado_counts.get(cat, 0) for cat in categorias],
            name='Sentimento Ponderado Normalizado',
            marker_color='#2ecc71'
        ))
        fig.update_layout(
            title_text='Comparação da Distribuição de Sentimentos',
            height=400,
            xaxis=dict(title='Sentimento'),
            yaxis=dict(title='Contagem'),
            barmode='group'
        )
        
        return render_template(
            'relatorio_ponderado.html',
            estatisticas=estatisticas,
            registros=registros,
            grafico_comparacao=json.dumps(fig.to_dict())
        )
        
    except Exception as e:
        logger.error(f"Erro ao gerar visualização do relatório ponderado: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return render_template('erro.html', erro=f"Erro ao processar relatório: {e}"), 500

@app.route('/api/relatorio-ponderado', methods=['GET'])
def api_relatorio_ponderado():
    """API para obter relatório em formato JSON"""
    try:
        peso_cliente = request.args.get('peso_cliente', type=int, default=30)
        peso_modelo = request.args.get('peso_modelo', type=int, default=80)
        
        df_relatorio = data_handler.gerar_relatorio_analise_ponderada(
            peso_cliente=peso_cliente, 
            peso_modelo=peso_modelo
        )
        
        if df_relatorio.empty:
            return jsonify({
                'sucesso': False,
                'erro': 'Não foi possível gerar o relatório'
            }), 500
        
        # Ler as estatísticas do arquivo JSON
        try:
            with open('data/estatisticas_ponderacao.json', 'r', encoding='utf-8') as f:
                estatisticas = json.load(f)
        except Exception as e:
            estatisticas = {'erro': f'Não foi possível carregar estatísticas: {e}'}
        
        # Retornar versão JSON do relatório com metadados
        return jsonify({
            'sucesso': True,
            'estatisticas': estatisticas,
            'dados': df_relatorio.to_dict('records')[:10],  # Limitar a 10 registros para a API
            'total_registros': len(df_relatorio),
            'colunas': df_relatorio.columns.tolist()
        })
        
    except Exception as e:
        logger.error(f"Erro na API de relatório ponderado: {e}")
        return jsonify({
            'sucesso': False,
            'erro': str(e)
        }), 500

class AspectExtractor:
    """Classe para extrair e analisar aspectos em textos"""
    
    # Dicionários de palavras-chave para cada aspecto
    ASPECTOS = {
        "produto": ["produto", "qualidade", "material", "durável", "design", "funcionalidade", "item"],
        "empresa": ["empresa", "marca", "loja", "vendedor", "confiança", "reputação", "compra"],
        "preço": ["preço", "valor", "custo", "caro", "barato", "desconto", "promoção"],
        "entrega": ["entrega", "prazo", "frete", "atraso", "chegou", "rápido", "transportadora", "envio", "demora"],
        "atendimento": ["atendimento", "suporte", "vendedor", "resposta", "educado", "comunicação", "contato"]
    }
    
    # Palavras indicadoras de problemas
    INDICADORES_PROBLEMA = [
        "problema", "defeito", "quebrou", "falhou", "erro", "decepção", 
        "horrível", "péssimo", "ruim", "não funciona", "demorou", "atraso", "reclamação"
    ]
    
    def __init__(self):
        """Inicializa o extrator de aspectos"""
        # Pré-processar palavras-chave para busca eficiente
        self.aspectos_flat = {}
        for aspecto, palavras in self.ASPECTOS.items():
            for palavra in palavras:
                self.aspectos_flat[palavra] = aspecto
    
    def _tokenizar_texto(self, texto):
        """Tokeniza o texto em palavras para melhorar a detecção de aspectos"""
        # Remover pontuação e quebrar em palavras
        import re
        texto_limpo = re.sub(r'[^\w\s]', ' ', texto.lower())
        return texto_limpo.split()
    
    def extrair_aspectos(self, texto, analise_sentimento):
        """Extrai aspectos mencionados no texto e associa sentimentos"""
        texto_lower = texto.lower()
        
        # Resultado da análise
        resultado = {
            "aspectos": {},
            "summary": {
                "aspects_detected": [],
                "all_problems": []
            }
        }
        
        # Tokenizar o texto para busca exata de palavras
        tokens = self._tokenizar_texto(texto_lower)
        token_set = set(tokens)  # Converter para conjunto para busca mais eficiente
        
        # Detectar menções a aspectos usando tokenização
        for palavra, aspecto in self.aspectos_flat.items():
            # Verificar se a palavra-chave exata está presente nos tokens
            if palavra in token_set:
                # Inicializar aspecto se for primeira menção
                if aspecto not in resultado["aspectos"]:
                    resultado["aspectos"][aspecto] = {
                        "mentions": 0,
                        "sentiment": 0,
                        "relevance": 0,
                        "context_phrases": [],
                        "problems": []
                    }
                    resultado["summary"]["aspects_detected"].append(aspecto)
                
                # Incrementar contagem de menções
                resultado["aspectos"][aspecto]["mentions"] += 1
                
                # Extrair frases de contexto
                frases = texto_lower.split('.')
                for frase in frases:
                    if palavra in frase and frase.strip() not in resultado["aspectos"][aspecto]["context_phrases"]:
                        resultado["aspectos"][aspecto]["context_phrases"].append(frase.strip())
        
        # Caso especial para palavras comuns que podem não ser detectadas pela tokenização
        if not resultado["summary"]["aspects_detected"]:
            # Verificação secundária com base em substrings para casos específicos
            for palavra, aspecto in self.aspectos_flat.items():
                if palavra in texto_lower and aspecto not in resultado["aspectos"]:
                    # Verificar se a palavra está como substring (caso não tenha sido detectada como token)
                    resultado["aspectos"][aspecto] = {
                        "mentions": 1,
                        "sentiment": 0,
                        "relevance": 0,
                        "context_phrases": [],
                        "problems": []
                    }
                    resultado["summary"]["aspects_detected"].append(aspecto)
                    
                    # Extrair frases de contexto
                    frases = texto_lower.split('.')
                    for frase in frases:
                        if palavra in frase and frase.strip() not in resultado["aspectos"][aspecto]["context_phrases"]:
                            resultado["aspectos"][aspecto]["context_phrases"].append(frase.strip())
        
        # Aplicar sentimento a cada aspecto
        # Esta é uma implementação simplificada; ideal seria usar modelos de NLP para sentimento por frase
        sentimento_geral = analise_sentimento.get("compound", 0)
        for aspecto in resultado["aspectos"]:
            resultado["aspectos"][aspecto]["sentiment"] = sentimento_geral
        
        # Detectar problemas
        for problema in self.INDICADORES_PROBLEMA:
            if problema in texto_lower:
                # Encontrar aspecto mais próximo do problema na frase
                frases = texto_lower.split('.')
                for frase in frases:
                    if problema in frase:
                        # Identificar aspecto relacionado ao problema
                        aspecto_relacionado = None
                        for palavra, aspecto in self.aspectos_flat.items():
                            if palavra in frase:
                                aspecto_relacionado = aspecto
                                break
                        
                        # Se não encontrou aspecto específico, use "geral"
                        if not aspecto_relacionado:
                            aspecto_relacionado = "geral"
                        
                        # Registrar problema
                        problema_info = {
                            "aspect": aspecto_relacionado,
                            "issue": problema,
                            "phrase": frase.strip()
                        }
                        
                        # Adicionar à lista geral de problemas
                        resultado["summary"]["all_problems"].append(problema_info)
                        
                        # Adicionar à lista específica do aspecto
                        if aspecto_relacionado in resultado["aspectos"]:
                            resultado["aspectos"][aspecto_relacionado]["problems"].append(problema_info)
        
        # Calcular relevância para cada aspecto
        for aspecto, dados in resultado["aspectos"].items():
            mencoes = dados["mentions"]
            frases_contexto = len(dados["context_phrases"])
            intensidade_sentimento = abs(dados["sentiment"])
            
            # Fórmula de relevância conforme especificação
            relevancia = (mencoes * 0.5) + (frases_contexto * 0.3) + (intensidade_sentimento * 0.2)
            resultado["aspectos"][aspecto]["relevance"] = round(relevancia, 2)
        
        # Identificar aspecto principal
        aspecto_principal = None
        max_relevancia = -1
        
        for aspecto, dados in resultado["aspectos"].items():
            if dados["relevance"] > max_relevancia:
                max_relevancia = dados["relevance"]
                aspecto_principal = aspecto
        
        resultado["summary"]["primary_aspect"] = aspecto_principal
        
        return resultado

@app.route('/analise-aspectos', methods=['GET'])
def visualizar_aspectos():
    """Endpoint para visualizar estatísticas de aspectos"""
    try:
        # Gerar estatísticas atualizadas
        estatisticas = data_handler.gerar_estatisticas_aspectos()
        
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
        logger.error(f"Erro ao gerar visualização de aspectos: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return render_template('erro.html', erro=f"Erro ao processar aspectos: {e}"), 500

@app.route('/feedback')
def feedback():
    """Exibe o formulário de feedback detalhado"""
    try:
        return render_template('feedback.html')
    except Exception as e:
        logger.error(f"Erro na rota /feedback: {e}")
        return render_template('erro.html', erro=f"Erro ao carregar página de feedback: {e}"), 500

@app.route('/api/feedback', methods=['POST'])
def feedback_api():
    """API para processar o feedback detalhado e retornar análise"""
    try:
        # Obter dados do request
        data = request.get_json()
        if not data:
            return jsonify({"success": False, "message": "Dados não fornecidos"}), 400
        
        texto = data.get('texto', '')
        rating = data.get('rating', 0)
        emotion = data.get('emotion', '')
        tags = data.get('tags', [])
        
        if not texto:
            return jsonify({"success": False, "message": "Texto do feedback não fornecido"}), 400
        
        # Realizar análise de sentimento
        analise = sentiment_analyzer.analisar_sentimento(texto)
        
        # Enriquecer análise com dados extras
        analise_enriquecida = analise.copy()
        analise_enriquecida['avaliacao_usuario'] = rating
        analise_enriquecida['emocao'] = emotion
        analise_enriquecida['tags_usuario'] = tags
        
        # Salvar o feedback com a análise
        data_handler.salvar_transcricao(texto, analise_enriquecida)
        
        return jsonify({
            "success": True,
            "message": "Feedback processado com sucesso",
            "analise": analise
        })
    except Exception as e:
        logger.error(f"Erro na API de feedback: {e}")
        return jsonify({"success": False, "message": str(e)}), 500

if __name__ == "__main__":
    # Verificar argumentos de linha de comando
    import sys
    
    # Verificar se é para gerar apenas relatório
    if len(sys.argv) > 1 and sys.argv[1] == "--gerar-relatorio":
        logger.info("Modo CLI: gerando relatório de análise ponderada...")
        
        # Pesos padrão ou personalizados
        peso_cliente = 30
        peso_modelo = 80
        
        if len(sys.argv) > 3:
            try:
                peso_cliente = int(sys.argv[2])
                peso_modelo = int(sys.argv[3])
                logger.info(f"Usando pesos personalizados: cliente={peso_cliente}%, modelo={peso_modelo}%")
            except Exception as e:
                logger.error(f"Erro ao processar pesos personalizados: {e}")
                logger.info("Usando pesos padrão: cliente=30%, modelo=80%")
        
        # Criar diretórios necessários
        criar_diretorios_essenciais()
        
        # Inicializar o analisador de dados
        data_handler_cli = DataHandler()
        
        # Gerar relatório
        df_relatorio = data_handler_cli.gerar_relatorio_analise_ponderada(
            peso_cliente=peso_cliente,
            peso_modelo=peso_modelo
        )
        
        if not df_relatorio.empty:
            logger.info(f"Relatório gerado com sucesso: {len(df_relatorio)} registros processados")
            logger.info(f"Arquivo salvo em: data/relatorio_ponderado.csv")
            logger.info(f"Estatísticas salvas em: data/estatisticas_ponderacao.json")
            sys.exit(0)
        else:
            logger.error("Falha ao gerar relatório")
            sys.exit(1)
    # Modo servidor web normal
    else:
        logger.info("Iniciando servidor ASCEND com XLM-RoBERTa...")
        logger.info("Acesse http://127.0.0.1:5000 no seu navegador")
        app.run(debug=True)