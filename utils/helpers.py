import os
import sys
from utils.config import logger

def criar_diretorios_essenciais():
    """Cria os diretórios essenciais para o funcionamento da aplicação"""
    try:
        diretorios = [
            'transcricoes',
            'static/images',
            'data',
            'models/cardiffnlp-xlm-roberta'
        ]
        
        for diretorio in diretorios:
            os.makedirs(diretorio, exist_ok=True)
            logger.info(f"Diretório '{diretorio}' verificado/criado")
        
        return True
    except Exception as e:
        logger.error(f"Erro ao criar diretórios essenciais: {e}")
        return False

def verificar_dependencias():
    """Verifica se as dependências críticas estão disponíveis"""
    try:
        # Verificar versão do Python sendo utilizada
        logger.info(f"Python executando de: {sys.executable}")
        logger.info(f"Versão do Python: {sys.version}")
        
        # Verificar dependências críticas
        dependencias = [
            ('flask', 'Flask'),
            ('nltk', 'NLTK'),
            ('transformers', 'Transformers'),
            ('pandas', 'Pandas'),
            ('plotly', 'Plotly'),
            ('numpy', 'NumPy'),
            ('speech_recognition', 'SpeechRecognition')
        ]
        
        resultados = {}
        
        for modulo, nome in dependencias:
            try:
                __import__(modulo)
                resultados[nome] = "OK"
            except ImportError:
                resultados[nome] = "Ausente"
                logger.warning(f"Dependência '{nome}' não encontrada")
        
        return resultados
    except Exception as e:
        logger.error(f"Erro ao verificar dependências: {e}")
        return {"erro": str(e)}

def formatar_sentimento(sentimento, capitalize=True):
    """
    Formata o sentimento para exibição
    
    Parâmetros:
    - sentimento: String com o sentimento ('positivo', 'neutro', 'negativo')
    - capitalize: Se True, capitaliza a primeira letra
    
    Retorna:
    - String formatada do sentimento
    """
    try:
        if sentimento is None:
            return "Não identificado" if capitalize else "não identificado"
            
        sentimento = sentimento.lower()
        
        if sentimento == "positivo":
            texto = "Positivo"
        elif sentimento == "negativo":
            texto = "Negativo"
        elif sentimento == "neutro":
            texto = "Neutro"
        else:
            texto = "Não identificado"
        
        return texto if capitalize else texto.lower()
    except Exception as e:
        logger.error(f"Erro ao formatar sentimento: {e}")
        return "Desconhecido" if capitalize else "desconhecido"

def formatar_confianca(confianca):
    """
    Formata o valor de confiança para exibição
    
    Parâmetros:
    - confianca: Valor de confiança (float entre 0 e 1)
    
    Retorna:
    - String formatada com a confiança em porcentagem
    """
    try:
        if confianca is None:
            return "N/A"
            
        # Converter para porcentagem
        porcentagem = confianca * 100
        
        # Formatar com 1 casa decimal
        return f"{porcentagem:.1f}%"
    except Exception as e:
        logger.error(f"Erro ao formatar confiança: {e}")
        return "N/A" 