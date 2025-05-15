import numpy as np
from utils.config import logger

class AnaliseEstatistica:
    """Classe com métodos estatísticos para análise de sentimentos"""
    
    @staticmethod
    def normalizar_pesos(pesos):
        """
        Normaliza pesos para soma = 1
        
        Parâmetros:
        - pesos: Lista ou dicionário de pesos
        
        Retorna:
        - Pesos normalizados no mesmo formato que a entrada
        """
        try:
            if isinstance(pesos, dict):
                total = sum(pesos.values())
                if total == 0:
                    # Evitar divisão por zero
                    valores = list(pesos.keys())
                    return {k: 1.0 / len(valores) for k in valores}
                return {k: v / total for k, v in pesos.items()}
            elif isinstance(pesos, (list, tuple, np.ndarray)):
                total = sum(pesos)
                if total == 0:
                    # Evitar divisão por zero
                    return [1.0 / len(pesos)] * len(pesos)
                return [v / total for v in pesos]
            else:
                raise ValueError(f"Formato não suportado: {type(pesos)}")
        except Exception as e:
            logger.error(f"Erro ao normalizar pesos: {e}")
            # Retornar valores iguais em caso de erro
            if isinstance(pesos, dict):
                return {k: 1.0 / len(pesos) for k in pesos.keys()}
            else:
                return [1.0 / len(pesos)] * len(pesos)
    
    @staticmethod
    def calcular_media_ponderada(valores, pesos, normalizar=True):
        """
        Calcula a média ponderada de valores
        
        Parâmetros:
        - valores: Lista de valores
        - pesos: Lista de pesos correspondentes
        - normalizar: Se True, normaliza os pesos antes do cálculo
        
        Retorna:
        - Média ponderada
        """
        try:
            if len(valores) != len(pesos):
                raise ValueError("Listas de valores e pesos devem ter o mesmo tamanho")
            
            if normalizar:
                pesos_norm = AnaliseEstatistica.normalizar_pesos(pesos)
            else:
                pesos_norm = pesos
            
            # Cálculo da média ponderada
            return sum(v * p for v, p in zip(valores, pesos_norm))
        except Exception as e:
            logger.error(f"Erro ao calcular média ponderada: {e}")
            # Em caso de erro, retorna média simples
            return sum(valores) / len(valores) if valores else 0
    
    @staticmethod
    def mapear_sentimento_para_valor(sentimento):
        """
        Mapeia string de sentimento para valor numérico
        
        Parâmetros:
        - sentimento: String ('positivo', 'neutro', 'negativo')
        
        Retorna:
        - Valor numérico (positivo=1, neutro=0, negativo=-1)
        """
        try:
            mapeamento = {
                'positivo': 1,
                'neutro': 0,
                'negativo': -1
            }
            
            return mapeamento.get(sentimento.lower(), 0)
        except Exception as e:
            logger.error(f"Erro ao mapear sentimento para valor: {e}")
            return 0 