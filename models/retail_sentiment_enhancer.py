"""
Módulo de aprimoramento da análise de sentimentos para feedbacks de varejo.
Contém funções e classes para melhorar a precisão da análise em feedbacks específicos do varejo.
"""

import re
import logging
from typing import Dict, Any, List, Tuple, Optional, Union

# Configurar logger
logger = logging.getLogger(__name__)

class RetailSentimentEnhancer:
    """
    Classe para melhorar a análise de sentimentos específica para feedbacks de varejo.
    Aplica regras e heurísticas baseadas em conhecimento de domínio do varejo.
    """
    
    def __init__(self):
        """Inicializa o aprimorador de sentimentos para varejo"""
        # Categorias de varejo e seus termos associados
        self.categorias_varejo = {
            "entrega_logistica": [
                "entrega", "prazo", "frete", "transportadora", "rastreamento", "pedido", 
                "chegou", "caixa", "embalagem", "proteção", "danificado", "tempo", 
                "receber", "endereço", "envio", "entregador", "motoboy", "portão",
                "pacote", "encomenda", "correios", "status"
            ],
            "produto": [
                "produto", "item", "qualidade", "defeito", "funcionar", "material", 
                "tamanho", "cor", "medidas", "tabela", "manual", "instruções", 
                "embalagem", "original", "autêntico", "modelo", "versão", "conjunto",
                "peças", "itens", "eletrodoméstico", "plugue", "padrão"
            ],
            "atendimento": [
                "atendimento", "SAC", "chat", "e-mail", "email", "resposta", "contato",
                "atendente", "telefonema", "transferido", "setor", "resolver", "problema",
                "reclamação", "mensagem", "robô", "humano", "horário", "comercial",
                "canal", "comunicação", "solução"
            ],
            "cobranca_pagamento": [
                "cobrado", "pagamento", "cartão", "PIX", "boleto", "desconto", "valor",
                "preço", "promocional", "estorno", "reembolso", "parcela", "juros",
                "taxa", "cancelamento", "aprovado", "cupom", "código", "checkout",
                "finalização", "promoção", "parcelamento"
            ],
            "site_app": [
                "site", "aplicativo", "app", "plataforma", "sistema", "travou", "erro",
                "pagamento", "login", "cadastro", "senha", "carrinho", "compras", 
                "finalizar", "botão", "filtro", "buscar", "pesquisar", "imagem",
                "descrição", "avaliação", "lento", "carregando", "página", "navegação"
            ],
            "promocoes_propaganda": [
                "promoção", "black friday", "desconto", "cupom", "e-mail", "propaganda",
                "anúncio", "publicidade", "banner", "oferta", "imperdível", "exclusivo",
                "limitado", "última unidade", "estoque", "frete grátis", "brinde",
                "regalo", "presente", "disponível", "indisponível", "valor", "preço"
            ],
            "trocas_devolucoes": [
                "troca", "devolução", "reembolso", "produto", "substituição", "política",
                "prazo", "frete", "retorno", "arrependimento", "defeito", "garantia",
                "burocracia", "processo", "procedimento", "embalagem", "original",
                "lacre", "etiqueta", "nota fiscal", "crédito", "restituição", "valor"
            ],
            "loja_fisica": [
                "loja", "física", "estabelecimento", "caixa", "fila", "espera", 
                "vendedor", "atendente", "produtos", "prateleira", "organização",
                "preço", "etiqueta", "provador", "ambiente", "climatização",
                "música", "sinalização", "seção", "departamento", "horário"
            ]
        }
        
        # Expressões negativas específicas por categoria
        self.expressoes_negativas = {
            "entrega_logistica": [
                "atraso na entrega", "prazo não cumprido", "entrega atrasada", 
                "produto danificado", "embalagem violada", "caixa amassada",
                "entregador jogou", "arremessou", "endereço errado", "rastreio desatualizado",
                "não receber", "não recebi", "produto incompleto", "sem proteção"
            ],
            "produto": [
                "produto defeituoso", "parou de funcionar", "diferente do anunciado",
                "qualidade inferior", "tamanho errado", "cor diferente", "medidas incorretas",
                "embalagem violada", "sem manual", "falta peças", "modelo anterior",
                "não original", "falsificado", "material de baixa qualidade", "plugue diferente"
            ],
            "atendimento": [
                "não respondeu", "sem resposta", "atendente despreparado", "SAC inexistente",
                "transferido várias vezes", "ninguém resolve", "espera excessiva",
                "atendimento robótico", "sem solução", "robô ineficiente", 
                "informações contraditórias", "prazo não cumprido", "chat offline"
            ],
            "cobranca_pagamento": [
                "cobrado duas vezes", "valor errado", "desconto não aplicado", 
                "estorno não realizado", "pagamento não reconhecido", "taxa não informada",
                "promoção não válida", "cupom recusado", "preço diferente", 
                "cobrado mesmo após cancelamento", "parcela com valor errado"
            ],
            "site_app": [
                "site travou", "app fecha sozinho", "erro no pagamento", "falha no sistema",
                "botão não funciona", "filtro inoperante", "imagens não carregam",
                "descrição incompleta", "carrinho esvazia", "site lento", "avaliações removidas"
            ],
            "promocoes_propaganda": [
                "falsa promoção", "mesmo preço de antes", "publicidade enganosa",
                "cupom não funciona", "letra miúda", "condições escondidas", 
                "produto indisponível", "frete grátis falso", "brinde não incluído",
                "desconto insignificante", "oferta enganosa", "estoque inexistente"
            ],
            "trocas_devolucoes": [
                "recusaram troca", "processo burocrático", "devolução negada",
                "reembolso não realizado", "prazo excessivo", "frete da devolução cobrado",
                "produto de troca com defeito", "exigem embalagem original danificada",
                "política confusa", "apenas crédito na loja", "arrependimento negado"
            ],
            "loja_fisica": [
                "fila enorme", "poucos atendentes", "vendedores ausentes",
                "loja desorganizada", "preços diferentes", "provadores ocupados",
                "ambiente desagradável", "sinalização confusa", "atendimento demorado",
                "horário incorreto", "produtos indisponíveis", "informação errada"
            ]
        }
        
        # Palavras de sarcasmo específicas para varejo
        self.palavras_sarcasmo_varejo = [
            "excelente atendimento", "super rápida entrega", "qualidade impecável",
            "ótimo trabalho", "perfeito estado", "maravilhosa experiência",
            "impressionante eficiência", "extremamente satisfeito", "apenas X dias",
            "somente Y horas", "só demorou", "incrível como", "parabéns pelo"
        ]
        
        # Limiar de confiança para cada categoria
        self.limiares_confianca = {
            "entrega_logistica": 0.75,
            "produto": 0.70,
            "atendimento": 0.75,
            "cobranca_pagamento": 0.70,
            "site_app": 0.65,
            "promocoes_propaganda": 0.65,
            "trocas_devolucoes": 0.70,
            "loja_fisica": 0.70
        }
        
        # Padrões que indicam intensidade de sentimento
        self.intensificadores_positivos = [
            "muito bom", "excelente", "fantástico", "incrível", "maravilhoso",
            "excepcional", "perfeito", "impecável", "surpreendente", "espetacular",
            "superou expectativas", "melhor", "ótimo", "nota 10", "recomendo muito"
        ]
        
        self.intensificadores_negativos = [
            "péssimo", "horrível", "terrível", "absurdo", "ridículo", "inaceitável",
            "vergonhoso", "decepcionante", "deplorável", "frustrante", "lamentável",
            "pior", "nunca mais", "fuja", "evite", "desastre", "catástrofe", "lixo"
        ]
        
        logger.info("RetailSentimentEnhancer inicializado com sucesso")
        
    def detectar_categoria_feedback(self, texto: str) -> Tuple[str, float]:
        """
        Detecta a categoria principal do feedback de varejo
        
        Args:
            texto: O texto do feedback
            
        Returns:
            Tupla contendo (categoria_detectada, confiança)
        """
        texto_lower = texto.lower()
        melhor_categoria = "geral"
        melhor_score = 0
        
        # Verificar cada categoria
        for categoria, termos in self.categorias_varejo.items():
            score = 0
            for termo in termos:
                if termo in texto_lower:
                    # Palavras exatas têm mais peso
                    if re.search(r'\b' + re.escape(termo) + r'\b', texto_lower):
                        score += 1.5
                    else:
                        score += 0.5
            
            # Verificar expressões negativas específicas
            if categoria in self.expressoes_negativas:
                for expressao in self.expressoes_negativas[categoria]:
                    if expressao in texto_lower:
                        score += 2.0  # Expressões completas têm peso maior
            
            # Normalizar score pelo número de termos (para não favorecer categorias com mais termos)
            score_normalizado = score / (len(termos) + 1)
            
            if score_normalizado > melhor_score:
                melhor_score = score_normalizado
                melhor_categoria = categoria
        
        # Converter score para uma medida de confiança entre 0 e 1
        confianca = min(melhor_score / 5.0, 1.0)
        
        return melhor_categoria, confianca
    
    def ajustar_analise_sentimento(self, resultado_original: Dict[str, Any], texto: str) -> Dict[str, Any]:
        """
        Ajusta o resultado de análise de sentimento considerando especificidades do varejo
        
        Args:
            resultado_original: Resultado original da análise de sentimento
            texto: Texto do feedback
            
        Returns:
            Resultado ajustado da análise de sentimento
        """
        # Clonar resultado para não modificar o original
        resultado = resultado_original.copy()
        texto_lower = texto.lower()
        
        # Detectar categoria do feedback
        categoria, confianca_categoria = self.detectar_categoria_feedback(texto)
        
        # Verificar intensificadores de sentimento
        tem_intensificador_positivo = any(termo in texto_lower for termo in self.intensificadores_positivos)
        tem_intensificador_negativo = any(termo in texto_lower for termo in self.intensificadores_negativos)
        
        # Verificar expressões negativas específicas da categoria
        tem_expressao_negativa = False
        if categoria in self.expressoes_negativas:
            tem_expressao_negativa = any(expressao in texto_lower for expressao in self.expressoes_negativas[categoria])
        
        # Ajustar sentimento baseado nas regras específicas de varejo
        sentimento_original = resultado.get('sentimento', 'neutro')
        confianca_original = resultado.get('confianca', 0.5)
        
        # Ajustar confiança com base nos intensificadores
        if tem_intensificador_positivo and sentimento_original == 'positivo':
            resultado['confianca'] = min(confianca_original + 0.15, 0.98)
        elif tem_intensificador_negativo and sentimento_original == 'negativo':
            resultado['confianca'] = min(confianca_original + 0.15, 0.98)
        
        # Se temos expressões negativas específicas, reforçar sentimento negativo
        if tem_expressao_negativa:
            if sentimento_original != 'negativo':
                # Se o sentimento original não era negativo, mas temos expressões negativas claras, ajustar
                resultado['sentimento'] = 'negativo'
                resultado['sentimento_original'] = sentimento_original
                resultado['confianca'] = max(confianca_original, 0.75)
            else:
                # Se já era negativo, aumentar a confiança
                resultado['confianca'] = min(confianca_original + 0.1, 0.98)
        
        # Regras específicas por categoria
        if categoria == 'entrega_logistica':
            # Menções a atraso devem ser negativas com alta confiança
            if any(termo in texto_lower for termo in ['atraso', 'atrasado', 'demora', 'demorou']):
                if sentimento_original != 'negativo':
                    resultado['sentimento'] = 'negativo'
                    resultado['sentimento_original'] = sentimento_original
                resultado['confianca'] = max(confianca_original, 0.8)
        
        elif categoria == 'produto':
            # Menções a defeito, quebra ou qualidade inferior devem ser negativas
            if any(termo in texto_lower for termo in ['defeito', 'quebrou', 'parou de funcionar', 'baixa qualidade']):
                if sentimento_original != 'negativo':
                    resultado['sentimento'] = 'negativo'
                    resultado['sentimento_original'] = sentimento_original
                resultado['confianca'] = max(confianca_original, 0.85)
        
        elif categoria == 'cobranca_pagamento':
            # Menções a cobranças indevidas, valores errados são quase sempre negativas
            if any(termo in texto_lower for termo in ['cobrado duas vezes', 'valor errado', 'estorno']):
                if sentimento_original != 'negativo':
                    resultado['sentimento'] = 'negativo'
                    resultado['sentimento_original'] = sentimento_original
                resultado['confianca'] = max(confianca_original, 0.8)
        
        # Adicionar metadados ao resultado
        resultado['categoria_varejo'] = categoria
        resultado['confianca_categoria'] = confianca_categoria
        
        return resultado
    
    def detectar_sarcasmo_varejo(self, texto: str, resultado_sentimento: Dict[str, Any]) -> Dict[str, Any]:
        """
        Detecta sarcasmo específico em feedbacks de varejo
        
        Args:
            texto: Texto do feedback
            resultado_sentimento: Resultado da análise de sentimento
            
        Returns:
            Dicionário indicando se há sarcasmo e sua probabilidade
        """
        texto_lower = texto.lower()
        resultado = {
            'e_sarcastico': False,
            'probabilidade': 0.0,
            'nivel': 'baixo',
            'tipo': None
        }
        
        # Padrões de sarcasmo específicos para varejo
        padroes_sarcasmo = [
            # Padrão para espera/demora com adjetivos positivos
            (r'(?i)(apenas|só|somente)?\s+([2-9]|1[0-9]|[2-9][0-9])\s+(horas?|dias?|semanas?).+(incrível|excelente|ótimo|maravilhoso|perfeito)', 0.85, 'tempo_positivo'),
            
            # Padrão para entregas atrasadas com tom positivo
            (r'(?i)(chegou|entregue).+(apenas|só|somente)?\s+([3-9]|[1-9][0-9])\s+(dias?|semanas?).+(depois|após).+(prazo|previsto)', 0.8, 'atraso_entrega'),
            
            # Padrão para problemas com produtos com tom positivo
            (r'(?i)(produto|item).+(quebrou|parou|falhou|defeito).+(apenas|só|somente)?\s+([1-9])\s+(dias?|horas?|semanas?)', 0.8, 'produto_defeituoso'),
            
            # Padrão para atendimento ruim com tom positivo
            (r'(?i)(atendimento|suporte).+(incrível|excelente|ótimo).+(não|nunca).+(resolveu|solucionou|atendeu)', 0.75, 'atendimento_ruim'),
            
            # Padrão para cobranças erradas com tom positivo
            (r'(?i)(cobrado|cobraram).+(duas|três|várias).+vezes.+(incrível|excelente|ótimo|maravilhoso)', 0.8, 'cobranca_errada'),
            
            # Padrão para site/app com problemas com tom positivo
            (r'(?i)(site|aplicativo|app).+(travou|caiu|falhou|erro).+(incrível|excelente|ótimo|experiência)', 0.75, 'site_problematico'),
            
            # Padrão para promoções falsas com tom positivo
            (r'(?i)(promoção|black friday|oferta).+(incrível|excelente|ótimo).+(mesmo preço|sem desconto|falsa)', 0.8, 'promocao_falsa'),
            
            # Padrão para troca/devolução problemática com tom positivo
            (r'(?i)(processo|política).+(troca|devolução).+(simples|fácil|rápido).+(não|nunca|impossível)', 0.8, 'devolucao_problematica')
        ]
        
        # Verificar se o texto contém padrões de sarcasmo
        for padrao, probabilidade, tipo in padroes_sarcasmo:
            if re.search(padrao, texto_lower):
                resultado['e_sarcastico'] = True
                resultado['probabilidade'] = max(resultado['probabilidade'], probabilidade)
                resultado['tipo'] = tipo
        
        # Verificar palavras de sarcasmo específicas
        for frase in self.palavras_sarcasmo_varejo:
            if frase in texto_lower:
                # Verificar se o sentimento geral é positivo, mas há indicações de problemas
                sentimento = resultado_sentimento.get('sentimento', 'neutro')
                if sentimento == 'positivo':
                    # Verificar se há palavras negativas ou problemas mencionados
                    palavras_problema = ["problema", "ruim", "péssimo", "horroroso", "terrível", 
                                          "demora", "atraso", "não funciona", "defeito", "quebrou", 
                                          "não recomendo", "nunca mais"]
                    
                    tem_problema = any(palavra in texto_lower for palavra in palavras_problema)
                    if tem_problema:
                        resultado['e_sarcastico'] = True
                        resultado['probabilidade'] = max(resultado['probabilidade'], 0.7)
                        resultado['tipo'] = 'positivo_com_problema'
        
        # Definir nível baseado na probabilidade
        if resultado['probabilidade'] >= 0.7:
            resultado['nivel'] = 'alto'
        elif resultado['probabilidade'] >= 0.4:
            resultado['nivel'] = 'medio'
        
        return resultado

def aplicar_melhorias_varejo(texto: str, resultado_analise: Dict[str, Any]) -> Dict[str, Any]:
    """
    Função principal para aplicar todas as melhorias específicas para análise de feedbacks de varejo
    
    Args:
        texto: Texto do feedback
        resultado_analise: Resultado original da análise de sentimento
        
    Returns:
        Resultado aprimorado da análise
    """
    enhancer = RetailSentimentEnhancer()
    
    # Ajustar análise de sentimento
    resultado_ajustado = enhancer.ajustar_analise_sentimento(resultado_analise, texto)
    
    # Verificar sarcasmo específico para varejo
    resultado_sarcasmo = enhancer.detectar_sarcasmo_varejo(texto, resultado_ajustado)
    
    # Se detectou sarcasmo com alta confiança, ajustar o sentimento
    if resultado_sarcasmo['e_sarcastico'] and resultado_sarcasmo['probabilidade'] >= 0.7:
        # Guardar sentimento original
        resultado_ajustado['sentimento_original'] = resultado_ajustado.get('sentimento', 'neutro')
        
        # Se sentimento era positivo, inverter para negativo com alta confiança
        if resultado_ajustado.get('sentimento') == 'positivo':
            resultado_ajustado['sentimento'] = 'negativo'
            resultado_ajustado['confianca'] = resultado_sarcasmo['probabilidade']
        
        # Adicionar informações do sarcasmo ao resultado
        resultado_ajustado['sarcasmo'] = resultado_sarcasmo
    
    return resultado_ajustado 