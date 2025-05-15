"""
Configuração para os detectores de sarcasmo.
Este módulo contém as configurações padrão para os detectores de sarcasmo.
"""

# Configuração do detector de sarcasmo
SARCASMO_CONFIG = {
    # Marcadores lexicais de sarcasmo em português
    'marcadores': [
        'sério mesmo',
        'tá bom',
        'aham',
        'sei',
        'claro',
        'nossa',
        'uau',
        'parabéns',
        'brilhante',
        'genial',
        'perfeito',
        'excelente',
        'maravilhoso',
        'incrível',
        'sensacional',
        'conta outra',
        'tá de brincadeira',
        'imagina',
        'jura',
        'fala sério',
        'tá certo',
        'valeu'
    ],
    
    # Padrões de expressão regular para identificar frases sarcásticas
    'padroes': [
        ('aspas irônicas', r'"([^"]{1,20})"'),
        ('ironia por repetição', r'(?i)(claro+|ótimo+|legal+|lindo+)'),
        ('falsa concordância', r'(?i)com certeza(,|\.|$)'),
        ('ironia condicional', r'(?i)como se (fosse|isso|eu|alguém)'),
        ('ironia interrogativa', r'(?i)sério\s?\?'),
        ('exagero', r'(?i)(super|hiper|mega|ultra)'),
        ('contraste', r'(?i)(imagina|imagina se)'),
        ('tempo_espera', r'(?i)(incrível|excelente|ótimo|maravilhoso|perfeito).+(apenas|só)?\s+\d+\s+(horas?|minutos?|dias?)'),
        ('falsa_valorização', r'(?i)(nossa|uau|caramba|puxa).+(que|como).+(incrível|excelente|ótimo|maravilhoso|perfeito)'),
        ('contradição_tempo', r'(?i)(apenas|só)\s+\d+\s+(horas?|minutos?|dias?)'),
        # Novos padrões para melhorar detecção de espera
        ('espera_positiva', r'(?i)(esperei|aguardei|fiquei).+(apenas|só)?\s+\d+\s+(horas?|minutos?|dias?|semanas?)'),
        ('atendimento_tempo', r'(?i)(atendimento|serviço)\s+(incrível|excelente|ótimo|maravilhoso|perfeito).+(espera|fila|aguard)'),
        ('fila_positiva', r'(?i)(incrível|excelente|ótimo|maravilhoso|perfeito).+(fila|espera)'),
        ('tempo_numero_grande', r'(?i)(apenas|só|somente)?\s+([2-9]|[1-9][0-9]+)\s+(horas?|dias?|semanas?|meses?)'),
        ('tempo_numero_exclamacao', r'(?i)\d+\s+(horas?|minutos?|dias?|semanas?).+!'),
        # Padrões para detectar contradção qualidade/problema
        ('problema_qualidade', r'(?i)(quebrou|quebrado|parou|estragou|falhou|defeito).+\d+.+(dias?|semanas?|meses?)'),
        ('problema_elogio', r'(?i)(quebrou|quebrado|parou|estragou|falhou|defeito|estrago|pane|danificado).+(qualidade|excelente|excepcional|ótima|perfeita|incrível)'),
        ('elogio_problema', r'(?i)(qualidade|excelente|excepcional|ótima|perfeita|incrível).+(quebrou|quebrado|parou|estragou|falhou|defeito)'),
        ('quebrou_dias', r'(?i)(quebrou|quebrado|parou|falhou).+\d+.+(dias?|semanas?|horas?|meses?).+(excepcional|excelente|ótimo|perfeito)'),
    ],
    
    # Contextos que indicam possível sarcasmo (combinações de palavras)
    'contextos': [
        # Adjetivos positivos + substantivos negativos
        ('incrível', 'demora'),
        ('ótimo', 'espera'),
        ('excelente', 'fila'),
        ('maravilhoso', 'lentidão'),
        ('perfeito', 'atraso'),
        
        # Problema de qualidade + adjetivos positivos
        ('quebrou', 'excelente'),
        ('quebrou', 'qualidade'),
        ('quebrou', 'perfeito'),
        ('quebrou', 'incrível'),
        ('quebrou', 'excepcional'),
        ('falhou', 'excelente'),
        ('falhou', 'qualidade'),
        ('defeito', 'excelente'),
        ('defeito', 'qualidade'),
        ('quebrado', 'excelente'),
        ('estragou', 'qualidade'),
        
        # Advérbios + indicadores de tempo
        ('apenas', 'horas'),
        ('só', 'minutos'),
        ('somente', 'dias'),
        ('rápido', 'espera'),
        ('eficiente', 'demora'),
        
        # Outros contextos contraditórios
        ('adorei', 'péssimo'),
        ('melhor', 'pior'),
        ('excelente', 'terrível'),
        ('nossa', 'horas'), 
        ('incrível', 'esperei'),
        ('ótimo', 'demorou'),
    ],
    
    # Thresholds para classificação de sarcasmo
    'thresholds': {
        'alto': 0.7,    # Sarcasmo forte
        'medio': 0.4,   # Sarcasmo moderado
        'baixo': 0.2    # Indício fraco de sarcasmo
    },
    
    # Palavras com sentimento positivo para detecção de contradição
    'palavras_positivas': [
        'bom', 'bem', 'ótimo', 'excelente', 'maravilhoso', 'incrível', 
        'fantástico', 'sensacional', 'adorável', 'impressionante', 'perfeito',
        'feliz', 'alegre', 'satisfeito', 'contente', 'agradável', 'positivo',
        'fabuloso', 'extraordinário', 'magnífico', 'esplêndido', 'divino',
        'surpreendente', 'grato', 'encantado', 'animado', 'esperançoso',
        'adorei', 'adoro', 'amo', 'gosto', 'confio', 'acredito', 'recomendo'
    ],
    
    # Palavras com sentimento negativo para detecção de contradição
    'palavras_negativas': [
        'ruim', 'mal', 'péssimo', 'terrível', 'horrível', 'detestável', 'odioso',
        'desagradável', 'defeituoso', 'falho', 'triste', 'decepcionante',
        'deplorável', 'desapontador', 'insatisfatório', 'negativo', 'indesejável',
        'inadequado', 'lamentável', 'medíocre', 'problemático', 'caótico',
        'confuso', 'instável', 'pobre', 'fraco', 'odeio', 'detesto', 'lamento',
        'não gosto', 'não confio', 'não acredito', 'não recomendo',
        "horas", "minutos", "fila", "demora", "lentidão", "péssimo",
        "ruim", "horrível", "terrível", "irritante", "cansativo",
        "problema", "erro", "falho", "dias", "semanas"
    ],
    
    # Pesos para cada método de detecção
    'pesos': {
        'markers': 0.4,       # Peso para detecção baseada em regras/marcadores
        'patterns': 0.5,  # Peso para detecção baseada em padrões
        'punctuation': 0.3        # Peso para detecção baseada em pontuação
    },
    
    # Pesos para combinar os diferentes métodos
    'combine_weights': {
        'rule': 0.4,       # Peso para detector baseado em regras
        'contradiction': 0.3,  # Peso para detector baseado em contradição
        'ml': 0.3        # Peso para detector baseado em modelo ML
    },
    
    # Valores máximos de score para normalização
    'max_score': 5.0,
    
    # Limiar para classificar um texto como sarcástico
    'limiar_sarcasmo': 0.4,
    
    # Caminhos para modelos
    'modelo': {
        'path_local': './models/sarcasm_model/',
        'hf_model': 'cardiffnlp/twitter-xlm-roberta-base-sentiment',
        'fallback': True  # Usar métodos alternativos se o modelo falhar
    },
    
    # Ajustes de sentimento baseados em sarcasmo
    'ajustes_sentimento': {
        'alto': -0.8,       # Inversão quase total para sarcasmo alto
        'moderado': -0.5,   # Inversão parcial para sarcasmo moderado
        'baixo': 0.0        # Sem ajuste para sarcasmo baixo
    },
    
    # Configurações de logging
    'logging': {
        'nivel': 'INFO',
        'detalhe_debug': False  # Habilitar para logs detalhados de detecção
    },
    
    "thresholds": {
        "high": 0.7,
        "medium": 0.4
    },

    # Limiares para classificação de sarcasmo
    'limiares': {
        'alto': 0.7,
        'moderado': 0.4,
        'baixo': 0.2
    },
    
    # Frases sarcasticamente positivas
    'frases_sarcasticas': [
        "que incrível", "que maravilha", "que bom", "que ótimo", "claro",
        "certamente", "obviamente", "com certeza", "exatamente", 
        "fantástico", "esperei apenas", "só esperei", "melhor impossível",
        "nossa", "uau", "caramba", "puxa", "genial", "brilhante"
    ],
    
    # Configurações avançadas
    'avancado': {
        'peso_regras': 0.6,
        'peso_ml': 0.2,
        'peso_contradicao': 0.2,
        'usar_ml': False,  # Desativado por padrão
        'usar_contradicao': True
    }
} 