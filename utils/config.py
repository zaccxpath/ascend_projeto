import os
import logging
import json
from pathlib import Path
import nltk
from nltk.corpus import stopwords
from nltk import download as nltk_download

# Configuração de logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler()
    ]
)
logger = logging.getLogger('ascend')

# Diretórios
DATA_DIR = 'data'
MODELS_DIR = 'models'
TRANSCRIPTIONS_DIR = 'transcricoes'
STATIC_IMAGES_DIR = 'static/images'

# Garantir que diretórios existam
os.makedirs(TRANSCRIPTIONS_DIR, exist_ok=True)
os.makedirs(STATIC_IMAGES_DIR, exist_ok=True)
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(os.path.join(MODELS_DIR, 'cardiffnlp-xlm-roberta'), exist_ok=True)

# Função para carregar stopwords do NLTK
def carregar_stopwords() -> set[str]:
    """
    Carrega as stopwords do NLTK em português. 
    Se o recurso não existir, tenta baixar. 
    Se ainda assim falhar, levanta LookupError.
    """
    try:
        # Tenta carregar diretamente
        sw = set(stopwords.words('portuguese'))
        logger.info(f"Stopwords carregadas do NLTK: {len(sw)} palavras")
        return sw

    except LookupError as e:
        logger.warning(f"Recurso NLTK 'stopwords' não encontrado: {e}. Tentando download...")
        try:
            nltk_download('stopwords', quiet=True)
            sw = set(stopwords.words('portuguese'))
            logger.info(f"Stopwords baixadas e carregadas: {len(sw)} palavras")
            return sw
        except Exception as e2:
            logger.error(f"Falha ao baixar stopwords do NLTK: {e2}")
            # Erro intencional se não puder usar NLTK
            raise LookupError("Não foi possível carregar as stopwords do NLTK; instale o pacote NLTK e rode nltk.download('stopwords').")

# Uso
STOPWORDS_PT = carregar_stopwords()


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

# Aspectos originais (mantido para retrocompatibilidade)
ASPECTOS = [
    "produto",
    "empresa",
    "preço",
    "entrega",
    "atendimento"
]

# Indicadores de problema originais (mantido para retrocompatibilidade)
INDICADORES_PROBLEMA = [
    "atraso", "demora", "defeito", "quebrado", "danificado",
    "não funciona", "péssimo", "ruim", "falsificado", "não original"
]

# Novas configurações para a versão refatorada
ASPECT_EXPRESSIONS = {
    'produto': [
        "produto", "qualidade", "material", "durável", "design", "funcionalidade", 
        "item", "mercadoria", "compra", "artigo", "bem feito", "mal feito", 
        "boa qualidade", "má qualidade", "baixa qualidade", "resistente", 
        "frágil", "quebrou", "defeito", "defeituoso", "funciona bem", "não funciona",
        # Adicionando termos específicos para perfumes
        "perfume", "aroma", "cheiro", "fragrância", "fixação", "fixa", "duração", 
        "dura", "horas", "falsificado", "original", "falso", "autêntico", "genuíno"
    ],
    'empresa': [
        "empresa", "marca", "loja", "vendedor", "confiança", "reputação", "compra",
        "atendimento da empresa", "política da empresa", "marca é", "site", 
        "e-commerce", "marketplace", "serviço", "garantia", "política de devolução"
    ],
    'preço': [
        "preço", "valor", "custo", "caro", "barato", "desconto", "promoção",
        "valor cobrado", "preço está", "custo benefício", "preço alto", 
        "preço baixo", "em conta", "econômico", "custa", "custou", "pagamento",
        "investi", "investimento", "valeu a pena", "não valeu", "caro demais"
    ],
    'entrega': [
        "entrega", "prazo", "frete", "atraso", "chegou", "rápido", "transportadora", 
        "envio", "demora", "demorou", "prazo de entrega", "tempo de entrega", 
        "entrega foi", "chegou rápido", "atrasou", "não chegou", "extraviado", 
        "entregue", "recebimento", "pacote", "embalagem", "enviado", "demorou muito", 
        "mais do que o prometido", "antes do prazo", "depois do prazo", "no prazo",
        "esperar", "esperando", "esperei", "espera", "aguardar", "aguardando", "aguardei",
        "receber", "recebi", "tempo", "dias", "semanas", "mês", "meses", 
        "esperar para receber", "tempo de espera", "demora para receber", 
        "espera longa", "espera demorada", "chegar em casa"
    ],
    'atendimento': [
        # Termos diretamente relacionados ao atendimento
        "atendimento", "suporte", "vendedor", "resposta", "educado", "comunicação", 
        "contato", "qualidade do atendimento", "atendente foi", "suporte técnico",
        "serviço ao cliente", "sac", "ouvidoria", "cordial", "receptivo", "atencioso",
        "grosso", "grosseria", "mal atendido", "bem atendido", "não respondeu", 
        "respondeu rápido", "demorou para responder", "assistência", "ajuda",
        "atender", "atendente", "cliente", "consumidor", "usuário", 
        
        # Expressões específicas de tempo de atendimento
        "tempo de atendimento", "espera para atendimento", "tempo de espera", 
        "demora no atendimento", "demorou para atender", "atendimento rápido",
        "atendimento demorado", "atendimento lento", "tempo de resposta",
        "horas para atender", "minutos para atender", "esperar atendimento",
        "fila de atendimento", "espera na linha", "aguardar atendimento",
        "atendimento imediato", "atendimento prioritário", "prioridade"
    ]
}

# Padrões negativos por aspecto
NEGATIVE_PATTERNS = {
    'produto': [
        'baixa qualidade', 'má qualidade', 'ruim', 'péssimo', 'quebrou', 'defeito', 
        'mal feito', 'não funciona', 'frágil', 'estragou', 'danificado', 'não gostei', 
        'decepcionou', 'decepcionante', 'problema', 'falha', 'piorou', 'produto quebrado',
        'quebrado', 'estragado', 'com defeito', 'defeituoso', 'produto danificado',
        'não fixa', 'não dura', 'não cheira', 'falsificado', 'falso', 'não original',
        'não é original', 'fake', 'sem fixação', 'pouca duração', 'não tem o mesmo aroma'
    ],
    'empresa': [
        'não recomendo', 'desconfiança', 'sem comprometimento', 'enganosa', 'golpe', 
        'péssima', 'ruim', 'fraude', 'sem ética', 'não cumpre', 'não honra'
    ],
    'preço': [
        'caro', 'absurdo', 'não vale', 'exagerado', 'elevado', 'abusivo', 'exorbitante', 
        'fora da realidade', 'não condiz', 'muito alto', 'caro demais', 'superfaturado'
    ],
    'entrega': [
        'atrasada', 'demorou', 'não chegou', 'extraviou', 'prazo excedido', 'demorou muito', 
        'mais do que o prometido', 'além do prazo', 'muito tempo', 'lentidão', 
        'extremamente demorada', 'péssima entrega', 'atraso', 'deixou a desejar'
    ],
    'atendimento': [
        'péssimo', 'lento', 'sem educação', 'não resolveu', 'despreparo', 'grosso', 
        'descaso', 'mal educado', 'grosseiro', 'sem solução', 'incompetente', 
        'não atendeu', 'insatisfatório', 'horrível'
    ]
}

# Padrões positivos por aspecto
POSITIVE_PATTERNS = {
    'produto': [
        'boa qualidade', 'excelente produto', 'bem feito', 'resistente', 'durável', 
        'funciona bem', 'perfeito', 'ótimo', 'excelente', 'supera expectativas'
    ],
    'empresa': [
        'confiável', 'recomendo', 'séria', 'comprometida', 'atenciosa', 'responsável', 
        'cumpre', 'honesta', 'transparente', 'satisfeito com a empresa'
    ],
    'preço': [
        'barato', 'em conta', 'justo', 'vale a pena', 'bom custo-benefício', 'acessível', 
        'econômico', 'ótimo preço', 'preço bom', 'valor justo', 'promoção'
    ],
    'entrega': [
        'rápida', 'pontual', 'antes do prazo', 'eficiente', 'chegou cedo', 'no tempo certo', 
        'ágil', 'cumpriu o prazo', 'entrega rápida', 'chegou rápido', 'sem atrasos'
    ],
    'atendimento': [
        'cordial', 'educado', 'prestativo', 'rápido', 'eficiente', 'atencioso', 
        'excelente', 'ótimo', 'gentil', 'resolveu', 'encantador', 'muito bom'
    ]
}

# Padrões regex complexos para detecção
REGEX_PATTERNS = {
    # Padrões existentes
    'entrega_rapida': r'cheg[a-z]+ (((rápido|antes|no prazo|cedo)))',
    'entrega_atrasada': r'cheg[a-z]+ (((tard[a-z]+|atras[a-z]+|depois)))',
    'demora_excessiva': r'(demor[a-z]+|atras[a-z]+) (muito|bastante|demasiado|mais)',
    'prazo_excedido': r'mais (do que|tempo|que) (o )?prometido',
    'espera_tempo': r'esper[a-z]+ .{1,20} (para )?receber',
    'tempo_especifico': r'esper[a-z]+ (um|uma|dois|três|quatro|cinco|\d+) (dia|dias|semana|semanas|mês|meses)',
    
    # Padrões para produto
    'produto_qualidade': r'(produto|item) (é|está|parece) (bom|ruim|excelente|ótimo|péssimo|terrível)',
    'produto_problema': r'produto (quebrado|danificado|defeituoso|com defeito|estragado|ruim|péssimo)',
    'recebido_problema': r'receb[a-z]+ .{0,20} (produto|item) .{0,10} (quebrado|danificado|defeituoso|com defeito)',
    'perfume_problematico': r'(perfume|aroma|cheiro|fragrância) (não )?(fixa|dura|cheira)',
    'duracao_perfume': r'(não )?(dura|fixa) (nem )?(\d+)? ?horas',
    'produto_falsificado': r'(falsificado|falso|não original|não é original|não é autêntico|fake)',
    
    # Novos padrões para tempo de atendimento
    'atendimento_demorado': r'(atendimento|atender|atendente|resposta) .{0,15} (demor[a-z]+|atras[a-z]+)',
    'tempo_atendimento': r'(demor[a-z]+|espera[a-z]+|aguard[a-z]+) .{0,20} (atend|resposta|contato)',
    'tempo_resposta': r'(tempo|hora[s]?|minuto[s]?) .{0,15} (atend|resposta|contato)',
    'tempo_espera_especifico': r'(esper[a-z]+|aguard[a-z]+) (um|uma|dois|três|quatro|cinco|\d+) (hora|horas|minuto|minutos)',
    'demora_excessiva_atendimento': r'(demor[a-z]+) (\d+) (hora|horas|minuto|minutos) (para|pra) (atender|responder|retornar)',
    'interacao_atendente': r'(atendente|vendedor|funcionário) .{0,20} (foi|estava|ficou) .{0,15} (grosseiro|educado|atencioso|prestativo)'
}

# Pesos para os diferentes tipos de identificação
WEIGHTS = {
    'unigram': 1.0,          # Peso para palavras únicas exatas
    'bigram': 2.0,           # Peso para expressões compostas
    'partial': 0.5,          # Peso para menções parciais
    'aspect_name_direct': 4.0, # Peso para menção direta ao nome do aspecto (ex: "atendimento")
    
    # Pesos especiais para padrões regex de entrega
    'entrega_rapida': 2.0,
    'entrega_atrasada': 2.0,
    'demora_excessiva': 2.5,
    'prazo_excedido': 2.0,
    'espera_tempo': 3.0,
    'tempo_especifico': 3.0,
    
    # Pesos especiais para padrões regex de produto
    'produto_qualidade': 2.0,
    'produto_problema': 3.0,
    'recebido_problema': 3.0,
    'perfume_problematico': 3.0,
    'duracao_perfume': 3.0,
    'produto_falsificado': 4.0,
    
    # Pesos para os novos padrões de atendimento
    'atendimento_demorado': 3.5,
    'tempo_atendimento': 3.5,
    'tempo_resposta': 3.0,
    'tempo_espera_especifico': 3.5,
    'demora_excessiva_atendimento': 4.0,
    'interacao_atendente': 2.5
}

# Dicionário de sub-aspectos enriquecido
SUB_ASPECTOS = {
    'produto': {
        'qualidade': [
            'qualidade', 'bem feito', 'mal feito', 'durável', 'resistente', 'frágil',
            'boa qualidade', 'má qualidade', 'baixa qualidade', 'alto padrão',
            'acabamento', 'durabilidade', 'robusto', 'consistente', 'confiável',
            'ótimo', 'excelente', 'ruim', 'péssimo', 'superior', 'inferior',
            'de primeira', 'barato demais', 'cara demais', 'premium', 'econômico'
        ],
        'funcionalidade': [
            'funciona', 'não funciona', 'funcionalidade', 'recursos', 'capacidade',
            'funções', 'utilidade', 'versátil', 'prático', 'útil', 'eficiente',
            'ineficiente', 'eficaz', 'desempenho', 'performance', 'potência',
            'serve', 'atende', 'cumpre', 'bug', 'travou', 'crash', 'instável'
        ],
        'design': [
            'design', 'aparência', 'visual', 'estética', 'bonito', 'feio', 'elegante',
            'moderno', 'ultrapassado', 'estilo', 'formato', 'cor', 'tamanho',
            'dimensões', 'compacto', 'volumoso', 'discreto', 'chamativo',
            'ergonômico', 'confortável', 'pesado', 'leve', 'textura', 'acabamento'
        ],
        'conectividade': [
            'conexão', 'conectividade', 'porta', 'portas', 'cabo', 'entrada', 'saída',
            'usb', 'hdmi', 'wifi', 'bluetooth', 'wireless', 'sem fio', 'compatível',
            'incompatível', 'adaptador', 'interface', 'conector', 'ethernet',
            'lan', 'rápido', 'lento', 'queda de sinal', 'emparelhamento'
        ],
        'instalação': [
            'instalação', 'instalar', 'configurar', 'configuração', 'setup', 'montar',
            'montagem', 'fácil de instalar', 'difícil de instalar', 'complicado',
            'intuitivo', 'instruções', 'manual', 'guia', 'suporte técnico',
            'plug and play', 'drivers', 'atualização de firmware'
        ],
        'embalagem': [
            'embalagem', 'caixa', 'proteção', 'amassado', 'plástico bolha',
            'intacto', 'danificado', 'sustentável', 'reciclável', 'excessivo',
            'ecofriendly', 'inadequado', 'poluente'
        ]
    },
    'entrega': {
        'prazo': [
            'prazo', 'tempo', 'atrasou', 'atrasada', 'demorou', 'rápido',
            'rápida', 'pontual', 'dentro do prazo', 'no prazo', 'antes do prazo',
            'expectativa', 'prometido', 'urgente', 'demorado', 'tenso', 'estendido'
        ],
        'rastreamento': [
            'rastreamento', 'rastrear', 'código', 'acompanhar', 'atualização',
            'status', 'informação', 'localização', 'paradeiro', 'desatualizado',
            'tracking', 'sem sinalização'
        ],
        'embalagem': [
            'embalagem', 'pacote', 'caixa', 'protegido', 'danificado', 'amassado',
            'lacre', 'etiqueta', 'identificação', 'fragilidade', 'vazio'
        ],
        'transportadora': [
            'transportadora', 'entregador', 'motoboy', 'correios', 'sedex',
            'empresa de entrega', 'serviço de entrega', 'logística', 'parceiro',
            'terceirizado', 'despreparo', 'atraso injustificado'
        ],
        'custos_envio': [
            'frete', 'custo frete', 'frete grátis', 'valor do frete',
            'frete caro', 'frete barato', 'taxas extras', 'taxa de entrega'
        ]
    },
    'atendimento': {
        'cordialidade': [
            'educado', 'gentil', 'atencioso', 'cordial', 'prestativo',
            'grosseiro', 'mal educado', 'rude', 'antipático', 'simpático',
            'amigável', 'hostil', 'frio', 'acolhedor'
        ],
        'resolução': [
            'resolução', 'resolver', 'solução', 'problema', 'resolvido',
            'pendência', 'pendente', 'solucionado', 'demora para resolver',
            'atraso na solução', 'efetivo', 'ineficaz'
        ],
        'tempo_resposta': [
            'tempo de resposta', 'rápido', 'demorou', 'espera', 'aguardar',
            'retorno', 'feedback', 'contato', 'imediato', 'lento', 'última milha'
        ],
        'canais': [
            'chat', 'email', 'telefone', 'whatsapp', 'sac', 'ouvidoria',
            'site', 'app', 'aplicativo', 'central', 'rede social', 'físico',
            'presencial', 'bot', 'autoatendimento'
        ],
        'empatia': [
            'entendimento', 'empático', 'educação', 'colaboração',
            'simpatia', 'conexão', 'consideração', 'escuta ativa'
        ]
    },
    'preço': {
        'custo_beneficio': [
            'custo-benefício', 'vale a pena', 'não vale a pena',
            'compensador', 'vantajoso', 'desvantajoso', 'retorno', 'investimento',
            'custo efetivo', 'valor agregado'
        ],
        'valor': [
            'caro', 'barato', 'acessível', 'inacessível', 'econômico', 'em conta',
            'alto', 'baixo', 'justo', 'injusto', 'abusivo', 'razoável',
            'inflacionado', 'subvalorizado'
        ],
        'desconto': [
            'desconto', 'promoção', 'oferta', 'cupom', 'redução', 'abatimento',
            'black friday', 'liquidação', 'saldos', 'ofertas relâmpago',
            'promoção contínua'
        ],
        'formas_pagamento': [
            'pagamento', 'cartão', 'boleto', 'pix', 'parcelado', 'à vista',
            'financiamento', 'crédito', 'débito', 'juros', 'taxas', 'min_parcela'
        ]
    },
    'empresa': {
        'reputação': [
            'reputação', 'confiável', 'não confiável', 'confiança', 'séria',
            'credibilidade', 'fidedigna', 'duvidosa', 'suspeita', 'respeitável',
            'renomada', 'anônima'
        ],
        'transparência': [
            'transparência', 'honesto', 'desonesto', 'claro', 'obscuro',
            'informação', 'comunicação', 'política', 'termos', 'contrato',
            'letra miúda'
        ],
        'pos_venda': [
            'pós-venda', 'garantia', 'suporte', 'assistência', 'manutenção',
            'reparo', 'conserto', 'substituição', 'troca', 'devolução',
            'cobertura', 'prazo de garantia', 'SLA'
        ],
        'responsabilidade_social': [
            'ético', 'sustentável', 'social', 'ambiental', 'compliance',
            'doação', 'inclusão', 'diversidade', 'governança'
        ]
    }
}
