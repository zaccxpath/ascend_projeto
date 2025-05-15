# ASCEND - Análise de Sentimentos com Normalização de Pesos

O ASCEND (Análise de Sentimentos e Classificação Em Narrativas Digitais) é uma aplicação para análise de sentimentos em textos em português utilizando o modelo XLM-RoBERTa ou uma análise de fallback baseada em regras.

## Estrutura do Projeto

O projeto foi refatorado para seguir uma arquitetura modular, separando as responsabilidades em diferentes arquivos e módulos.

```
├── app.py                     # Aplicativo principal com todas as funcionalidades
├── app_modular.py             # Versão modular do aplicativo
├── feedbacks_varejo_refatorado.py # Implementação refatorada das análises de varejo
├── models/                    # Pacote para modelos e processamento
│   ├── __init__.py
│   ├── sentiment.py           # Análise de sentimentos
│   ├── speech.py              # Reconhecimento de fala
│   ├── data_handler.py        # Manipulação de dados e estatísticas
│   ├── aspect_extractor.py    # Extração de aspectos em textos
│   ├── estatisticas.py        # Funções estatísticas
│   ├── retail_sentiment_enhancer.py # Melhorias para análise de varejo
│   ├── sarcasm_config.py      # Configurações para detecção de sarcasmo
│   ├── sarcasm_detectors.py   # Detectores específicos de sarcasmo
│   ├── sarcasm_factory.py     # Factory pattern para detectores de sarcasmo
│   ├── sarcasm_integration.py # Integração da detecção de sarcasmo
│   ├── detectors.py           # Implementação de detectores 
│   └── cardiffnlp-xlm-roberta/ # Modelo de NLP baixado
├── utils/                     # Pacote para utilitários
│   ├── __init__.py
│   ├── config.py              # Configurações e constantes
│   └── helpers.py             # Funções auxiliares
├── static/                    # Arquivos estáticos (CSS, JS, imagens)
├── templates/                 # Templates HTML
│   ├── index.html             # Página principal
│   ├── dashboard.html         # Dashboard de visualização
│   ├── diagnostico.html       # Página de diagnóstico
│   ├── varejo.html            # Interface para análise de varejo
│   ├── teste_sarcasmo.html    # Interface para testar detecção de sarcasmo
│   ├── relatorio_ponderado.html # Visualização de relatórios ponderados
│   └── outros templates...    # Outros templates da aplicação
├── transcricoes/              # Transcrições de áudio salvas
├── data/                      # Dados processados e estatísticas
│   ├── analises.csv           # Dados de análises de sentimentos
│   ├── relatorio_ponderado.csv # Relatório com pesos normalizados
│   └── estatisticas_ponderacao.json # Estatísticas das análises ponderadas
├── requirements.txt           # Dependências do projeto
└── testar_*.py                # Scripts para testes de funcionalidades
```

## Funcionalidades Principais

- Análise de sentimentos com XLM-RoBERTa (Hugging Face)
- Processamento de áudio para transcrição e análise
- Dashboard interativo para visualização de dados
- Página de diagnóstico do sistema
- Análise ponderada com normalização de pesos
- Análise especializada para feedbacks de varejo
- Detecção avançada de sarcasmo em diferentes contextos

## Módulos Principais

### models/sentiment.py
- `ModeloFallbackSimplificado`: Implementação simplificada para análise de sentimentos quando o modelo XLM-RoBERTa não está disponível
- `SentimentAnalyzer`: Classe principal para análise de sentimentos usando o modelo XLM-RoBERTa ou fallback

### models/speech.py
- `SpeechHandler`: Classe para captura e processamento de áudio do microfone

### models/data_handler.py
- `DataHandler`: Classe para manipulação de dados, salvar/carregar histórico, e gerar gráficos e estatísticas

### models/aspect_extractor.py
- `AspectExtractor`: Classe para identificar aspectos mencionados em textos (produto, empresa, preço, entrega, atendimento)

### models/estatisticas.py
- `AnaliseEstatistica`: Classe com métodos estatísticos para processamento de dados

### models/retail_sentiment_enhancer.py
- `RetailSentimentEnhancer`: Módulo especializado para melhorar a análise de sentimentos em feedbacks de varejo

### Componentes de Detecção de Sarcasmo
- `models/sarcasm_config.py`: Configurações para detecção de sarcasmo
- `models/sarcasm_detectors.py`: Implementações de diferentes detectores de sarcasmo
- `models/sarcasm_factory.py`: Factory para criação de detectores especializados
- `models/sarcasm_integration.py`: Integração da detecção de sarcasmo com o sistema

## Normalização de Pesos em Análise Multifonte

Uma característica importante desta aplicação é a capacidade de combinar diferentes fontes de avaliação (cliente e modelo de IA) usando pesos normalizados.

### O Problema de Pesos Não-Normalizados

Quando os pesos atribuídos a diferentes fontes somam mais que 100%, a média ponderada resultante pode apresentar distorções. Por exemplo, se atribuirmos:

- Peso do cliente: 30%
- Peso do modelo: 80%

A soma desses pesos é 110%, o que pode levar a interpretações errôneas dos dados.

### Nossa Solução: Normalização Proporcional

O sistema automaticamente normaliza os pesos para que somem exatamente 100%, preservando a proporção relativa entre eles:

```
Peso Normalizado (Cliente) = (30 / 110) × 100 ≈ 27,27%
Peso Normalizado (Modelo) = (80 / 110) × 100 ≈ 72,73%
```

### Benefícios

- **Integridade Estatística**: Garante que a influência total seja exatamente 100%
- **Preservação de Proporções**: Mantém a relação relativa entre as fontes
- **Resultados Confiáveis**: Evita viés por excesso ou falta de representação

## Melhorias na Análise de Sentimento para Feedbacks de Varejo

### Categorias de Varejo

O sistema identifica 8 categorias principais de feedbacks de varejo:

- **Entrega e Logística**: Entrega, prazo, frete, transportadora, etc.
- **Produto**: Qualidade, defeito, material, tamanho, cor, etc.
- **Atendimento**: SAC, chat, e-mail, resposta, tempo de atendimento
- **Cobrança e Pagamento**: Cobrança, cartão, PIX, boleto, valor, estorno
- **Site/App**: Plataforma, travamento, erro, login, cadastro, interface
- **Promoções e Propaganda**: Descontos, Black Friday, anúncios, ofertas
- **Trocas e Devoluções**: Política, devolução, reembolso, garantia
- **Loja Física**: Organização, fila, atendimento presencial, ambiente

### Detecção de Sarcasmo Especializada para Varejo

Foi implementada uma detecção de sarcasmo contextualizada para varejo, que identifica:

- Sarcasmo relacionado a tempo de espera
- Sarcasmo sobre qualidade de produtos
- Sarcasmo relacionado a problemas comuns no varejo
- Sarcasmo em feedback sobre atendimento

## Como Usar

### Interface Web

A aplicação disponibiliza uma interface web completa em `http://localhost:5000` após a inicialização.

- **Página principal**: Gravação e análise de áudio
- **Dashboard**: Visualização de estatísticas e gráficos
- **Relatório Ponderado**: Análise de sentimentos com pesos normalizados
- **Diagnóstico**: Verificação do status do sistema
- **Varejo**: Interface específica para análise de feedbacks de varejo em `/varejo`
- **Teste de Sarcasmo**: Interface para testar a detecção de sarcasmo em `/teste_sarcasmo`

### Linha de Comando

A geração de relatórios também pode ser executada via linha de comando:

```bash
# Relatório com pesos padrão (cliente=30%, modelo=80%)
python app.py --gerar-relatorio

# Relatório com pesos personalizados
python app.py --gerar-relatorio 40 60
```

### API para Varejo

Para usar via API, faça uma requisição POST para `/api/analisar-varejo`:

```json
{
  "texto": "Seu feedback de varejo aqui",
  "categoria": "produto" // opcional
}
```

### Scripts de Teste

O projeto inclui vários scripts para testar funcionalidades específicas:

- `testar_sarcasmo.py`: Testa a detecção básica de sarcasmo
- `testar_sarcasmo_qualidade.py`: Testa a detecção de sarcasmo em feedbacks sobre qualidade
- `testar_api_sarcasmo.py`: Testa a API de detecção de sarcasmo
- `testar_extrator_aspectos.py`: Testa a extração de aspectos de feedbacks
- `teste_novo_feedback.py` e `teste_feedback_corrigido.py`: Testes adicionais

## Como Executar

Para executar a versão modular do aplicativo:

```bash
python app_modular.py
```

## Vantagens da Arquitetura Modular

1. **Separação de Responsabilidades**: Cada módulo tem uma função específica e bem definida
2. **Manutenção Simplificada**: Alterações em uma funcionalidade não afetam outras partes do código
3. **Testabilidade**: Facilita a criação de testes unitários para cada módulo
4. **Reutilização de Código**: Os módulos podem ser reutilizados em outros projetos
5. **Legibilidade**: Código mais limpo e organizado, facilitando a compreensão
6. **Escalabilidade**: Facilita a adição de novas funcionalidades sem modificar o código existente

## Arquivos Gerados

A análise ponderada gera dois arquivos principais:

- `data/relatorio_ponderado.csv`: Relatório detalhado com todos os registros analisados
- `data/estatisticas_ponderacao.json`: Resumo estatístico da análise com metadados

## Requisitos

- Python 3.7+
- Flask
- SpeechRecognition
- PyTorch
- Transformers
- NLTK
- Pandas
- Matplotlib
- Seaborn
- Plotly
- PyAudio (para captura de áudio)
- Huggingface_hub (para download de modelos)

## Próximos Passos

1. Verificar todas as funcionalidades para garantir que estão operando corretamente
2. Atualizar os testes existentes para a nova estrutura
3. Criar datasets de treinamento específicos para cada categoria de varejo
4. Implementar mais expressões e padrões baseados em novos exemplos
5. Desenvolver um modelo de fine-tuning específico para varejo usando transformers
6. Expandir a detecção de sarcasmo para mais contextos e idiomas 