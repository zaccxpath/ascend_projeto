# ASCEND - Análise de Sentimentos com Normalização de Pesos

O ASCEND (Análise de Sentimentos e Classificação Em Narrativas Digitais) é uma aplicação para análise de sentimentos em textos em português utilizando o modelo XLM-RoBERTa ou uma análise de fallback baseada em regras.

## Funcionalidades Principais

- Análise de sentimentos com XLM-RoBERTa (Hugging Face)
- Processamento de áudio para transcrição e análise
- Dashboard interativo para visualização de dados
- Página de diagnóstico do sistema
- Análise ponderada com normalização de pesos

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

## Uso

### Interface Web

A aplicação disponibiliza uma interface web completa em `http://localhost:5000` após a inicialização.

- **Página principal**: Gravação e análise de áudio
- **Dashboard**: Visualização de estatísticas e gráficos
- **Relatório Ponderado**: Análise de sentimentos com pesos normalizados
- **Diagnóstico**: Verificação do status do sistema

### Linha de Comando

A geração de relatórios também pode ser executada via linha de comando:

```bash
# Relatório com pesos padrão (cliente=30%, modelo=80%)
python app.py --gerar-relatorio

# Relatório com pesos personalizados
python app.py --gerar-relatorio 40 60
```

## Detalhes Técnicos

O cálculo da média ponderada normalizada é realizado pela classe `AnaliseEstatistica`, que:

1. Normaliza os pesos para somarem 100%
2. Converte sentimentos textuais para valores numéricos (Positivo=3, Neutro=2, Negativo=1)
3. Calcula a média ponderada usando os pesos normalizados
4. Converte o valor numérico de volta para categoria de sentimento

## Arquivos Gerados

A análise ponderada gera dois arquivos principais:

- `data/relatorio_ponderado.csv`: Relatório detalhado com todos os registros analisados
- `data/estatisticas_ponderacao.json`: Resumo estatístico da análise com metadados

## Requisitos

- Python 3.7+
- Flask
- PyTorch
- Transformers
- NLTK
- Pandas
- Plotly
- SpeechRecognition
- PyAudio (para captura de áudio)
- Huggingface_hub (para download de modelos) 