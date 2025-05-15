# Configuração de Modelos Locais (LLM) para Análise de Sentimentos

Este documento contém instruções para configurar e usar modelos de linguagem locais (LLM) como DeepSeek e LLaMA em substituição ao GPT para análise de sentimentos e detecção de sarcasmo.

## Índice

1. [Requisitos](#requisitos)
2. [Modelos Recomendados](#modelos-recomendados)
3. [Instalação](#instalação)
4. [Download de Modelos](#download-de-modelos)
5. [Configuração na Interface](#configuração-na-interface)
6. [Resolução de Problemas](#resolução-de-problemas)

## Requisitos

Para usar modelos locais, você precisará de:

- Python 3.8 ou superior
- Pelo menos 8GB de RAM (16GB recomendado)
- Para modelos maiores ou uso de GPU: NVIDIA GPU com pelo menos 8GB VRAM
- Aproximadamente 5-10GB de espaço em disco para armazenar os modelos

## Modelos Recomendados

### Opção 1: DeepSeek (Recomendado para a maioria dos casos)

- **DeepSeek 7B Base**: Excelente desempenho geral, bom suporte ao português
- **DeepSeek Coder 6.7B**: Boa opção para análise estruturada 

### Opção 2: LLaMA 3 (Recomendado para análise mais contextual)

- **LLaMA 3 8B**: Ótima compreensão contextual, bom para detecção de sarcasmo

### Opção 3: BERTimbau (Opção mais leve, específica para português)

- **BERTimbau Base**: Modelo específico para português desenvolvido pela NeuralMind

## Instalação

### 1. Instale as dependências necessárias

```bash
pip install -r requirements.txt
```

Ou instale manualmente as bibliotecas necessárias:

```bash
pip install transformers torch huggingface_hub sentencepiece
pip install llama-cpp-python  # Para modelos LLaMA
```

### 2. Verificar a instalação

Para verificar se as bibliotecas necessárias estão instaladas corretamente:

```bash
python -c "import transformers; print(f'transformers {transformers.__version__}'); import torch; print(f'torch {torch.__version__}'); import huggingface_hub; print(f'huggingface_hub {huggingface_hub.__version__}')"
```

## Download de Modelos

### Método 1: Script automatizado

Usamos o script `download_models.py` para baixar e configurar os modelos automaticamente:

```bash
# Baixar DeepSeek 7B (opção padrão)
python download_models.py

# Baixar DeepSeek Coder
python download_models.py --modelo deepseek-ai/deepseek-coder-6.7b-instruct

# Baixar LLaMA 3
python download_models.py --modelo meta-llama/Llama-3-8b --llama

# Especificar diretório de saída personalizado
python download_models.py --saida /caminho/para/modelos
```

### Método 2: Download manual via Hugging Face

Você também pode baixar os modelos manualmente do [Hugging Face Hub](https://huggingface.co/):

1. Acesse a página do modelo (ex: [deepseek-ai/deepseek-llm-7b-base](https://huggingface.co/deepseek-ai/deepseek-llm-7b-base))
2. Clique em "Files and versions"
3. Baixe os arquivos do modelo (config.json, model.safetensors, tokenizer.json, etc.)
4. Organize-os em um diretório dentro da pasta `models/`

## Configuração na Interface

1. Inicie o aplicativo:
   ```bash
   python app_modular.py
   ```

2. Abra seu navegador e acesse: `http://localhost:5000`

3. Na barra de navegação, clique em "Configurar LLM Local"

4. Selecione um dos modelos pré-configurados ou insira o caminho para o seu modelo baixado

5. Configure as opções avançadas (se necessário):
   - **Dispositivo**: Auto (recomendado), CPU ou CUDA (para GPU NVIDIA)
   - **Quantização**: Use 8-bit para economizar memória ou 4-bit para máxima economia

6. Clique em "Ativar Modelo"

7. Teste o modelo com alguns textos para verificar seu funcionamento

## Resolução de Problemas

### Problema: Modelo muito grande para RAM disponível

**Solução**: Use quantização para reduzir o tamanho do modelo:
- Selecione "8-bit" ou "4-bit" na opção de quantização

### Problema: Erro "CUDA out of memory"

**Solução**:
- Use um modelo menor
- Ative a quantização 4-bit
- Mude para CPU (mais lento, mas funciona com menos memória)

### Problema: Modelo lento na CPU

**Solução**:
- Use modelos GGUF via llama.cpp que são otimizados para CPU
- Reduza o tamanho do contexto nas configurações avançadas

### Problema: Modelo não encontrado

**Solução**:
- Verifique se o caminho do modelo está correto
- Certifique-se de que o diretório contém todos os arquivos necessários (config.json, model.safetensors, tokenizer.json)

## Comparação com GPT

| Aspecto | Modelos Locais | GPT (OpenAI) |
|---------|---------------|--------------|
| Privacidade | Alta (dados ficam no seu servidor) | Baixa (dados enviados para OpenAI) |
| Custo | Gratuito (após download) | Baseado em tokens/uso |
| Velocidade | Depende do hardware local | Geralmente mais rápido |
| Precisão em PT-BR | Boa (especialmente DeepSeek e LLaMA 3) | Excelente |
| Requisitos | Hardware local significativo | Apenas conexão à internet |

## Referências

- [DeepSeek AI](https://github.com/deepseek-ai)
- [Meta LLaMA 3](https://llama.meta.com/llama-downloads/)
- [Hugging Face Transformers](https://huggingface.co/docs/transformers/index)
- [llama.cpp](https://github.com/ggerganov/llama.cpp) 