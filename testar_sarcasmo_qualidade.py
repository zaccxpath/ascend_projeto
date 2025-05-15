#!/usr/bin/env python
# -*- coding: utf-8 -*-

from models.sentiment import SentimentAnalyzer
from models.sarcasm_integration import analisar_texto_com_sarcasmo

def testar_casos_qualidade():
    print('===== TESTES DE SARCASMO EM PROBLEMAS DE QUALIDADE =====')
    
    # Inicializar o analisador de sentimentos
    analyzer = SentimentAnalyzer()
    
    # Lista de exemplos para teste
    exemplos = [
        'O produto quebrou em 2 dias, qualidade excepcional!',  # Exemplo que falhou anteriormente
        'Qualidade incrível, quebrou na primeira semana!',  # Variação com ordem invertida
        'Este celular é excelente, parou de funcionar no segundo dia',  # Variação sem exclamação
        'Comprei um produto de qualidade superior que estragou em 3 dias!',  # Variação com outra estrutura
        'Produto resistente e durável, quebrou quando tirei da caixa!',  # Contradição clara
        'O notebook é muito bom, só travou 10 vezes hoje'  # Outro tipo de problema
    ]
    
    # Analisar cada exemplo
    for i, texto in enumerate(exemplos):
        resultado = analisar_texto_com_sarcasmo(texto, analyzer, True)
        
        print(f'Exemplo {i+1}: {texto}')
        print(f'Sentimento original: {resultado["sentimento_original"]}')
        print(f'Sentimento final: {resultado["sentimento"]}')
        print(f'É sarcasmo: {resultado["sarcasmo"]["detectado"]} (Probabilidade: {resultado["sarcasmo"]["probabilidade"]:.2f})')
        
        # Exibir evidências se for sarcasmo
        if resultado["sarcasmo"]["detectado"]:
            print(f'Evidências: {resultado["evidencias_sarcasmo"][:3]}')
            if "detalhes_sarcasmo" in resultado and "rule" in resultado["detalhes_sarcasmo"]:
                print(f'Score: {resultado["detalhes_sarcasmo"]["rule"].get("score", 0)}')
        
        print()

if __name__ == "__main__":
    testar_casos_qualidade() 