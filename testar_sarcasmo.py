#!/usr/bin/env python
# -*- coding: utf-8 -*-

from models.sentiment import SentimentAnalyzer
from models.sarcasm_integration import analisar_texto_com_sarcasmo

def testar_exemplos():
    print('===== EXEMPLOS PARA TESTE =====')
    
    # Inicializar o analisador de sentimentos
    analyzer = SentimentAnalyzer()
    
    # Lista de exemplos para teste
    exemplos = [
        'Nossa, que atendimento incrível, esperei apenas 2 horas na fila!',  # Sarcasmo claro
        'O atendimento foi rápido e eficiente, não esperei muito',  # Sem sarcasmo
        'O produto chegou em 3 dias, muito bom',  # Potencialmente ambíguo
        'Esperei 50 minutos para ser atendido, excelente serviço!',  # Sarcasmo sobre tempo de espera
        'O app é fantástico, só demorou 5 dias para entregar!'  # Sarcasmo com só/apenas
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
        
        print()

if __name__ == "__main__":
    testar_exemplos() 