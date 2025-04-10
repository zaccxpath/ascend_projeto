import os
import pandas as pd
import json
import datetime
from app import sentiment_analyzer, data_handler

def criar_feedback_teste():
    """Cria um feedback de teste com aspectos extraídos explicitamente"""
    print("Criando feedback de teste com aspectos...")
    
    # Texto de exemplo que menciona empresa e produto
    texto = "A empresa entregou o produto com qualidade excelente. O atendimento foi ótimo e o preço justo. Recomendo a marca para todos."
    
    # Analisar sentimento e extrair aspectos
    analise = sentiment_analyzer.analisar_sentimento(texto)
    
    # Verificar a análise
    print(f"Sentimento: {analise['sentimento']}")
    print(f"Compound: {analise['compound']}")
    
    # Verificar aspectos
    if 'aspectos' in analise and 'summary' in analise['aspectos']:
        aspectos_detectados = analise['aspectos']['summary'].get('aspects_detected', [])
        aspecto_principal = analise['aspectos']['summary'].get('primary_aspect', None)
        print(f"Aspectos detectados: {aspectos_detectados}")
        print(f"Aspecto principal: {aspecto_principal}")
    else:
        print("Nenhum aspecto detectado na análise!")
    
    # Salvar no histórico
    data_handler.salvar_transcricao(texto, analise)
    print("Feedback salvo com sucesso!")
    
    # Verificar o número total de registros
    csv_path = 'data/analises.csv'
    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path)
        print(f"Total de registros no CSV: {len(df)}")
        print(f"Último registro: {df.iloc[-1].to_dict()}")

if __name__ == "__main__":
    criar_feedback_teste() 