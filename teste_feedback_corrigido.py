import os
import pandas as pd
import json
import datetime
import sys
import logging
from app import sentiment_analyzer, data_handler

# Configurar logging para exibir na console
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
logger.addHandler(logging.StreamHandler(sys.stdout))

def criar_feedback_teste():
    """Cria um feedback de teste com o texto que falhou anteriormente"""
    print("\n" + "=" * 50)
    print("TESTANDO FEEDBACK COM ASPECTOS CORRIGIDOS")
    print("=" * 50)
    
    # Texto que falhou na detecção de aspectos
    texto = "O produto é bom, mas a entrega demorou muito mais do que o prometido. Isso deixou muito a desejar."
    print(f"\nTexto de teste: '{texto}'")
    
    try:
        # Analisar sentimento e extrair aspectos
        print("\nAnalisando sentimento e extraindo aspectos...")
        analise = sentiment_analyzer.analisar_sentimento(texto)
        
        # Verificar a análise
        print(f"\nResultado da análise:")
        print(f"Sentimento: {analise['sentimento']}")
        print(f"Compound: {analise['compound']}")
        
        # Verificar aspectos
        if 'aspectos' in analise and 'summary' in analise['aspectos']:
            aspectos_detectados = analise['aspectos']['summary'].get('aspects_detected', [])
            aspecto_principal = analise['aspectos']['summary'].get('primary_aspect', None)
            print(f"\nAspectos detectados: {aspectos_detectados}")
            print(f"Aspecto principal: {aspecto_principal}")
            
            # Verificar detalhes dos aspectos
            if aspectos_detectados:
                print("\nDetalhes dos aspectos:")
                for aspecto in aspectos_detectados:
                    relevancia = analise['aspectos']['aspectos'][aspecto]['relevance']
                    sentimento = analise['aspectos']['aspectos'][aspecto]['sentiment']
                    mencoes = analise['aspectos']['aspectos'][aspecto]['mentions']
                    print(f"  Aspecto '{aspecto}': relevância={relevancia}, sentimento={sentimento}, menções={mencoes}")
            else:
                print("\nNenhum aspecto específico detectado!")
        else:
            print("\nO campo 'aspectos' não está presente na análise!")
            print(f"Campos disponíveis: {list(analise.keys())}")
        
        # Salvar no histórico
        print("\nSalvando feedback no histórico...")
        data_handler.salvar_transcricao(texto, analise)
        print("Feedback salvo com sucesso!")
        
        # Verificar o número total de registros
        csv_path = 'data/analises.csv'
        if os.path.exists(csv_path):
            df = pd.read_csv(csv_path)
            print(f"\nTotal de registros no CSV: {len(df)}")
            
            # Verificar o último registro salvo
            ultimo = df.iloc[-1].to_dict()
            print("\nDados do último registro salvo:")
            print(f"  Texto: {ultimo.get('texto', '')[:50]}...")
            print(f"  Sentimento: {ultimo.get('sentimento', '')}")
            print(f"  Aspectos detectados: {ultimo.get('aspectos_detectados', '')}")
            print(f"  Aspecto principal: {ultimo.get('aspecto_principal', '')}")
            
            print("\nO extrator de aspectos está funcionando corretamente!" if ultimo.get('aspectos_detectados', '') else "\nO extrator de aspectos ainda NÃO está funcionando corretamente!")
    except Exception as e:
        print(f"\nERRO: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    criar_feedback_teste()
    print("\nTeste concluído.") 