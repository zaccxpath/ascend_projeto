#!/usr/bin/env python
# -*- coding: utf-8 -*-

import requests
import json
import sys

def testar_api_sarcasmo(texto):
    """Testa a API de detecção de sarcasmo com o texto fornecido"""
    
    url = "http://localhost:5000/api/analisar-sarcasmo"
    headers = {"Content-Type": "application/json"}
    
    payload = {
        "texto": texto
    }
    
    try:
        # Enviar a requisição POST
        response = requests.post(url, json=payload, headers=headers)
        
        # Verificar se a requisição foi bem-sucedida
        if response.status_code == 200:
            # Extrair e formatar a resposta
            resultado = response.json()
            print("Resultado da API:")
            print(json.dumps(resultado, indent=2, ensure_ascii=False))
            
            # Mostrar informações relevantes
            analise = resultado.get("analise", {})
            sarcasmo = analise.get("sarcasmo", {})
            
            print("\n=== Análise de Sarcasmo ===")
            print(f"Texto: {texto}")
            print(f"Sentimento original: {analise.get('sentimento')}")
            print(f"Confiança: {analise.get('compound', 0.0):.4f}")
            print(f"\nSarcasmo detectado: {sarcasmo.get('detectado')}")
            print(f"Probabilidade: {sarcasmo.get('probabilidade', 0.0):.4f}")
            print(f"Classificação: {sarcasmo.get('classificacao', 'baixo')}")
            
            # Exibir componentes da análise
            componentes = sarcasmo.get("componentes", {})
            print("\nComponentes da análise:")
            print(f"  Regras: {componentes.get('regras', 0.0):.4f}")
            print(f"  Contradição: {componentes.get('contradicao', 0.0):.4f}")
            print(f"  Modelo: {componentes.get('modelo', 0.0):.4f}")
            
            # Exibir evidências, se houver
            evidencias = sarcasmo.get("evidencias", [])
            if evidencias:
                print("\nEvidências encontradas:")
                for i, ev in enumerate(evidencias[:5], 1):
                    print(f"  {i}. {ev}")
            
            # Exibir resultado final
            resultado_final = resultado.get("resultado", {})
            print("\n=== Resultado Final ===")
            print(f"Sentimento original: {resultado_final.get('sentimento_original')}")
            print(f"Sentimento final: {resultado_final.get('sentimento_final')}")
            print(f"Invertido: {resultado_final.get('invertido')}")
            
        else:
            print(f"Erro na requisição: {response.status_code}")
            print(f"Resposta: {response.text}")
    
    except Exception as e:
        print(f"Erro ao fazer a requisição: {e}")

if __name__ == "__main__":
    # Verificar se foi fornecido um texto via linha de comando
    if len(sys.argv) > 1:
        texto = " ".join(sys.argv[1:])
    else:
        # Usar exemplos pré-definidos
        exemplos = [
            "O produto quebrou em 2 dias, qualidade excepcional!",
            "Nossa, que atendimento incrível, esperei apenas 2 horas na fila!",
            "Este produto é simplesmente maravilhoso, recomendo a todos!",
            "Claro, com certeza vou comprar de novo nessa loja que sempre atrasa."
        ]
        
        print("=== Exemplos para teste ===")
        for i, exemplo in enumerate(exemplos, 1):
            print(f"{i}. {exemplo}")
        
        try:
            escolha = int(input("\nEscolha um exemplo (1-4) ou digite 0 para inserir seu próprio texto: "))
            if escolha == 0:
                texto = input("Digite o texto para análise: ")
            elif 1 <= escolha <= len(exemplos):
                texto = exemplos[escolha-1]
            else:
                texto = exemplos[0]
                print(f"Escolha inválida. Usando o exemplo 1.")
        except ValueError:
            texto = exemplos[0]
            print(f"Entrada inválida. Usando o exemplo 1.")
    
    print(f"\nAnalisando: '{texto}'\n")
    testar_api_sarcasmo(texto) 