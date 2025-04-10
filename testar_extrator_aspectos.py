from app import AspectExtractor

def testar_extrator():
    """Testa o extrator de aspectos com diferentes exemplos"""
    print("=" * 50)
    print("TESTANDO EXTRATOR DE ASPECTOS")
    print("=" * 50)
    
    # Criar uma instância do extrator
    extrator = AspectExtractor()
    
    # Exemplo que falhou na análise original
    texto1 = "O produto é bom, mas a entrega demorou muito mais do que o prometido. Isso deixou muito a desejar."
    # Outros exemplos para teste
    texto2 = "A empresa entregou o produto com qualidade excelente. O atendimento foi ótimo."
    texto3 = "Atendimento péssimo! O produto veio com defeito e não quiseram trocar."
    
    # Análise de sentimento mock para testes
    analise_sentimento_mock = {
        "compound": -0.5834,
        "sentimento": "negativo"
    }
    
    # Testar com o texto que falhou
    print("\n1. Testando texto que falhou anteriormente:")
    print(f"Texto: '{texto1}'")
    resultado1 = extrator.extrair_aspectos(texto1, analise_sentimento_mock)
    print(f"Aspectos detectados: {resultado1['summary']['aspects_detected']}")
    print(f"Aspecto principal: {resultado1['summary']['primary_aspect']}")
    
    # Testar com um texto positivo com múltiplos aspectos
    print("\n2. Testando texto positivo com múltiplos aspectos:")
    print(f"Texto: '{texto2}'")
    analise_sentimento_mock["compound"] = 0.8963
    analise_sentimento_mock["sentimento"] = "positivo"
    resultado2 = extrator.extrair_aspectos(texto2, analise_sentimento_mock)
    print(f"Aspectos detectados: {resultado2['summary']['aspects_detected']}")
    print(f"Aspecto principal: {resultado2['summary']['primary_aspect']}")
    
    # Testar com um texto negativo sobre atendimento e produto
    print("\n3. Testando texto negativo sobre atendimento:")
    print(f"Texto: '{texto3}'")
    analise_sentimento_mock["compound"] = -0.9466
    analise_sentimento_mock["sentimento"] = "negativo"
    resultado3 = extrator.extrair_aspectos(texto3, analise_sentimento_mock)
    print(f"Aspectos detectados: {resultado3['summary']['aspects_detected']}")
    print(f"Aspecto principal: {resultado3['summary']['primary_aspect']}")
    
    # Mostrar detalhes do primeiro resultado (que anteriormente falhou)
    if resultado1["summary"]["aspects_detected"]:
        print("\nDetalhes do primeiro exemplo (anteriormente falhou):")
        for aspecto in resultado1["summary"]["aspects_detected"]:
            relevancia = resultado1["aspectos"][aspecto]["relevance"]
            sentimento = resultado1["aspectos"][aspecto]["sentiment"]
            mencoes = resultado1["aspectos"][aspecto]["mentions"]
            print(f"Aspecto '{aspecto}': relevância={relevancia}, sentimento={sentimento}, menções={mencoes}")
    else:
        print("\nNenhum aspecto foi detectado no primeiro exemplo! Isso indica que o problema persiste.")

if __name__ == "__main__":
    testar_extrator() 