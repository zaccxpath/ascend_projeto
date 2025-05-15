# Criação de um arquivo Python contendo feedbacks categorizados de varejo
# Baseado em pesquisas de comportamento do consumidor brasileiro

# Categorias de feedbacks que uma varejista pode receber, baseado em dados do mercado

feedbacks_varejo = {
    # 1. Problemas de Entrega e Logística
    "entrega_logistica": [
        "Meu pedido deveria ter chegado há 10 dias, e até agora nada. Isso é o que chamam de entrega expressa?",
        "Recebi metade dos produtos que pedi. A caixa parecia que tinha sido aberta no caminho.",
        "O rastreamento indica que meu produto está parado no mesmo lugar há uma semana.",
        "Paguei frete premium para receber em 24h e o pedido chegou após 5 dias úteis.",
        "O entregador arremessou o pacote por cima do meu portão, produto danificado.",
        "A transportadora tentou entregar em horário comercial quando solicitei expressamente entrega noturna.",
        "Comprei um produto frágil que chegou totalmente quebrado por falta de proteção na embalagem.",
        "Minha encomenda foi entregue no endereço errado e ninguém resolve.",
        "O status de rastreamento nunca é atualizado, parece que meu pedido sumiu.",
        "A caixa chegou completamente molhada, comprometendo todos os produtos."
    ],

    # 2. Problemas com Produto
    "produto": [
        "Comprei um eletrônico que parou de funcionar após três dias de uso.",
        "O produto que recebi é completamente diferente do que foi anunciado no site.",
        "A qualidade deste item é muito inferior ao que mostram nas fotos do site.",
        "Comprei uma roupa tamanho M que parece um tamanho PP, tabela de medidas incorreta.",
        "O produto veio com a embalagem original violada e sem manual de instruções.",
        "A cor real do produto é totalmente diferente da mostrada no anúncio.",
        "O eletrodoméstico veio com um plugue de padrão diferente do brasileiro.",
        "O material descrito como 'premium' é visivelmente de baixa qualidade.",
        "Recebi um produto de modelo anterior ao anunciado, e não a versão mais recente.",
        "Comprei um conjunto de 6 peças e vieram apenas 5 na caixa."
    ],

    # 3. Problemas de Atendimento ao Cliente
    "atendimento": [
        "Estou tentando contato há 3 dias e ninguém responde meus e-mails ou mensagens.",
        "O chat online sempre indica 'todos os atendentes estão ocupados' seja qual for o horário.",
        "O atendente me tratou com total indiferença quando relatei meu problema.",
        "Fui transferido(a) para 5 setores diferentes e ninguém conseguiu resolver meu caso.",
        "A resposta automática não resolveu meu problema e não há opção de falar com humano.",
        "Recebi informações contraditórias de diferentes atendentes sobre a mesma questão.",
        "O prazo de resposta de 24h não foi cumprido, estou esperando há 5 dias.",
        "O SAC só funciona em horário comercial, impossível para quem trabalha no mesmo período.",
        "Não há nenhum canal de atendimento telefônico, só robôs de mensagem.",
        "Depois que a venda foi feita, parece que vocês simplesmente desaparecem."
    ],

    # 4. Problemas de Cobrança e Pagamento
    "cobranca_pagamento": [
        "Fui cobrado duas vezes pelo mesmo produto e não consigo o estorno.",
        "O desconto promocional não foi aplicado na finalização da compra.",
        "A parcela no cartão veio com valor diferente do que foi aprovado na compra.",
        "Cancelei o pedido imediatamente após a compra mas fui cobrado mesmo assim.",
        "Vocês cobram taxa de frete adicional que não estava indicada durante a compra.",
        "Meu pagamento foi aprovado há semanas, mas o pedido continua como 'aguardando pagamento'.",
        "Comprei com cupom de desconto válido que foi recusado sem explicação.",
        "O preço anunciado na promoção era diferente do cobrado no checkout.",
        "Paguei pelo PIX mas o sistema não reconheceu meu pagamento automaticamente.",
        "A opção de parcelamento prometida no anúncio não estava disponível na finalização."
    ],

    # 5. Problemas com Site/Aplicativo
    "site_app": [
        "O site travou no momento do pagamento e fui cobrado sem receber confirmação.",
        "O aplicativo fecha sozinho toda vez que tento finalizar uma compra.",
        "Impossível filtrar produtos adequadamente, os filtros não funcionam.",
        "As imagens dos produtos não carregam corretamente no aplicativo.",
        "Tentei comprar pelo celular mas o botão de finalizar compra não responde.",
        "As descrições dos produtos estão incompletas ou têm informações contraditórias.",
        "Impossível fazer login na minha conta, sistema sempre dá erro.",
        "O carrinho de compras esvazia sozinho após alguns minutos de navegação.",
        "As avaliações negativas dos produtos parecem ter sido removidas do site.",
        "O site é extremamente lento, demora minutos para carregar cada página."
    ],
    
    # 6. Promoções e Publicidade Enganosa
    "promocoes_propaganda": [
        "O desconto anunciado como 'Black Friday' é o mesmo preço regular de semanas atrás.",
        "Recebi um cupom exclusivo por e-mail que não funciona no site.",
        "A promoção 'compre 1 leve 2' tinha letras miúdas que limitavam a produtos específicos.",
        "O produto anunciado como 'última unidade' voltou ao estoque normal no dia seguinte.",
        "A propaganda mostrava funcionalidades que o produto real não possui.",
        "O banner no site anunciava frete grátis, mas na finalização havia cobrança.",
        "O site mostra o produto como disponível, mas após o pagamento informam que está em falta.",
        "A oferta 'imperdível' tinha um valor maior que o da concorrência.",
        "A publicidade prometia brinde que não veio com minha compra.",
        "O desconto alardeado como 'o maior do ano' era de apenas 5%."
    ],

    # 7. Problemas com Trocas e Devoluções
    "trocas_devolucoes": [
        "Solicitei a troca há 30 dias e ainda não recebi o produto substituto.",
        "Devolvi o produto dentro do prazo, mas o reembolso não foi realizado.",
        "A política de trocas no site é diferente da informada pelo atendente.",
        "Precisei pagar frete para devolver um produto com defeito de fábrica.",
        "O processo de devolução é extremamente burocrático e confuso.",
        "Não aceitaram a troca mesmo estando dentro do prazo legal de arrependimento.",
        "Troquei por um produto mais caro e fui cobrado pelo frete, que deveria ser grátis.",
        "O produto de troca veio com o mesmo defeito do anterior.",
        "Vocês exigem a embalagem original para troca, mas ela foi danificada na entrega.",
        "Solicitei devolução e vocês estão oferecendo apenas crédito na loja, não o reembolso."
    ],

    # 8. Feedbacks Positivos
    "positivos": [
        "Estou impressionado com a rapidez da entrega, chegou antes do prazo!",
        "O atendimento por chat foi excepcional, resolveram meu problema em minutos.",
        "A qualidade do produto superou minhas expectativas, vale cada centavo.",
        "O processo de troca foi muito simples e rápido, sem burocracia.",
        "Adorei a embalagem sustentável e a preocupação com o meio ambiente.",
        "O site é muito intuitivo, consegui encontrar e comprar facilmente o que precisava.",
        "A promoção era real, economizei muito em comparação a outras lojas.",
        "O SAC resolveu meu problema imediatamente, sem transferir para outros setores.",
        "Recebi exatamente o que estava nas imagens, com todas as características descritas.",
        "A política de frete grátis para compras acima de R$100 faz toda diferença!"
    ],

    # 9. Sugestões de Melhoria
    "sugestoes": [
        "Seria ótimo ter a opção de agendar a entrega para um horário específico.",
        "Poderiam implementar um sistema de cashback para clientes frequentes.",
        "Sugiro incluírem mais fotos dos produtos, mostrando diferentes ângulos.",
        "Vocês poderiam oferecer a opção de embalagem para presente em todas as compras.",
        "O site ficaria melhor se tivesse a funcionalidade de comparar produtos lado a lado.",
        "Seria útil ter uma seção de perguntas frequentes mais completa no site.",
        "Sugiro criarem um programa de fidelidade com pontos acumulativos.",
        "Poderiam disponibilizar vídeos demonstrativos dos produtos em funcionamento.",
        "O aplicativo poderia ter uma função de escaneamento de código de barras para busca rápida.",
        "Seria interessante ter uma opção de compra recorrente com desconto para itens de uso contínuo."
    ],
    
    # 10. Experiência na Loja Física
    "loja_fisica": [
        "As filas nos caixas são sempre enormes, precisam contratar mais funcionários.",
        "O vendedor foi extremamente prestativo e conhecia bem os produtos.",
        "A loja estava desorganizada, com produtos fora do lugar e difíceis de encontrar.",
        "Não havia ninguém disponível para me auxiliar quando precisei de informações.",
        "Os preços na loja física são mais caros que no site de vocês.",
        "O ambiente da loja é agradável, com boa climatização e música ambiente.",
        "Os provadores estavam todos ocupados, com fila de espera de mais de 20 minutos.",
        "A sinalização dentro da loja é confusa, difícil achar as seções que procurava.",
        "Excelente iniciativa ter espaço kids para as crianças enquanto os pais fazem compras.",
        "O horário de funcionamento estendido aos sábados é muito conveniente."
    ]
}

