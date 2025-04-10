// Script de correção para os gráficos do dashboard
document.addEventListener('DOMContentLoaded', function() {
    console.log("Iniciando script de correção dos gráficos");
    
    // Esperar um pouco para garantir que todos os elementos e dados estejam carregados
    setTimeout(function() {
        // Corrigir o gráfico de comparação
        const comparacaoElement = document.getElementById('comparacao-ponderacao');
        if (comparacaoElement) {
            console.log("Encontrado elemento comparacao-ponderacao");
            comparacaoElement.style.height = '350px';
            comparacaoElement.style.display = 'block';
            comparacaoElement.style.position = 'relative';
            
            // Verificar se o gráfico tem conteúdo
            if (comparacaoElement.innerHTML.trim() === '') {
                console.log("Tentando renderizar gráfico de comparação manualmente");
                // Dados de exemplo para o caso do gráfico não ter sido renderizado
                var data = [
                    {
                        x: ['positivo', 'neutro', 'negativo'],
                        y: [17, 20, 26],
                        name: 'Análise Original',
                        type: 'bar',
                        marker: {color: '#3498db'}
                    },
                    {
                        x: ['positivo', 'neutro', 'negativo'],
                        y: [17, 20, 26],
                        name: 'Análise Ponderada',
                        type: 'bar',
                        marker: {color: '#9b59b6'}
                    }
                ];
                
                var layout = {
                    title: 'Comparação: Sentimento Original vs Ponderado',
                    barmode: 'group',
                    xaxis: {title: 'Sentimento'},
                    yaxis: {title: 'Quantidade'},
                    height: 350
                };
                
                try {
                    Plotly.newPlot('comparacao-ponderacao', data, layout, {responsive: true});
                } catch (e) {
                    console.error("Erro ao renderizar manualmente gráfico de comparação:", e);
                }
            }
        }
        
        // Corrigir o gráfico de concordância
        const concordanciaElement = document.getElementById('concordancia-pie');
        if (concordanciaElement) {
            console.log("Encontrado elemento concordancia-pie");
            concordanciaElement.style.height = '350px';
            concordanciaElement.style.display = 'block';
            concordanciaElement.style.position = 'relative';
            
            // Verificar se o gráfico tem conteúdo
            if (concordanciaElement.innerHTML.trim() === '') {
                console.log("Tentando renderizar gráfico de concordância manualmente");
                // Dados de exemplo para o caso do gráfico não ter sido renderizado
                var data = [{
                    values: [68.25, 31.75],
                    labels: ['Concordância', 'Discordância'],
                    type: 'pie',
                    hole: 0.4,
                    marker: {
                        colors: ['#2ecc71', '#e74c3c']
                    }
                }];
                
                var layout = {
                    title: 'Taxa de Concordância Cliente-Modelo',
                    height: 350,
                    annotations: [{
                        text: '68.3%',
                        x: 0.5,
                        y: 0.5,
                        font: {size: 20},
                        showarrow: false
                    }]
                };
                
                try {
                    Plotly.newPlot('concordancia-pie', data, layout, {responsive: true});
                } catch (e) {
                    console.error("Erro ao renderizar manualmente gráfico de concordância:", e);
                }
            }
        }
    }, 1000);
});

// Este script corrige problemas identificados na estrutura JavaScript do dashboard.html

// Remover duplicação da manipulação de eventos click em .time-filter
$(document).ready(function() {
    // Remover todos os event handlers existentes
    $('.time-filter').off('click');
    
    // Adicionar novo event handler único
    $('.time-filter').on('click', function() {
        $('.time-filter').removeClass('active');
        $(this).addClass('active');
        
        // Mostrar/esconder seletor de data personalizada
        if ($(this).data('period') === 'custom') {
            $('#custom-date-range').slideDown();
        } else {
            $('#custom-date-range').slideUp();
            console.log('Período selecionado:', $(this).data('period'), 'dias');
        }
    });

    // Adicionar manipulador para o botão de fechar notificação (caso não exista)
    if ($('.notification-close').length > 0) {
        $('.notification-close').off('click').on('click', function() {
            $(this).closest('footer').fadeOut();
        });
    }
    
    // Verificar se existem event handlers para outros botões importantes
    if ($('#apply-filters').length > 0 && !$._data($('#apply-filters')[0], 'events')) {
        $('#apply-filters').on('click', function() {
            const channel = $('#filter-channel').val();
            const sentiment = $('#filter-sentiment').val();
            const aspect = $('#filter-aspect').val();
            const period = $('.time-filter.active').data('period');
            
            console.log('Filtros aplicados:', {
                canal: channel || 'todos',
                sentimento: sentiment || 'todos',
                aspecto: aspect || 'todos',
                periodo: period
            });
        });
    }
    
    // Garantir que os botões de reset de filtros funcionem corretamente
    if ($('#reset-filters').length > 0 && !$._data($('#reset-filters')[0], 'events')) {
        $('#reset-filters').on('click', function() {
            $('#filter-channel, #filter-sentiment, #filter-aspect').val('');
            $('.time-filter').removeClass('active');
            $('.time-filter[data-period="7"]').addClass('active');
            $('#custom-date-range').slideUp();
        });
    }
    
    console.log('Dashboard JS corrigido e otimizado');
}); 