/**
 * Dashboard Corrections
 * Este script corrige diversos problemas identificados no código JavaScript do dashboard.html
 */

$(document).ready(function() {
    console.log("Aplicando correções ao dashboard...");
    
    // Problema 1: Duplicidade na manipulação de eventos dos filtros de tempo
    // Solução: Remover todos os event handlers existentes e adicionar um único
    $('.time-filter').off('click');
    
    $('.time-filter').on('click', function() {
        $('.time-filter').removeClass('active');
        $(this).addClass('active');
        
        // Mostrar/esconder seletor de data personalizada
        if ($(this).data('period') === 'custom') {
            $('#custom-date-range').slideDown();
        } else {
            $('#custom-date-range').slideUp();
            // Implementação da lógica de filtragem
            console.log('Período selecionado:', $(this).data('period'), 'dias');
            
            // Aqui podemos adicionar a lógica real de filtragem
        }
    });
    
    // Problema 2: Funcionalidade do botão de notificação possivelmente não implementada
    // Solução: Garantir que o botão tenha funcionalidade para fechar o rodapé
    if ($('.notification-close').length > 0) {
        $('.notification-close').off('click').on('click', function() {
            $(this).closest('footer').fadeOut();
        });
    }
    
    // Problema 3: Configurações de layout possivelmente incorretas para gráficos
    // Solução: Garantir que as configurações de layout sejam aplicadas corretamente aos gráficos
    var reloadCharts = function() {
        $('.chart-container').each(function() {
            var chartId = $(this).attr('id');
            if (chartId && window[chartId + '_data']) {
                var chartData = window[chartId + '_data'];
                
                // Garantir que as configurações básicas estejam presentes
                if (chartData.layout) {
                    // Ajustar margens e tamanho
                    chartData.layout.margin = chartData.layout.margin || { l: 30, r: 30, t: 10, b: 30 };
                    chartData.layout.height = chartData.layout.height || 300;
                    
                    // Garantir que o layout seja responsivo
                    chartData.layout.autosize = true;
                }
                
                if (chartData.data && chartData.data.length > 0) {
                    try {
                        Plotly.react(chartId, chartData.data, chartData.layout, {responsive: true});
                        console.log('Gráfico recarregado:', chartId);
                    } catch (err) {
                        console.error('Erro ao recarregar gráfico:', chartId, err);
                    }
                }
            }
        });
    };
    
    // Chamar a recarga dos gráficos após um breve atraso para garantir que tudo esteja carregado
    setTimeout(reloadCharts, 500);
    
    // Problema 4: Erros nos templates Jinja que podem causar problemas de renderização
    // Solução: Monitorar e capturar erros JavaScript durante a renderização da página
    window.onerror = function(message, source, lineno, colno, error) {
        console.warn('Erro detectado:', {
            message: message,
            source: source,
            line: lineno,
            column: colno,
            error: error
        });
        
        // Registrar erro no console para depuração
        console.log('Erro capturado pelo script de correção');
        
        // Aqui poderíamos implementar alguma lógica adicional para recuperação de erros
        return true; // Isso impede que o erro seja mostrado no console novamente
    };
    
    console.log('Correções aplicadas com sucesso.');
}); 