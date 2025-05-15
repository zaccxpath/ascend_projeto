/**
 * Script de correções para o dashboard ASCEND
 * Este script corrige vários problemas identificados no dashboard.html
 */

$(document).ready(function() {
    console.log("Aplicando correções ao dashboard ASCEND...");
    
    // 1. Problema: Duplicidade na manipulação de eventos dos filtros de tempo
    // Solução: Remover todos os event handlers existentes e adicionar um único handler
    $('.time-filter').off('click'); // Remove todos os event handlers anteriores
    
    // Adiciona um único event handler para os filtros de tempo
    $('.time-filter').on('click', function() {
        $('.time-filter').removeClass('active');
        $(this).addClass('active');
        
        // Mostrar/esconder seletor de data personalizada
        if ($(this).data('period') === 'custom') {
            $('#custom-date-range').slideDown();
        } else {
            $('#custom-date-range').slideUp();
            console.log('Período selecionado:', $(this).data('period'), 'dias');
            
            // Aqui poderia ser implementada a lógica real de filtragem por período
            // Exemplo: atualizarDadosPorPeriodo($(this).data('period'));
        }
    });
    
    // 2. Problema: Botão de notificação sem funcionalidade
    // Solução: Adicionar evento de click para fechar o rodapé com notificação
    $('.notification-close').off('click').on('click', function() {
        $(this).closest('footer').fadeOut();
    });
    
    // 3. Problema: Erros em templates Jinja que podem afetar os gráficos
    // Solução: Monitorar e tratar erros JavaScript durante a execução
    window.onerror = function(message, source, lineno, colno, error) {
        console.warn('Erro detectado:', {
            message: message,
            source: source,
            line: lineno,
            column: colno
        });
        
        return true; // Previne que o erro apareça no console novamente
    };
    
    // 4. Verificar se os gráficos estão presentes e tentar recuperá-los
    function verificarGraficos() {
        console.log("Verificando status dos gráficos...");
        
        // Lista de IDs de gráficos principais
        var graficosIds = [
            'sentimento-pie', 
            'sentimento-tempo', 
            'palavras-top', 
            'aspectos-radar', 
            'sentimento-por-aspecto',
            'comparacao-ponderacao',
            'concordancia-pie'
        ];
        
        // Cria um mapeamento entre IDs dos elementos HTML e chaves dos dados
        var idParaChave = {
            'sentimento-pie': 'sentimento_pie',
            'sentimento-tempo': 'sentimento_tempo',
            'palavras-top': 'palavras_top',
            'aspectos-radar': 'aspectos_radar',
            'sentimento-por-aspecto': 'sentimento_por_aspecto',
            'comparacao-ponderacao': 'comparacao_ponderacao',
            'concordancia-pie': 'concordancia_pie'
        };
        
        // Verifica cada gráfico
        graficosIds.forEach(function(id) {
            var container = document.getElementById(id);
            if (!container) {
                console.log(`Container para gráfico ${id} não encontrado no DOM`);
                return;
            }
            
            // Obter a chave correspondente para acessar dados
            var chave = idParaChave[id];
            
            // Verificar se o container está vazio
            if (container.innerHTML.trim() === '' || $(container).find('.plotly-notifier').length > 0) {
                console.log(`Tentando renderizar gráfico ${id} com chave ${chave}`);
                
                // Verificar se temos dados para o gráfico no objeto global do dashboard
                if (window.dados_dashboard && window.dados_dashboard.graficos && window.dados_dashboard.graficos[chave]) {
                    var dados = window.dados_dashboard.graficos[chave];
                    console.log(`Dados encontrados para ${id}, tentando renderizar`);
                    
                    try {
                        // Tornar os contêineres de gráficos visíveis
                        container.style.height = '300px';
                        container.style.display = 'block';
                        
                        // Renderizar o gráfico com os dados disponíveis
                        Plotly.newPlot(id, dados.data, dados.layout, {responsive: true});
                        console.log(`Gráfico ${id} renderizado com sucesso a partir dos dados JSON`);
                    } catch (e) {
                        console.error(`Erro ao renderizar gráfico ${id}:`, e);
                    }
                } else {
                    console.log(`Nenhum dado encontrado para ${id} na chave ${chave}`);
                }
            } else {
                console.log(`Gráfico ${id} já está renderizado`);
            }
        });
    }
    
    // 5. Problema: Problemas de interação com os botões de exportação e ampliação
    // Solução: Reforçar a implementação das funções
    window.exportarGrafico = function(id) {
        try {
            let gd = document.getElementById(id);
            if (gd) {
                Plotly.downloadImage(gd, {
                    format: 'png',
                    filename: 'ascend-' + id + '-' + new Date().toISOString().split('T')[0]
                });
            } else {
                console.error('Elemento não encontrado para exportação:', id);
            }
        } catch (e) {
            console.error('Erro ao exportar gráfico:', e);
        }
    };
    
    window.ampliarGrafico = function(id) {
        try {
            let gd = document.getElementById(id);
            if (gd && gd.data) {
                Plotly.newPlot(
                    gd, 
                    gd.data, 
                    Object.assign({}, gd.layout, {
                        width: window.innerWidth * 0.9,
                        height: window.innerHeight * 0.8
                    }), 
                    {responsive: true}
                );
            } else {
                console.error('Elemento não encontrado ou sem dados para ampliação:', id);
            }
        } catch (e) {
            console.error('Erro ao ampliar gráfico:', e);
        }
    };
    
    // Adicionar dados do dashboard como variável global para acesso em funções de correção
    if (typeof dados_json !== 'undefined') {
        try {
            // Verificar se já não está disponível globalmente (adicionado no template)
            if (!window.dados_dashboard) {
                window.dados_dashboard = JSON.parse(dados_json);
                console.log("Dados do dashboard disponibilizados globalmente");
            }
            
            // Chamar verificação de gráficos para garantir que todos estão renderizados
            setTimeout(verificarGraficos, 500);
        } catch (e) {
            console.error("Erro ao analisar dados_json:", e);
        }
    }
    
    console.log('Correções aplicadas com sucesso');
}); 