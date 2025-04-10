// Script de depuração para o dashboard
document.addEventListener('DOMContentLoaded', function() {
    console.log("=== Iniciando depuração do dashboard ===");
    
    // Verificar se o elemento para o gráfico concordancia-pie existe
    var concordanciaPieElement = document.getElementById('concordancia-pie');
    if (concordanciaPieElement) {
        console.log("✅ Elemento concordancia-pie encontrado no DOM");
        console.log("- Largura: " + concordanciaPieElement.offsetWidth);
        console.log("- Altura: " + concordanciaPieElement.offsetHeight);
        console.log("- Visibilidade: " + window.getComputedStyle(concordanciaPieElement).visibility);
        console.log("- Display: " + window.getComputedStyle(concordanciaPieElement).display);
        
        // Visualizar o conteúdo do elemento
        console.log("- HTML interno: " + concordanciaPieElement.innerHTML);
        
        // Forçar visualização
        concordanciaPieElement.style.display = 'block';
        concordanciaPieElement.style.height = '400px';
        concordanciaPieElement.style.border = '2px solid red';
        
        // Tentar renderizar um gráfico de teste
        setTimeout(function() {
            try {
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
                    title: 'Teste - Taxa de Concordância',
                    height: 300
                };
                
                Plotly.newPlot('concordancia-pie', data, layout);
                console.log("✅ Gráfico de teste renderizado com sucesso");
            } catch (e) {
                console.error("❌ Erro ao renderizar gráfico de teste:", e);
            }
        }, 1000);
    } else {
        console.error("❌ Elemento concordancia-pie NÃO encontrado no DOM");
        
        // Verificar todos os contêineres de gráfico
        var chartContainers = document.getElementsByClassName('chart-container');
        console.log("Total de contêineres de gráfico encontrados: " + chartContainers.length);
        
        // Listar IDs de todos os contêineres
        for (var i = 0; i < chartContainers.length; i++) {
            console.log("- Contêiner #" + i + ": ID=" + chartContainers[i].id);
        }
        
        // Criar dinamicamente um contêiner de teste
        var mainContainer = document.querySelector('.container');
        if (mainContainer) {
            var testDiv = document.createElement('div');
            testDiv.innerHTML = `
                <div class="mt-4 p-3" style="border: 2px dashed blue;">
                    <h3>Contêiner de Teste</h3>
                    <div id="test-pie" style="height:300px;"></div>
                </div>
            `;
            mainContainer.appendChild(testDiv);
            
            // Renderizar um gráfico no contêiner de teste
            setTimeout(function() {
                var data = [{
                    values: [70, 30],
                    labels: ['Teste A', 'Teste B'],
                    type: 'pie'
                }];
                
                var layout = {
                    title: 'Gráfico de Teste',
                    height: 300
                };
                
                Plotly.newPlot('test-pie', data, layout);
                console.log("✅ Gráfico de teste criado com sucesso");
            }, 1000);
        }
    }
    
    // Listar todos os scripts carregados
    var scripts = document.getElementsByTagName('script');
    console.log("Total de scripts carregados: " + scripts.length);
    
    // Verificar se Plotly está disponível
    if (typeof Plotly !== 'undefined') {
        console.log("✅ Biblioteca Plotly encontrada, versão: " + Plotly.version);
    } else {
        console.error("❌ Biblioteca Plotly NÃO encontrada!");
    }
    
    console.log("=== Fim da depuração do dashboard ===");
}); 