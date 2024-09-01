# Coder-Revisor RL Training Framework

## Descrição do Projeto

Este projeto visa desenvolver um framework de treinamento para dois agentes de inteligência artificial: o **Coder** e o **Revisor**. O objetivo é permitir que esses agentes aprendam a gerar e revisar código de forma autônoma, utilizando técnicas de aprendizado por reforço, especificamente o método ator-crítico.

### O Que Será Feito

1. **Desenvolvimento do Agente Coder**: Um agente responsável por gerar código baseado em descrições de tarefas e variáveis específicas.

2. **Desenvolvimento do Agente Revisor**: Um agente que revisa o código gerado pelo Coder, fornecendo feedback e sugerindo melhorias.

3. **Criação de um Ambiente de Treinamento**: Um ambiente simulado onde o código gerado pelo Coder será executado e avaliado, e onde as recompensas serão calculadas com base no desempenho do código.

4. **Treinamento com a Equação de Bellman**: Utilização da Equação de Bellman para calcular o valor dos estados e otimizar as políticas dos agentes, levando em consideração as recompensas imediatas e futuras.

5. **Implementação do Ciclo de Treinamento**: Um processo iterativo onde o Coder gera código, o ambiente executa e avalia, o Revisor fornece feedback (se necessário), e as políticas dos agentes são ajustadas para maximizar as recompensas ao longo do tempo.

