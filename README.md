# Projeto de Treinamento de Agentes LLM com Reinforcement Learning

Este projeto tem como objetivo desenvolver um sistema de aprendizado por reforço (Reinforcement Learning) para treinar dois agentes baseados em modelos de linguagem de grande escala (Large Language Models - LLMs). Os agentes cooperam na resolução de tarefas de análise de dados, onde um "Codificador" gera código e um "Revisor" sugere melhorias. O treinamento é conduzido em um ambiente de simulação que avalia o desempenho dos agentes com base na qualidade do código gerado e nas visualizações produzidas.

## Sumário

- [Agentes](#agentes)
- [Ambiente de RL](#ambiente-de-rl)
- [Datasets](#datasets)
- [Tarefas](#tarefas)
- [Algoritmo PPO](#algoritmo-ppo)

## Agentes

O sistema conta com dois agentes principais:

- **Codificador:** Responsável por gerar o código necessário para resolver uma tarefa de análise de dados, como a criação de visualizações (gráficos).
  
  O Codificador utiliza prompts e uma base de conhecimento gerada pela análise de dados fornecidos para gerar um código coerente e funcional. Ele interage com o ambiente para criar gráficos e outras representações analíticas.

- **Revisor:** Avalia o código gerado pelo Codificador e sugere melhorias para corrigir erros, otimizar o desempenho ou garantir que os requisitos da tarefa sejam atendidos.

  O Revisor usa feedback de recompensas fornecido pelo ambiente para ajustar o código com base nos critérios de qualidade, como legibilidade, eficiência e corretude.

Ambos os agentes utilizam modelos pré-treinados da biblioteca `transformers` da Hugging Face, com o "Codificador" e o "Revisor" ajustados para suas funções específicas no ambiente de aprendizado por reforço.

## Ambiente de RL

O ambiente de simulação foi desenvolvido usando a biblioteca `gym`, fornecendo uma plataforma onde os agentes interagem e aprendem com base em feedbacks e recompensas.

- O **Codificador** gera código para resolver uma tarefa específica.
- O **Revisor** faz a revisão e sugere mudanças ou melhorias.
- O processo é repetido até que uma solução final seja submetida ou o número máximo de interações seja alcançado.

### Ações dos Agentes

Cada agente pode tomar três tipos de ação durante o treinamento:

1. **Gerar código (Codificador):** O Codificador cria uma nova porção de código baseada nos dados fornecidos e na descrição da tarefa.
2. **Revisar código (Revisor):** O Revisor avalia o código atual e fornece melhorias ou correções.
3. **Finalizar (Ambos):** O agente decide submeter o código final para avaliação e determinar se a solução está correta.

### Avaliação e Recompensas

O ambiente avalia a solução gerada com base nos seguintes critérios:

- **Completude:** Se todos os componentes obrigatórios da tarefa foram implementados no código.
- **Correção:** Se o código executa sem erros de sintaxe e gera as visualizações ou análises corretas.
- **Qualidade:** O nível de otimização e clareza do código final.

Recompensas positivas são atribuídas por soluções corretas e completas, enquanto penalidades ocorrem por erros de sintaxe, execuções incompletas ou código ineficiente.

## Datasets

Os agentes são treinados com um conjunto de datasets de treino. Estes datasets contêm informações diversas que exigem diferentes abordagens analíticas, como geração de gráficos, relatórios e outros tipos de visualizações.

### Exemplos de datasets usados no treinamento:

- `Walmart.csv`: Dados de vendas de uma rede de varejo.
- `bestsellers_with_categories.csv`: Dados de livros mais vendidos com suas respectivas categorias.
- `sell_house.csv`: Informações sobre preços de venda de imóveis.

Durante o teste, um **dataset de teste** separado é usado para avaliar a capacidade generalizada dos agentes.

### Dataset de Teste:

- `laptop_price.csv`: Contém dados sobre preços de laptops.

## Tarefas

As tarefas são baseadas em desafios de visualização e análise de dados. Cada tarefa define os componentes necessários para uma solução correta e recompensas são atribuídas quando esses componentes são implementados.

### Exemplos de tarefas:

- **Gráficos de linhas (line_chart):** O agente deve gerar um gráfico de linha representando uma relação temporal entre variáveis.
- **Gráficos de dispersão (scatter_plot):** O agente cria um gráfico de dispersão para visualizar correlações entre duas variáveis.
- **Histogramas (histogram):** O agente deve construir um histograma para analisar a distribuição dos dados.

Cada tarefa tem uma descrição clara e um conjunto de requisitos obrigatórios que precisam ser atendidos para que a solução seja considerada válida.

## Algoritmo PPO

Para o treinamento dos agentes, é utilizado o algoritmo **Proximal Policy Optimization (PPO)**, um dos algoritmos mais eficientes e amplamente utilizados em problemas de aprendizado por reforço.

- **PPO** é um algoritmo que busca encontrar o equilíbrio entre a exploração de novas estratégias e a exploração de estratégias já conhecidas.
- Ele ajusta os parâmetros dos agentes com base nas recompensas obtidas, evitando mudanças drásticas que poderiam desestabilizar o aprendizado.
- O treinamento é feito através da biblioteca `stable-baselines3`, que fornece uma implementação eficiente e fácil de usar do PPO.

O PPO é particularmente adequado para este projeto devido à natureza discreta das ações dos agentes e à necessidade de aprender com feedbacks contínuos do ambiente.

