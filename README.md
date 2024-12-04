# **Treinamento de Agentes Cooperativos com Reinforcement Learning**

---

## **Visão Geral do Projeto**

Este projeto explora o treinamento de dois agentes **baseados em LLM (Large Language Models)** de forma cooperativa utilizando **Reinforcement Learning (RL)**. Os agentes, **EncoderAgent** e **ReviewerAgent**, colaboram para gerar, revisar e refinar códigos Python, garantindo alta qualidade nas saídas. O objetivo principal é criar um sistema no qual os agentes possam melhorar seu desempenho iterativamente, utilizando técnicas de RL com foco em métricas como correção de código, eficiência e aderência a padrões de codificação.

---

## **Componentes Principais**

### 1. **Agentes**
- **EncoderAgent**:
  - Gera código Python para tarefas fornecidas, utilizando um modelo LLM pré-treinado.
  - Segue padrões como PEP 8 para garantir qualidade.
  - Realiza tarefas como limpeza de dados, cálculos estatísticos e algoritmos de aprendizado de máquina.

- **ReviewerAgent**:
  - Revisa o código gerado pelo EncoderAgent.
  - Identifica erros de sintaxe, falhas lógicas e sugere melhorias.
  - Avalia a qualidade geral do código.

### 2. **Ambiente**
- **CooperativeEnvironment**:
  - Simula interações entre os agentes.
  - Monitora métricas como recompensas, escores de qualidade e violações utilizando ferramentas como `pylint`, `flake8` e `mypy`.
  - Oferece feedback para que os agentes melhorem seu desempenho ao longo dos episódios.

### 3. **Recompensas**
  - Avalia a qualidade do código gerado e revisado com base em:
    - Pontuação do `pylint` (aderência ao PEP 8).
    - Violações de estilo (`flake8`).
    - Violações de tipagem (`mypy`).
    - Feedback do ReviewerAgent.
  - Penaliza erros como funções grandes, excesso de loops aninhados ou imports não utilizados.
  - Recompensa boas práticas, estruturas eficientes e correção lógica.

---

## **Configuração e Instalação**

### **Dependências**
1. PyTorch (com suporte a CUDA para aceleração por GPU, se disponível).
2. Biblioteca Transformers da Hugging Face.
3. Ferramentas adicionais:
   - `pylint`
   - `flake8`
   - `mypy`

