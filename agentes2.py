# agentes.py

import ollama
from typing import List, Dict

class LLMAgent:
    """
    Classe base para agentes de linguagem natural.
    """
    def __init__(self, model: str = "llama3.1"):
        """
        Inicializa o agente com um modelo de linguagem e memória vazia.
        
        :param model: Nome do modelo de linguagem a ser utilizado.
        """
        self.model = model
        self.memory: List[Dict[str, str]] = []  # Histórico de interações

    def add_to_memory(self, role: str, content: str):
        """
        Adiciona uma nova entrada ao histórico de memória.
        
        :param role: Papel na conversa ('user' ou 'assistant').
        :param content: Conteúdo da mensagem.
        """
        self.memory.append({'role': role, 'content': content})

    def generate(self, prompt: str) -> str:
        """
        Gera uma resposta do modelo de linguagem com base no prompt e na memória atual.
        
        :param prompt: Texto de entrada para o modelo.
        :return: Resposta gerada pelo modelo.
        """
        # Combina a memória e o prompt atual
        messages = self.memory + [{'role': 'user', 'content': prompt}]
        
        # Chama o modelo de linguagem
        response = ollama.chat(model=self.model, messages=messages)
        
        # Adiciona o prompt e a resposta à memória
        self.add_to_memory('user', prompt)
        self.add_to_memory('assistant', response['response'])
        
        return response['response']

class Coder(LLMAgent):
    """
    Agente responsável por gerar código para resolver problemas de análise de dados.
    """
    def __init__(self, model: str = "llama3.1"):
        """
        Inicializa o agente Coder com um prompt base específico.
        
        :param model: Nome do modelo de linguagem a ser utilizado.
        """
        super().__init__(model)
        self.base_prompt = (
            "Você é um desenvolvedor Python e cientista de dados experiente. "
            "Seu trabalho é escrever código bem documentado e otimizado para resolver problemas de análise de dados."
        )

    def generate_code(self, task_description: str, problem_type: str, variables: Dict[str, str]) -> str:
        """
        Gera código Python para uma tarefa específica de análise de dados.
        
        :param task_description: Descrição detalhada da tarefa.
        :param problem_type: Tipo de problema ('regressão', 'classificação', 'clusterização').
        :param variables: Dicionário contendo informações específicas do problema.
        :return: Código Python gerado como uma string.
        """
        # Seleciona o template de prompt adequado com base no tipo de problema
        prompt_templates = {
            'regressão': (
                f"{self.base_prompt} Sua tarefa é construir um modelo de regressão para prever {variables['target_variable']} "
                f"usando as variáveis {variables['input_variables']}. Certifique-se de incluir etapas de pré-processamento, "
                "treinamento e avaliação do modelo usando métricas apropriadas como MAE e RMSE."
            ),
            'classificação': (
                f"{self.base_prompt} Sua tarefa é construir um modelo de classificação para prever {variables['target_variable']} "
                f"usando as características {variables['input_variables']}. Inclua etapas de pré-processamento, treinamento "
                "e avaliação do modelo usando métricas como acurácia e F1-score."
            ),
            'clusterização': (
                f"{self.base_prompt} Sua tarefa é realizar uma análise de clusterização nos dados usando as características "
                f"{variables['input_variables']}. Identifique {variables['n_clusters']} clusters distintos e avalie a qualidade "
                "dos clusters usando métricas como Silhouette Score."
            )
        }
        
        # Monta o prompt final
        prompt = prompt_templates.get(problem_type.lower(), self.base_prompt + " " + task_description)
        
        # Gera o código usando o modelo de linguagem
        code = self.generate(prompt)
        
        return code

class Revisor(LLMAgent):
    """
    Agente responsável por revisar o código gerado pelo Coder e fornecer feedback construtivo.
    """
    def __init__(self, model: str = "llama3.1"):
        """
        Inicializa o agente Revisor com um prompt base específico.
        
        :param model: Nome do modelo de linguagem a ser utilizado.
        """
        super().__init__(model)
        self.base_prompt = (
            "Você é um desenvolvedor Python sênior e cientista de dados experiente. "
            "Seu trabalho é revisar o código de outros desenvolvedores e fornecer feedback detalhado, "
            "incluindo sugestões de melhorias, correções de erros e aderência às melhores práticas."
        )

    def static_analysis(self, code: str) -> str:
        """
        Realiza análise estática do código usando ferramentas como pylint ou flake8.
        
        :param code: Código Python a ser analisado.
        :return: Resultados da análise estática como uma string.
        """
        # Implementação fictícia da análise estática
        # Em uma implementação real, você pode integrar ferramentas como pylint ou flake8 aqui
        analysis_results = "Análise estática concluída. Nenhum erro crítico encontrado, mas há espaço para melhorias em otimização e legibilidade."
        return analysis_results

    def review_code(self, code: str, problem_type: str) -> str:
        """
        Fornece feedback detalhado sobre o código fornecido, incluindo resultados da análise estática.
        
        :param code: Código Python a ser revisado.
        :param problem_type: Tipo de problema associado ao código.
        :return: Feedback detalhado como uma string.
        """
        # Realiza análise estática do código
        analysis_results = self.static_analysis(code)
        
        # Monta o prompt de revisão
        prompt = (
            f"{self.base_prompt} O seguinte código foi escrito para resolver um problema de {problem_type}.\n\n"
            f"Código:\n```python\n{code}\n```\n\n"
            f"Resultados da Análise Estática:\n{analysis_results}\n\n"
            "Forneça um feedback detalhado sobre o código acima, incluindo sugestões de melhorias e correções necessárias."
        )
        
        # Gera o feedback usando o modelo de linguagem
        feedback = self.generate(prompt)
        
        return feedback
    

"""Explicação Detalhada:
LLMAgent

Classe base que define a estrutura comum para os agentes que utilizam modelos de linguagem natural.
Possui uma memória interna (self.memory) que armazena o histórico de interações, permitindo que o modelo mantenha contexto entre as chamadas.
O método generate constrói uma lista de mensagens combinando a memória com o novo prompt, chama o modelo de linguagem através da biblioteca ollama e atualiza a memória com o prompt e a resposta gerada.
Coder

Herda de LLMAgent e especializa-se na geração de código Python para diferentes tarefas de análise de dados.
O método generate_code recebe uma descrição da tarefa, o tipo de problema e variáveis específicas, selecionando o template de prompt adequado e solicitando ao modelo que gere o código correspondente.
Revisor

Também herda de LLMAgent e é responsável por revisar o código gerado pelo Coder.
O método static_analysis simula a execução de ferramentas de análise estática, retornando um resumo dos resultados.
O método review_code monta um prompt que inclui o código original e os resultados da análise estática, solicitando ao modelo que forneça um feedback detalhado sobre possíveis melhorias.
"""
