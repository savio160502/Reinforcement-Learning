# train.py

from agentes import Coder, Revisor
from ambiente import AnalysisEnvironment
import pandas as pd
from typing import List, Dict

class ActorCriticTrainer:
    """
    Classe que implementa o treinamento usando o padrão ator-crítico.
    """
    def __init__(self, coder: Coder, revisor: Revisor, environment: AnalysisEnvironment, learning_rate: float = 0.01):
        """
        Inicializa o treinador com os agentes e o ambiente.
        
        :param coder: Instância do agente Coder.
        :param revisor: Instância do agente Revisor.
        :param environment: Instância do ambiente de análise.
        :param learning_rate: Taxa de aprendizado para atualização das políticas.
        """
        self.coder = coder
        self.revisor = revisor
        self.environment = environment
        self.learning_rate = learning_rate
        self.rewards_history: List[float] = []

    def train(self, tasks: List[Dict[str, Any]], epochs: int = 10):
        """
        Executa o processo de treinamento por um número definido de épocas.
        
        :param tasks: Lista de tarefas contendo descrições e variáveis específicas.
        :param epochs: Número de épocas de treinamento.
        """
        for epoch in range(epochs):
            print(f"===== Época {epoch + 1}/{epochs} =====")
            total_reward = 0
            for task in tasks:
                print(f"\nTarefa: {task['description']}")
                
                # Passo 1: Coder gera o código
                code = self.coder.generate_code(
                    task_description=task['description'],
                    problem_type=self.environment.problem_type,
                    variables=task['variables']
                )
                
                print("Código gerado pelo Coder:")
                print(code)
                
                # Passo 2: Ambiente executa e avalia o código
                execution_result = self.environment.execute_code(code)
                
                if not execution_result['success']:
                    print(f"Erro na execução do código: {execution_result['error']}")
                    reward = -1.0  # Penalidade por falha na execução
                else:
                    metrics = self.environment.evaluate_output(execution_result['output'])
                    reward = self.environment.calculate_reward(metrics)
                    print(f"Métricas obtidas: {metrics}")
                    print(f"Recompensa calculada: {reward}")
                
                total_reward += reward
                
                # Passo 3: Revisor fornece feedback se a recompensa for insuficiente
                if reward < 0:
                    feedback = self.revisor.review_code(code, self.environment.problem_type)
                    print("Feedback do Revisor:")
                    print(feedback)
                    
                    # Coder gera código revisado com base no feedback
                    revised_code = self.coder.generate_code(
                        task_description=feedback,
                        problem_type=self.environment.problem_type,
                        variables=task['variables']
                    )
                    
                    print("Código revisado pelo Coder:")
                    print(revised_code)
                    
                    # Reavalia o código revisado
                    execution_result = self.environment.execute_code(revised_code)
                    
                    if execution_result['success']:
                        metrics = self.environment.evaluate_output(execution_result['output'])
                        reward = self.environment.calculate_reward(metrics)
                        print(f"Métricas obtidas após revisão: {metrics}")
                        print(f"Recompensa recalculada: {reward}")
                        total_reward += reward
                    else:
                        print(f"Erro na execução do código revisado: {execution_result['error']}")
                        total_reward += -1.0  # Penalidade adicional
                        
                # Passo 4: Atualiza políticas dos agentes (simulação)
                self.update_policies(reward)
            
            average_reward = total_reward / len(tasks)
            self.rewards_history.append(average_reward)
            print(f"\nRecompensa média na época {epoch + 1}: {average_reward}")

    def update_policies(self, reward: float):
        """
        Atualiza as políticas dos agentes com base na recompensa recebida.
        (Nota: Implementação simplificada para demonstração)
        
        :param reward: Recompensa obtida na interação atual.
        """
        # Em uma implementação real, aqui ajustaríamos os pesos do modelo com base na recompensa.
        # Para este exemplo, apenas registramos a recompensa.
        print(f"Atualizando políticas com base na recompensa: {reward}")



"""Explicação Detalhada:
Inicialização

A classe ActorCriticTrainer é inicializada com instâncias dos agentes Coder e Revisor, além do ambiente e uma taxa de aprendizado definida.
Mantém um histórico de recompensas para monitorar o progresso ao longo das épocas.
train

Recebe uma lista de tarefas e o número de épocas para treinamento.
Para cada tarefa:
Geração de Código: O Coder gera o código baseado na descrição da tarefa e nas variáveis específicas.
Execução e Avaliação: O ambiente executa o código e avalia os resultados, calculando a recompensa.
Revisão e Ajuste: Se a recompensa for negativa, o Revisor fornece feedback, e o Coder gera uma versão revisada do código, que é reavaliada.
Atualização de Políticas: As políticas dos agentes são atualizadas com base na recompensa obtida (simulado neste exemplo).
Após todas as tarefas de uma época, calcula e registra a recompensa média.
update_policies

Função simplificada que representa a atualização das políticas dos agentes com base na recompensa.
Em uma implementação completa, este método incluiria lógica para ajustar os parâmetros dos modelos de linguagem.
Observações:

Esta implementação fornece uma estrutura clara para o treinamento, mas não implementa algoritmos de aprendizado por reforço complexos. Para uma abordagem mais aprofundada, seria necessário integrar técnicas específicas de RL e possivelmente treinar modelos customizados."""


------------------------------------------------------------------------------------------------------
# Tentando usar Equação de Belmann
from agentes import Coder, Revisor
from ambiente import AnalysisEnvironment
from typing import List, Dict, Any

class ActorCriticTrainer:
    """
    Classe que implementa o treinamento usando o padrão ator-crítico com a Equação de Bellman.
    """
    def __init__(self, coder: Coder, revisor: Revisor, environment: AnalysisEnvironment, learning_rate: float = 0.01, gamma: float = 0.99):
        """
        Inicializa o treinador com os agentes e o ambiente.
        
        :param coder: Instância do agente Coder.
        :param revisor: Instância do agente Revisor.
        :param environment: Instância do ambiente de análise.
        :param learning_rate: Taxa de aprendizado para atualização das políticas.
        :param gamma: Fator de desconto para a Equação de Bellman.
        """
        self.coder = coder
        self.revisor = revisor
        self.environment = environment
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.rewards_history: List[float] = []
        self.value_memory: List[float] = []  # Memória para armazenar valores de recompensa

    def train(self, tasks: List[Dict[str, Any]], epochs: int = 10):
        """
        Executa o processo de treinamento por um número definido de épocas.
        
        :param tasks: Lista de tarefas contendo descrições e variáveis específicas.
        :param epochs: Número de épocas de treinamento.
        """
        for epoch in range(epochs):
            print(f"===== Época {epoch + 1}/{epochs} =====")
            total_reward = 0
            for task in tasks:
                print(f"\nTarefa: {task['description']}")
                
                # Passo 1: Coder gera o código
                code = self.coder.generate_code(
                    task_description=task['description'],
                    problem_type=self.environment.problem_type,
                    variables=task['variables']
                )
                
                print("Código gerado pelo Coder:")
                print(code)
                
                # Passo 2: Ambiente executa e avalia o código
                execution_result = self.environment.execute_code(code)
                
                if not execution_result['success']:
                    print(f"Erro na execução do código: {execution_result['error']}")
                    reward = -1.0  # Penalidade por falha na execução
                else:
                    metrics = self.environment.evaluate_output(execution_result['output'])
                    reward = self.environment.calculate_reward(metrics)
                    print(f"Métricas obtidas: {metrics}")
                    print(f"Recompensa calculada: {reward}")
                
                total_reward += reward
                self.value_memory.append(reward)
                
                # Passo 3: Revisor fornece feedback se a recompensa for insuficiente
                if reward < 0:
                    feedback = self.revisor.review_code(code, self.environment.problem_type)
                    print("Feedback do Revisor:")
                    print(feedback)
                    
                    # Coder gera código revisado com base no feedback
                    revised_code = self.coder.generate_code(
                        task_description=feedback,
                        problem_type=self.environment.problem_type,
                        variables=task['variables']
                    )
                    
                    print("Código revisado pelo Coder:")
                    print(revised_code)
                    
                    # Reavalia o código revisado
                    execution_result = self.environment.execute_code(revised_code)
                    
                    if execution_result['success']:
                        metrics = self.environment.evaluate_output(execution_result['output'])
                        reward = self.environment.calculate_reward(metrics)
                        print(f"Métricas obtidas após revisão: {metrics}")
                        print(f"Recompensa recalculada: {reward}")
                        total_reward += reward
                        self.value_memory.append(reward)
                    else:
                        print(f"Erro na execução do código revisado: {execution_result['error']}")
                        total_reward += -1.0  # Penalidade adicional
                        self.value_memory.append(-1.0)
                        
                # Passo 4: Atualiza políticas dos agentes
                self.update_policies(reward)
            
            average_reward = total_reward / len(tasks)
            self.rewards_history.append(average_reward)
            print(f"\nRecompensa média na época {epoch + 1}: {average_reward}")

    def calculate_value(self, reward: float, next_value: float) -> float:
        """
        Calcula o valor do estado atual usando a Equação de Bellman.
        
        :param reward: Recompensa atual.
        :param next_value: Valor estimado do próximo estado.
        :return: Valor calculado do estado atual.
        """
        return reward + self.gamma * next_value

    def update_policies(self, reward: float):
        """
        Atualiza as políticas dos agentes com base na recompensa recebida.
        
        :param reward: Recompensa obtida na interação atual.
        """
        # Verifica se há valores anteriores na memória para calcular o próximo valor
        next_value = self.value_memory[-1] if len(self.value_memory) > 1 else 0
        
        # Calcula o valor do estado atual
        value = self.calculate_value(reward, next_value)
        
        # Simulação da atualização de políticas (substitua por lógica de RL)
        adjusted_reward = reward + self.learning_rate * (value - reward)
        print(f"Atualizando políticas com recompensa ajustada: {adjusted_reward}")
        
        # Atualize a memória com o valor ajustado
        self.value_memory[-1] = adjusted_reward

"""Resumo das Otimizações:
Equação de Bellman: Foi adicionada para calcular o valor esperado dos estados, considerando as recompensas futuras esperadas.
Memória de Valores: Introduzida para armazenar os valores de recompensa em cada estado, permitindo a atualização das políticas com base em recompensas ajustadas.
Atualização de Políticas: Melhorada para ajustar as recompensas dos agentes usando uma taxa de aprendizado e a diferença entre o valor calculado e a recompensa recebida.
Essa abordagem oferece um processo mais robusto de aprendizado,
 permitindo que os agentes ajustem suas ações não só com base na recompensa imediata, mas também levando em consideração as recompensas futuras esperadas, o que é fundamental em problemas de decisão sequencial."""