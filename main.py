# Este arquivo implementa o fluxo de treinamento usando Reinforcement Learning.
# Ele orquestra a interação entre os agentes e o ambiente.

from agente import Coder, Revisor
from ambiente import DataAnalysisEnvironment

def main():
    # Inicializa os agentes e o ambiente
    problem_description = "Clean and analyze sales data."
    coder = Coder(model="llama3.1", problem_description=problem_description)
    revisor = Revisor(model="llama3.1", problem_description=problem_description)
    
    # Supondo que temos um dataset e uma solução correta para o problema
    dataset = "sales_data.csv"
    correct_solution = "Correct analysis and cleaned data"
    environment = DataAnalysisEnvironment(dataset, correct_solution)

    # Loop de treinamento
    for _ in range(100):  # Número de iterações de treinamento
        # Agente codificador gera código
        task_description = "Write a Python script to clean and analyze the data."
        generated_code = coder.generate_code(task_description)

        # Avalia o código gerado
        reward = environment.calculate_reward(generated_code, correct_solution)

        # Se houver penalidade, o revisor intervém
        if reward < 0:
            feedback = revisor.review_code(generated_code)
            # Agente codificador revisa o código com base no feedback
            generated_code = coder.generate_code(feedback)
        
        # A cada iteração, atualize as políticas dos agentes com base na recompensa recebida

    # Finaliza o treinamento e avalia a solução final
    final_reward = environment.calculate_reward(generated_code, correct_solution)
    print(f"Final Reward: {final_reward}")

if __name__ == "__main__":
    main()
