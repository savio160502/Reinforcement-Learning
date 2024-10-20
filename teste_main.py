# Usaremos a biblioteca stable-baselines3 para realizar o treinamento dos agentes.
# Usaremos o algoritmo PPO (Proximal Policy Optimization), que é adequado para esse tipo de problema

from teste_agente import LLMAgent
from teste_ambiente import DataAnalysisEnv
from stable_baselines3 import PPO
import torch

# Verifica se a GPU está disponível e escolhe o dispositivo correto
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def train_agents():
    """
    Treina os agentes codificador e revisor no ambiente de análise de dados.
    """
    # Instanciando os agentes LLM
    codificador = LLMAgent(agent_type="codificador")
    revisor = LLMAgent(agent_type="revisor")
    
    # Instanciando o ambiente de RL
    dataset = "Walmart.csv"
    #expected_output = "Solução correta de visualização de dados"
    expected_output = {
    "type": "visualization",
    "description": "Gráficos de linha mostrando as vendas ao longo do tempo, separados por categorias de produtos, com uma linha de tendência e intervalos de confiança.",
    "components": [
        {"chart_type": "line", "title": "Vendas ao Longo do Tempo"},
        {"chart_type": "bar", "title": "Vendas por Categoria"},
        {"chart_type": "heatmap", "title": "Análise de Sazonalidade"}
    ]
}
    env = DataAnalysisEnv(dataset, expected_output, codificador, revisor)
    
    # Configurando o modelo de aprendizado por reforço PPO para usar GPU, se disponível
    model = PPO("MlpPolicy", env, verbose=1, device=device)

    # Treinando o modelo por 10.000 passos
    model.learn(total_timesteps=10000)
    
    # Salvando o modelo treinado
    model.save("ppo_agents")
    
    # Carregar o modelo para uso posterior
    # model = PPO.load("ppo_agents")

# teste do resultado final
def test_trained_model():
    """
    Testa o modelo treinado em um novo episódio.
    """
    # Carrega o modelo treinado
    model = PPO.load("ppo_agents")
    
    # Instanciando os agentes
    codificador = LLMAgent(agent_type="codificador")
    revisor = LLMAgent(agent_type="revisor")
    
    # Instanciando o ambiente
    dataset = "sales_data.csv"
    expected_output = "Solução correta de visualização de dados"
    env = DataAnalysisEnv(dataset, expected_output, codificador, revisor)
    
    obs = env.reset()
    
    for _ in range(10):
        action, _states = model.predict(obs)
        obs, reward, done, info = env.step(action)
        print(f"Ação tomada: {action}, Recompensa: {reward}")
        
        if done:
            print("Episódio finalizado.")
            break

import time

if __name__ == "__main__":
    start_time = time.time()
    train_agents()
    end_time = time.time()
    tempo = end_time - start_time
    print(f"Tempo para o treinamento : {tempo/60}") # minutos

    # Teste o modelo treinado
    #test_trained_model()

