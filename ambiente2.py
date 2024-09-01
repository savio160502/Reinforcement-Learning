# ambiente.py

import subprocess
import sys
import tempfile
from typing import Dict, Any
from sklearn.metrics import mean_absolute_error, mean_squared_error, accuracy_score, f1_score, silhouette_score
import numpy as np
import pandas as pd

class AnalysisEnvironment:
    """
    Classe que define o ambiente de execução e avaliação para códigos de análise de dados.
    """
    def __init__(self, problem_type: str, dataset: pd.DataFrame, target_column: str = None, n_clusters: int = None):
        """
        Inicializa o ambiente com o tipo de problema e dados necessários.
        
        :param problem_type: Tipo de problema ('regressão', 'classificação', 'clusterização').
        :param dataset: Conjunto de dados a ser utilizado.
        :param target_column: Nome da coluna alvo (para regressão e classificação).
        :param n_clusters: Número de clusters (para clusterização).
        """
        self.problem_type = problem_type.lower()
        self.dataset = dataset
        self.target_column = target_column
        self.n_clusters = n_clusters

    def execute_code(self, code: str) -> Dict[str, Any]:
        """
        Executa o código fornecido em um ambiente seguro e captura os resultados.
        
        :param code: Código Python a ser executado.
        :return: Dicionário com os resultados da execução ou erros ocorridos.
        """
        try:
            # Cria um arquivo temporário para o código
            with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as temp_file:
                temp_file.write(code)
                temp_file_path = temp_file.name
            
            # Executa o código em um subprocesso
            result = subprocess.run(
                [sys.executable, temp_file_path],
                capture_output=True,
                text=True,
                timeout=300  # Tempo máximo de execução em segundos
            )
            
            # Verifica se a execução foi bem-sucedida
            if result.returncode != 0:
                return {'success': False, 'error': result.stderr}
            
            # Captura a saída do código (assumindo que o resultado é impresso no stdout)
            output = result.stdout
            return {'success': True, 'output': output}
        
        except subprocess.TimeoutExpired:
            return {'success': False, 'error': 'Execution timed out.'}
        
        except Exception as e:
            return {'success': False, 'error': str(e)}

    def evaluate_output(self, output: str) -> Dict[str, float]:
        """
        Avalia a saída do código executado usando métricas apropriadas.
        
        :param output: Saída produzida pelo código executado.
        :return: Dicionário com as métricas calculadas.
        """
        try:
            # Converte a saída em formato apropriado (assumindo que é uma string JSON)
            results = eval(output)  # ATENÇÃO: Em produção, use json.loads() e garanta que a saída esteja em JSON
            
            if self.problem_type == 'regressão':
                y_true = self.dataset[self.target_column].values
                y_pred = np.array(results['predictions'])
                mae = mean_absolute_error(y_true, y_pred)
                rmse = mean_squared_error(y_true, y_pred, squared=False)
                return {'MAE': mae, 'RMSE': rmse}
            
            elif self.problem_type == 'classificação':
                y_true = self.dataset[self.target_column].values
                y_pred = np.array(results['predictions'])
                accuracy = accuracy_score(y_true, y_pred)
                f1 = f1_score(y_true, y_pred, average='weighted')
                return {'Accuracy': accuracy, 'F1-Score': f1}
            
            elif self.problem_type == 'clusterização':
                X = self.dataset.values
                labels = np.array(results['labels'])
                silhouette = silhouette_score(X, labels)
                return {'Silhouette Score': silhouette}
            
            else:
                return {'error': 'Tipo de problema desconhecido.'}
        
        except Exception as e:
            return {'error': str(e)}

    def calculate_reward(self, metrics: Dict[str, float]) -> float:
        """
        Calcula a recompensa baseada nas métricas de avaliação.
        
        :param metrics: Dicionário com as métricas calculadas.
        :return: Valor numérico da recompensa.
        """
        if 'error' in metrics:
            return -1.0  # Penalidade por erro na avaliação
        
        # Calcula recompensa com base nas métricas (a lógica pode ser ajustada conforme necessário)
        if self.problem_type == 'regressão':
            reward = -metrics['RMSE']  # Menor RMSE implica maior recompensa
        elif self.problem_type == 'classificação':
            reward = metrics['Accuracy']  # Maior acurácia implica maior recompensa
        elif self.problem_type == 'clusterização':
            reward = metrics['Silhouette Score']  # Maior silhouette score implica maior recompensa
        else:
            reward = -1.0  # Penalidade padrão
        
        return reward


"""Explicação Detalhada:
Inicialização

A classe AnalysisEnvironment é inicializada com o tipo de problema, o conjunto de dados relevante e informações adicionais como a coluna alvo ou o número de clusters, dependendo do problema.
execute_code

O código recebido é salvo em um arquivo temporário e executado em um subprocesso seguro, capturando tanto a saída padrão quanto os erros.
Limita o tempo de execução para evitar loops infinitos ou execuções prolongadas.
Retorna um dicionário indicando sucesso ou falha e inclui a saída ou o erro ocorrido.
evaluate_output

Processa a saída do código executado e calcula métricas apropriadas usando bibliotecas como scikit-learn.
Para simplificação, assume que a saída é uma string contendo um dicionário com os resultados necessários (como previsões ou labels).
Retorna as métricas calculadas ou um erro caso ocorra alguma exceção.
calculate_reward

Converte as métricas de avaliação em uma recompensa numérica que será usada para ajustar as políticas dos agentes.
A lógica de conversão é simples e direta, mas pode ser ajustada conforme as necessidades do projeto.
Nota de Segurança:

O uso de eval na função evaluate_output é potencialmente perigoso se a saída não for confiável. Em um ambiente real, é recomendado usar json.loads após garantir que a saída esteja em formato JSON seguro.
A execução de código arbitrário pode ser arriscada. Certifique-se de implementar medidas de segurança adicionais conforme necessário."""