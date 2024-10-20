# Definindo os agentes Codificador e Revisor

from transformers import AutoModelForCausalLM, AutoTokenizer

class LLMAgent:
    def __init__(self, model_name="gpt2", agent_type="codificador"):
        """
        Inicializa o agente LLM com o modelo GPT-2.
        
        Parameters:
        model_name (str): Nome do modelo a ser utilizado (GPT-2 neste exemplo).
        agent_type (str): O tipo do agente ("codificador" ou "revisor").
        """
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name)
        self.agent_type = agent_type
        self.history = []  # Histórico de interações entre os agentes

        # Definir o pad_token_id explicitamente
        self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

    def generate_code(self, prompt):
        """
        Gera um código ou feedback baseado no prompt fornecido.
        
        Parameters:
        prompt (str): O enunciado do problema ou o código a ser revisado.
        
        Returns:
        str: O código gerado pelo agente.
        """
        # Tokenizar o prompt com attention_mask
        inputs = self.tokenizer.encode_plus(
            prompt,
            return_tensors="pt",  # Retorna tensores do PyTorch
            padding=True,  # Adiciona padding automaticamente
            truncation=True,  # Corta se o prompt for muito longo
            max_length=512  # Limita o comprimento dos tokens se necessário
        )

        input_ids = inputs["input_ids"]
        attention_mask = inputs["attention_mask"]

        # Geração com pad_token_id e attention_mask definidos
        outputs = self.model.generate(
            input_ids,
            attention_mask=attention_mask,
            max_new_tokens=100, 
            num_return_sequences=1,
            pad_token_id=self.tokenizer.pad_token_id  # Evita o warning do pad_token_id
        )

        #outputs = self.model.generate(inputs, max_length=150, num_return_sequences=1)
        #outputs = self.model.generate(inputs, max_new_tokens=100, num_return_sequences=1)
    
        result = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        self.history.append(result)  # Armazena a interação no histórico
        return result

    def review_code(self, code):
        """
        Revê o código gerado, sugerindo melhorias.
        
        Parameters:
        code (str): O código gerado pelo agente codificador.
        
        Returns:
        str: O feedback ou revisão do código.
        """
        review_prompt = f"Reveja o seguinte código e sugira melhorias:\n\n{code}"
        return self.generate_code(review_prompt)

# Exemplo de uso:
# codificador = LLMAgent(agent_type="codificador")
# revisor = LLMAgent(agent_type="revisor")
