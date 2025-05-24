# logic/generator.py
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

# Modelo poderoso, mas exige cuidado com limite de tokens
model_id = "stabilityai/stable-code-3b"

# Carregamento otimizado
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
    device_map="auto"
)

def generate_code(prompt):
    try:
        # Instrução formatada para guiar o modelo
        formatted_prompt = f"# Escreva um código Python que faça o seguinte:\n# {prompt}\n"

        # Tokenização com truncamento seguro para evitar overflow
        inputs = tokenizer(
            formatted_prompt,
            return_tensors="pt",
            truncation=True,
            max_length=512  # entrada limitada para evitar travamento
        ).to(model.device)

        # Geração com finalização forçada via EOS token
        outputs = model.generate(
            **inputs,
            max_new_tokens=256,  # reduzido para caber nos limites de GPU
            do_sample=True,
            temperature=0.3,
            top_k=50,
            top_p=0.95,
            eos_token_id=tokenizer.eos_token_id  # 🚨 Essencial para evitar loop eterno
        )

        # Decodifica e retorna o texto limpo
        result = tokenizer.decode(outputs[0], skip_special_tokens=True)
        return result.strip()

    except Exception as e:
        return f"Erro ao gerar código: {str(e)}"
