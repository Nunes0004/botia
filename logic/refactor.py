# logic/refactor.py
from logic.generator import tokenizer, model
import torch

def refactor_code(code):
    try:
        prompt = f"# Refatore o seguinte código Python para melhorar clareza e desempenho:\n{code}\n# Código refatorado:\n"
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        outputs = model.generate(
            **inputs,
            max_new_tokens=256,
            do_sample=True,
            temperature=0.3,
            top_k=50,
            top_p=0.95
        )
        result = tokenizer.decode(outputs[0], skip_special_tokens=True)
        return result.strip()
    except Exception as e:
        return f"Erro ao tentar refatorar o código: {str(e)}"
