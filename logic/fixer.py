# logic/fixer.py
from logic.generator import tokenizer, model
import torch

def fix_code(code):
    try:
        prompt = f"# Corrija o seguinte código Python com erros:\n{code}\n# Código corrigido:\n"
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
        return f"Erro ao tentar corrigir o código: {str(e)}"
