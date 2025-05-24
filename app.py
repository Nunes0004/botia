# app.py
import gradio as gr
from logic.generator import generate_code, tokenizer, model
from logic.fixer import fix_code
from logic.refactor import refactor_code
from logic.file_reader import read_uploaded_file
import torch

def process_file_command(arquivos, prompt):
    if not arquivos:
        return "Nenhum arquivo foi enviado."

    conteudo_total = ""
    for file in arquivos:
        conteudo = read_uploaded_file(file)
        conteudo_total += f"\n# ===== {file.name} =====\n{conteudo}\n"

    comando = f"""
# Abaixo estão os códigos enviados:
{conteudo_total}

# Instrução: {prompt}
# Código resultante:
"""

    inputs = tokenizer(comando, return_tensors="pt", truncation=True, max_length=512).to(model.device)
    outputs = model.generate(
        **inputs,
        max_new_tokens=256,
        temperature=0.3,
        top_k=50,
        top_p=0.95,
        eos_token_id=tokenizer.eos_token_id
    )
    return tokenizer.decode(outputs[0], skip_special_tokens=True).strip()

with gr.Blocks(title="Python Code Assistant", css="#file-output textarea { overflow-y: scroll !important; }") as demo:
    gr.Markdown("# 🧠 Python Code Assistant\nUma IA especializada em programação Python — gere, corrija, refatore e interprete arquivos.")

    with gr.Tab("📌 Gerar Código"):
        with gr.Row():
            prompt = gr.Textbox(label="Descreva o que o código deve fazer", placeholder="Ex: Ler um CSV e plotar um gráfico", lines=3)
        output_gen = gr.Code(label="Código gerado")
        btn_gen = gr.Button("Gerar")
        btn_gen.click(fn=generate_code, inputs=prompt, outputs=output_gen)

    with gr.Tab("🛠️ Corrigir Código"):
        buggy_code = gr.Code(label="Cole o código com erro")
        output_fix = gr.Code(label="Código corrigido")
        btn_fix = gr.Button("Corrigir")
        btn_fix.click(fn=fix_code, inputs=buggy_code, outputs=output_fix)

    with gr.Tab("🧹 Refatorar Código"):
        original_code = gr.Code(label="Cole o código para refatorar")
        output_refac = gr.Code(label="Código refatorado")
        btn_refac = gr.Button("Refatorar")
        btn_refac.click(fn=refactor_code, inputs=original_code, outputs=output_refac)

    with gr.Tab("📂 Interpretar e Comandar Arquivos"):
        file_input = gr.File(
            label="Arraste ou selecione arquivos (.py, .txt, .zip)",
            file_types=[".py", ".txt", ".zip"],
            file_count="multiple"
        )

        file_summary = gr.Textbox(
            label="📄 Resumo dos arquivos enviados",
            lines=15,
            max_lines=1000,
            interactive=False,
            show_copy_button=True
        )

        command_file = gr.Textbox(
            label="💬 O que deseja fazer com os arquivos?",
            placeholder="Ex: Corrija todos e transforme em uma API com FastAPI"
        )

        command_output = gr.Code(label="⚙️ Resultado gerado pela IA")
        btn_command = gr.Button("Executar comando")

        # Resumo automático dos arquivos
        file_input.change(
            fn=lambda files: "\n\n".join(read_uploaded_file(f)[:1000] for f in files),
            inputs=file_input,
            outputs=file_summary
        )

        # Executa comando com base nos arquivos carregados
        btn_command.click(
            fn=process_file_command,
            inputs=[file_input, command_file],
            outputs=command_output
        )

demo.launch()
