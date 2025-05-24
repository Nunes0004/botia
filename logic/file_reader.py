# logic/file_reader.py
import os
import zipfile

def read_uploaded_file(file):
    filepath = file.name

    def safe_read(path):
        try:
            with open(path, "r", encoding="utf-8", errors="ignore") as f:
                return f.read()
        except Exception as e:
            return f"Erro ao ler {path}: {str(e)}"

    if filepath.endswith(".zip"):
        output = ""
        try:
            with zipfile.ZipFile(filepath, "r") as zip_ref:
                zip_ref.extractall("unzipped")
                for name in zip_ref.namelist():
                    if name.endswith((".py", ".txt")):
                        full_path = os.path.join("unzipped", name)
                        if os.path.isfile(full_path):
                            content = safe_read(full_path)
                            output += f"\n# ===== {name} =====\n{content}\n"
            return output.strip() if output else "Nenhum arquivo .py ou .txt encontrado no ZIP."
        except zipfile.BadZipFile:
            return "Erro: Arquivo ZIP inválido ou corrompido."
    elif filepath.endswith((".py", ".txt")):
        return safe_read(filepath)
    else:
        return "Tipo de arquivo não suportado. Envie .py, .txt ou .zip."
