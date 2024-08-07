import os
import json

def dividir_conteudo(conteudo, max_length=1024):
    """
    Divide o conteúdo em partes menores para evitar exceder o comprimento máximo.
    
    Args:
        conteudo (str): O conteúdo a ser dividido.
        max_length (int): O comprimento máximo de cada parte.
        
    Returns:
        List[str]: Uma lista de partes do conteúdo.
    """
    palavras = conteudo.split()
    partes = []
    parte = ""
    
    for palavra in palavras:
        if len(parte) + len(palavra) + 1 > max_length:
            partes.append(parte)
            parte = palavra
        else:
            parte += " " + palavra if parte else palavra
    
    if parte:
        partes.append(parte)
    
    return partes

def txt_to_json(input_dir, output_path, max_length=1024):
    """
    Converte arquivos .txt em um diretório para um arquivo JSON, dividindo conteúdos grandes em partes menores.
    
    Args:
        input_dir (str): O diretório contendo os arquivos .txt.
        output_path (str): O caminho onde o arquivo JSON será salvo.
        max_length (int): O comprimento máximo de cada parte do conteúdo.
    """
    documents = []
    
    if not os.path.exists(input_dir):
        print(f"O diretório de entrada {input_dir} não existe.")
        return
    
    doc_id = 1
    for filename in os.listdir(input_dir):
        if filename.endswith(".txt"):
            txt_path = os.path.join(input_dir, filename)
            print(f"Lendo o arquivo: {filename}")
            try:
                with open(txt_path, 'r', encoding='utf-8') as file:
                    content = file.read()
                    partes = dividir_conteudo(content, max_length)
                    
                    for parte in partes:
                        document = {
                            "id": str(doc_id),
                            "titulo": filename,
                            "conteudo": parte
                        }
                        documents.append(document)
                        doc_id += 1
            except Exception as e:
                print(f"Erro ao ler o arquivo {filename}: {e}")
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as json_file:
        json.dump({"documentos": documents}, json_file, ensure_ascii=False, indent=4)
    
    print(f"Arquivos .txt convertidos e salvos em {output_path}")

# Especificar o diretório de entrada e o caminho de saída
input_dir = "env/processed-files" 
output_path = "env/docs3.json"  # Certifique-se de incluir o nome do arquivo JSON

# Executar a função de conversão
txt_to_json(input_dir, output_path)
