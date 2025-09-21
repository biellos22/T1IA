import numpy as np
from keras.models import load_model
import os

def prever_admissao():
    nome_arquivo_modelo = 'modelo_treinado.keras'

    if not os.path.exists(nome_arquivo_modelo):
        print(f"\nErro Crítico: O arquivo do modelo '{nome_arquivo_modelo}' não foi encontrado.")
        print("Por favor, garanta que o arquivo esteja no mesmo diretório que este script.")
        return

    try:
        modelo = load_model(nome_arquivo_modelo)
        print(f"Modelo '{nome_arquivo_modelo}' carregado com sucesso.\n")

        colunas = [
            "GRE Score", "TOEFL Score", "University Rating",
            "SOP", "LOR", "CGPA", "Research"
        ]

        print("--- Previsão de Admissão ---")
        print("Por favor, insira os dados do candidato conforme solicitado.")
        dados_candidato = []
        for coluna in colunas:
            while True:
                try:
                    valor_str = input(f"  - {coluna}: ")
                    valor = float(valor_str)

                    if coluna == "GRE Score" and not (260 <= valor <= 340):
                        print("  [Entrada Inválida] O GRE Score deve estar no intervalo de 260 a 340.")
                    elif coluna == "TOEFL Score" and not (0 <= valor <= 120):
                        print("  [Entrada Inválida] O TOEFL Score deve estar no intervalo de 0 a 120.")
                    elif coluna in ["University Rating", "SOP", "LOR"] and not (1 <= valor <= 5 and valor.is_integer()):
                        print(f"  [Entrada Inválida] O {coluna} deve ser um número inteiro entre 1 e 5.")
                    elif coluna == "CGPA" and not (0.0 <= valor <= 10.0):
                        print("  [Entrada Inválida] O CGPA deve ser um número entre 0.0 e 10.0.")
                    elif coluna == "Research" and valor not in [0, 1]:
                        print("  [Entrada Inválida] O valor para Research deve ser 0 (Não) ou 1 (Sim).")
                    else:
                        dados_candidato.append(valor)
                        break
                except ValueError:
                    print("  [Entrada Inválida] Por favor, digite um valor numérico.")

        entrada_array = np.array([dados_candidato]) 

        previsao = modelo.predict(entrada_array)
        chance_admissao = previsao[0][0]

        print("\n---------------------------------------")
        print("      Resultado da Análise")
        print("---------------------------------------")
        for i, coluna in enumerate(colunas):
            print(f"{coluna+':':<20} {dados_candidato[i]}")
        print("---------------------------------------")
        print(f"Chance prevista de admissão: {chance_admissao:.2%}")
        print("---------------------------------------")

    except Exception as e:
        print(f"\nOcorreu um erro inesperado durante a execução: {e}")
        print("Verifique a integridade do arquivo do modelo e as dependências instaladas (keras, tensorflow, numpy).")


if __name__ == "__main__":
    prever_admissao()