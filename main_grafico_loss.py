import numpy as np
import os
import argparse
import matplotlib.pyplot as plt

output_dir = "vetores_loss"
os.makedirs(output_dir, exist_ok=True)

def salvar_vetor (nome_arquivo, vetor):
    np.save(f"{output_dir}/{nome_arquivo}.npy", vetor)
    print("----VETOR LOSS SALVO----")
    return

if __name__ == "__main__":

    parse = argparse.ArgumentParser(description="Mostra gráfico de vetor Loss")

    parse.add_argument("vetor", type=str, help="Nome do arquivo do vetor .npy")

    args = parse.parse_args()

    try:
        vetor = np.load(f"{output_dir}/{args.vetor}")

        eixo_x = np.arange(1, len(vetor) + 1)

        plt.plot(eixo_x, vetor, marker='o', linestyle='-', color='b', label='Valores')

        plt.title("Gráfico dos Valores do Vetor")
        plt.xlabel("Épocas (X)")
        plt.ylabel("Valores Loss (Y)")
        plt.grid(True)
        plt.legend()
        plt.tight_layout()

        # Exibir o gráfico
        plt.show()


    except Exception as e:
        print(f"Erro ao carregar arquivo .npy :\n {e}")