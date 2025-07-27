import numpy as np
import os
import argparse
import matplotlib.pyplot as plt

output_dir = "vetores_loss"
os.makedirs(output_dir, exist_ok=True)

def salvar_vetor(nome_arquivo, vetor):
    np.save(f"{output_dir}/{nome_arquivo}.npy", vetor)
    return

if __name__ == "__main__":

    parse = argparse.ArgumentParser(description="Mostra gráfico de vetor Loss")

    parse.add_argument("vetor", type=str, help="Nome do arquivo do vetor .npy")

    args = parse.parse_args()

    try:
        vetor = np.load(f"{output_dir}/{args.vetor}")
        eixo_x = np.arange(1, len(vetor) + 1)

        plt.figure(figsize=(10, 6))
        plt.plot(eixo_x, vetor, linestyle='-', color='b', label='Loss')

        # Identificar mínimo
        min_loss = np.min(vetor)
        min_epoch = np.argmin(vetor) + 1  # +1 pois epochs começam em 1

        # Marcar ponto mínimo
        plt.scatter(min_epoch, min_loss, color='red',
                    label=f'Mínimo Loss: {min_loss:.4f} (Epoch {min_epoch})')

        # Linha tracejada horizontal
        plt.axhline(y=min_loss, color='red', linestyle='--', linewidth=1)

        plt.title("Gráfico dos Valores do Loss")
        plt.xlabel("Épocas")
        plt.ylabel("Loss")
        plt.grid(True)
        plt.legend()
        plt.tight_layout()

        # Salvar gráfico
        plt.savefig(f"{output_dir}/grafico_{args.vetor}.png")
        print(f"Imagem do gráfico salva! Menor loss = {min_loss:.4f} na epoch {min_epoch}.")

        # plt.show()

    except Exception as e:
        print(f"Erro ao carregar arquivo .npy :\n {e}")
