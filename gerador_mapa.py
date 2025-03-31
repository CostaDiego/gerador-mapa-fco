import sys
from src.main import gerar_mapa_com_padrao

MENSAGEM_USO = "Erro: entrada inválida. Uso: python3 wave_function_collapse.py <caminho_entrada> <tamanho_padrao> " \
            "<altura_saida> <largura_saida> [inverter] [rotacionar] [render_iteracoes] [render_video]"

if __name__ == "__main__":
    try:
        if len(sys.argv) > 9:
            raise IndexError
        caminho_entrada, tamanho_padrao, altura_saida, largura_saida = sys.argv[1:5]
        tamanho_padrao, altura_saida, largura_saida = int(tamanho_padrao), int(altura_saida), int(largura_saida)
        inverter = sys.argv[5] == 'True' if len(sys.argv) >= 6 else False
        rotacionar = sys.argv[6] == 'True' if len(sys.argv) >= 7 else False
        render_iteracoes = sys.argv[7] == 'True' if len(sys.argv) >= 8 else True
        render_video = sys.argv[8] == 'True' if len(sys.argv) == 9 else True
    except (TypeError, ValueError, IndexError):
        print(MENSAGEM_USO)
    else:
        gerar_mapa_com_padrao(caminho_entrada, tamanho_padrao, altura_saida, largura_saida, inverter, rotacionar, render_iteracoes,
                               render_video)
