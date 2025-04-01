import ntpath
from typing import List

import numpy as np
from PIL import Image as Img
from matplotlib import pyplot as plt
from moviepy.video.io.ImageSequenceClip import ImageSequenceClip

from .constantes import Constantes

def salvar_onda_colapsada(
    matriz_coeficientes: np.ndarray, caminho_entrada: str, padroes: np.ndarray
) -> np.ndarray:
    """
    Salva uma imagem da onda colapsada, mantendo a proporção da imagem de entrada.

    Parametros:
    - matriz_coeficientes: A matriz de coeficientes da onda
    - caminho_entrada: O caminho da imagem de entrada
    - padroes: O array numpy dos padrões

    Retorna:
    - Uma imagem da onda colapsada
    """
    # Extrai as dimensões da matriz de coeficientes
    largura, altura, _ = matriz_coeficientes.shape
    num_channels = padroes[0].ndim

    # Croia a imagem de saida como um array
    imagem_final = padroes[np.where(matriz_coeficientes[:, :])[2]][:, 0, 0, :].reshape(
        largura, altura, num_channels
    )

    # Calula o parametro de upscale para manter a proporção da imagem de entrada
    parametro_upscale = (
        Constantes.ALTURA_PADRAO_VIDEO,
        (min(largura, altura) * Constantes.ALTURA_PADRAO_VIDEO) // max(largura, altura),
    )

    # Cria uma imagem PIL a partir do array numpy
    img = Img.fromarray(imagem_final).resize(parametro_upscale, resample=Img.NONE)

    # Salva a imagem
    nome_arquivo = f"mapa_{ntpath.basename(caminho_entrada)}"
    img.save(nome_arquivo)

    return imagem_final


def mostrar_iteracao(
    iteracao: int, padroes: np.ndarray, matriz_coeficientes: np.ndarray
) -> np.ndarray:
    """
    Mostra o estado da onda na iteração atual do algoritmo.

    Parametros:
    - iteracao: A iteração atual do algoritmo
    - padroes: O array numpy dos padrões.
    - matriz_coeficientes: A matriz de coeficientes representando a onda

    Retorna:
    - Um array numpy representando a imagem da onda na iteração atual, todas as céluals não colapsadas estão com a
    cor resultante da média dos padrões válidos para a célula.
    """
    res = converter_coeficientes_para_imagem(matriz_coeficientes, padroes)
    largura, altura, _ = res.shape
    fig, axs = plt.subplots()
    axs.imshow(res)
    celulas_colapsadas = num_celulas_colapsadas(matriz_coeficientes)
    axs.set_title(
        f"Células colapsadas: {celulas_colapsadas} de {largura * altura}. Finalizados {round(100 * celulas_colapsadas / (largura * altura), 2)}%"
    )
    fig.suptitle(f"Iteração número: {iteracao}")
    plt.show()

    return res


def salvar_iteracoes_em_video(imagens: List[np.ndarray], caminho_entrada: str) -> None:
    """
    Salva todas as imagens das iterações do algoritmo em um vídeo.

    Parametros:
    - imagens: Uma lista de ndarrays do estado da onda durante as iterações do algoritmo
    - caminho_entrada: O caminho da imagem de entrada
    """
    largura, altura, _ = imagens[0].shape

    # Calculo do parametro de upscale para manter a proporção da imagem de entrada
    parametro_upscale = Constantes.ALTURA_PADRAO_VIDEO // max(largura, altura)

    # Calcula o parametro_amostragem_tempo para selecionar os frames que vão fazer parte do vídeo
    parametro_amostragem_tempo = 1
    if len(imagens) > Constantes.FPS_PADRAO * Constantes.TAMANHO_PADRAO_VIDEO:
        parametro_amostragem_tempo = len(imagens) // (Constantes.FPS_PADRAO * Constantes.TAMANHO_PADRAO_VIDEO)

    # Aumenta o tamanho da imagem e seleciona um subconjunto de imagens
    imagens = np.array(imagens)
    imagens = np.kron(
        imagens[::parametro_amostragem_tempo, :, :, :],
        np.ones((parametro_upscale, parametro_upscale, 1)),
    )
    imagens = [imagens[i] for i in range(imagens.shape[0])]

    # Salva o video
    video_name = f"mapa_{ntpath.basename(caminho_entrada).split('.')[0]}.mp4"
    clip = ImageSequenceClip(imagens, fps=Constantes.FPS_PADRAO)
    clip.write_videofile(video_name, fps=Constantes.FPS_PADRAO)


def converter_coeficientes_para_imagem(
    matriz_coeficientes: np.ndarray, padroes: np.ndarray
) -> np.ndarray:
    """
    Gera uma imagem do estado da função de onda. Cada célular é inicializada com a média dos padrões válidas para ela.

    Parametros:
    - matriz_coeficientes: A matriz de coeficientes da função de onda.
    - padroes: O array numpy dos padrões

    Retorna:
    - Uma imagem do estado da função de onda, onde cada célula é inicializada com a média dos padrões válidas para ela.
    """

    linhas, colunas, _ = matriz_coeficientes.shape
    imagem = np.empty((linhas, colunas, 3))

    borda_padroes = padroes[:, 0, 0]

    # Itera sobre todas as células da matriz_coeficientes
    for linha in range(linhas):
        for coluna in range(colunas):
            # Para cada célula, encontre os padrões válidos nela
            padroes_validos = np.where(matriz_coeficientes[linha, coluna])[0]

            # Seta a célula com o valor médio de todos os padrões válidos
            imagem[linha, coluna] = np.mean(borda_padroes[padroes_validos], axis=0)

    return imagem.astype(int)


def num_celulas_colapsadas(matriz_coeficientes: np.ndarray) -> int:
    """
    Calula o número de células colapsadas na matriz de coeficientes.

    Parametros:
    - matriz_coeficientes: A matriz de coeficientes da função de onda.

    Retorna:
    - O número de células colapsadas na matriz de coeficientes.
    """
    return np.count_nonzero(np.sum(matriz_coeficientes, axis=2) == 1)


def barra_progresso(max_work: int, curr_work: int) -> None:
    """
    Printa a barra de progresso de execução do algoritmo na tela.

    Parametros:
    - trabalho_total: O total de trabalho a ser feito, neste caso, número de células a serem colapsadas
    - trabalho_atual: A quantidade de trabalho feito até agora, neste caso, número de células colapsadas
    """
    progresso = int(100 * (curr_work / float(max_work)))

    # Create and print the progress bar
    bar = "█" * progresso + "-" * (100 - progresso)
    print(f"\r|{bar}| {round(progresso, 2)}% ", end="\b")
