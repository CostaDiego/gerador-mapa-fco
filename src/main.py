import numpy as np

from src.funcao_onda import FuncaoOnda
from src.iteracao import observe, propagate
from src.estado_onda import EstadoOnda
from src.renderizacao import (
    converter_coeficientes_para_imagem,
    barra_progresso,
    mostrar_iteracao,
    salvar_iteracoes_em_video,
    salvar_onda_colapsada,
    ITERACOES_ENTRE_RENDERIZACAO,
    num_celulas_colapsadas,
)

MENSAGEM_ONDA_COLAPSADA = "\nOnda Colapsou!"
MENSAGEM_CONTRADICAO_ENCONTRADA = (
    "\nContradição encontrada! O algoritmo não pode continuar."
)


def gerar_mapa_com_padrao(
    caminho_entrada: str,
    tamanho_padrao: int,
    altura_saida: int,
    largura_saida: int,
    inverter: bool,
    rotacionar: bool,
    render_iteracoes: bool,
    render_video: bool,
) -> np.ndarray:
    """
    Função principal do programa, irá executar o algoritmo de colapso da função de onda.
    Dada uma imagem de entrada, a função irá gerar aleatoriamente uma imagem de saída de qualquer tamanho, onde cada pixel
    da imagem de saída se assemelha a um pequeno ambiente local na imagem de entrada.

    Parametros:
    - caminho_entrada: O caminho da espaço de entrada da imagem de entrada do algoritmo, da qual extrair os padrões.
    - tamanho_padrao: O tamanho (largura e altura) dos padrões da imagem de entrada, deve ser o menor possível para eficiência,
    mas grande o suficiente para capturar as características principais da imagem de entrada.
    - altura_saida: A altura da imagem de saída
    - largura_saida: A largura da imagem de saída
    - inverter: Defina como True para calcular todas as inversões possíveis do padrão como padrões adicionais.
    - rotacionar: Defina como True para calcular todas as rotações possíveis do padrão como padrões adicionais.
    - render_iteracoes: Defina como True para renderizar imagens em tempo de execução a cada ITERACOES_ENTRE_RENDERIZACAO iterações.
    - render_video: Defina como True para renderizar vídeo da execução do algoritmo.

    Retorna:
    - Um array numpy representando a imagem de saída.
    """

    # Cria e inicializa a função de onda
    funcao_onda = FuncaoOnda(
        caminho_entrada,
        tamanho_padrao,
        altura_saida,
        largura_saida,
        inverter,
        rotacionar,
    )
    funcao_onda.inicializar()

    # Inicializa as variáveis de controle do algoritmo
    status = EstadoOnda.INCIALIZADA
    iteration = 0

    # If render_video, initialize the images list
    if render_video:
        images = [
            converter_coeficientes_para_imagem(
                funcao_onda.matriz_coeficientes, funcao_onda.padroes
            )
        ]

    # Itera sobre os passo básicos do algoritmo: Observa, colapsa, propaga e repete até a onda colapsar.
    while status != EstadoOnda.COLAPSADA:
        iteration += 1
        # Observa e colapsa a onda
        min_entropy_pos, funcao_onda.matriz_coeficientes, status = observe(
            funcao_onda.matriz_coeficientes, funcao_onda.frequencias
        )

        if status == EstadoOnda.CONTRADICAO:
            print(MENSAGEM_CONTRADICAO_ENCONTRADA)
            exit(-1)

        # Recupera o estado atual do progresso
        image = converter_coeficientes_para_imagem(
            funcao_onda.matriz_coeficientes, funcao_onda.padroes
        )

        number_of_collapsed_cells = num_celulas_colapsadas(
            funcao_onda.matriz_coeficientes
        )

        # Atualiza a barra de progresso
        barra_progresso(altura_saida * largura_saida, number_of_collapsed_cells)

        # Se render_video for True, adicionar a imagem atual à lista de imagens
        if render_video and not status == EstadoOnda.COLAPSADA:
            images.append(image)

        # Se a onda colapsou, renderizar a iteração final
        if status == EstadoOnda.COLAPSADA:
            print(MENSAGEM_ONDA_COLAPSADA)
            mostrar_iteracao(
                iteration, funcao_onda.padroes, funcao_onda.matriz_coeficientes
            )

            # If render_video, save the image list to a video
            if render_video:
                images.append(image)
                salvar_iteracoes_em_video(images, caminho_entrada)

            # Salva a imagem de output e a retorna
            return salvar_onda_colapsada(
                funcao_onda.matriz_coeficientes, caminho_entrada, funcao_onda.padroes
            )

        # Se render_iteracoes é True e ITERACOES_ENTRE_RENDERIZACAO passaram desde o último render, renderizar essa iteração
        if render_iteracoes and iteration % ITERACOES_ENTRE_RENDERIZACAO == 0:
            mostrar_iteracao(
                iteration, funcao_onda.padroes, funcao_onda.matriz_coeficientes
            )

        # Propagar onda
        funcao_onda.matriz_coeficientes = propagate(
            min_entropy_pos,
            funcao_onda.matriz_coeficientes,
            funcao_onda.regras,
            funcao_onda.direcoes,
        )
