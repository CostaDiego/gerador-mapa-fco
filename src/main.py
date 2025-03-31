import numpy as np

from src.funcao_onda import FuncaoOnda
from src.iteracao import WAVE_COLLAPSED, observe, CONTRADICTION, propagate
from src.renderizacao import image_from_coefficients, progress_bar, show_iteration, save_iterations_to_video, \
    save_collapsed_wave, NUM_OF_ITERATIONS_BETWEEN_RENDER, num_of_collapsed_cells

MENSAGEM_ONDA_COLAPSADA = "\nOnda Colapsou!"
MENSAGEM_CONTRADICAO_ENCONTRADA = "\nContradição encontrada! O algoritmo não pode continuar."


def gerar_mapa_com_padrao(caminho_entrada: str, tamanho_padrao: int, altura_saida: int, largura_saida: int, inverter: bool,
                           rotacionar: bool, render_iteracoes: bool, render_video: bool) -> np.ndarray:
    """
    The main function of the program, will preform the wave function collapse algorithm.
    Given an input_examples image, the function will randomly generate an output image of any size, where each pixel in the
    output image resembles a small, local environment in the input_examples image.

    :param inverter: Set to True to calculate all possible flips of pattern as additional patterns
    :param rotacionar: Set to True to calculate all possible rotation of pattern as additional patterns
    :param caminho_entrada: The path of the input_examples image of the algorithm, from which to extract the patterns
    :param tamanho_padrao: The size (width and height) of the patterns from the input_examples image, should be as small as
    possible for efficiency, but large enough to catch key features in the input_examples image
    :param altura_saida: The height of the output image
    :param largura_saida: The width of the output image
    :param render_iterations:Set to True to render images in runtime every NUM_OF_ITERATIONS_BETWEEN_RENDER iteratoins
    :param render_video: Set to True to render video of the run of the algorithm
    :return: A numpy array representing the output image
    """

    # Cria e inicializa a função de onda
    funcao_onda = FuncaoOnda(caminho_entrada, tamanho_padrao, altura_saida, largura_saida, inverter, rotacionar)
    funcao_onda.inicializar()

    # Initialize control parameters
    status = 1
    iteration = 0

    # If render_video, initialize the images list
    if render_video:
        images = [image_from_coefficients(funcao_onda.matriz_coeficientes, funcao_onda.padroes)]

    # Iterate over the steps of the algorithm: observe, collapse, propagate until the wave collapses
    while status != WAVE_COLLAPSED:
        iteration += 1
        # Observe and collapse
        min_entropy_pos, funcao_onda.matriz_coeficientes, status = observe(funcao_onda.matriz_coeficientes, funcao_onda.frequencias)

        if status == CONTRADICTION:
            print(MENSAGEM_CONTRADICAO_ENCONTRADA)
            exit(-1)

        # Get current progress status
        image = image_from_coefficients(funcao_onda.matriz_coeficientes, funcao_onda.padroes)

        number_of_collapsed_cells = num_of_collapsed_cells(funcao_onda.matriz_coeficientes)

        # Update the progress bar
        progress_bar(altura_saida * largura_saida, number_of_collapsed_cells)

        # If render_video, add current iteration to the images list
        if render_video and not status == WAVE_COLLAPSED:
            images.append(image)

        # If the wave collapsed, stop the iterations
        if status == WAVE_COLLAPSED:
            print(MENSAGEM_ONDA_COLAPSADA)
            show_iteration(iteration, funcao_onda.padroes, funcao_onda.matriz_coeficientes)

            # If render_video, save the image list to a video
            if render_video:
                images.append(image)
                save_iterations_to_video(images, caminho_entrada)

            # Save the output image and return it
            return save_collapsed_wave(funcao_onda.matriz_coeficientes, caminho_entrada, funcao_onda.padroes)

        # If render_iterations and NUM_OF_ITERATIONS_BETWEEN_RENDER passed from last render, render this iteration
        if render_iteracoes and iteration % NUM_OF_ITERATIONS_BETWEEN_RENDER == 0:
            show_iteration(iteration, funcao_onda.padroes, funcao_onda.matriz_coeficientes)

        # Propagate
        funcao_onda.matriz_coeficientes = propagate(min_entropy_pos, funcao_onda.matriz_coeficientes, funcao_onda.regras, funcao_onda.direcoes)
