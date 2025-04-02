from enum import Enum
import queue
from typing import List, Tuple, Dict, Set

import numpy as np

from .estado_onda import EstadoOnda


def coordenadas_celula_minima_entropia(
    matriz_coeficientes: np.ndarray, frequencias: List[int]
) -> Tuple[int, int, type[EstadoOnda]]:
    """
    Calcula as coordenadas da células com a menor entropia e retorna a linha, coluna e o estado da onda (colapsada/em andamento)

    Parametros:
    - matriz_coeficientes: A matriz de coeficientes representando o estado da onda
    - frequencias: Lista de frequencias dos padrões na onda

    Retorna:
    - Tupla de dois inteiros e um estado da onda: linha, coluna, estado da onda (colapsada/em andamento)
    """
    # Calcula a probabilidade de cada padrão
    probabilidades = np.array(frequencias) / np.sum(frequencias)

    # Calcula a entropia de cada célula
    entropias = np.sum(matriz_coeficientes.astype(int) * probabilidades, axis=2)

    # Seta a entropia de células colapsadas para 0
    entropias[np.sum(matriz_coeficientes, axis=2) == 1] = 0

    if np.sum(entropias) == 0:
        return -1, -1, EstadoOnda.COLAPSADA

    # Recupera todos os indices de células não nulas com entropia mínima
    indices_minimos = np.argwhere(
        entropias == np.amin(entropias, initial=np.max(entropias), where=entropias > 0)
    )

    # Checa se a entropia de todas as células é 0, ou seja, a onda está totalmente colapsada
    if indices_minimos.shape[0] == 0:
        return -1, -1, EstadoOnda.COLAPSADA

    # Escolha um indice aleatório da lista de indices minimos
    min_index = indices_minimos[np.random.randint(0, indices_minimos.shape[0])]
    return min_index[0], min_index[1], EstadoOnda.EM_ANDAMENTO


def celula_colapsada(
    matriz_coeficientes: np.ndarray, posicao_celula: Tuple[int, int]
) -> bool:
    """
    Checa se a célula localizada em `posicao_celula` na `matriz_coeficientes` está colapsada.

    Parametros:
    - matriz_coeficientes: A matriz de coeficientes representando o estado da onda.
    - cell_pos: Tupla de inteiros representando a posição da célula na matriz (x, y).
    Retorna:
    - Um booleano indicando se a célula está colapsada (True) ou não (False).
    """
    return np.sum(matriz_coeficientes[posicao_celula[0], posicao_celula[1], :]) == 1


def propagar_celula(
    celula_origem: Tuple[int, int],
    direcao: Tuple[int, int],
    matriz_coeficientes: np.ndarray,
    regras: List[Dict[Tuple[int, int], Set[int]]],
) -> np.ndarray:
    """
    Propaga as restrições de padrões de uma célula para uma célula adjacente em uma determinada direção de acordo
    com um conjunto de regras

    Parametros:
    - celula_origem: A posição da célula original (x, y) na matriz de coeficientes
    - direcao: A direção para propagar padrões (dx, dy)
    - matriz_coeficientes: Uma matriz de coeficientes representando os padrões possíveis em cada célula
    - regras: Uma lista de dicionários, onde cada dicionário representa os padrões possíveis que podem ser
    propagados em uma direção específica para um padrão específico na célula original.

    Retorna:
    - Os padrões propagados para a célula adjacente, na forma de uma célula na matriz de coeficientes
    """
    # Calcula a coordenada da celula adjacente
    posicao_celula_adjacente = (
        celula_origem[0] + direcao[0],
        celula_origem[1] + direcao[1],
    )

    # recupera o padrão válido na célula adjacente
    padrao_valido_celula_adjacente = matriz_coeficientes[posicao_celula_adjacente]

    # recupera os padrões válidos na célula adjacente pelo indice
    possivel_padrao_celula_origem = np.where(matriz_coeficientes[celula_origem])[0]

    # Vetor cheio de False com  o mesmo tamanho da célula alvo
    possibilidade_direcao = np.zeros(matriz_coeficientes[celula_origem].shape, bool)

    # Acumule todos os padrões possíveis na direcão
    for padrao in possivel_padrao_celula_origem:
        possibilidade_direcao[list(regras[padrao][direcao])] = True

    # Multiplica a celula alvo pelos padrões possíveis da célular original
    return np.multiply(possibilidade_direcao, padrao_valido_celula_adjacente)


def na_matriz(posicao: Tuple[int, int], dimensao: Tuple[int, ...]) -> bool:
    """
    Checa se a posição 'posicao' está dentro dos limites da matriz com dimensões 'dimensao'

    Parametros:
    - posicao: Tupla de inteiros representando a posição de um elemento em uma matriz
    - dimensao: Tupla de inteiros representando as dimensões da matriz

    Retorna:
    - Booelano: Indica se a posição dada está dentro dos limites da matriz
    """
    return 0 <= posicao[0] < dimensao[0] and 0 <= posicao[1] < dimensao[1]


def propague(
    pos_min_entropia: Tuple[int, int],
    matriz_coeficientes: np.ndarray,
    regras: List[Dict[Tuple[int, int], Set[int]]],
    direcoes: List[Tuple[int, int]],
) -> np.ndarray:
    """
    Esta função executa a parte de propagação do algoritmo de função de colapso de onda. Após colapsar uma célula
    em 'observe', esta função irá propagar essa mundaça para todas as células relevantes

    Parametros:
    - pos_min_entropia: Tupla de inteiros representando a posição da célula que foi colapsada em 'observe'.
    - matriz_coeficientes: A matriz de coeficientes representando o estado da onda.
    - regras: Uma lista de dicionários, onde cada dicionário representa um padrão e contém um mapeamento
    de direções para um conjunto de padrões possíveis na direção do padrão do dicionario.
    - direcoes: Uma lista de tuplas de inteiros representando as direções nas quais a célula está sendo propagada.

    Retorna:
    - NumPy Array: a matriz de coeficientes atualizada, após a propagação.
    """
    # Cria uma fila de posições de células a serem atualizadas
    fila_propagacao = queue.Queue()
    fila_propagacao.put(pos_min_entropia)

    while not fila_propagacao.empty():
        celula = fila_propagacao.get()

        # Em todas as direções a partir da célula
        for direcao in direcoes:
            pos_celula_adjacente = celula[0] + direcao[0], celula[1] + direcao[1]

            # Se a célula adjacente não foi colapsada, propague
            if na_matriz(
                pos_celula_adjacente, matriz_coeficientes.shape
            ) and not celula_colapsada(matriz_coeficientes, pos_celula_adjacente):
                celula_atualizada = propagar_celula(
                    celula, direcao, matriz_coeficientes, regras
                )

                # Se a propagação da célula adjacente mudou a onda, atualiza a matriz de coeficientes
                if not np.array_equal(
                    matriz_coeficientes[pos_celula_adjacente], celula_atualizada
                ):
                    matriz_coeficientes[pos_celula_adjacente] = celula_atualizada

                    # Se a célula adjacente não está na fila de propagação, addiciona ela
                    if pos_celula_adjacente not in fila_propagacao.queue:
                        fila_propagacao.put(pos_celula_adjacente)
    return matriz_coeficientes


def observe(
    matriz_coeficientes: np.ndarray, frequencias: List[int]
) -> Tuple[Tuple[int, int], np.ndarray, type[EstadoOnda]]:
    """
    Esta função realiza a fase de 'observar' do algoritmode função do colapso de onda. Procura uma célula com entropia
    mínima e a colapsa com base em possíveis padrões da célula e suas frequências.

    Parametros:
    - matriz_coeficientes: A matriz de coeficientes representando o estado da onda.
    - frequencias: Uma lista de inteiros representando a frequência de cada padrão dentro da imagem de entrada.

    Retorna:
    - Uma tuple: A tupla contém:
        1. Uma tupla de inteiros representando a posição da célula com a menor entropia.
        2. A matriz de coeficientes atualizada representando o estado da onda.
        3. Um 'EstadoOnda' representando o estado da função de onda.
    """
    # Se contradição for encontrada
    if np.any(~np.any(matriz_coeficientes, axis=2)):
        return (-1, -1), matriz_coeficientes, EstadoOnda.CONTRADICAO

    # Calcula a posição da célula com a menor entropia
    pos_min_entropia_x, pos_min_entropia_y, status = coordenadas_celula_minima_entropia(
        matriz_coeficientes, frequencias
    )
    pos_min_entropia = (pos_min_entropia_x, pos_min_entropia_y)

    if status == EstadoOnda.COLAPSADA:
        return (
            (pos_min_entropia_x, pos_min_entropia_y),
            matriz_coeficientes,
            EstadoOnda.COLAPSADA,
        )

    # Colapsa a célula na pos_min_entropia
    matriz_coeficientes = colapse_celula(
        matriz_coeficientes, frequencias, pos_min_entropia
    )
    return pos_min_entropia, matriz_coeficientes, EstadoOnda.EM_ANDAMENTO


def colapse_celula(
    matriz_coeficientes: np.ndarray,
    frequencias: List[int],
    pos_min_entropia: Tuple[int, int],
) -> np.ndarray:
    """
    Colapsa uma única célula na pos_min_entropia em um único padrão ponderado aleatóriamente pelas frequências.

    Parametros:
    - matriz_coeficientes: A matriz de coeficientes representando o estado da onda.
    - frequencias: Uma lista de inteiros representando a frequência de cada padrão dentro da imagem de entrada.
    - pos_min_entropia: A posição da célula para ser colapsada

    Retorna:
    - NumPy Array: A matriz do estado da onda atualizada com a nova célula colapsada
    """
    # Pega indices de padrões opcionais na pos_min_entropia
    indices_relevantes = np.where(matriz_coeficientes[pos_min_entropia])[0]

    # Pega a frequencia para padrões relevantes na pos_min_entropia
    frequencias_relevantes = np.array(frequencias)[indices_relevantes]

    # Colapsa a célula em um padrão aleatóriamente escolhido a partir das frequencias
    indice_padrao_escolhido = np.random.choice(
        indices_relevantes, p=frequencias_relevantes / np.sum(frequencias_relevantes)
    )

    # Seta a possibiliade de todos os outros padrões para False
    matriz_coeficientes[pos_min_entropia] = np.full(
        matriz_coeficientes[pos_min_entropia].shape[0], False, dtype=bool
    )
    matriz_coeficientes[
        pos_min_entropia[0], pos_min_entropia[1], indice_padrao_escolhido
    ] = True

    return matriz_coeficientes
