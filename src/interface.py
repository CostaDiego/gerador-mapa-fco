from typing import Tuple

import numpy as np

from src.algoritmo import observe_algoritmo, propague_algoritmo, EstadoOnda
from src.funcao_onda import FuncaoOndaBase


def observe(
    funcao_onda: type[FuncaoOndaBase],
) -> Tuple[Tuple[int, int], np.ndarray, type[EstadoOnda]]:
    """
    Interface com o algoritmo que performa a fase de observação da Função de colapso de onda.

    Parametros:
    - funcao_onda: Objeto do tipo função de onda para ser manipulado

    Retorna:
    - Uma tupla: A tupla contém:
        1. Uma tupla de inteiros representando a posição da célula com a menor entropia.
        2. A matriz de coeficientes atualizada representando o estado da onda.
        3. Um 'EstadoOnda' representando o estado da função de onda.
    """
    if not isinstance(funcao_onda, FuncaoOndaBase):
        raise TypeError(
            "Esperado objeto do tipo 'FuncaoOndaBase' e classes que a herdam."
        )

    matriz_coeficientes = funcao_onda.matriz_coeficientes
    frequencias = funcao_onda.frequencias

    min_entropy_pos, matriz_coeficientes_atualizada, status = observe_algoritmo(
        matriz_coeficientes, frequencias
    )

    return min_entropy_pos, matriz_coeficientes_atualizada, status


def propague(
    pos_min_entropia: Tuple[int, int], funcao_onda: type[FuncaoOndaBase]
) -> np.ndarray:
    """
    Interface com o algoritmo que performa a fase de propagação da Função de colapso de onda.

    Parametros:
    - pos_min_entropia: Tupla de inteiros representando a posição da célula que foi colapsada em 'observe'.
    - funcao_onda: Objeto do tipo função de onda para ser manipulado

    Retorna:
    - NumPy Array: a matriz de coeficientes atualizada, após a propagação.
    """

    if not isinstance(funcao_onda, FuncaoOndaBase):
        raise TypeError(
            "Esperado objeto do tipo 'FuncaoOndaBase' e classes que a herdam."
        )

    matriz_coeficientes = funcao_onda.matriz_coeficientes
    regras = funcao_onda.regras
    direcoes = funcao_onda.direcoes

    matriz_coeficientes_atualizada = propague_algoritmo(
        pos_min_entropia,
        matriz_coeficientes,
        regras,
        direcoes,
    )

    return matriz_coeficientes_atualizada
