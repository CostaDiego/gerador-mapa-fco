from typing import Tuple, List, Dict, Set, Any, Union

import numpy as np
from PIL import Image as Img

from .funcao_onda_base import FuncaoOndaBase


class FuncaoOnda(FuncaoOndaBase):
    def __init__(
        self,
        caminho_entrada: str,
        tamanho_padrao: int,
        altura_saida: int,
        largura_saida: int,
        inverter: bool,
        rotacionar: bool,
    ):
        super().__init__()

        self.caminho_entrada = caminho_entrada
        self.tamanho_padrao = tamanho_padrao
        self.altura_saida = altura_saida
        self.largura_saida = largura_saida
        self.inverter = inverter
        self.rotacionar = rotacionar

    def inicializar(
        self,
    ):
        """
        Inicializa o algoritmo da função do colapson de onde, do ingles wave function collapse. Este método
        agrega os padrões da imagem de entrada, calcula suas frequências e inicia a matriz de coeficientes.

        Parametros:
        - caminho_entrada: O caminho para a imagem de entrada
        - tamanho_padrao: O tamanho (largura e altura) dos padrões da imagem de entrada. Deve ser o menor possível
        por questões de eficiencia, mas grande o suficiente para capturar as principais características da image de
        entrada.
        - altura_saida: A altura da imagem de saída.
        - largura_saida: A largura da imagem de saída.
        - inverter: Defina como 'True' para calcular todas as possíveis inversões do padrão como padrões adicionais.
        - rotacionar: Defina como 'True' para calcular todas as possíveis rotações do padrão como padrões adicionais.
        """
        # Calcule todas as possíveis direcoes para os padroes
        self.direcoes = self.gerar_direcoes(self.tamanho_padrao)

        # Get a dictionary mapping padroes to their respective frequencias, then separate them
        pattern_to_freq = self.gerar_padroes_e_frequencias(
            self.caminho_entrada, self.tamanho_padrao, self.inverter, self.rotacionar
        )
        self.padroes, self.frequencias = np.array(
            np.array([self.para_ndarray(tup) for tup in pattern_to_freq.keys()])
        ), list(pattern_to_freq.values())

        # recupera as regras de correspondência entre os padrões
        self.regras = self.get_regras(self.padroes, self.direcoes)

        # inicializa a matriz_coeficientes, representando a função de onda
        self.matriz_coeficientes = np.full(
            (self.altura_saida, self.largura_saida, len(self.padroes)), True, dtype=bool
        )

        self._inicializado = True
        print("Inicialização completa.")

    def gerar_direcoes(self, n: int) -> List[Tuple[int, int]]:
        """
        Gerar direções para os padrões.
        Esta função retorna uma lista de todas as direções possíveis para um padrão de tamanho 'n', começando do canto
        superior esquerdo e terminando no canto inferior direito. O ponto central (0,0) é excluído da lista.
        As direções são representadas como tuplas (x,y), onde x e y são os deslocamentos em relação ao ponto central.

        Parametros:
        - n: O tamanho do padrão.

        Retorna:
        - Uma lista de coordenadas ao redor do padrão.
        """
        direcoes = [(i, j) for j in range(-n + 1, n) for i in range(-n + 1, n)]
        direcoes.remove((0, 0))
        return direcoes

    def gerar_padroes_e_frequencias(
        self,
        caminho_imagem: str,
        N: int,
        inverter: bool = True,
        rotacionar: bool = True,
    ) -> Dict[tuple[Any, ...], int]:
        """
        Extrai N por N subimagens de uma imagem a partir de um caminho dado e retorna um dicionário de padrões para sua
        frequência. Opcionalmente inclui versões invertidas e rotacionadas das subimagens.

        Parametros:
        - caminho_imagem: O caminho para a imagem de entrada
        :param path: A string containing the path to the image file
        :param N: An integer specifying the size of the subimages
        :param inverter: A boolean indicating whether to include inverterped versions of the subimages (defaults to True)
        :param rotacionar: A boolean indicating whether to include rotacionard versions of the subimages (defaults to True)
        :return: A tuple with:
        1. The input_examples image as numpy array.
        2. A dictionary mapping each pattern as a tuple to its respective frequency
        """
        # Open the image using PIL and convert the image to a numpy array
        imagem = np.array(Img.open(caminho_imagem))

        # Check if the array has more than 3 channels
        if imagem.shape[2] > 3:
            # If the array has more than 3 channels, reduce the number of channels to 3
            imagem = imagem[:, :, :3]

        padroes = self.extrair_padroes_da_image(imagem, N, inverter, rotacionar)

        padroes = [self.para_tupla(padrao) for padrao in padroes]
        pattern_to_freq = {item: padroes.count(item) for item in padroes}
        return pattern_to_freq

    def extrair_padroes_da_image(
        self, im: np.ndarray, N: int, inverter: bool = True, rotacionar: bool = True
    ) -> List[np.ndarray]:
        """
        Extracts N by N subimages from an image and returns a dictionary of padroes to their frequency.
        Optionally includes virarped and rotacionard versions of the subimages.

        :param im: The image from which to extract the padroes, as a numpy array
        :param N: An integer specifying the size of the subimages
        :param virar: A boolean indicating whether to include virarped versions of the subimages (defaults to True)
        :param rotacionar: A boolean indicating whether to include rotacionard versions of the subimages (defaults to True)
        :return: A list of all the padroes of size N*N inside the input_examples image
        """
        # Generate a list of indices for the rows and columns of the image
        linha_indices = np.arange(im.shape[0] - N + 1)
        coluna_indices = np.arange(im.shape[1] - N + 1)
        # Reshape the array of tiles into a list
        padroes = []
        for i in linha_indices:
            for j in coluna_indices:
                padroes.append(im[i : i + N, j : j + N, :])
        # Optionally include virarped and rotacionard versions of the tiles
        invertido, rotacionado = [], []
        if inverter:
            invertido = [
                np.flip(pattern, axis=axis) for axis in [0, 1] for pattern in padroes
            ]
        if rotacionar:
            rotacionado = [
                np.rot90(pattern, k=k) for k in range(1, 4) for pattern in padroes
            ]
        if inverter:
            padroes += invertido
        if rotacionar:
            padroes += rotacionado
        return padroes

    def get_regras(
        self, padroes: np.ndarray, direcoes: List[Tuple[int, int]]
    ) -> List[Dict[Tuple[int, int], Set[int]]]:
        """
        Creates the regras data structure, which is a list where entry i holds a dictionary that maps offset (x,y) to a
        set of indices of all padroes matching there

        :param direcoes: An array of all surrounding possible offsets
        :param padroes: The list of all the padroes
        :return:The regras list
        """
        regras = [{direcao: set() for direcao in direcoes} for _ in range(len(padroes))]
        for i in range(len(padroes)):
            for direcao in direcoes:
                for j in range(i, len(padroes)):
                    if self.check_for_match(padroes[i], padroes[j], direcao):
                        regras[i][direcao].add(j)
                        regras[j][self.inverter_direcao(direcao)].add(i)
        return regras

    def para_ndarray(self, tup: Tuple) -> Union[Tuple, np.ndarray]:
        """
        Convert tuple to NumPy ndarray.

        :param tup: The tuple to be converted. Can be a nested tuple.
        :return: The input_examples tuple as a NumPy ndarray with the same structure.
        """
        if isinstance(tup, tuple):
            return np.array(list(map(self.para_ndarray, tup)))
        else:
            return tup

    def para_tupla(self, array: np.ndarray) -> Union[Tuple, np.ndarray]:
        """
        Convert array to tuple.

        :param array: The array to be converted. Can be a NumPy ndarray or a nested sequence.
        :return: The input_examples array as a tuple with the same structure.
        """
        if isinstance(array, np.ndarray):
            return tuple(map(self.para_tupla, array))
        else:
            return array

    def check_for_match(
        self, p1: np.ndarray, p2: np.ndarray, offset: Tuple[int, int]
    ) -> bool:
        """
        checks whether 2 patterns with a given offset from each other match

        :param p1: first pattern
        :param p2: second pattern
        :param offset: offset of the first pattern
        :return: true if p2 equals to p1_ind with offset
        """
        p1_offset = self.mask_with_offset(p1, offset)
        p2_offset = self.mask_with_offset(p2, self.inverter_direcao(offset))
        return np.all(np.equal(p1_offset, p2_offset))

    def inverter_direcao(self, d: Tuple[int, int]) -> Tuple[int, int]:
        """
        Flips the direction of the given 2D vector d

        :param d: A 2D vector
        :return: The input_examples vector multiplied by -1,-1
        """
        return -1 * d[0], -1 * d[1]

    def mask_with_offset(
        sefl, pattern: np.ndarray, offset: Tuple[int, int]
    ) -> np.ndarray:
        """
        Get a subarray of a pattern based on an offset.
        This function returns a subarray of `pattern`, which is all entries that are inside the intersection of the
        `pattern` with another pattern offset by `offset`.

        :param pattern: an N*N*channels ndarray.
        :param offset: a 2D vector.
        :return: a subarray of `pattern` that intersects with another pattern by 'offset'.
        """
        x_offset, y_offset = offset
        if abs(x_offset) > len(pattern) or abs(y_offset) > len(pattern[0]):
            return np.array([[]])
        return pattern[
            max(0, x_offset) : min(len(pattern) + x_offset, len(pattern)),
            max(0, y_offset) : min(len(pattern[0]) + y_offset, len(pattern[0])),
            :,
        ]
