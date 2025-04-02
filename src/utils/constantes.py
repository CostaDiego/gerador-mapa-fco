from dataclasses import dataclass


@dataclass(frozen=True)
class Constantes:
    """
    Classe de constantes usadas para renderização e processamento do video.
    Atributos:
    - FPS_PADRAO: Frames por segundo do vídeo de saída.
    - TAMANHO_PADRAO_VIDEO: Tamanho em segundos do vídeo de saída.
    - ALTURA_PADRAO_VIDEO: Tamanho vertical do vídeo de saída, preserva a proporção.
    - ITERACOES_ENTRE_RENDERIZACAO: Número de iterações entre cada renderização durante a execução.
    """
    FPS_PADRAO = 30
    TAMANHO_PADRAO_VIDEO = 30
    ALTURA_PADRAO_VIDEO = 1000
    ITERACOES_ENTRE_RENDERIZACAO = 15
