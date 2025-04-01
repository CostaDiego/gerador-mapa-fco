from enum import Enum


class EstadoOnda(Enum):
    INCIALIZADA = -1
    COLAPSADA = 1
    EM_ANDAMENTO = 0
    CONTRADICAO = -2
