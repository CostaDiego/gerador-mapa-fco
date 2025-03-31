class FuncaoOndaBase:
    """
    Classe base para a função do colapson de onda, do ingles wave function collapse. Esta classe é responsável por
    inicializar os padrões, calcular suas frequências e gerar a matriz de coeficientes.
    """

    def __init__(self):
        self.matriz_coeficientes = None
        self.direcoes = None
        self.frequencias = None
        self.padroes = None
        self.regras = None

        self._inicializado = False

    def inicializar(self):
        """
        Inicializa o algoritmo da função do colapson de onda. Este método agrega os padrões da imagem de entrada,
        calcula suas frequências e inicia a matriz de coeficientes.
        """
        raise NotImplementedError("Método não implementado na classe base.")
