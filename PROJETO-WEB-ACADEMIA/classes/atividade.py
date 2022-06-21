# Classe Atividade
class Atividade:

    # Método construtor para inicialização dos atributos
    def __init__(self, nome, instrutor, data, duracao, capacidade, tipo_plano, ativo, id = None):
        self.nome = nome
        self.instrutor = instrutor
        self.data = data
        self.duracao = duracao
        self.capacidade = capacidade
        self.tipo_plano = tipo_plano
        self.ativo = ativo
        self.id = id