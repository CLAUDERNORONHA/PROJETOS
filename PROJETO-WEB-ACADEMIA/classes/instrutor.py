# Classe Instrutor
class Instrutor:

    # Método construtor para inicialização dos atributos
    def __init__(self, nome, sobrenome, data_nascimento, endereco, telefone, id = None):
        self.nome = nome
        self.sobrenome = sobrenome
        self.data_nascimento = data_nascimento
        self.endereco = endereco
        self.telefone = telefone
        self.id = id