# Classe Membro
class Membro:

    # Método construtor para inicialização dos atributos
    def __init__(self, nome, sobrenome, data_nascimento, endereco, telefone, email, tipo_plano, data_inicio, ativo, id = None):
        self.nome = nome
        self.sobrenome = sobrenome
        self.data_nascimento = data_nascimento
        self.endereco = endereco
        self.telefone = telefone
        self.email = email
        self.tipo_plano = tipo_plano
        self.data_inicio = data_inicio
        self.ativo = ativo
        self.id = id
