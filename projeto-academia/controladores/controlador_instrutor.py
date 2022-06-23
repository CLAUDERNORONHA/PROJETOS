# Controlador de Instrutores  

# Imports
from flask import render_template, request, redirect
from flask import Blueprint
from classes.instrutor import Instrutor
import conectores.conector_instrutor as conector_instrutor


# Cria o blueprint (instância da classe)
instrutores_blueprint = Blueprint("instrutores", __name__)


# Rota para a página index.html
@instrutores_blueprint.route("/instrutores")
def instrutores_index():
    instrutores = conector_instrutor.get_all()
    return render_template("instrutores/index.html", instrutores = instrutores, title = "Instrutores")


# Rota para a página de cadastro de instrutor
@instrutores_blueprint.route("/instrutores/novo")
def novo_instrutor():
    return render_template("instrutores/novo.html", title = "Novo Instrutor")


# Método para envio dos dados de um novo instrutor via POST
@instrutores_blueprint.route("/instrutores", methods = ["POST"])
def cria_instrutor():
    nome = request.form["nome"]
    sobrenome = request.form["sobrenome"]
    data_nascimento = request.form["data_nascimento"]
    endereco = request.form["endereco"]
    telefone = request.form["telefone"]
    novo_instrutor = Instrutor(nome, sobrenome, data_nascimento, endereco, telefone)
    conector_instrutor.new(novo_instrutor)
    return redirect("/instrutores")


# Rota para a página de edição de um instrutor
@instrutores_blueprint.route("/instrutores/<id>/edit")
def edita_instrutor(id):
    instrutor = conector_instrutor.get_one(id)
    return render_template("/instrutores/editar.html", instrutor = instrutor, title = "Editar Detalhes do Instrutor")


# Método para envio dos dados de alteração de um instrutor via POST
@instrutores_blueprint.route("/instrutores/<id>", methods = ["POST"])
def atualiza_instrutor(id):
    nome = request.form["nome"]
    sobrenome = request.form["sobrenome"]
    data_nascimento = request.form["data_nascimento"]
    endereco = request.form["endereco"]
    telefone = request.form["telefone"]
    instrutor_atualizado = Instrutor(nome, sobrenome, data_nascimento, endereco, telefone, id)
    conector_instrutor.edit(instrutor_atualizado)
    return redirect("/instrutores")


# Rota para a página de visualização de instrutores
@instrutores_blueprint.route("/instrutores/<id>")
def mostra_detalhes(id):
    instrutor_atividades = conector_instrutor.get_activities(id)
    instrutor = conector_instrutor.get_one(id)
    return render_template("/instrutores/mostrar.html", instrutor = instrutor, instrutor_atividades = instrutor_atividades, title = "Detalhes do Instrutor")


# Rota para deletar um instrutor
@instrutores_blueprint.route("/instrutores/<id>/delete")
def deleta_instrutor(id):
    conector_instrutor.delete_one(id)
    return redirect("/instrutores")



