# Módulo de conexão entre a classe e o banco de dados


# Imports
import conectores.conector_plano as plano
from database.run_sql import run_sql
from classes.atividade import Atividade
from classes.membro import Membro


# Listar todas as atividades
def get_all():

    atividades = []

    sql = "SELECT * FROM webuser.TB_ATIVIDADES"
    results = run_sql(sql)

    for row in results:
        
        tipo_plano = plano.get_one(row["tipo_plano"])
        
        atividade = Atividade(result["nome"],
                              result["instrutor"],
                              result["data"],
                              result["duracao"],
                              result["capacidade"],
                              tipo_plano,
                              result["ativo"],
                              result["id"])

        atividades.append(atividade)

    return atividades

# Função para obter membros
def get_members(id):

    membros = []

    sql = "SELECT webuser.TB_MEMBROS.* FROM webuser.TB_MEMBROS INNER JOIN webuser.TB_AGENDAMENTOS ON membros.id = webuser.TB_AGENDAMENTOS.membro WHERE webuser.TB_AGENDAMENTOS.atividade = %s"
    value = [id]

    results = run_sql(sql, value)

    for row in results:
        membro = Membro(row["nome"],
                        row["sobrenome"],
                        row["data_nascimento"],
                        row["endereco"],
                        row["telefone"],
                        row["email"],
                        row["tipo_plano"],
                        row["data_inicio"],
                        row["ativo"],
                        row["id"])

        membros.append(membro)
        
    return membros


# Função para obter atividades ativas ordenadas por data
def get_all_active():

    atividades = []

    sql = "SELECT * FROM webuser.TB_ATIVIDADES WHERE ativo = true ORDER BY data ASC"
    results = run_sql(sql)

    for row in results:
        
        tipo_plano = plano.get_one(row["tipo_plano"])
        
        atividade = Atividade(row["nome"],
                              row["instrutor"],
                              row["data"],
                              row["duracao"],
                              row["capacidade"],
                              tipo_plano,
                              row["ativo"],
                              row["id"])

        atividades.append(atividade)

    return atividades


# Função para obter atividades inativas
def get_all_inactive():

    atividades = []

    sql = "SELECT * FROM webuser.TB_ATIVIDADES WHERE ativo = false ORDER BY data ASC"
    results = run_sql(sql)

    for row in results:
        
        tipo_plano = plano.get_one(row["tipo_plano"])
        
        atividade = Atividade(result["nome"],
                              result["instrutor"],
                              result["data"],
                              result["duracao"],
                              result["capacidade"],
                              tipo_plano,
                              result["ativo"],
                              result["id"])

        atividades.append(atividade)

    return atividades


# Função para obter uma atividade ativa
def get_one(id):
    
    sql = "SELECT * FROM webuser.TB_ATIVIDADES WHERE ativo = true AND id = %s"
    value = [id]
    
    result = run_sql(sql, value)[0]

    if result is not None:
        tipo_plano = plano.get_one(result["tipo_plano"])
        
        atividade = Atividade(result["nome"],
                              result["instrutor"],
                              result["data"],
                              result["duracao"],
                              result["capacidade"],
                              tipo_plano,
                              result["ativo"],
                              result["id"])

    return atividade


# Função para cadastrar atividade
def new(atividade):
    
    sql = "INSERT INTO webuser.TB_ATIVIDADES( nome, instrutor, data, duracao, capacidade, tipo_plano, ativo ) VALUES (%s, %s, %s, %s, %s, %s, %s) RETURNING *;"
    values = [atividade.nome, atividade.instrutor, atividade.data, atividade.duracao, atividade.capacidade, atividade.tipo_plano, atividade.ativo]

    results = run_sql(sql, values)

    atividade.id = results[0]["id"]

    return atividade


# Função para deletar uma atividade
def delete_one(id):
    
    sql = "DELETE FROM webuser.TB_ATIVIDADES WHERE id = %s"
    value = [id]
    
    run_sql(sql, value)


# Função para editar uma atividade
def edit(atividade):
    
    sql = "UPDATE webuser.TB_ATIVIDADES SET ( nome, instrutor, data, duracao, capacidade, tipo_plano, ativo ) = (%s, %s, %s, %s, %s, %s, %s) WHERE id = %s;"
    values = [atividade.nome, atividade.instrutor, atividade.data, atividade.duracao, atividade.capacidade, atividade.tipo_plano, atividade.ativo]
    
    run_sql(sql, values)


