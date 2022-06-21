# Módulo de conexão entre a classe e o banco de dados


# Imports
from classes.plano import TipoPlano
from database.run_sql import run_sql#conector do banco de dados 


# Função para obter todos os planos
def get_all():

    tipos_planos = []

    sql = "SELECT * FROM webuser.TB_PLANOS"
    results = run_sql(sql)

    for row in results:
        tipo_plano = TipoPlano(row["plano"], row["id"])
        tipos_planos.append(tipo_plano)

    return tipos_planos


# Função para obter um plano
def get_one(id):
    
    sql = "SELECT * FROM webuser.TB_PLANOS WHERE id = %s"
    value = [id]
    
    result = run_sql(sql, value)[0]

    if result is not None:
        tipo_plano = TipoPlano(result["plano"], result["id"])

    return tipo_plano


# Função para cria um tipo de plano
def new(tipo_plano):
    
    sql = "INSERT INTO webuser.TB_PLANOS ( plano ) VALUES ( %s ) RETURNING *;"
    values = [tipo_plano.plano]
    
    results = run_sql(sql, values)
    
    tipo_plano.id = results[0]["id"]

    return tipo_plano


# Função para deletar um plano
def delete_one(id):
    
    sql = "DELETE FROM webuser.TB_PLANOS WHERE id = %s"
    value = [id]
    
    run_sql(sql, value)


# Função para alterar um plano
def edit(tipo_plano):
    
    sql = "UPDATE webuser.TB_PLANOS SET (plano) = (%s) WHERE id = %s;"
    values = [tipo_plano.plano, tipo_plano.id]

    run_sql(sql, values)


