# Módulo de conexão ao banco de dados

# pip install psycopg2
# https://pypi.org/project/psycopg2/

# Imports
import os
import psycopg2
import psycopg2.extras as ext

# Função
def run_sql(sql, values = None):

    # Variáveis de controle
    conn = None
    results = []

    # Conexão ao banco de dados
    try:
        conn = psycopg2.connect("host=localhost port=5432 dbname=dbapp user=postgres password=dsacademy")
        cur = conn.cursor(cursor_factory = ext.DictCursor)
        cur.execute(sql, values)
        conn.commit()
        results = cur.fetchall()
        cur.close()
    except (Exception, psycopg2.DatabaseError) as error:
        print(error)
    finally:
        if conn is not None:
            conn.close()
            
    return results
