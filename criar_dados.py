import pandas as pd
import numpy as np
import random

N_TRANSACOES = 5000  #Número de transações a gerar
N_CLIENTES = 50      #Número de clientes no sistema

#Listas de opções para simular variedade
LISTA_MCCS = ['5812', '5813', '8299', '7512', '5411', '5941', '8011', '4511']
LISTA_PAYMENT_METHODS = ['credit', 'debit', 'pix_debit', 'pix_credit', 'voucher']
CLIENTES_IDS = [f'C{100 + i}' for i in range(N_CLIENTES)]

print(f"Gerando {N_TRANSACOES} transações para {N_CLIENTES} clientes.")

#1. Criação Dataset `regras.csv` (Clientes)

#Decisão aleatória de quem tem isenção documentada
regras_data = []
for cliente in CLIENTES_IDS:
    # 80% de chance de ter taxa padrão, 20% de ter isenção documentada
    if random.random() < 0.20:
        taxa_esperada = 0.0
        isencao = True
    else:
        taxa_esperada = random.choice([0.025, 0.028, 0.030]) #Diferentes taxas
        isencao = False
    
    regras_data.append({
        'cliente_id': cliente,
        'taxa_padrao_esperada': taxa_esperada,
        'isencao_documentada': isencao
    })

df_regras = pd.DataFrame(regras_data)

#2. Criação Dataset `transacoes.csv`

transacoes_data = {
    'trans_id': [f'T{100000 + i}' for i in range(N_TRANSACOES)],
    'cliente_id': np.random.choice(CLIENTES_IDS, N_TRANSACOES),
    'valor': np.random.lognormal(5, 1, N_TRANSACOES).round(2) + 10.0, 
    'mcc': np.random.choice(LISTA_MCCS, N_TRANSACOES),
    'payment_method': np.random.choice(LISTA_PAYMENT_METHODS, N_TRANSACOES, p=[0.4, 0.3, 0.15, 0.1, 0.05]) 
}
df_transacoes = pd.DataFrame(transacoes_data)


# 3. Criação Dataset `cobrancas.csv`: cobrancas do mundo "real" (com erros)

#Juntamos transações e regras para simular a cobrança
df_sim = pd.merge(df_transacoes, df_regras, on='cliente_id')

# REGRA OCULTA 1: Todos do MCC '8299' (Educação) ficam isentos, mesmo que não devessem.
regra_oculta_mcc = '8299'

# REGRA OCULTA 2: Todas as transações 'pix_debit' ficam isentas.
regra_oculta_payment = 'pix_debit'

# FALHA DE SISTEMA 1: Clientes que DEVERIAM ser isentos (ex: 'C105', 'C110') são cobrados.
clientes_com_falha_isencao = np.random.choice(df_regras[df_regras['isencao_documentada'] == True]['cliente_id'], 2, replace=False).tolist()
print(f"Clientes que falharão na isenção (serão cobrados indevidamente): {clientes_com_falha_isencao}")


def simular_cobranca_real(row):
    taxa_esperada = row['taxa_padrao_esperada']
    
    # Aplicando Regras Ocultas (Isenção Fantasma)
    if row['mcc'] == regra_oculta_mcc:
        return 0.0 # Erro: Isenção fantasma (MCC)
    
    if row['payment_method'] == regra_oculta_payment:
        return 0.0 # Erro: Isenção fantasma (PIX)

    # Aplicando Falhas do Sistema (Cobrança Indevida)
    if row['cliente_id'] in clientes_com_falha_isencao:
        # Este cliente deveria ser isento (taxa_esperada = 0.0), mas será cobrado
        return 0.025 # Erro: Falha na isenção
        
    # Caso Normal (Sem erros)
    # Se não caiu em nenhuma exceção, cobra o que estava documentado
    return taxa_esperada

df_sim['taxa_cobrada_real'] = df_sim.apply(simular_cobranca_real, axis=1)

# Criando o dataframe final de cobranças
df_cobrancas = df_sim[['trans_id', 'taxa_cobrada_real']]
print("Base de Cobranças (com erros) creada.")

# 4. Salvar os arquivos
df_transacoes.to_csv("transacoes.csv", index=False)
df_regras.to_csv("regras.csv", index=False)
df_cobrancas.to_csv("cobrancas.csv", index=False)

print("\nArquivos CSV criados com sucesso!")
print(f" - transacoes.csv ({len(df_transacoes)} linhas)")
print(f" - regras.csv ({len(df_regras)} linhas)")
print(f" - cobrancas.csv ({len(df_cobrancas)} linhas)")