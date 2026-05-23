import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import IsolationForest
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder

# *** Carregar dados ***
def carregar_dados_reais():
    """Carrega dados de arquivos CSV."""
    try:
        df_transacoes = pd.read_csv("transacoes.csv")
        df_regras = pd.read_csv("regras.csv")
        df_cobrancas = pd.read_csv("cobrancas.csv")

        df_merged = pd.merge(df_transacoes, df_regras, on='cliente_id', how='left')
        df_final = pd.merge(df_merged, df_cobrancas, on='trans_id', how='left')

        df_final['taxa_padrao_esperada'] = df_final['taxa_padrao_esperada'].fillna(0.025) 
        df_final['taxa_cobrada_real'] = df_final['taxa_cobrada_real'].fillna(0.0) 
        df_final['isencao_documentada'] = df_final['isencao_documentada'].fillna(False)

        return df_final
    except FileNotFoundError as e:
        print(f"Erro: Arquivo não encontrado. Verifique se os arquivos CSV estão na mesma pasta.")
        print(f"Arquivo faltante: {e.filename}")
        return pd.DataFrame() 

# *** Motor de Regras e Análise de Discrepância ***
def analisar_discrepancias(df):
    """Calcula a taxa esperada e a discrepância."""
    df['taxa_esperada'] = df['taxa_padrao_esperada']
    df['discrepancia'] = df['taxa_cobrada_real'] - df['taxa_esperada']
    
    def classificar_problema(row):
        if round(row['discrepancia'], 5) == 0:
            return "Correto"
        if row['discrepancia'] > 0:
            return "Falha na Isencao" #(Cobrado a mais)
        if row['discrepancia'] < 0:
            return "Isencao Fantasma" #(Cobrado a menos)
    
    df['status_cobranca'] = df.apply(classificar_problema, axis=1)
    return df

# *** Análise 1: Análise de Causa Raiz (Agregação Simples) ***
def encontrar_regras_agregadas_e_plotar(df_anomalias):
    """Usa Agregação Simples (Mineração de Padrões) para encontrar padrões."""
    print("\n--- Análise 1: Mineração de Padrões ---")
    
    df_anomalias['causa_combinada'] = df_anomalias['mcc'].astype(str) + ' + ' + df_anomalias['payment_method'].astype(str)
    sugestoes_regras = df_anomalias['causa_combinada'].value_counts().head(10).reset_index()
    sugestoes_regras.columns = ['causa_combinada', 'contagem']
    
    print("Possíveis Regras de Isenção Não Documentadas (Top 10):")
    print(sugestoes_regras.to_markdown(index=False))

    try:
        if sugestoes_regras.empty:
            print("Não há dados de 'Isenção Fantasma' para plotar.")
            return None
        plt.figure(figsize=(12, 8)) 
        sns.barplot(data=sugestoes_regras, x='contagem', y='causa_combinada', orient='h', palette='viridis')
        plt.title("Top 10 Causas de 'Isenção Fantasma'", fontsize=16)
        plt.xlabel("Número de Transações com Erro", fontsize=12)
        plt.ylabel("Causa (MCC + Método de Pagamento)", fontsize=12)
        plt.tight_layout()
        plt.savefig("causas_agregadas_fantasma.png")
        plt.close()
    except Exception as e:
        print(f"Erro ao gerar gráfico de causas agregadas: {e}")
    return sugestoes_regras

# *** Função para plotar o resumo da auditoria ***
def plotar_resumo_auditoria(df):
    """Gera um gráfico de barras com o resumo da auditoria."""
    try:
        ordem_status = ['Correto', 'Isencao Fantasma', 'Falha na Isencao']
        plt.figure(figsize=(10, 6))
        
        ax = sns.countplot(
            data=df, 
            x='status_cobranca', 
            order=ordem_status, 
            palette='mako',
            hue='status_cobranca', 
            legend=False         
        )

        for p in ax.patches:
            height = p.get_height() 
            ax.annotate(
                f'{height:.0f}',    
                (p.get_x() + p.get_width() / 2., height),
                ha='center',        
                va='bottom',        
                xytext=(0, 5),     
                textcoords='offset points'
            )

        plt.title('Resumo da Auditoria de Cobranças', fontsize=16)
        plt.ylabel('Contagem de Transações', fontsize=12)
        plt.xlabel('Status da Cobrança', fontsize=12)
        plt.tight_layout()
        plt.savefig("resumo_auditoria.png")
        plt.close()
    except Exception as e:
        print(f"Erro ao gerar gráfico de resumo: {e}")

# *** Helper para preparar dados para ML ***
def preparar_dados_para_ml(df):
    """Label Encoding"""
    df_ml = df.copy()
    encoders = {}
    features_categoricas = ['cliente_id', 'mcc', 'payment_method', 'status_cobranca']
    
    for col in features_categoricas:
        if col in df_ml.columns:
            le = LabelEncoder()
            df_ml[col] = df_ml[col].astype(str) 
            df_ml[col] = le.fit_transform(df_ml[col])
            encoders[col] = le
            
    if 'isencao_documentada' in df_ml.columns:
        df_ml['isencao_documentada'] = df_ml['isencao_documentada'].astype(int)
        
    return df_ml, encoders

# *** Análise 2: Isolation Forest ***
def rodar_isolation_forest(df):
    """Aplica Isolation Forest para detectar anomalias e plota o resultado."""
    print("\n--- Análise 2: Isolation Forest ---")
    
    df_ml, _ = preparar_dados_para_ml(df.copy()) # Usa a cópia com LabelEncoder
    
    features = ['valor', 'mcc', 'payment_method', 'cliente_id', 'taxa_esperada', 'discrepancia']
    df_features = df_ml[features].fillna(0)
    
    actual_contamination = (df['status_cobranca'] != 'Correto').mean()
    if actual_contamination == 0: actual_contamination = 'auto'
    
    model = IsolationForest(contamination=actual_contamination, random_state=42)
    model.fit(df_features)
    
    df['anomaly_score'] = model.decision_function(df_features)
    df['anomaly_flag'] = model.predict(df_features) # -1 = Anomalia, 1 = Normal

    print(f"Modelo Isolation Forest treinado com contaminação de {actual_contamination:.2f}")

    try:
        
        df = df.rename(columns={
            'status_cobranca': 'Status de Cobranca',
            'anomaly_flag': 'Deteccao do Modelo'
        })

        tolerancia = 1e-6
        df['Deteccao do Modelo'] = np.where(
            abs(df['taxa_esperada'] - df['taxa_cobrada_real']) <= tolerancia,
            'Normal',
            'Anomalia'
        )

        plt.figure(figsize=(16, 9))
        sns.set_theme(style="whitegrid")

        ax = sns.scatterplot(
            data=df,
            x='taxa_esperada',
            y='taxa_cobrada_real',
            hue='Status de Cobranca',
            style='Deteccao do Modelo',
            markers={'Anomalia': 'X', 'Normal': 'o'},
            palette={'Correto': '#2ca02c', 'Isencao Fantasma': '#1f77b4', 'Falha na Isencao': '#d62728'},
            s=120,
            edgecolor='black',
            alpha=0.8
        )

        plt.plot(
            [df['taxa_esperada'].min(), df['taxa_esperada'].max()],
            [df['taxa_esperada'].min(), df['taxa_esperada'].max()],
            color='gray',
            linestyle='--',
            linewidth=2,
            label='Cobrança Correta (Esperado = Real)'
        )

        plt.title("Análise de Anomalias com Isolation Forest", fontsize=18, weight='bold', pad=15)
        plt.xlabel("Taxa Esperada (Regra)", fontsize=13)
        plt.ylabel("Taxa Cobrada (Real)", fontsize=13)
        plt.grid(True, linestyle='--', alpha=0.6)

        plt.legend(
            bbox_to_anchor=(1.03, 1),
            loc='upper left',
            borderaxespad=0,
            frameon=True,
            shadow=True
        )

        plt.tight_layout(rect=[0, 0, 0.82, 1])
        plt.savefig("analise_isolation_forest.png", dpi=300, bbox_inches="tight")
        plt.close()


    except Exception as e:
        print(f"Erro ao gerar gráfico do Isolation Forest: {e}")
    
    return df

# *** Análise 3: Árvore de Decisão ***
def rodar_arvore_decisao(df):
    """Treina uma Árvore de Decisão com OneHotEncoder para gerar um gráfico legível."""
    print("\n--- Análise 3: Árvore de Decisão (Supervisionado) ---")
    
    df_ml = df.copy()
    
    # 3.1 Preparar o Target (y)
    target = 'status_cobranca'
    le = LabelEncoder()
    y = le.fit_transform(df_ml[target])
    class_names = le.classes_
    
    # 3.2 Preparar as Features (X)
    features_categoricas = ['mcc', 'payment_method']
    features_numericas = ['valor', 'isencao_documentada']
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('ohe', OneHotEncoder(handle_unknown='ignore', sparse_output=False), features_categoricas),
            ('num', 'passthrough', features_numericas)
        ],
        remainder='drop'
    )
    
    X_raw = df_ml[features_categoricas + features_numericas]
    X_raw[features_categoricas] = X_raw[features_categoricas].fillna('Missing')
    X_raw[features_numericas] = X_raw[features_numericas].fillna(0)

    X = preprocessor.fit_transform(X_raw)
    
    # 3.3 Obter os nomes legíveis das features
    try:
        feature_names_out = preprocessor.named_transformers_['ohe'].get_feature_names_out(features_categoricas)
    except AttributeError:
        feature_names_out = preprocessor.named_transformers_['ohe'].get_feature_names()
        
    all_feature_names = list(feature_names_out) + features_numericas
    
    # 3.4 Treinar o modelo
    model = DecisionTreeClassifier(max_depth=4, random_state=42)
    model.fit(X, y)
    
    print("Modelo de Árvore de Decisão treinado com OneHotEncoder.")

    # 3.5 Plotar o gráfico 
    try:
        plt.figure(figsize=(35, 20)) 
        plot_tree(
            model,
            feature_names=all_feature_names,
            class_names=class_names,
            filled=True,
            rounded=True,
            fontsize=23,
            precision=2
        )
        
        plt.title("Árvore de Decisão: Regras Ocultas", fontsize=35, weight='bold')
        plt.savefig("arvore_decisao.png", dpi=200) # DPI maior
        plt.close()
    except Exception as e:
        print(f"Erro ao gerar gráfico da Árvore de Decisão: {e}")

if __name__ == "__main__":
    
    # 1. Obter dados
    df_principal = carregar_dados_reais()

    if not df_principal.empty:
        
        try:
            df_principal.to_csv("dados_carregados_merged.csv", index=False)
            print("\nArquivo 'dados_carregados_merged.csv' salvo com sucesso (dados brutos antes da análise).")
        except Exception as e:
            print(f"Erro ao salvar CSV bruto: {e}")

        # 2. Analisar
        df_analisado = analisar_discrepancias(df_principal)

        # 3. Exibir Resumo (Texto)
        print("\n--- Resumo da AuditorIA de Cobranças (PaySmart) ---")
        print(df_analisado['status_cobranca'].value_counts(normalize=True).to_markdown(floatfmt=".2f"))

        # --- Gráfico 1: Resumo ---
        plotar_resumo_auditoria(df_analisado)

        # --- Análise 1: Agregação (Mineração de Padrões) ---
        df_fantasmas = df_analisado[df_analisado['status_cobranca'] == 'Isencao Fantasma']
        if not df_fantasmas.empty:
            encontrar_regras_agregadas_e_plotar(df_fantasmas)
        else:
            print("\nNenhuma 'Isenção Fantasma' encontrada.")
            
        # --- Análise 2: Isolation Forest ---
        df_com_anomalias = rodar_isolation_forest(df_analisado.copy()) 
        
        # --- Análise 3: Árvore de Decisão ---
        rodar_arvore_decisao(df_analisado.copy()) 

        # 5. Focar nas "Falhas na Isenção" (Texto)
        df_falhas_isencao = df_analisado[df_analisado['status_cobranca'] == 'Falha na Isencao']
        if not df_falhas_isencao.empty:
            print(f"\nClientes cobrados indevidamente (falha na isenção):")
            print(list(df_falhas_isencao['cliente_id'].unique()))
        else:
            print("\nNenhuma 'Falha na Isenção' encontrada.")