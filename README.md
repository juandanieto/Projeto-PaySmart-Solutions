# AuditIA de Cobranças — Confiabilidade na Cobrança de Taxas de Transação

> © 2025 Juan David Nieto García. A apresentação e os materiais contidos neste repositório são de autoria exclusiva de **Juan David Nieto García**. Todos os direitos reservados. É proibida a reprodução total ou parcial sem autorização prévia do autor.

---

## Descrição do Projeto

Este projeto implementa um pipeline de auditoria automatizada para detectar discrepâncias entre as regras de isenção documentadas e as cobranças efetivamente aplicadas em transações financeiras. O sistema utiliza um motor de regras combinado com técnicas de Machine Learning para identificar dois tipos de falha: isenções aplicadas indevidamente (*Isenção Fantasma*) e isenções contratuais não aplicadas (*Falha na Isenção*).

---

## Problema Abordado

Em sistemas de pagamento, a principal fonte de receita são as taxas de transação. A falta de documentação consolidada nas regras de isenção gera dois riscos financeiros simultâneos:

- **Vazamento de Receita** — isenções aplicadas pelo sistema sem respaldo contratual
- **Risco de Faturamento** — clientes isentos sendo cobrados indevidamente

---

## Arquitetura do Pipeline

O projeto é executado em três fases:

**Fase 1 — Geração e ETL dos Dados**  
O script `criar_dados.py` gera dados sintéticos simulando um ambiente realista com regras ocultas intencionais. São criados três arquivos CSV que representam as fontes da verdade do sistema:

| Arquivo | Conteúdo |
|---|---|
| `transacoes.csv` | 5.000 transações com cliente, valor, MCC e método de pagamento |
| `regras.csv` | Regras contratuais de isenção por cliente |
| `cobrancas.csv` | Taxas efetivamente cobradas pelo sistema de faturamento |

**Fase 2 — Classificação por Motor de Regras**  
O script `auditoria_paysmart.py` realiza o merge dos três datasets e calcula a discrepância entre a taxa esperada e a taxa cobrada real. Cada transação é classificada em uma de três categorias: *Correto*, *Isenção Fantasma* ou *Falha na Isenção*.

**Fase 3 — Análise com Machine Learning**  
Dois métodos são aplicados para descobrir padrões e validar os achados:

- **Isolation Forest** (não supervisionado) — detecta transações anômalas com base nas features numéricas e categóricas, sem utilizar os rótulos de classificação
- **Árvore de Decisão** (supervisionado) — treinada com `OneHotEncoder` para identificar as combinações de MCC e método de pagamento que explicam as isenções fantasmas

---

## Estrutura do Repositório

```
├── criar_dados.py                     # Geração dos dados sintéticos
├── auditoria_paysmart.py              # Pipeline principal de auditoria
├── transacoes.csv                     # Dataset de transações
├── regras.csv                         # Dataset de regras de isenção
├── cobrancas.csv                      # Dataset de cobranças reais
├── dados_carregados_merged.csv        # DataFrame mestre após merge e classificação
├── resumo_auditoria.png               # Gráfico de barras com o status das cobranças
├── causas_agregadas_fantasma.png      # Top 10 causas de Isenção Fantasma (MCC + método)
├── analise_isolation_forest.png       # Dispersão taxa esperada vs. taxa real com anomalias
├── arvore_decisao.png                 # Árvore de Decisão com as regras ocultas descobertas
└── Projeto.pdf                        # Apresentação executiva do projeto
```

---

## Dependências

```
pandas
numpy
matplotlib
seaborn
scikit-learn
```

Instale com:
```bash
pip install pandas numpy matplotlib seaborn scikit-learn
```

---

## Como Executar

**1. Gerar os dados sintéticos:**
```bash
python criar_dados.py
```

**2. Executar a auditoria completa:**
```bash
python auditoria_paysmart.py
```

Os quatro gráficos e o CSV consolidado serão gerados automaticamente na mesma pasta.

---

## Visualizações Geradas

- `resumo_auditoria.png` — distribuição geral das transações por status de cobrança
- `causas_agregadas_fantasma.png` — mineração de padrões nas isenções fantasmas por combinação de MCC e método de pagamento
- `analise_isolation_forest.png` — dispersão da taxa esperada versus taxa cobrada, com detecção de anomalias pelo Isolation Forest
- `arvore_decisao.png` — árvore de decisão com profundidade máxima de 4, revelando as regras ocultas do sistema de faturamento

---

© 2025 Juan David Nieto García — Todos os direitos reservados.
