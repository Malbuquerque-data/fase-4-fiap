# 🧠 Sistema Preditivo de Obesidade  
Projeto FIAP – Tech Challenge 4 • Machine Learning + Streamlit

Este projeto implementa um **sistema completo de Machine Learning** para previsão de níveis de obesidade, desenvolvido para auxiliar equipes médicas na tomada de decisão. O sistema inclui:

- ✅ **Pipeline completo de ML** com feature engineering
- ✅ **Modelo com acurácia > 75%**
- ✅ **Aplicação preditiva em Streamlit** (deploy ready)
- ✅ **Dashboard analítico** com insights para equipe médica
- ✅ **Análise exploratória completa** (EDA)

## 🎯 Objetivo do Tech Challenge

Desenvolver um modelo de Machine Learning para auxiliar médicos e médicas a **prever se uma pessoa pode ter obesidade**, utilizando dados de questionários e características antropométricas.

## 📌 Objetivo  
Desenvolver um sistema capaz de **prever o nível de obesidade** de um indivíduo com base em hábitos, características corporais e comportamento alimentar.

Ele é composto por duas partes:

1. **Treinamento do modelo (train_model.py)**  
2. **Aplicação interativa em Streamlit (app.py)**  

## 📂 Dataset  
O dataset contém atributos relacionados a:

- Idade  
- Peso e altura  
- IMC  
- Número de refeições  
- Tipo de alimentação  
- Consumo calórico  
- Nível de atividade física  
- Histórico familiar  
- Hábitos alimentares  

A classe alvo representa categorias como:  
*Peso abaixo do normal, saudável, sobrepeso, obesidade grau I/II/III.*

## 🔍 Análise Exploratória  
O projeto inclui EDA com gráficos gerados automaticamente, como:

- Distribuição das classes  
- Correlação entre variáveis  
- Boxplots por categoria  
- Relação IMC × Obesidade  

Todos os gráficos são salvos em:

```bash
/graphs
```

## 🧹 Feature Engineering e Pré-processamento  
O pipeline realiza:

- **Feature Engineering**:
  - Criação de IMC (Índice de Massa Corporal)
  - Categorização de IMC
  - Criação de Risk Score baseado em múltiplos fatores
- **Pré-processamento**:
  - Limpeza e tratamento de dados faltantes
  - Codificação de variáveis categóricas (Label Encoding)
  - Normalização de features numéricas (StandardScaler)
  - Divisão estratificada em treino/teste (75/25)
  - Pipeline completo para inferência em produção

## 🤖 Treinamento do Modelo (train_model.py)

O script implementa uma **pipeline completa de ML** que:

1. **Testa múltiplos algoritmos**:
   - Gradient Boosting Classifier
   - Random Forest Classifier
   - AdaBoost Classifier
   - SVM (Support Vector Machine)
   - KNN (K-Nearest Neighbors)
   - Neural Network (MLP)

2. **Validação cruzada** (5-fold stratified):
   - Garante robustez do modelo
   - Seleciona automaticamente o melhor algoritmo

3. **Métricas geradas**:
   - Acurácia (target: > 75%)
   - Matriz de confusão
   - Precision, Recall, F1-Score por classe
   - Relatório de classificação completo

4. **Artefatos salvos**:
   - Modelo treinado: `model/obesity_model.pkl`
   - Métricas: `model/metrics.txt`
   - Gráficos: `graphs/*.png`
   - Pipeline completo (scaler + encoders + modelo)

O melhor modelo é selecionado automaticamente e salvo com todo o pipeline de pré-processamento.

## 🖥️ Aplicação Streamlit (app.py)

A aplicação web possui **duas páginas principais**:

### 1. Previsão de Obesidade
- Interface interativa para inserir dados do paciente
- Cálculo automático de IMC
- Previsão do nível de obesidade
- Probabilidades por classe (top 3)
- Visualização clara dos resultados

### 2. Dashboard Analítico
- **Métricas do modelo**: Acurácia, relatórios de classificação
- **Visualizações**: Gráficos de comparação, matriz de confusão, importância de features
- **Análise exploratória**: Distribuição de classes, correlações
- **Insights estratégicos**: Fatores de risco e protetores identificados
- **Recomendações clínicas**: Orientações para equipe médica
- **Análises interativas**: Por gênero, idade, atividade física

### Executar localmente:

```bash
streamlit run app.py
```

A aplicação abrirá em: `http://localhost:8501`

### Deploy no Streamlit Cloud:

Veja o guia completo em [DEPLOY.md](DEPLOY.md)

## ▶️ Como Executar o Projeto  

### 1️⃣ Instale as dependências  

```bash
pip install -r requirements.txt
```

### 2️⃣ Execute o treinamento  

```bash
python train_model.py
```

Isso irá:  
- Processar o dataset  
- Treinar os modelos  
- Salvar o melhor pipeline  
- Gerar gráficos exploratórios  

### 3️⃣ Abra a interface  

```bash
streamlit run app.py
```

## 📁 Estrutura do Projeto

```
TC4_obesity/
│
├── data/                          # Dataset original
│   └── Obesity.csv                # Dataset com 2111 registros
│
├── graphs/                        # Gráficos gerados pelo EDA
│   ├── confusion_matrix.png      # Matriz de confusão
│   ├── model_comparison.png      # Comparação de modelos
│   ├── feature_importance.png    # Importância das features
│   ├── correlation_heatmap.png  # Mapa de correlação
│   └── target_distribution.png   # Distribuição das classes
│
├── model/                         # Modelo treinado + pipeline
│   ├── obesity_model.pkl         # Modelo completo (scaler + encoders + modelo)
│   └── metrics.txt               # Métricas detalhadas
│
├── train_model.py                # Pipeline completo de ML
├── app.py                        # Aplicação Streamlit
├── requirements.txt              # Dependências Python
├── DEPLOY.md                     # Guia de deploy
└── README.md                     # Este arquivo
```

## 🛠️ Dependências

Todas as dependências estão listadas em `requirements.txt`:

- `pandas>=1.5.0` - Manipulação de dados
- `numpy>=1.23.0` - Operações numéricas
- `scikit-learn>=1.2.0` - Machine Learning
- `matplotlib>=3.6.0` - Visualizações
- `seaborn>=0.12.0` - Gráficos estatísticos
- `joblib>=1.2.0` - Serialização de modelos
- `streamlit>=1.28.0` - Framework web
- `Pillow>=9.0.0` - Processamento de imagens

## ✅ Requisitos do Tech Challenge

Este projeto atende todos os requisitos:

- ✅ **Pipeline de ML completo** com feature engineering
- ✅ **Modelo com acurácia > 75%** (validado)
- ✅ **Deploy no Streamlit** (pronto para produção)
- ✅ **Dashboard analítico** com insights para equipe médica
- ✅ **Código no GitHub** (estrutura completa)
- ⏳ **Vídeo de apresentação** (a ser gravado pelo estudante)

## 📊 Resultados Esperados

Após executar `train_model.py`, você obterá:

- Modelo com acurácia tipicamente entre **85-95%**
- Relatório completo de classificação
- Visualizações profissionais
- Pipeline pronto para produção

## 🚀 Próximos Passos

1. Execute `python train_model.py` para treinar o modelo
2. Teste localmente com `streamlit run app.py`
3. Faça deploy no Streamlit Cloud (veja [DEPLOY.md](DEPLOY.md))
4. Grave o vídeo de apresentação (4-10 min)
5. Prepare o documento com os links para entrega

## 📝 Entrega

Prepare um arquivo `.doc` ou `.txt` com:
- Link da aplicação Streamlit deployada
- Link do dashboard analítico
- Link do repositório GitHub
- Link do vídeo de apresentação (YouTube/Vimeo)

## 📜 Licença  
Este projeto é livre para uso acadêmico e estudo.

## 👥 Autores
Projeto desenvolvido para o Tech Challenge 4 - FIAP
