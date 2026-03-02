import streamlit as st
import pandas as pd
import pickle
import numpy as np
from PIL import Image
import plotly.express as px
import plotly.graph_objects as go

# Configuração da página
st.set_page_config(
    page_title="Sistema Preditivo de Obesidade",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Carregar modelo e artefatos ---
@st.cache_data
def load_model():
    """Carrega o modelo e pré-processador"""
    with open('model/obesity_model.pkl', 'rb') as f:
        data = pickle.load(f)
    return data

try:
    model_data = load_model()
    model = model_data['model']
    scaler = model_data['scaler']
    label_encoders = model_data['label_encoders']
    columns = model_data['columns']
    model_name = model_data.get('model_name', 'Random Forest')
    model_accuracy = model_data.get('accuracy', 0.9886)
except Exception as e:
    st.error(f"Erro ao carregar modelo: {str(e)}")
    st.stop()

# Mapeamento de níveis de obesidade para português
OBESITY_LEVELS_PT = {
    'Normal_Weight': 'Peso Normal',
    'Overweight_Level_I': 'Sobrepeso Nível I',
    'Overweight_Level_II': 'Sobrepeso Nível II',
    'Obesity_Type_I': 'Obesidade Tipo I',
    'Obesity_Type_II': 'Obesidade Tipo II',
    'Obesity_Type_III': 'Obesidade Tipo III',
    'Insufficient_Weight': 'Peso Insuficiente'
}

# Sidebar com informações
with st.sidebar:
    st.header("ℹ️ Sobre o Sistema")
    st.markdown(f"""
    Este sistema foi desenvolvido como parte do Tech Challenge 4.
    
    **Funcionalidades:**
    - Predição do nível de obesidade
    - Análise de probabilidades por classe
    - Dashboard com insights analíticos
    - Recomendações baseadas nos dados
    
    **Modelo:**
    - Algoritmo: {model_name}
    - Acurácia: {model_accuracy:.2%}
    """)
    st.markdown("---")
    st.markdown("**Desenvolvido para auxiliar profissionais de saúde**")

# Criar abas
tab1, tab2, tab3 = st.tabs(["🏠 Início", "🔮 Predição", "📊 Insights e Métricas"])

# ===== ABA 1: INÍCIO =====
with tab1:
    st.header("Bem-vindo ao Sistema Preditivo de Obesidade")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        ### 🎯 Objetivo
        
        Este sistema utiliza Machine Learning para auxiliar médicos e médicas 
        na previsão do nível de obesidade de pacientes, fornecendo ferramentas 
        para auxiliar na tomada de decisão clínica.
        
        ### 🔮 Predição
        
        Na aba **Predição**, você pode:
        - Preencher dados do paciente
        - Obter predição do nível de obesidade
        - Ver probabilidades por classe
        - Receber recomendações personalizadas
        """)
    
    with col2:
        st.markdown("""
        ### 📊 Insights e Métricas
        
        Na aba **Insights e Métricas**, você encontra:
        - Visualizações interativas dos dados
        - Análises e insights sobre obesidade
        - Métricas do modelo
        - Recomendações clínicas
        
        ### 📈 Recursos
        
        - Modelo com {:.2%} de acurácia
        - Interface intuitiva e profissional
        - Análises baseadas em dados reais
        """.format(model_accuracy))
    
    st.markdown("---")
    
    st.subheader("🚀 Como Usar")
    
    st.markdown("""
    1. **Para fazer uma predição:**
       - Navegue para a aba "🔮 Predição"
       - Preencha o formulário com os dados do paciente
       - Clique em "Fazer Predição"
       - Analise os resultados e recomendações
    
    2. **Para análise de dados:**
       - Navegue para a aba "📊 Insights e Métricas"
       - Explore os gráficos e insights apresentados
       - Analise as métricas do modelo
    """)
    
    st.markdown("---")
    
    st.subheader("📋 Informações Técnicas")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Acurácia do Modelo", f"{model_accuracy:.2%}")
    
    with col2:
        try:
            df_temp = pd.read_csv('data/Obesity.csv')
            st.metric("Total de Registros", f"{len(df_temp):,}")
        except:
            st.metric("Total de Registros", "2.111")
    
    with col3:
        st.metric("Variáveis de Entrada", len(columns) if isinstance(columns, list) else len(columns))

# ===== ABA 2: PREDIÇÃO =====
with tab2:
    st.header("🔮 Predição de Nível de Obesidade")
    st.markdown("Preencha os dados abaixo para obter uma predição do nível de obesidade.")
    
    # Função para criar formulário
    def create_input_form():
        """Cria formulário de entrada de dados"""
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("📊 Dados Demográficos")
            gender = st.selectbox("Gênero", ["Male", "Female"])
            age = st.number_input("Idade", min_value=1, max_value=120, value=30)
            height = st.number_input("Altura (metros)", min_value=0.5, max_value=2.5, value=1.70, step=0.01)
            weight = st.number_input("Peso (kg)", min_value=10.0, max_value=300.0, value=70.0, step=0.1)
            
            # Calcular IMC
            if height > 0:
                bmi = weight / (height ** 2)
                st.info(f"**IMC Calculado:** {bmi:.2f} kg/m²")
        
        with col2:
            st.subheader("🍽️ Hábitos Alimentares")
            family_history = st.selectbox("Histórico familiar de excesso de peso", ["yes", "no"])
            favc = st.selectbox("Come alimentos altamente calóricos com frequência?", ["yes", "no"])
            fcvc = st.number_input("Frequência de consumo de vegetais (1-3)", min_value=1.0, max_value=3.0, value=2.0, step=0.1)
            ncp = st.number_input("Número de refeições principais diárias (1-4)", min_value=1.0, max_value=4.0, value=3.0, step=0.1)
            caec = st.selectbox("Come algo entre as refeições?", ["no", "Sometimes", "Frequently", "Always"])
            ch2o = st.number_input("Quantidade de água diária (1-3)", min_value=1.0, max_value=3.0, value=2.0, step=0.1)
            scc = st.selectbox("Monitora as calorias ingeridas?", ["yes", "no"])
        
        st.markdown("---")
        
        col3, col4 = st.columns(2)
        
        with col3:
            st.subheader("🏃 Estilo de Vida")
            smoke = st.selectbox("Fuma?", ["yes", "no"])
            faf = st.number_input("Frequência de atividade física (0-3)", min_value=0.0, max_value=3.0, value=1.0, step=0.1)
            tue = st.number_input("Tempo em dispositivos tecnológicos (0-2)", min_value=0.0, max_value=2.0, value=1.0, step=0.1)
            calc = st.selectbox("Frequência de consumo de álcool", ["no", "Sometimes", "Frequently", "Always"])
        
        with col4:
            st.subheader("🚗 Transporte")
            mtrans = st.selectbox("Meio de transporte", [
                "Public_Transportation",
                "Automobile",
                "Walking",
                "Motorbike",
                "Bike"
            ])
        
        return {
            'Gender': gender,
            'Age': age,
            'Height': height,
            'Weight': weight,
            'family_history': family_history,
            'FAVC': favc,
            'FCVC': fcvc,
            'NCP': ncp,
            'CAEC': caec,
            'SMOKE': smoke,
            'CH2O': ch2o,
            'SCC': scc,
            'FAF': faf,
            'TUE': tue,
            'CALC': calc,
            'MTRANS': mtrans
        }
    
    # Função para fazer predição
    def make_prediction(input_data):
        """Faz predição usando o modelo treinado"""
        try:
            df_input = pd.DataFrame([input_data])
            
            # --- Feature Engineering (igual ao treinamento) ---
            # 1. Criar IMC
            df_input['BMI'] = df_input['Weight'] / (df_input['Height'] ** 2)
            
            # 2. Criar categoria de IMC
            def categorize_bmi(bmi):
                if bmi < 18.5:
                    return 'Underweight'
                elif bmi < 25:
                    return 'Normal'
                elif bmi < 30:
                    return 'Overweight'
                elif bmi < 35:
                    return 'Obese_I'
                elif bmi < 40:
                    return 'Obese_II'
                else:
                    return 'Obese_III'
            
            df_input['BMI_Category'] = df_input['BMI'].apply(categorize_bmi)
            
            # 3. Codificar todas as variáveis categóricas (exceto Obesity)
            for col, le in label_encoders.items():
                if col in df_input.columns and col != 'Obesity':
                    try:
                        df_input[col] = le.transform(df_input[col].astype(str))
                    except:
                        df_input[col] = 0
            
            # 4. Criar Risk Score após codificação
            try:
                favc_le = label_encoders.get('FAVC')
                family_le = label_encoders.get('family_history')
                
                if favc_le is not None and family_le is not None:
                    yes_favc_idx = None
                    yes_family_idx = None
                    
                    for i, val in enumerate(favc_le.classes_):
                        if str(val).lower() == 'yes':
                            yes_favc_idx = i
                            break
                    
                    for i, val in enumerate(family_le.classes_):
                        if str(val).lower() == 'yes':
                            yes_family_idx = i
                            break
                    
                    if yes_favc_idx is None:
                        yes_favc_idx = 1 if len(favc_le.classes_) > 1 else 0
                    if yes_family_idx is None:
                        yes_family_idx = 1 if len(family_le.classes_) > 1 else 0
                    
                    df_input['Risk_Score'] = (
                        (df_input['FAVC'] == yes_favc_idx).astype(int) +
                        (df_input['family_history'] == yes_family_idx).astype(int) -
                        (df_input['FAF'] / 3.0) +
                        (df_input['TUE'] / 2.0)
                    )
                else:
                    df_input['Risk_Score'] = df_input['FAVC'] + df_input['family_history'] - (df_input['FAF'] / 3.0) + (df_input['TUE'] / 2.0)
            except:
                df_input['Risk_Score'] = df_input.get('FAVC', 0) + df_input.get('family_history', 0) - (df_input.get('FAF', 0) / 3.0) + (df_input.get('TUE', 0) / 2.0)
            
            # 5. Garantir que todas as colunas esperadas estejam presentes e na ordem correta
            expected_cols = columns if isinstance(columns, list) else list(columns)
            for col in expected_cols:
                if col not in df_input.columns:
                    df_input[col] = 0
            
            # Reordenar colunas na ordem esperada pelo modelo
            df_input = df_input[expected_cols]
            
            df_scaled = scaler.transform(df_input)
            prediction = model.predict(df_scaled)[0]
            probabilities = model.predict_proba(df_scaled)[0]
            classes = model.classes_
            
            return prediction, probabilities, classes
            
        except Exception as e:
            st.error(f"Erro ao fazer predição: {str(e)}")
            return None, None, None
    
    # Interface principal
    st.subheader("📝 Formulário de Entrada")
    
    # Criar formulário
    input_data = create_input_form()
    
    # Botão de predição
    st.markdown("---")
    col_btn1, col_btn2, col_btn3 = st.columns([1, 2, 1])
    
    with col_btn2:
        predict_button = st.button("🔮 Fazer Predição", type="primary", use_container_width=True)
    
    # Fazer predição
    if predict_button:
        with st.spinner("Processando predição..."):
            prediction, probabilities, classes = make_prediction(input_data)
            
            if prediction is not None:
                st.markdown("---")
                st.header("📊 Resultado da Predição")
                
                # Converter prediction para string se necessário
                # Garantir que seja sempre uma string Python, não numpy type
                if isinstance(prediction, (np.integer, int, np.int64, np.int32)):
                    # Se prediction é um índice, converter para string usando classes
                    pred_idx = int(prediction)  # Converter para int Python
                    if pred_idx < len(classes):
                        prediction_str = str(classes[pred_idx])  # Converter explicitamente para string
                    else:
                        prediction_str = str(prediction)
                else:
                    prediction_str = str(prediction)
                
                # Garantir que prediction_str é uma string Python, não numpy
                prediction_str = str(prediction_str)
                
                # Resultado principal
                prediction_pt = OBESITY_LEVELS_PT.get(prediction_str, prediction_str)
                
                # Container para resultado
                result_container = st.container()
                with result_container:
                    col1, col2, col3 = st.columns([1, 2, 1])
                    with col2:
                        st.markdown(f"### 🎯 Nível de Obesidade Previsto:")
                        st.markdown(f"# {prediction_pt}")
                        
                        # Probabilidade da classe predita
                        if isinstance(prediction, (np.integer, int)):
                            pred_idx = prediction
                        else:
                            pred_idx = list(classes).index(prediction_str) if prediction_str in classes else 0
                        confidence = probabilities[pred_idx] * 100
                        st.progress(confidence / 100)
                        st.caption(f"Confiança: {confidence:.2f}%")
                
                # Probabilidades por classe
                st.markdown("---")
                st.subheader("📈 Probabilidades por Classe")
                
                # Criar DataFrame com probabilidades
                prob_df = pd.DataFrame({
                    'Nível de Obesidade': [OBESITY_LEVELS_PT.get(c, c) for c in classes],
                    'Probabilidade (%)': [p * 100 for p in probabilities]
                }).sort_values('Probabilidade (%)', ascending=False)
                
                # Gráfico de barras
                fig = px.bar(
                    prob_df,
                    x='Nível de Obesidade',
                    y='Probabilidade (%)',
                    title='Probabilidades por Classe',
                    color='Probabilidade (%)',
                    color_continuous_scale='Reds'
                )
                fig.update_layout(showlegend=False)
                st.plotly_chart(fig, width='stretch')
                
                # Tabela
                st.dataframe(prob_df, width='stretch', hide_index=True)
                
                # Recomendações
                st.markdown("---")
                st.subheader("💡 Recomendações")
                
                if 'Obesity' in prediction_str or 'Overweight' in prediction_str:
                    st.warning("""
                    **Atenção:** O modelo indica risco de sobrepeso/obesidade. Recomenda-se:
                    - Consultar um profissional de saúde
                    - Avaliar hábitos alimentares
                    - Aumentar atividade física regular
                    - Monitorar peso e IMC periodicamente
                    """)
                elif prediction_str == 'Normal_Weight':
                    st.success("""
                    **Peso Normal:** Mantenha hábitos saudáveis:
                    - Continue com alimentação balanceada
                    - Mantenha atividade física regular
                    - Monitore peso periodicamente
                    """)
                else:
                    st.info("""
                    **Peso Insuficiente:** Consulte um nutricionista para:
                    - Avaliar necessidades nutricionais
                    - Desenvolver plano alimentar adequado
                    - Monitorar ganho de peso saudável
                    """)

# ===== ABA 3: INSIGHTS E MÉTRICAS =====
with tab3:
    st.title("📊 Dashboard Analítico - Previsão de Obesidade")
    st.markdown("### Visão estratégica para equipe médica")
    
    # Carregar dados para análise
    try:
        df = pd.read_csv('https://raw.githubusercontent.com/Malbuquerque-data/fase-4-fiap/refs/heads/main/Obesity.csv')
        
        # Métricas principais
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total de Pacientes", len(df))
        with col2:
            st.metric("Taxa de Obesidade", f"{(df['Obesity'].str.contains('Obesity', case=False).sum() / len(df) * 100):.1f}%")
        with col3:
            avg_age = df['Age'].mean()
            st.metric("Idade Média", f"{avg_age:.1f} anos")
        with col4:
            avg_bmi = (df['Weight'] / (df['Height'] ** 2)).mean()
            st.metric("IMC Médio", f"{avg_bmi:.1f} kg/m²")
        
        st.markdown("---")
        
        # Seção 1: Desempenho do Modelo
        st.header("🎯 Desempenho do Modelo Preditivo")
        
        try:
            with open('model/metrics.txt', 'r', encoding='utf-8') as f:
                metrics_text = f.read()
            st.text_area("Métricas Detalhadas", metrics_text, height=200)
        except:
            st.info("Execute train_model.py para gerar as métricas")
        
        st.markdown("### 🔹 Comparação de Acurácia entre Modelos")
        try:
            img_comp = Image.open("graphs/model_comparison.png")
            st.image(img_comp, caption="Comparação de Acurácia entre os Modelos", width='stretch')
        except:
            st.warning("Imagem não encontrada. Execute train_model.py primeiro.")

        st.markdown("### 🔹 Matriz de Confusão")
        try:
            img_conf = Image.open("graphs/confusion_matrix.png")
            st.image(img_conf, caption="Matriz de Confusão do Melhor Modelo", width='stretch')
        except:
            st.warning("Matriz de confusão não encontrada.")

        st.markdown("### 🔹 Importância das Features")
        try:
            img_feat = Image.open("graphs/feature_importance.png")
            st.image(img_feat, caption="Top 15 Features Mais Importantes", width='stretch')
        except:
            st.warning("Gráfico de importância não encontrado.")

        st.markdown("---")
        
        # Seção 2: Análise Exploratória
        st.header("📈 Análise Exploratória dos Dados")
        
        st.markdown("### 🔹 Distribuição das Classes de Obesidade")
        try:
            img_dist = Image.open("graphs/target_distribution.png")
            st.image(img_dist, caption="Distribuição das Classes", width='stretch')
        except:
            # Criar gráfico inline se não existir
            import matplotlib.pyplot as plt
            fig, ax = plt.subplots(figsize=(10, 6))
            df['Obesity'].value_counts().plot(kind='bar', ax=ax, color='steelblue')
            ax.set_title("Distribuição das Classes de Obesidade")
            ax.set_xlabel("Nível de Obesidade")
            ax.set_ylabel("Frequência")
            plt.xticks(rotation=45, ha='right')
            plt.tight_layout()
            st.pyplot(fig)
        
        # Gráfico interativo com Plotly
        obesity_counts = df['Obesity'].value_counts()
        fig_dist = px.bar(
            x=obesity_counts.index,
            y=obesity_counts.values,
            labels={'x': 'Nível de Obesidade', 'y': 'Frequência'},
            title='Distribuição de Níveis de Obesidade (Interativo)',
            color=obesity_counts.values,
            color_continuous_scale='Reds'
        )
        fig_dist.update_layout(showlegend=False)
        st.plotly_chart(fig_dist, width='stretch')
        
        st.markdown("### 🔹 Correlação entre Variáveis")
        try:
            img_corr = Image.open("graphs/correlation_heatmap.png")
            st.image(img_corr, caption="Mapa de Correlação entre Variáveis", width='stretch')
        except:
            st.warning("Mapa de correlação não encontrado.")
        
        st.markdown("---")
        
        # Seção 3: Análises Interativas
        st.header("📊 Análises Interativas")
        
        # Análise por gênero
        st.markdown("#### Distribuição por Gênero")
        gender_obesity = pd.crosstab(df['Gender'], df['Obesity'], normalize='index') * 100
        fig_gender = px.bar(
            gender_obesity,
            barmode='group',
            title='Distribuição de Obesidade por Gênero',
            labels={'value': 'Percentual (%)', 'Gender': 'Gênero'}
        )
        st.plotly_chart(fig_gender, width='stretch')
        
        # Análise por idade
        st.markdown("#### Relação Idade vs Obesidade")
        df['BMI'] = df['Weight'] / (df['Height'] ** 2)
        df['Age_Group'] = pd.cut(df['Age'], bins=[0, 20, 30, 40, 50, 100], labels=['<20', '20-30', '30-40', '40-50', '50+'])
        age_bmi = df.groupby('Age_Group', observed=True)['BMI'].mean()
        fig_age = px.line(
            x=age_bmi.index,
            y=age_bmi.values,
            title='IMC Médio por Faixa Etária',
            labels={'x': 'Faixa Etária', 'y': 'IMC Médio'},
            markers=True
        )
        st.plotly_chart(fig_age, width='stretch')
        
        # Scatter plot: Idade vs IMC
        fig_scatter = px.scatter(
            df,
            x='Age',
            y='BMI',
            color='Obesity',
            title='Relação entre Idade e IMC',
            labels={'Age': 'Idade', 'BMI': 'IMC'},
            hover_data=['Gender', 'Weight', 'Height']
        )
        st.plotly_chart(fig_scatter, width='stretch')
        
        # Análise de atividade física
        st.markdown("#### Impacto da Atividade Física")
        activity_obesity = pd.crosstab(df['FAF'], df['Obesity'].str.contains('Obesity', case=False), normalize='index') * 100
        fig_activity = px.bar(
            activity_obesity,
            title='Taxa de Obesidade por Nível de Atividade Física',
            labels={'value': 'Taxa de Obesidade (%)', 'FAF': 'Frequência de Atividade Física'}
        )
        st.plotly_chart(fig_activity, width='stretch')
        
        st.markdown("---")
        
        # Seção 4: Insights para Equipe Médica
        st.header("💡 Insights Estratégicos para Equipe Médica")
        
        # Análises específicas
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("🔍 Fatores de Risco Identificados")
            st.markdown("""
            - **Histórico Familiar**: Pacientes com histórico familiar têm maior risco
            - **Alimentos Calóricos (FAVC)**: Consumo frequente aumenta significativamente o risco
            - **Sedentarismo**: Baixa atividade física (FAF) está correlacionada com obesidade
            - **Tempo em Dispositivos (TUE)**: Maior tempo de uso aumenta o risco
            - **Poucas Refeições (NCP)**: Menos refeições principais pode indicar padrões não saudáveis
            """)
        
        with col2:
            st.subheader("✅ Fatores Protetores")
            st.markdown("""
            - **Atividade Física Regular (FAF)**: Reduz significativamente o risco
            - **Consumo de Vegetais (FCVC)**: Hábito protetor importante
            - **Monitoramento Calórico (SCC)**: Consciência alimentar ajuda na prevenção
            - **Hidratação Adequada (CH2O)**: Importante para metabolismo
            - **Transporte Ativo**: Caminhar ou usar bicicleta reduz risco
            """)
        
        st.markdown("---")
        
        # Recomendações
        st.header("🎯 Recomendações Clínicas")
        st.markdown("""
        ### Para Prevenção e Tratamento:
        
        1. **Triagem Familiar**: Priorizar pacientes com histórico familiar de obesidade
        2. **Educação Alimentar**: Focar em redução de alimentos altamente calóricos
        3. **Promoção de Atividade Física**: Incentivar exercícios regulares
        4. **Monitoramento de IMC**: Acompanhamento regular para detecção precoce
        5. **Redução de Tempo em Dispositivos**: Orientar sobre tempo de tela
        6. **Padrões Alimentares**: Encorajar refeições regulares e balanceadas
        
        ### Uso do Modelo Preditivo:
        - O modelo pode auxiliar na **identificação precoce** de risco
        - Use como **ferramenta complementar** ao diagnóstico clínico
        - Considere os fatores de risco identificados no **aconselhamento** ao paciente
        - **Validação clínica** sempre necessária para decisões de tratamento
        """)
        
    except Exception as e:
        st.error(f"Erro ao carregar dados: {str(e)}")
        st.info("Certifique-se de que o arquivo data/Obesity.csv existe")

# Rodapé
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: gray;'>
    <p>Sistema desenvolvido para o Tech Challenge 4 - FIAP | Uso exclusivo para fins educacionais</p>
</div>
""", unsafe_allow_html=True)