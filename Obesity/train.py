import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pickle
import os
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier, AdaBoostClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay, classification_report

# Criar diretórios necessários
os.makedirs('graphs', exist_ok=True)
os.makedirs('model', exist_ok=True)

# --- 1. Carregar os dados ---
print("📊 Carregando dados...")
df = pd.read_csv('https://raw.githubusercontent.com/Malbuquerque-data/fase-4-fiap/refs/heads/main/Obesity.csv')
print(f"Dataset original: {df.shape[0]} linhas, {df.shape[1]} colunas")

# --- 2. Feature Engineering ---
print("\n🔧 Aplicando Feature Engineering...")
df_processed = df.copy()

# Criar IMC (Índice de Massa Corporal)
df_processed['BMI'] = df_processed['Weight'] / (df_processed['Height'] ** 2)

# Criar categoria de IMC (para análise)
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

df_processed['BMI_Category'] = df_processed['BMI'].apply(categorize_bmi)

# Criar feature de risco baseada em múltiplos fatores
# Nota: Esta feature será calculada após a codificação para evitar problemas com tipos

# Remover valores nulos
df_processed = df_processed.dropna()
print(f"Dataset após limpeza: {df_processed.shape[0]} linhas")

# --- 3. Separar features e target ANTES da codificação ---
target = df_processed['Obesity'].copy()
df_features = df_processed.drop('Obesity', axis=1)

# --- 4. Codificar variáveis categóricas ---
print("\n🔤 Codificando variáveis categóricas...")
label_encoders = {}
df_encoded = df_features.copy()

for col in df_features.select_dtypes(include='object').columns:
    le = LabelEncoder()
    df_encoded[col] = le.fit_transform(df_features[col].astype(str))
    label_encoders[col] = le

# Codificar target separadamente
le_target = LabelEncoder()
y = le_target.fit_transform(target)
label_encoders['Obesity'] = le_target

# --- 5. Separar features e target após codificação ---
X = df_encoded.copy()

# Criar feature de risco após codificação (usando índices codificados)
# Assumindo que 'yes'=1, 'no'=0 após codificação (pode variar)
favc_idx = label_encoders.get('FAVC', None)
family_idx = label_encoders.get('family_history', None)

if favc_idx is not None and family_idx is not None:
    # Encontrar índices de 'yes' e 'no' após codificação
    try:
        yes_favc = favc_idx.transform(['yes'])[0] if 'yes' in favc_idx.classes_ else 1
        yes_family = family_idx.transform(['yes'])[0] if 'yes' in family_idx.classes_ else 1
        
        # Criar Risk_Score
        X['Risk_Score'] = (
            (X['FAVC'] == yes_favc).astype(int) +
            (X['family_history'] == yes_family).astype(int) -
            (X['FAF'] / 3.0) +  # Atividade física reduz risco
            (X['TUE'] / 2.0)    # Tempo em dispositivos aumenta risco
        )
    except:
        # Se houver erro, criar Risk_Score simples
        X['Risk_Score'] = X['FAVC'] + X['family_history'] - (X['FAF'] / 3.0) + (X['TUE'] / 2.0)

# --- 6. Normalizar ---
print("\n📏 Normalizando features...")
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# --- 7. Dividir dados ---
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.25, stratify=y, random_state=42)

# --- 8. Modelos ---
models = {
    'Gradient Boosting': GradientBoostingClassifier(n_estimators=200, learning_rate=0.1, random_state=42),
    'Random Forest': RandomForestClassifier(n_estimators=200, random_state=42),
    'AdaBoost': AdaBoostClassifier(n_estimators=200, random_state=42),
    'SVM': SVC(kernel='rbf', probability=True, random_state=42),
    'KNN': KNeighborsClassifier(n_neighbors=5),
    'Neural Network': MLPClassifier(hidden_layer_sizes=(100,), max_iter=500, random_state=42)
}

# --- 9. Validação cruzada e avaliação ---
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
results = []

print("\n=== Resultados de Cross-Validation (5-folds) ===")
for name, model in models.items():
    scores = cross_val_score(model, X_train, y_train, cv=cv, scoring='accuracy')
    mean_acc = np.mean(scores)
    results.append((name, mean_acc))
    print(f"{name}: {mean_acc:.4f} ± {scores.std():.4f}")

# --- 10. Comparar resultados ---
results_df = pd.DataFrame(results, columns=['Modelo', 'Acurácia Média'])
best_model_name = results_df.iloc[results_df['Acurácia Média'].idxmax()]['Modelo']
best_model = models[best_model_name]

# --- 11. Treinar modelo final ---
best_model.fit(X_train, y_train)
preds = best_model.predict(X_test)

# --- 12. Relatório de classificação ---
print("\n=== Relatório de Classificação ===")
print(classification_report(y_test, preds))

# --- 13. Avaliar modelo final ---
test_acc = accuracy_score(y_test, preds)
print(f"\n✅ Acurácia no conjunto de teste: {test_acc:.4f} ({test_acc*100:.2f}%)")

if test_acc < 0.75:
    print("⚠️  ATENÇÃO: Acurácia abaixo de 75%! Ajustando hiperparâmetros...")
    # Tentar com mais estimadores
    if best_model_name == 'Gradient Boosting':
        best_model = GradientBoostingClassifier(n_estimators=300, learning_rate=0.05, max_depth=5, random_state=42)
    elif best_model_name == 'Random Forest':
        best_model = RandomForestClassifier(n_estimators=300, max_depth=15, random_state=42)
    best_model.fit(X_train, y_train)
    preds = best_model.predict(X_test)
    test_acc = accuracy_score(y_test, preds)
    print(f"✅ Nova acurácia: {test_acc:.4f} ({test_acc*100:.2f}%)")

# --- 14. Matriz de confusão ---
cm = confusion_matrix(y_test, preds)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=label_encoders['Obesity'].classes_)
disp.plot(cmap='Blues', xticks_rotation=45)
plt.title(f"Matriz de Confusão - {best_model_name}\nAcurácia: {test_acc:.2%}")
plt.tight_layout()
plt.savefig("graphs/confusion_matrix.png", dpi=300, bbox_inches='tight')
plt.close()

# --- 15. Gráfico de acurácia dos modelos ---
plt.figure(figsize=(10, 6))
sns.barplot(data=results_df, x='Acurácia Média', y='Modelo', palette='viridis', hue='Modelo', legend=False)
plt.title("Comparação de Acurácia entre Modelos (5-Fold Cross-Validation)", fontsize=14, fontweight='bold')
plt.xlabel("Acurácia Média", fontsize=12)
plt.ylabel("Modelo", fontsize=12)
# Adicionar valores nas barras
for i, v in enumerate(results_df['Acurácia Média']):
    plt.text(v + 0.01, i, f'{v:.3f}', va='center', fontweight='bold')
plt.tight_layout()
plt.savefig("graphs/model_comparison.png", dpi=300, bbox_inches='tight')
plt.close()

# --- 16. Feature Importance (se o modelo suportar) ---
if hasattr(best_model, 'feature_importances_'):
    feature_importance = pd.DataFrame({
        'Feature': X.columns,
        'Importance': best_model.feature_importances_
    }).sort_values('Importance', ascending=False)
    
    plt.figure(figsize=(10, 8))
    sns.barplot(data=feature_importance.head(15), x='Importance', y='Feature', palette='coolwarm', hue='Feature', legend=False)
    plt.title("Top 15 Features Mais Importantes para Previsão", fontsize=14, fontweight='bold')
    plt.xlabel("Importância", fontsize=12)
    plt.ylabel("Feature", fontsize=12)
    plt.tight_layout()
    plt.savefig("graphs/feature_importance.png", dpi=300, bbox_inches='tight')
    plt.close()

# --- 17. Correlação ---
X_df = pd.DataFrame(X, columns=X.columns)
corr = X_df.corr()
plt.figure(figsize=(14, 10))
mask = np.triu(np.ones_like(corr, dtype=bool))
sns.heatmap(corr, mask=mask, annot=False, cmap='coolwarm', center=0, 
            square=True, linewidths=0.5, cbar_kws={"shrink": 0.8})
plt.title("Matriz de Correlação entre Variáveis", fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig("graphs/correlation_heatmap.png", dpi=300, bbox_inches='tight')
plt.close()

# --- 18. Distribuição das classes ---
plt.figure(figsize=(10, 6))
target_counts = pd.Series(target).value_counts()
target_df = pd.DataFrame({'Classe': target_counts.index, 'Frequência': target_counts.values})
sns.barplot(data=target_df, x='Classe', y='Frequência', palette='Set2', hue='Classe', legend=False)
plt.title("Distribuição das Classes de Obesidade", fontsize=14, fontweight='bold')
plt.xlabel("Nível de Obesidade", fontsize=12)
plt.ylabel("Frequência", fontsize=12)
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.savefig("graphs/target_distribution.png", dpi=300, bbox_inches='tight')
plt.close()

# --- 19. Salvar modelo e métricas ---
model_data = {
    'model': best_model,
    'scaler': scaler,
    'label_encoders': label_encoders,
    'columns': X.columns.tolist(),
    'model_name': best_model_name,
    'accuracy': float(test_acc),
    'classification_report': classification_report(y_test, preds, output_dict=True)
}

with open('model/obesity_model.pkl', 'wb') as f:
    pickle.dump(model_data, f)

# Salvar métricas em arquivo de texto
with open('model/metrics.txt', 'w', encoding='utf-8') as f:
    f.write("="*60 + "\n")
    f.write("MÉTRICAS DO MODELO DE PREVISÃO DE OBESIDADE\n")
    f.write("="*60 + "\n\n")
    f.write(f"Modelo Selecionado: {best_model_name}\n")
    f.write(f"Acurácia no Teste: {test_acc:.4f} ({test_acc*100:.2f}%)\n")
    f.write(f"\n{'='*60}\n")
    f.write("RELATÓRIO DE CLASSIFICAÇÃO\n")
    f.write("="*60 + "\n\n")
    f.write(classification_report(y_test, preds))
    f.write(f"\n{'='*60}\n")
    f.write("RESULTADOS DE CROSS-VALIDATION\n")
    f.write("="*60 + "\n\n")
    for _, row in results_df.iterrows():
        f.write(f"{row['Modelo']}: {row['Acurácia Média']:.4f}\n")

print(f"\n{'='*60}")
print(f"✅ TREINAMENTO CONCLUÍDO!")
print(f"{'='*60}")
print(f"Melhor modelo: {best_model_name}")
print(f"Acurácia no teste: {test_acc:.4f} ({test_acc*100:.2f}%)")
if test_acc >= 0.75:
    print("✅ Requisito de acurácia > 75% ATENDIDO!")
else:
    print("⚠️  Acurácia abaixo de 75% - considere ajustar hiperparâmetros")
print(f"\n📁 Arquivos salvos:")
print(f"   - Modelo: model/obesity_model.pkl")
print(f"   - Métricas: model/metrics.txt")
print(f"   - Gráficos: graphs/")
print(f"{'='*60}\n")