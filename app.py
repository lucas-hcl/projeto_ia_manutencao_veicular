import streamlit as st
import joblib
import pandas as pd
import numpy as np

# 1. Definir Parâmetros: Medianas, Modas e CODED_COLUMNS dos resultados do treinamento

# --- Medians/Modes ---
# SUBSTITUIR [AQUI VAI O DICIONÁRIO DE MEDIANAS PARA CARRO] PELO VALOR IMPRESSO NO PASSO ANTERIOR
numerical_features_medians = {
    'Carro': {'Mileage': 54903.0, 'Reported_Issues': 3.0, 'Vehicle_Age': 5.0, 'Service_History': 6.0, 'Accident_History': 2.0},
    'Caminhão': {'Mileage': 55115.5, 'Reported_Issues': 2.0, 'Vehicle_Age': 5.0, 'Service_History': 6.0, 'Accident_History': 2.0},
    'Ônibus': {'Mileage': 54299.0, 'Reported_Issues': 3.0, 'Vehicle_Age': 6.0, 'Service_History': 6.0, 'Accident_History': 1.0},
    'Motocicleta': {'Mileage': 54879.0, 'Reported_Issues': 2.0, 'Vehicle_Age': 5.0, 'Service_History': 5.0, 'Accident_History': 1.0}
}


# SUBSTITUIR [AQUI VAI O DICIONÁRIO DE MODAS PARA CARRO] E OUTROS PELOS VALORES IMPRESSOS NO PASSO ANTERIOR
categorical_features_modes = {
    'Carro': {'Maintenance_History': 'Poor', 'Tire_Condition': 'Worn Out', 'Brake_Condition': 'New', 'Battery_Status': 'New'},
    'Caminhão': {'Maintenance_History': 'Good', 'Tire_Condition': 'Worn Out', 'Brake_Condition': 'Worn Out', 'Battery_Status': 'Weak'},
    'Ônibus': {'Maintenance_History': 'Good', 'Tire_Condition': 'New', 'Brake_Condition': 'Worn Out', 'Battery_Status': 'Good'},
    'Motocicleta': {'Maintenance_History': 'Average', 'Tire_Condition': 'New', 'Brake_Condition': 'Good', 'Battery_Status': 'New'}
}

# --- Colunas Finais ---
# SUBSTITUIR [AQUI VAI A LISTA COMPLETA CODED_COLUMNS] PELA LISTA IMPRESSA NO PASSO ANTERIOR
CODED_COLUMNS = [
    'Mileage',
    'Reported_Issues',
    'Vehicle_Age',
    'Service_History',
    'Accident_History',
    'Maintenance_History_Average',
    'Maintenance_History_Good',
    'Maintenance_History_Poor',
    'Tire_Condition_Good',
    'Tire_Condition_New',
    'Tire_Condition_Worn Out',
    'Brake_Condition_Good',
    'Brake_Condition_New',
    'Brake_Condition_Worn Out',
    'Battery_Status_Good',
    'Battery_Status_New',
    'Battery_Status_Weak',
]


# Definir listas de features
numerical_features = ['Mileage', 'Reported_Issues', 'Vehicle_Age', 'Service_History', 'Accident_History']
categorical_features = ['Maintenance_History', 'Tire_Condition', 'Brake_Condition', 'Battery_Status'] # Fuel_Efficiency removido

# Definir mapeamentos para features categóricas (para st.selectbox)
maintenance_options = ['Poor', 'Average', 'Good']
tire_options = ['Worn Out', 'Good', 'New']
brake_options = ['Worn Out', 'Good', 'New']
battery_options = ['Weak', 'Good', 'New']


# 2. Carregar os modelos treinados
try:
    # Ajustar nomes dos arquivos conforme o passo anterior
    model_car = joblib.load('modelo_car_decision_tree.joblib') # Ajustado
    model_truck = joblib.load('modelo_trucks_decision_tree.joblib')
    model_bus = joblib.load('modelo_bus_decision_tree.joblib')
    model_motorcycle = joblib.load('modelo_motorcycle_decision_tree.joblib')

except FileNotFoundError as e:
    st.error(f"Um ou mais arquivos de modelo não encontrados: {e}. Por favor, garanta que os arquivos de modelo estejam no mesmo diretório que app.py")
    st.stop()


# Interface da Aplicação Streamlit (em Português do Brasil)
st.title('Previsão de Manutenção de Veículos')
st.header('Preveja se um veículo precisa de manutenção com base em suas características.')

# Seleção do Tipo de Veículo
vehicle_type = st.selectbox(
    'Selecione o Tipo de Veículo:',
    ('Carro', 'Caminhão', 'Ônibus', 'Motocicleta') # Ajustado para 'Carro'
)

# Campos de entrada com base no tipo de veículo selecionado
st.subheader(f'Insira os detalhes para o veículo selecionado ({vehicle_type}):')

mileage = st.slider('Quilometragem', 0, 100000, 50000)
reported_issues = st.slider('Problemas Relatados', 0, 10, 0)
vehicle_age = st.slider('Idade do Veículo (Anos)', 0, 20, 5)
maintenance_history = st.selectbox('Histórico de Manutenção', maintenance_options)
service_history = st.slider('Histórico de Serviços (Número de serviços)', 0, 20, 5)
accident_history = st.slider('Histórico de Acidentes (Número de acidentes)', 0, 10, 0)
tire_condition = st.selectbox('Condição do Pneu', tire_options)
brake_condition = st.selectbox('Condição do Freio', brake_options)
battery_status = st.selectbox('Status da Bateria', battery_options)
# Fuel_Efficiency removido


# Botão de Previsão
if st.button('Prever Necessidade de Manutenção'):
    # Preparar dicionário de dados de entrada (apenas as 8 features)
    input_data = {
        'Mileage': mileage,
        'Reported_Issues': reported_issues,
        'Vehicle_Age': vehicle_age,
        'Service_History': service_history,
        'Accident_History': accident_history,
        'Maintenance_History': maintenance_history,
        'Tire_Condition': tire_condition,
        'Brake_Condition': brake_condition,
        'Battery_Status': battery_status
    }

    # Criar um DataFrame a partir dos dados de entrada
    input_df = pd.DataFrame([input_data])

    # 5. Aplicar Imputação (Mediana/Moda) manualmente
    # Features numéricas: preencher com a mediana
    for col in numerical_features:
         input_df[col] = input_df[col].fillna(numerical_features_medians[vehicle_type][col])

    # Features categóricas: preencher com a moda
    for col in categorical_features:
        if col in input_df.columns:
             # Usar a moda dos dados de treinamento para o tipo de veículo específico
             input_df[col] = input_df[col].fillna(categorical_features_modes[vehicle_type][col])


    # 5. Aplicar One-Hot Encoding (pd.get_dummies) manualmente
    # Aplicar get_dummies apenas às features categóricas definidas
    input_df_encoded = pd.get_dummies(input_df, columns=categorical_features, dummy_na=False)

    # 5. Usar CODED_COLUMNS para reindexar e alinhar o DataFrame de entrada
    # Adicionar colunas ausentes (com valor 0) que estavam presentes no treinamento, mas não na entrada atual
    # Garantir que a ordem das colunas seja a mesma que em CODED_COLUMNS
    input_df_final = input_df_encoded.reindex(columns=CODED_COLUMNS, fill_value=0)


    # Selecionar o modelo apropriado com base no tipo de veículo
    model_to_use = None
    if vehicle_type == 'Carro': # Ajustado
        model_to_use = model_car # Ajustado
    elif vehicle_type == 'Caminhão':
        model_to_use = model_truck
    elif vehicle_type == 'Ônibus':
        model_to_use = model_bus
    elif vehicle_type == 'Motocicleta':
        model_to_use = model_motorcycle

    if model_to_use is not None:
        # Fazer a previsão
        prediction = model_to_use.predict(input_df_final)

        # Exibir o resultado (em Português do Brasil)
        st.subheader('Resultado da Previsão:')
        if prediction[0] == 0:
            st.success('Status: OK (Não precisa de manutenção imediata)')
        else:
            st.error('Status: Necessita de Manutenção')
    else:
        st.error("Erro: Não foi possível carregar o modelo para o tipo de veículo selecionado.")
