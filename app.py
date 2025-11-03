import streamlit as st
import joblib
import pandas as pd
import numpy as np

# --- 1. CONFIGURAÇÃO DE PARÂMETROS E MODELOS (MANTENHA OS SEUS VALORES!) ---

# --- Dicionários de Mediana (Imputação de Colunas Numéricas) ---
MEDIANS_CARRO = {'Mileage': 54885.0, 'Reported_Issues': 3.0, 'Vehicle_Age': 5.0, 'Service_History': 6.0, 'Accident_History': 2.0} # SUBSTITUA!
MEDIANS_CAMINHÃO = {'Mileage': 55115.5, 'Reported_Issues': 2.0, 'Vehicle_Age': 5.0, 'Service_History': 6.0, 'Accident_History': 2.0} # SUBSTITUA!
MEDIANS_ÔNIBUS = {'Mileage': 54299.0, 'Reported_Issues': 3.0, 'Vehicle_Age': 6.0, 'Service_History': 6.0, 'Accident_History': 1.0} # SUBSTITUA!
MEDIANS_MOTOCICLETA = {'Mileage': 54879.0, 'Reported_Issues': 2.0, 'Vehicle_Age': 5.0, 'Service_History': 5.0, 'Accident_History': 1.0} # SUBSTITUA!

# --- Dicionários de Moda (Imputação de Colunas Categóricas) ---
MODES_CARRO = {'Maintenance_History': 'Good', 'Tire_Condition': 'New', 'Brake_Condition': 'New', 'Battery_Status': 'New'} # SUBSTITUA!
MODES_CAMINHÃO = {'Maintenance_History': 'Good', 'Tire_Condition': 'Worn Out', 'Brake_Condition': 'Worn Out', 'Battery_Status': 'Weak'} # SUBSTITUA!
MODES_ÔNIBUS = {'Maintenance_History': 'Good', 'Tire_Condition': 'New', 'Brake_Condition': 'Worn Out', 'Battery_Status': 'Good'} # SUBSTITUA!
MODES_MOTOCICLETA = {'Maintenance_History': 'Average', 'Tire_Condition': 'New', 'Brake_Condition': 'Good', 'Battery_Status': 'New'} # SUBSTITUA!

# --- Colunas Esperadas Pelo Modelo (Resolve o ValueError) ---
CODED_COLUMNS = ['Mileage', 'Reported_Issues', 'Vehicle_Age', 'Service_History', 'Accident_History', 
                 'Maintenance_History_Average', 'Maintenance_History_Good', 'Maintenance_History_Poor', 
                 'Tire_Condition_Good', 'Tire_Condition_New', 'Tire_Condition_Worn Out', 
                 'Brake_Condition_Good', 'Brake_Condition_New', 'Brake_Condition_Worn Out', 
                 'Battery_Status_Good', 'Battery_Status_New', 'Battery_Status_Weak'] # SUBSTITUA!

# --- DICIONÁRIO DE TRADUÇÃO (Português -> Inglês e Vice-versa para Mensagens) ---
TRADUCAO_INVERSA = {
    'Bom': 'Good', 'Médio': 'Average', 'Ruim': 'Poor',
    'Novo': 'New', 'Gasto': 'Worn Out',
    'Fraca': 'Weak',
}

# --- Carregamento dos Modelos (.joblib) ---
try:
    models = {
        'Carro': joblib.load('modelo_car_decision_tree.joblib'),
        'Caminhão': joblib.load('modelo_trucks_decision_tree.joblib'),
        'Ônibus': joblib.load('modelo_bus_decision_tree.joblib'),
        'Motocicleta': joblib.load('modelo_motorcycle_decision_tree.joblib')
    }
except FileNotFoundError as e:
    st.error(f"Erro ao carregar o modelo: {e}. Verifique se os arquivos .joblib estão na mesma pasta.")
    st.stop()


# --- 2. FUNÇÃO DE PRÉ-PROCESSAMENTO E PREVISÃO (LÓGICA INALTERADA) ---

def preprocess_and_predict(data, vehicle_type):
    data_translated = data.copy()
    data_translated['Maintenance_History'] = TRADUCAO_INVERSA.get(data['Maintenance_History'], data['Maintenance_History'])
    data_translated['Tire_Condition'] = TRADUCAO_INVERSA.get(data['Tire_Condition'], data['Tire_Condition'])
    data_translated['Brake_Condition'] = TRADUCAO_INVERSA.get(data['Brake_Condition'], data['Brake_Condition'])
    data_translated['Battery_Status'] = TRADUCAO_INVERSA.get(data['Battery_Status'], data['Battery_Status'])

    params = {
        'Carro': {'medians': MEDIANS_CARRO, 'modes': MODES_CARRO},
        'Caminhão': {'medians': MEDIANS_CAMINHÃO, 'modes': MODES_CAMINHÃO},
        'Ônibus': {'medians': MEDIANS_ÔNIBUS, 'modes': MODES_ÔNIBUS},
        'Motocicleta': {'medians': MEDIANS_MOTOCICLETA, 'modes': MODES_MOTOCICLETA}
    }[vehicle_type]
    
    df_input = pd.DataFrame([data_translated]) 
    
    for col, value in params['medians'].items():
        df_input[col] = df_input[col].fillna(value)
    for col, value in params['modes'].items():
        df_input[col] = df_input[col].fillna(value)

    categorical_cols = ['Maintenance_History', 'Tire_Condition', 'Brake_Condition', 'Battery_Status']
    df_encoded = pd.get_dummies(df_input, columns=categorical_cols, drop_first=False)
    
    df_final = df_encoded.reindex(columns=CODED_COLUMNS, fill_value=0)
    
    model = models[vehicle_type]
    prediction = model.predict(df_final)
    
    return prediction[0]

# --- 3. ANÁLISE DE FATORES DE RISCO (LÓGICA INALTERADA) ---

def analyze_risk_factors(input_data):
    """Analisa os inputs do usuário e retorna uma lista de fatores de risco com sugestões."""
    
    critical_factors = []
    
    # 1. Análise Numérica (Baseado em thresholds lógicos e nos dados do input)
    # Assumimos que KM acima de 150.000 é alta (pode ser ajustado)
    if input_data['Mileage'] > 150000:
        critical_factors.append({
            'falha': 'Quilometragem Elevada',
            'explicacao': f"A alta quilometragem ({input_data['Mileage']:,} KM) aumenta o desgaste de todos os componentes mecânicos e estruturais.",
            'solucao': 'Focar na troca de fluidos, correias e inspeção de suspensão e transmissão. Priorizar a manutenção preventiva geral.'
        })
    # Assumimos que mais de 5 problemas relatados é crítico
    if input_data['Reported_Issues'] >= 5:
        critical_factors.append({
            'falha': 'Falhas Recorrentes no Sistema',
            'explicacao': f"O alto número de problemas relatados ({input_data['Reported_Issues']} falhas) sugere um erro persistente em sensores ou sistemas eletrônicos.",
            'solucao': 'Realizar diagnóstico eletrônico completo (OBD/scanner) para identificar a origem das falhas e corrigir o componente sensor defeituoso.'
        })
    # Assumimos que a idade acima de 10 anos é alta
    if input_data['Vehicle_Age'] >= 10:
        critical_factors.append({
            'falha': 'Idade do Veículo Elevada',
            'explicacao': f"A idade ({input_data['Vehicle_Age']} anos) aumenta a probabilidade de corrosão, fadiga de materiais e falha em borrachas e vedantes.",
            'solucao': 'Verificar mangueiras, vedantes e o estado geral da estrutura. Substituir peças de borracha e plástico ressecadas.'
        })
        
    # 2. Análise Categórica (Fatores de maior risco)
    if input_data['Maintenance_History'] == 'Ruim':
        critical_factors.append({
            'falha': 'Histórico de Manutenção Ruim',
            'explicacao': 'Um histórico ruim de manutenção preventiva é o maior preditor de falhas futuras, pois problemas pequenos não foram corrigidos.',
            'solucao': 'Criar um novo plano de manutenção agressivo, com troca imediata de óleo, filtros e revisão completa do motor.'
        })
    if input_data['Tire_Condition'] == 'Gasto':
        critical_factors.append({
            'falha': 'Pneus Desgastados',
            'explicacao': 'Pneus gastos comprometem a segurança (frenagem e aquaplanagem) e indicam a necessidade de troca imediata.',
            'solucao': 'Substituição imediata dos pneus. Alinhar e balancear após a troca para evitar desgaste irregular.'
        })
    if input_data['Brake_Condition'] == 'Gasto':
        critical_factors.append({
            'falha': 'Freios Desgastados',
            'explicacao': 'Freios gastos (pastilhas ou lonas) são um risco de segurança. O modelo prioriza a manutenção quando há risco de falha na frenagem.',
            'solucao': 'Inspecionar e substituir pastilhas/lonas e verificar o nível e qualidade do fluido de freio.'
        })
    if input_data['Battery_Status'] == 'Fraca':
        critical_factors.append({
            'falha': 'Bateria Fraca',
            'explicacao': 'O status "Fraca" da bateria prediz falha na ignição e no sistema elétrico, especialmente em climas extremos.',
            'solucao': 'Testar a capacidade da bateria e o alternador. Se a capacidade estiver baixa, a substituição é recomendada.'
        })
        
    return critical_factors


# --- 4. INTERFACE STREAMLIT (DESIGN AUTOMOTIVO) ---

st.set_page_config(
    page_title="Manutenção Preditiva", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- TÍTULO DO PAINEL ---
st.markdown("<h1 style='text-align: center; color: #FF4B4B;'>MANUTENÇÃO PREDITIVA DE VEÍCULOS </h1>", unsafe_allow_html=True)
st.markdown("<h4 style='text-align: center; color: #E0E0E0;'>Análise de Risco de Falha em Tempo Real</h4>", unsafe_allow_html=True)
st.markdown("---") 


# --- SELEÇÃO DE VEÍCULO (Barra Lateral) ---
with st.sidebar:
    # --- IMAGEM REMOVIDA AQUI ---
    # st.image("icons8-car-service-96.png", width=100) 
    st.markdown("## Seleção de Modelo")
    
    vehicle_type = st.radio(
        "Selecione o Tipo de Veículo para Inspeção:",
        ('Carro', 'Caminhão', 'Ônibus', 'Motocicleta'),
        key='vehicle_selector'
    )
    
    st.markdown(f"<p style='color:#FF4B4B; font-weight:bold;'>Modelo Ativo: {vehicle_type}</p>", unsafe_allow_html=True)
    st.markdown("---")


# --- INPUT DE DADOS NO PAINEL PRINCIPAL ---
st.subheader(f"Formulário de Inspeção - {vehicle_type}")

# Colunas para Inputs
col_km, col_idade, col_service, col_acidente = st.columns(4)

# Variável para armazenar a quilometragem limpa
mileage_cleaned = None

# --- Inputs Numéricos ---
with col_km:
    mileage_input = st.text_input(
        "Quilometragem (KM)", 
        value="50.000",
        help="Digite a quilometragem. O sistema aceita a formatação de milhares (Ex: 150.000)."
    )
    
    # Lógica de limpeza da Quilometragem (text_input -> int)
    try:
        mileage_cleaned = int(mileage_input.replace('.', '').replace(',', ''))
        st.caption(f"Valor para o modelo: {mileage_cleaned:,}")
    except ValueError:
        st.error("🚨 ERRO: KM inválida. Insira apenas números.")
        mileage_cleaned = None
        
with col_idade:
    vehicle_age = st.number_input("Idade (Anos)", min_value=0, max_value=30, value=5, help="Tempo de uso do veículo em anos.")
    
with col_service:
    service_history = st.number_input("Nº Serviços Anteriores", min_value=0, max_value=30, value=5, help="Quantas manutenções já recebeu.")

with col_acidente:
    accident_history = st.number_input("Nº Acidentes Registrados", min_value=0, max_value=10, value=0, help="Número de acidentes leves ou graves.")


st.markdown("<br>", unsafe_allow_html=True) # Espaçamento

# Colunas para Componentes Categóricos
col_comp1, col_comp2, col_comp3, col_comp4, col_relatorio = st.columns(5)

with col_comp1:
    maintenance_history = st.selectbox("Manutenção Geral", ['Bom', 'Médio', 'Ruim'])

with col_comp2:
    tire_condition = st.selectbox("Condição do Pneu", ['Novo', 'Bom', 'Gasto'])

with col_comp3:
    brake_condition = st.selectbox("Condição do Freio", ['Novo', 'Bom', 'Gasto'])

with col_comp4:
    battery_status = st.selectbox("Status da Bateria", ['Novo', 'Bom', 'Fraca'])

with col_relatorio:
    reported_issues = st.number_input("Falhas Reportadas (Nº)", min_value=0, max_value=10, value=0, help="Falhas reportadas pelo sistema de bordo.")

st.markdown("<br>", unsafe_allow_html=True) 

# --- BOTÃO DE PREVISÃO E ÁREA DE RESULTADO (MAIOR DESTAQUE) ---
if st.button("EXECUTAR DIAGNÓSTICO PREDITIVO", type="primary", use_container_width=True):
    
    if mileage_cleaned is None:
        st.warning("⚠️ O campo Quilometragem possui um erro de formatação. Ajuste para continuar.")
        st.stop()
        
    # 1. Coletar Dados para ambas as funções
    input_data = {
        'Mileage': mileage_cleaned, 
        'Reported_Issues': reported_issues,
        'Vehicle_Age': vehicle_age,
        'Service_History': service_history,
        'Accident_History': accident_history,
        'Maintenance_History': maintenance_history,
        'Tire_Condition': tire_condition,
        'Brake_Condition': brake_condition,
        'Battery_Status': battery_status
    }
    
    # 2. Fazer Previsão
    prediction = preprocess_and_predict(input_data, vehicle_type)
    
    # 3. Exibir Resultado com Design Impactante
    st.markdown("### 📊 Resultado do Diagnóstico Preditivo")
    st.markdown("---")
    
    col_risco, col_msg = st.columns([1, 2])

    if prediction == 1:
        # ALTO RISCO - EXPLICABILIDADE APLICADA
        with col_risco:
            st.metric(
                label="Nível de Risco de Falha",
                value="ALTO",
                delta="REQUER ATENÇÃO",
                delta_color="inverse"
            )
        
        with col_msg:
            st.error(f"**🚨 AÇÃO IMEDIATA NECESSÁRIA!** O modelo prediz alta probabilidade de falha ou necessidade de serviço iminente para o veículo **{vehicle_type}**.")
            
            # --- BLOCO DE EXPLICABILIDADE ---
            st.markdown("#### Fatores Críticos Detectados (Por que o risco é alto?)")
            
            risk_factors = analyze_risk_factors(input_data)
            
            if risk_factors:
                for factor in risk_factors:
                    with st.expander(f"🔴 **{factor['falha']}**"):
                        st.markdown(f"**Explicação:** {factor['explicacao']}")
                        st.markdown(f"**Solução Recomendada:** {factor['solucao']}")
            else:
                st.warning("O modelo prediz alto risco, mas os inputs atuais não mostram fatores óbvios acima dos limites de alerta. Recomenda-se inspeção geral.")
            # --- FIM DO BLOCO ---
            
    else:
        # BAIXO RISCO
        with col_risco:
            st.metric(
                label="Nível de Risco de Falha",
                value="BAIXO",
                delta="ESTÁVEL",
                delta_color="normal"
            )
        with col_msg:
            st.success(f"**✅ CONDIÇÃO ESTÁVEL.** O veículo **{vehicle_type}** está com baixo risco de precisar de manutenção corretiva no curto prazo.")
            st.write("Recomendação: Manter o cronograma de manutenção preventiva.")