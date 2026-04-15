import streamlit as st
import pandas as pd
import pickle

# Загрузка модели
with open('loan_model.pkl', 'rb') as f:
    model_data = pickle.load(f)

model = model_data['model']
education_map = model_data['education_map']
self_employed_map = model_data['self_employed_map']
features = model_data['features']

# Настройка страницы
st.set_page_config(page_title="Кредитный скоринг", page_icon="🏦", layout="wide")

st.title("Предсказание одобрения кредита")
st.markdown("Заполните данные о заёмщике, и модель предскажет, одобрит ли банк кредит.")

# Боковая панель для выбора режима
mode = st.sidebar.radio("Выберите режим ввода:", ["Ручной ввод", "Загрузка CSV-файла"])

def predict_single(data_dict):
    """Предсказание для одного заёмщика"""
    df = pd.DataFrame([data_dict])
    df['education'] = df['education'].map(education_map)
    df['self_employed'] = df['self_employed'].map(self_employed_map)
    df = df[features]
    pred = model.predict(df)[0]
    prob = model.predict_proba(df)[0]
    return pred, prob

# ==================== РУЧНОЙ ВВОД ====================
if mode == "Ручной ввод":
    st.header("Введите данные заёмщика")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        no_of_dependents = st.number_input("Количество иждивенцев", min_value=0, max_value=5, value=0, step=1)
        education = st.selectbox("Образование", ["Graduate", "Not Graduate"])
        self_employed = st.selectbox("Самозанятый", ["No", "Yes"])
        income_annum = st.number_input("Годовой доход (₽)", min_value=0, value=500000, step=100000)
    
    with col2:
        loan_amount = st.number_input("Сумма кредита (₽)", min_value=0, value=10000000, step=1000000)
        loan_term = st.number_input("Срок кредита (лет)", min_value=1, max_value=20, value=10, step=1)
        cibil_score = st.number_input("Кредитный рейтинг (CIBIL, 300-900)", min_value=300, max_value=900, value=700, step=10)
        residential_assets_value = st.number_input("Стоимость жилой недвижимости (₽)", min_value=0, value=5000000, step=500000)
    
    with col3:
        commercial_assets_value = st.number_input("Стоимость коммерческой недвижимости (₽)", min_value=0, value=2000000, step=500000)
        luxury_assets_value = st.number_input("Стоимость предметов роскоши (₽)", min_value=0, value=3000000, step=500000)
        bank_asset_value = st.number_input("Сбережения в банке (₽)", min_value=0, value=2000000, step=500000)
    
    if st.button("Предсказать", type="primary"):
        data = {
            'no_of_dependents': no_of_dependents,
            'education': education,
            'self_employed': self_employed,
            'income_annum': income_annum,
            'loan_amount': loan_amount,
            'loan_term': loan_term,
            'cibil_score': cibil_score,
            'residential_assets_value': residential_assets_value,
            'commercial_assets_value': commercial_assets_value,
            'luxury_assets_value': luxury_assets_value,
            'bank_asset_value': bank_asset_value
        }
        
        pred, prob = predict_single(data)
        
        st.markdown("---")
        if pred == 1:
            st.success(f"**Кредит ОДОБРЕН!**")
            st.metric("Вероятность одобрения", f"{prob[1]:.1%}")
        else:
            st.error(f"**Кредит ОТКАЗАН!**")
            st.metric("Вероятность отказа", f"{prob[0]:.1%}")

# ==================== ЗАГРУЗКА CSV ====================
else:
    st.header("Загрузите CSV-файл с данными")
    st.markdown("Файл должен содержать колонки с теми же названиями, что и в обучающих данных.")
    
    uploaded_file = st.file_uploader("Выберите CSV-файл", type="csv")
    
    if uploaded_file is not None:
        df_input = pd.read_csv(uploaded_file)
        st.write("**Загруженные данные:**")
        st.dataframe(df_input.head(10))
        
        required_cols = ['no_of_dependents', 'education', 'self_employed', 'income_annum',
                         'loan_amount', 'loan_term', 'cibil_score', 'residential_assets_value',
                         'commercial_assets_value', 'luxury_assets_value', 'bank_asset_value']
        
        missing_cols = [col for col in required_cols if col not in df_input.columns]
        if missing_cols:
            st.error(f"В файле отсутствуют колонки: {missing_cols}")
        else:
            df_pred = df_input[required_cols].copy()
            df_pred['education'] = df_pred['education'].map(education_map)
            df_pred['self_employed'] = df_pred['self_employed'].map(self_employed_map)
            df_pred = df_pred.fillna(0)
            
            predictions = model.predict(df_pred)
            probabilities = model.predict_proba(df_pred)
            
            df_input['Предсказание'] = ['Approved' if p == 1 else 'Rejected' for p in predictions]
            df_input['Вероятность одобрения'] = [f"{p[1]:.1%}" for p in probabilities]
            
            st.success("Предсказания выполнены!")
            st.dataframe(df_input[['loan_id', 'Предсказание', 'Вероятность одобрения'] + required_cols].head(20))
            
            col1, col2 = st.columns(2)
            with col1:
                approved_count = sum(predictions)
                st.metric("Одобрено заявок", f"{approved_count} из {len(predictions)}")
            
            csv_output = df_input.to_csv(index=False)
            st.download_button(
                label="Скачать результаты в CSV",
                data=csv_output,
                file_name="predictions.csv",
                mime="text/csv"
            )