import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import pickle

# 1. Загружаем данные
df = pd.read_csv('loan_data.csv')

# Очистка: убираем лишние пробелы в названиях колонок
df.columns = df.columns.str.strip()

# Очистка: убираем лишние пробелы в текстовых колонках
for col in df.select_dtypes(include=['object', 'string']).columns:
    df[col] = df[col].str.strip()

print("Данные загружены. Формат:", df.shape)
print("Колонки:", list(df.columns))

# Проверяем уникальные значения в текстовых колонках
print("\n--- Уникальные значения в колонках ---")
print("education:", df['education'].unique())
print("self_employed:", df['self_employed'].unique())
print("loan_status:", df['loan_status'].unique())

# 2. Выбираем признаки (X) и целевую переменную (y)
features = ['no_of_dependents', 'education', 'self_employed', 'income_annum',
            'loan_amount', 'loan_term', 'cibil_score', 'residential_assets_value',
            'commercial_assets_value', 'luxury_assets_value', 'bank_asset_value']

X = df[features].copy()
y = df['loan_status'].copy()

print("\nПризнаки:", features)

# 3. Кодируем текстовые данные в числа
print("\n--- Кодирование признаков ---")

# Кодируем education
print("До кодирования education:", X['education'].value_counts().to_dict())
X['education'] = X['education'].map({'Graduate': 1, 'Not Graduate': 0})
print("После кодирования education:", X['education'].value_counts().to_dict())

# Кодируем self_employed
print("\nДо кодирования self_employed:", X['self_employed'].value_counts().to_dict())
X['self_employed'] = X['self_employed'].map({'Yes': 1, 'No': 0})
print("После кодирования self_employed:", X['self_employed'].value_counts().to_dict())

# Проверяем, сколько NaN появилось
print(f"\nNaN в education после кодирования: {X['education'].isna().sum()}")
print(f"NaN в self_employed после кодирования: {X['self_employed'].isna().sum()}")

# Заполняем NaN нулями (на случай, если есть другие значения)
X['education'] = X['education'].fillna(0)
X['self_employed'] = X['self_employed'].fillna(0)

# 4. Кодируем целевую переменную
print("\nДо кодирования loan_status:", y.value_counts().to_dict())
y = y.map({'Approved': 1, 'Rejected': 0})
print("После кодирования loan_status:", y.value_counts().to_dict())

# Заполняем NaN нулями и удаляем строки с NaN
y = y.fillna(0)

# 5. Финальная проверка
print(f"\nИтоговый размер данных: {len(X)} строк")
print(f"NaN в X: {X.isnull().sum().sum()}")
print(f"NaN в y: {y.isnull().sum()}")

# Удаляем строки с NaN (если остались)
mask = ~(X.isnull().any(axis=1) | y.isnull())
X = X[mask]
y = y[mask]

print(f"Размер после удаления NaN: {len(X)} строк")

# 6. Обучаем модель
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X, y)

print(f"\nМодель обучена!")
print(f"Точность на обучающих данных: {model.score(X, y):.3f}")

# 7. Сохраняем модель
with open('loan_model.pkl', 'wb') as f:
    pickle.dump({
        'model': model,
        'education_map': {'Graduate': 1, 'Not Graduate': 0},
        'self_employed_map': {'Yes': 1, 'No': 0},
        'features': features
    }, f)

print("\nМодель сохранена в файл 'loan_model.pkl'")

# 8. Тест предсказания
print("\n--- Тест предсказания ---")
sample = X.iloc[0:1]
pred = model.predict(sample)
prob = model.predict_proba(sample)
print(f"Первая строка данных:")
print(f"  Предсказание: {'Approved' if pred[0] == 1 else 'Rejected'}")
print(f"  Вероятность Rejected: {prob[0][0]:.3f}")
print(f"  Вероятность Approved: {prob[0][1]:.3f}")

# 9. Важность признаков
print("\n--- Важность признаков ---")
importance = pd.DataFrame({
    'Признак': features,
    'Важность': model.feature_importances_
}).sort_values('Важность', ascending=False)
print(importance.to_string(index=False))