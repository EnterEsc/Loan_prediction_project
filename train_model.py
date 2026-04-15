import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import pickle

# 1. Загружаем данные
df = pd.read_csv('loan_data.csv')
df.columns = df.columns.str.strip()

print("Данные загружены. Формат:", df.shape)

# 2. Выбираем признаки и целевую переменную
features = ['no_of_dependents', 'education', 'self_employed', 'income_annum',
            'loan_amount', 'loan_term', 'cibil_score', 'residential_assets_value',
            'commercial_assets_value', 'luxury_assets_value', 'bank_asset_value']

X = df[features].copy()
y = df['loan_status'].copy()

# 3. Кодируем текстовые данные
X['education'] = X['education'].map({'Graduate': 1, 'Not Graduate': 0})
X['self_employed'] = X['self_employed'].map({'Yes': 1, 'No': 0})
y = y.map({'Approved': 1, 'Rejected': 0})

# Заполняем возможные пустые значения
X = X.fillna(0)
y = y.fillna(0)

print(f"Размер данных: {len(X)} строк")

# 4. Разделяем на обучающую (80%) и тестовую (20%) выборки
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print(f"Обучающая выборка: {len(X_train)} строк")
print(f"Тестовая выборка: {len(X_test)} строк")

# 5. Обучаем модель
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# 6. Оцениваем качество
train_score = model.score(X_train, y_train)
test_score = model.score(X_test, y_test)

print(f"\nМодель обучена!")
print(f"Точность на обучающих данных: {train_score:.3f}")
print(f"Точность на ТЕСТОВЫХ данных: {test_score:.3f}")

# 7. Сохраняем модель
with open('loan_model.pkl', 'wb') as f:
    pickle.dump({
        'model': model,
        'education_map': {'Graduate': 1, 'Not Graduate': 0},
        'self_employed_map': {'Yes': 1, 'No': 0},
        'features': features
    }, f)

print("\nМодель сохранена в файл 'loan_model.pkl'")