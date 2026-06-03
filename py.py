import pandas as pd

# Wczytaj dane
df = pd.read_csv('train_u6lujuX_CVtuZ9i.csv')

# Policz brakujące dane
missing_values_count = df.isnull().sum().sum()
print(f"Liczba brakujących danych: {missing_values_count}")

# Znajdź kolumny numeryczne i kategoryczne
numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns
categorical_cols = df.select_dtypes(include=['object']).columns

print(f"Liczba kolumn numerycznych: {len(numeric_cols)}")
print(f"Liczba kolumn kategorycznych: {len(categorical_cols)}")
print(f"Liczba rekordów: {df.shape[0]}")