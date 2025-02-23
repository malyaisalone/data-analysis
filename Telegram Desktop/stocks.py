import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
file1 = r'C:\Users\user1718\Desktop\all_stocks_5yr.csv'
df = pd.read_csv(file1)

print(df.columns)

# Рассчитываем простое скользящее среднее (SMA) с окном 10
df['SMA_10'] = df['close'].rolling(window=10).mean()
print(df[['close', 'SMA_10']].head(15))
# Рассчитываем стандартное отклонение цен закрытия
std_dev = df['close'].std()
print('Стандартное отклонение цен закрытия:', std_dev)

# Рассчитываем ежедневную доходность (процентное изменение цены закрытия)
df['Daily Returns'] = df['close'].pct_change()
print(df)

def calculate_rsi(data, window=14):
    diff = data['close'].diff(1)
    up = diff.where(diff > 0, 0)
    down = -diff.where(diff < 0, 0)
    avg_gain = up.rolling(window=window).mean()
    avg_loss = down.rolling(window=window).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi
# RSI является популярным техническим индикатором, который измеряет скорость и изменение цен,
# а также указывает на перекупленность или перепроданность актива.
# Рассчитываем RSI (14-дневный) и добавляем в DataFrame
df['RSI'] = calculate_rsi(df)
print(df)

# График цен закрытия
plt.figure(figsize=(12, 6))
plt.plot(df['date'], df['close'], label='Цена закрытия', color='blue')
plt.title('Изменение цен закрытия')
plt.xlabel('Дата')
plt.ylabel('Цена закрытия')
plt.legend()
plt.show()

# Описательные статистики
print(df[['open', 'high', 'low', 'close', 'volume']].describe())
# Рассчитываем матрицу корреляции
correlation_matrix = df[['open', 'high', 'low', 'close', 'volume']].corr()
print(correlation_matrix)

# Построение графика ежедневных доходностей
plt.figure(figsize=(10, 6))
df['Daily Returns'].plot()
plt.title('Ежедневная инвестиционная доходность')
plt.xlabel('Дата')
plt.ylabel('Ежедневный доход')
plt.show()

# Построение гистограммы распределения доходности
plt.figure(figsize=(8, 6))
df['Daily Returns'].hist(bins=50, alpha=0.5)
plt.title('Распределение ежедневной инвестиционной доходности')
plt.xlabel('Ежедневная доходность')
plt.ylabel('Частота')
plt.show()

# Удаление строк с пропущенными значениями
df.dropna(inplace=True)
# Заполнение пропущенных значений средним
mean_value = df['Daily Returns'].mean()
df['Daily Returns'].fillna(mean_value, inplace=True)

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error

# Создаем признаки (X) и целевую переменную (y)
X = df[['close']]
y = df['Daily Returns']

# Разделение данных на обучающий и тестовый наборы
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Инициализация модели линейной регрессии
model = LinearRegression()

# Обучение модели на обучающих данных
model.fit(X_train, y_train)

# Предсказание на тестовом наборе
y_pred = model.predict(X_test)

# Оценка производительности модели
r2 = r2_score(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)

print(f"Коэффициент детерминации (R^2): {r2}")
print(f"Среднеквадратичная ошибка (MSE): {mse}")



df['date'] = pd.to_datetime(df['date'])
df.set_index('date', inplace=True)
df['50_MA'] = df['close'].rolling(window=50).mean()

# Определить тренд
df['trend'] = 'Боковой тренд'

df.loc[df['close'] > df['50_MA'], 'trend'] = 'Восходящий'
df.loc[df['close'] < df['50_MA'], 'trend'] = 'Нисходящий'
#Восходящий или «бычий» — цена растет. Трейдеру важно предсказать его начало и купить акции
# на самом первом этапе роста.
# Нисходящий или «медвежий» — цена падает. Когда достигнут пик, и котировки движутся вниз,
# нужно или продать имеющиеся акции или сыграть на понижение.
# Боковой — цена незначительно колеблется вокруг какой-то цифры.
# Заметную прибыль получить нельзя, остается только ждать, пока цена куда-то сдвинется.

# Визуализировать цены закрытия и скользящую среднюю с отображением тренда
plt.figure(figsize=(12, 6))
plt.plot(df.index, df['close'], label='Цена закрытия')
plt.plot(df.index, df['50_MA'], label='50-дневная скользящая средняя')
plt.title('Цены акций и 50-дневная скользящая средняя')
plt.xlabel('Дата')
plt.ylabel('Цена')
plt.legend()

# Вывести на экран
plt.show()

# Вывести результаты определения тренда
print("Trend Analysis:")
print(df['trend'].value_counts())

descending_stocks = df[df['trend'] == 'Нисходящий']['Name'].unique()

print("Акции, которые необходимо продать:")
for stock in descending_stocks:
    print(stock)






