from sklearn import tree
from sklearn.linear_model import LinearRegression, SGDClassifier
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_absolute_error, accuracy_score, precision_recall_fscore_support
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
import graphviz

df = pd.read_csv('../learn_dataset (1).csv', delimiter=';')

# Удаляем строки с нулевой прибылью
# df1 = df[['capital-gain', 'hours-per-week']]
# df1 = df1[df1['capital-gain'] != 0]

# Точечная диаграмма зависимости прибыли от продолжительности
# рабочей недели
# sns.scatterplot(data=df1, x='hours-per-week', y='capital-gain')

# Матрица корреляции
# print(df1.corr())

# Превращаем столбец в горизонтальный вектор
# print(df1['hours-per-week'].values)

# Горизонтальный вектор превращаем в вертикальный набор
# print(df1['hours-per-week'].values.reshape(-1, 1))

# Вызываем метод линейной регрессии
# lr = LinearRegression
# result = lr.fit(df1['hours-per-week'].values.reshape(-1, 1), y=df1['capital-gain'].values)
# df1['predicted-capital-gain'] = result.predict(df1['hours-per-week'].values.reshape(-1, 1))
# df1.head(20)

# Вычисление средней абсолютной ошибки
# mean_absolute_error(df1['capital-gain'], df1['predicted-capital-gain'])


# Множественная регрессиия
# df1 = df[['capital-gain', 'hours-per-week', 'education-num']]
# df1 = df1[df1['capital-gain'] != 0]
# lr = LinearRegression
# result = lr.fit(df1['hours-per-week', 'education-num'].values.reshape(-1, 2), y=df1['capital-gain'].values)

# Результат регрессии
# print(result.coef_, result.intercept_)

# Средняя абсолютная ошибка
# df1['predicted-capital-gain'] = result.predict(df1[['hours-per-week', 'education-num']].values.reshape(-1, 2))
# df1.head(20)

# Формируем датасет df_test
# df_test = pd.read_csv('../learn_dataset (1).csv', delimiter=';')
# df_test = df_test[['capital-gain', 'hours-per-week', 'education-num']]
# df_test['predicted-capital-gain'] = result.predict(df_test[['hours-per-week', 'education-num']].values.reshape(-1, 2))
#
# Вычисляем ошибку на тестовой выборке
# mean_absolute_error(df_test['capital-gain'], df_test['predicted-capital-gain'])

# Строим новый датасет
# df1 = df[['capital-gain', 'height', 'Sex']]
#
# # Точка и ее ближайшие соседи
# sns.scatterplot(data=df1, x='height', y='capital-gain', hue='Sex')
#
# # Подключение библиотеки и нормирование
# scaler = StandardScaler()
# scaler.fit(df1[['height', 'capital-gain']].values.reshape(-1, 2))
# arr = scaler.transform(df1[['height', 'capital-gain']].values.reshape(-1, 2))
# print(arr)

# Применение метода k ближайших соседей
# model = KNeighborsClassifier(n_neighbors=1)
# model.fit(arr, y=df1['Sex'].values)

# Формирование тестового датасета и его нормирование
# df_test = df[['capital-gain', 'height', 'Sex']]
# df_test = df_test.dropna()
#
# # arr_test = scaler.transform(df_test[['height', 'capital-gain']].values.reshape(-1, 2))
#
# # df_test['Predicted sex'] = model.predict(arr_test)
# # df_test.tail(20)
#
# # Подсчет точности предсказания
# # print(accuracy_score(df_test['Predicted sex'], df_test['Sex']))
#
# # Матрица ошибок
# # pd.crosstab(df_test['Predicted sex'], df_test['Sex'])
#
# # Формирование столбца ‘Code’
# # df_test['Code'] = 0
# # df_test.loc[(df_test['Sex'] == 'Male') & (df_test['Predicted sex'] == 'Female'), 'Code'] = '1'
# # df_test.loc[(df_test['Sex'] == 'Female') & (df_test['Predicted sex'] == 'male'), 'Code'] = '2'
#
# # Визуализация ошибок
# sns.scatterplot(data=df_test, x='height', y='capital-gain', hue='Code')
#
# # Поиск 7 ближайших соседей
# model = KNeighborsClassifier(n_neighbors=7)
# model.fit(arr_test, y=df1['Sex'].values)
#
# # Точность предсказания
# print(accuracy_score(df_test['Predicted sex'], df_test['Sex']))
#
# # Матрица ошибок
# pd.crosstab(df_test['Predicted sex'], df_test['Sex'])

# df1 = df[['capital-gain', 'height', 'Sex']]
# df1 = df1.dropna()
#
# # Нормирование данных
# scaler = StandardScaler()
# scaler.fit(df1[['height', 'capital-gain']].values.reshape(-1, 2))
# arr = scaler.transform(df1[['height', 'capital-gain']].values.reshape(-1, 2))
#
# # Создание и обучение модели линейной классификации
# model = SGDClassifier()
# model.fit(arr, y=df1['Sex'].values)
#
# # Формирование тестовой выборки и ее нормирование
# df_test = pd.read_csv('../learn_dataset (1).csv', delimiter=';')
# df_test = df_test[['capital-gain', 'height', 'Sex']]
# df_test = df_test.dropna()
#
# arr_test = scaler.transform(df_test[['height', 'capital-gain']].values.reshape(-1, 2))
#
# # Предсказание на тестовой выборке
# df_test['Predicted sex'] = model.predict(arr_test)
#
# # Матрица ошибок
# pd.crosstab(df_test['Predicted sex'], df_test['Sex'])
#
# # Вычисление точности предсказания
# print(accuracy_score(df_test['Predicted sex'], df_test['Sex']))

# Формирование выборки и точечная диаграмма
# df1 = df[['education-num', 'capital-gain', 'hours-per-week', 'height', 'Sex']]
# # sns.pairplot(data=df1, hue='Sex')
#
# # Создание и обучение модели
# model = tree.DecisionTreeClassifier()
# model.fit(df1[['education-num', 'capital-gain', 'hours-per-week', 'height']].values.reshape(-1, 4), y=df1['Sex'].values)
#
# # Вывод решающего дерева на экран
# dot_data = tree.export_graphviz(model, out_file=None,
#                                 feature_names=['education-num', 'capital-gain', 'hours-per-week', 'height'],
#                                 class_names=['Female', 'Male'],
#                                 filled=True, rounded=True,
#                                 special_characters=True)
# graph = graphviz.Source(dot_data)
# print(graph)
#
# # Формирование тестовой выборки
# df_test = pd.read_csv('../learn_dataset (1).csv', delimiter=';')
# df_test1 = df_test[['education-num', 'capital-gain', 'hours-per-week', 'height', 'Sex']]
# df_test1.dropna()
#
# # Предсказание на тестовой выборке
# df_test1['Predicted sex'] = model.predict(df_test1[['education-num', 'capital-gain', 'hours-per-week', 'height']].values.reshape(-1, 4))
#
# # Матрица ошибок
# pd.crosstab(df_test1['Predicted sex'], df_test1['Sex'])
#
# # Изменение глубины решающего дерева
# model = tree.DecisionTreeClassifier(max_depth=3)
# model.fit(df1[['education-num', 'capital-gain', 'hours-per-week', 'height']].values.reshape(-1, 4), y=df1['Sex'].values)
#
# # Вычисление метрик предсказания
# precision_recall_fscore_support(df_test1['Sex'], df_test1['Predicted sex'])

plt.show()
