import pandas as pd
import seaborn as sns
from sklearn import tree, preprocessing
import graphviz
from sklearn.metrics import mean_absolute_error, accuracy_score, precision_score, recall_score
from matplotlib import pyplot as plt
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, ExtraTreesClassifier

# Загрузка датасета и подготовка выборки
df = pd.read_csv('../learn_dataset (1).csv', delimiter=';')
#
# df1 = df[['education-num', 'hours-per-week', 'height', 'capital-gain']]
#
# # Обучение модели и вывод на экран решающего дерева
# model = tree.DecisionTreeRegressor(max_depth=10)
# model.fit(df1[['education-num', 'hours-per-week', 'height']].values.reshape(-1, 3), y=df1['capital-gain'].values)
#
# dot_data = tree.export_graphviz(model, out_file=None,
#                                 feature_names=['education-num', 'hours-per-week', 'height'],
#                                 class_names=['capital-gain'],
#                                 filled=True, rounded=True,
#                                 special_characters=True)
# graph = graphviz.Source(dot_data)
# print(graph)
#
# # Подготовка тестовой выборки
# df_test = pd.read_csv('../learn_dataset (1).csv', delimiter=';')
# df_test1 = df_test[['education-num', 'hours-per-week', 'height', 'capital-gain']]
# df_test1.dropna()
#
# # Применение обученной модели к тестовой выборке
# df_test1['Predicted capital gain'] = model.predict(df_test1[['education-num',
#                                                              'hours-per-week',
#                                                              'height']].values.reshape(-1, 3))
#
# # Средняя абсолютная ошибка
# mean_absolute_error(df_test1['Predicted capital gain'], df_test1['capital-gain'])

# Подготовка выборки
# df = pd.read_csv('../learn_dataset (1).csv', delimiter=';')
# df1 = df[['capital-gain', 'hours-per-week', 'education-num', 'height', 'Sex']]
# df1.dropna()

# # Обучение модели
# model = RandomForestClassifier(max_depth=9, random_state=0)
# model.fit(df1[['height', 'hours-per-week', 'education-num', 'capital-gain']].values.reshape(-1, 4), y=df1['Sex'].values)
#
# # Формирование тестовой выборки
# df_test1 = df[['height', 'capital-gain', 'hours-per-week', 'education-num', 'Sex']]
#
# # Предсказание пола
# df_test1['Predicted sex'] = model.predict(df_test1[['height',
#                                                     'capital-gain',
#                                                     'hours-per-week',
#                                                     'education-num']].values.reshape(-1, 4))
#
# # Вычисление метрики точности
# print(accuracy_score(df_test1['Predicted sex'], df_test1['Sex']))
#
# # Точность по категориям и полнота
# print(precision_score(df_test1['Predicted sex'], df_test1['Sex'], average=None, zero_division=1))
# print(recall_score(df_test1['Predicted sex'], df_test1['Sex'], average=None, zero_division=1))
#
# # Матрица ошибок
# pd.crosstab(df_test1['Predicted sex'], df_test1['Sex'])

# # Подготовка выборки
# df1 = df[['capital-gain', 'hours-per-week', 'education-num', 'height', 'Sex']]
# df1.dropna()
#
# # Обучение модели градиентного бустинга
# model = GradientBoostingClassifier(random_state=0)
# model.fit(df1[['height', 'hours-per-week', 'education-num', 'capital-gain']].values.reshape(-1, 4), y=df1['Sex'].values)
#
# # Подготовка тестовой выборки
# df_test = pd.read_csv('../learn_dataset (1).csv', delimiter=';')
# df_test1 = df_test[['height', 'capital-gain', 'hours-per-week', 'education-num', 'Sex']]
#
# # Предсказание
# df_test1['Predicted sex'] = model.predict(df_test1[['height', 'capital-gain', 'hours-per-week', 'education-num']].values.reshape(-1, 4))
#
# # Точность без учета категории
# print(accuracy_score(df_test1['Predicted sex'], df_test1['Sex']))
#
# # Точность и полнота
# print(precision_score(df_test1['Predicted sex'], df_test1['Sex'], average=None, zero_division=1))
# print(recall_score(df_test1['Predicted sex'], df_test1['Sex'], average=None, zero_division=1))
#
# # Матрица ошибок
# pd.crosstab(df_test1['Predicted sex'], df_test1['Sex'])

# # Создание выборки
# df1 = df[['capital-gain', 'hours-per-week', 'education-num', 'height', 'Sex']]
# df1.dropna()
#
# # Обучение случайного леса
# model = RandomForestClassifier(max_depth=2, random_state=0)
# model.fit(df1[['height', 'hours-per-week', 'education-num', 'capital-gain']].values.reshape(-1, 4), y=df1['Sex'].values)
#
# # Получение тестовой выбрки
# df_test1 = df[['height', 'capital-gain', 'hours-per-week', 'education-num', 'Sex']]
#
# # Предсказание вероятности принадлежности к классу
# result = model.predict_proba(df_test1[['height', 'capital-gain', 'hours-per-week', 'education-num']].values.reshape(-1, 4))
# print(result)
#
# # Два новых столбца в тестовой выборке
# df_test1['pr 0'] = result[:,0]
# df_test1['pr 1'] = result[:,1]
#
# # Преобразование признака «раса» в числовой
# coder = preprocessing.LabelEncoder()
# coder.fit(df['race'])
# coder.transform(df['race'])
#
# # Замена столбца «race» на числовые значения
# df['race'] = coder.transform(df['race'])
#
# # Замена категориальных признаков на числовые в цикле
# for n in ['marital-status', 'relationship', 'Sex']:
#     coder.fit(df[n])
#     df[n] = coder.transform(df[n])
#
# df1 = df[['marital-status', 'relationship', 'Sex']]
# df1.dropna()
#
# # Сценарий предсказания пола по  двум категориальным признакам
# model = tree.DecisionTreeClassifier(max_depth=10)
# model.fit(df1[['marital-status', 'relationship']].values.reshape(-1, 2), y=df1['Sex'].values)
#
# dot_data = tree.export_graphviz(model, out_file=None,
#                                 feature_names=['marital-status', 'relationship'],
#                                 class_names=['f', 'm'],
#                                 filled=True, rounded=True,
#                                 special_characters=True)
# graph = graphviz.Source(dot_data)
# print(graph)
#
# coder = preprocessing.LabelEncoder()
#
# # Перекодирование категориальных признаков в числовые
# for n in ['workclass', 'education', 'marital-status', 'occupation', 'relationship', 'race', 'Sex', 'native-country', 'salary']:
#     coder.fit(df[n])
#     df[n] = coder.transform(df[n])
#
# # Массив категориальных признаков
# df.dropna()
# mas = ['occupation', 'relationship', 'race', 'Sex', 'native-country', 'marital-status', 'education', 'education-num', 'salary']
#
# # Определение значимости признаков по отношению к признаку
# # «education»
# selector = ExtraTreesClassifier()
# result = selector.fit(df[mas], df['education'])
# result.feature_importances_
#
# # Вывод значимости в удобной форме
# features_table = pd.DataFrame(result.feature_importances_, index=mas, columns=['importance'])
# print(features_table)
#
# # Упорядоченная таблица значимости признаков
# features_table.sort_values(by='importance', ascending=False)
#
# df1 = df[['occupation', 'marital-status', 'education', 'education-num']]
#
# # Подготовка выборки использующей только значимые признаки
# for n in ['education', 'marital-status', 'occupation', 'education-num']:
#     coder.fit(df[n])
#     df[n] = coder.transform(df[n])

# # Выборка с признаками «пол» и «рост»
# df1 = df[['Sex', 'height']]
# df1.dropna()
#
# # Быстрое кодирование и его результат
# df1 = pd.get_dummies(df1)
# df1.dropna()
#
# # Удаление избыточности
# df1 = pd.get_dummies(df1, drop_first=True)
# df1.dropna()
#
# # Предсказание признака «Sex_Male»
# model = tree.DecisionTreeClassifier(max_depth=5)
# model.fit(df1['height'].values.reshape(-1, 1), y=df1['Sex_Male'].values)
#
# # Формирование тестовой выборки
# df_test1 = df[['Sex', 'height']]
# df_test1.dropna()
#
# # Метод быстрого кодирования примененный к тестовой выборке
# df_test1 = pd.get_dummies(df_test1, drop_first=True)
# df_test1.head()
#
# # Предсказание
# df_test1['Predicted sex_male'] = model.predict(df_test1[['height']].values.reshape(-1, 1))
#
# # Матрица ошибок
# pd.crosstab(df_test1['Predicted sex_male'], df_test1['Sex_Male'])
