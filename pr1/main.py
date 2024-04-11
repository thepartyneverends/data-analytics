import pandas as pd

df = pd.read_csv('learn_dataset (1).csv', delimiter=';')

df.info()


# Размер датасета
print(df.shape)


# Вывод всей таблицы
print(df)


# Вывод первых пяти строк
print(df.head())


# Вывод последних пяти строк
print(df.tail())


#Вывод одного признака
print(df['Age'])


# Вывод описательных статистик по количественному признаку «Возраст»
print(df['Age'].describe())


# Получение общих данных по признаку «Раса»
print(df['race'].value_counts()


# Получение среднего по росту
print(df['height'].mean())


# Вычисление медианы по росту
print(df['height'].median())


# Получение сразу нескольких величин: минимум, максимум и среднеквадратичное отклонение
print(df['height'].min(), df['height'].max(), df['height'].std())


# Выбор нескольких столбцов из таблицы
print(df[['Age', 'Sex', 'education']])

# Перенос всех женщин и мужчин в отдельные датасеты
females = df[df['Sex'] == 'Female']
males = df[df['Sex'] == 'Male']


# Вычисление продолжительности образования для мужчин и женщин отдельно
print(males['education-num'].mean())
print(females['education-num'].mean())


# Описательные статистики по росту для мужчин и женщин
print(males['height'].describe())
print(females['height'].describe())


# Среднее значение и медиана количества рабочих часов в неделю
print(males['hours-per-week'].mean(), females['hours-per-week'].mean())
print(males['hours-per-week'].median(), females['hours-per-week'].median())


# Количество разведенных среди мужчин и женщин
print(males['marital-status'].value_counts())
print(females['marital-status'].value_counts())


# Получение людей с возрастом выше среднего
print(df[df['Age'] > df['Age'].mean()])


# Люди выше среднего и ниже среднего. Сравнение
print(df[df['Age'] > df['Age'].mean()].shape, df[df['Age'] < df['Age'].mean()].shape)


# Люди с максимальным весом
df1 = df[df['weight'] == df['weight'].max()]
print(df1[['Age', 'education', 'height', 'weight']])


# Сортировка по продолжительности образования от меньшего к большему
df2 = df[['Age', 'education', 'education-num']]
print(df2.sort_values(by='education-num', ascending=False))


# Сортировка по возрасту и продолжительности образования
df2 = df[['Age', 'education', 'education-num']]
df2_sort = df2.sort_values(by=['education-num', 'Age'], ascending=[True, False])
print(df2_sort.tail(10))


# Человек с самым большим возрастом
df2 = df[['Age', 'education', 'education-num']]
df2_sort = df2.sort_values(by='Age', ascending=False)
print(df2_sort.iloc[2, 0])


# Подсчет индекса массы тела
df['imt'] = 10000*df['weight']/(df['height']*df['height'])
print(df[['Age', 'height', 'weight', 'imt']])


# Корреляция между размером прибыли и количеством рабочих часов в неделю
df3 = df[['Age', 'hours-per-week', 'capital-gain']]
print(df3.corr())


# Корреляция между весом и индексом массы тела
df4 = df[['Age', 'imt', 'height', 'weight']]
print(df4.corr())


# Группировка по расе с вычислением средней продолжительности образования
print(df.groupby('race')['education-num'].mean())


# Группировка по расе с вычислением средних продолжительности образования и числа рабочих часов в неделю
print(df.groupby(['Sex','race'])['education-num','hours-per-week'].mean())


# Группировка по двум признакам(расе и полу) с вычислением средних продолжительности образования и числа рабочих часов в неделю
print(df.groupby(['Sex','race'])['education-num','hours-per-week'].mean())