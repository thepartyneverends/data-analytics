import pandas as pd
import seaborn as sns
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt  #модуль для отображения диаграмм

df = pd.read_csv('../learn_dataset (1).csv', delimiter=';')

# Код для построения столбчатой диаграммы по продолжительности образования
# sns.displot(data=df, x='education-num')

# Параметр kind
# sns.displot(data=df, x='education-num', kind='kde')

# Код для построения точечной диаграммы
# sns.scatterplot(data=df, x='hours-per-week', y='capital-gain')

# Параметр hue в методе skatterplot
# sns.scatterplot(data=df, x='hours-per-week', y='capital-gain', hue='Sex')

# Точечная диаграмма по росту и весу с разбивкой по расам
# sns.scatterplot(data=df, x='height', y='weight', hue='race')

# Столбчатая диаграмма для категориального признака «пол»
# sns.countplot(data=df, x='Sex')

# Столбчатая диаграмма по признаку пола с разбивкой по семейному статусу
# sns.countplot(data=df, x='Sex', hue='marital-status')

# Построение парной точечной диаграммы
# df4 = df[['education-num', 'capital-gain', 'hours-per-week']]
# sns.pairplot(df4)

# Разбивка облаков точек по полу
# df4 = df[['education-num', 'capital-gain', 'hours-per-week', 'Sex']]
# sns.pairplot(df4, hue='Sex')

# Формирование нового датасета без пропущенных данных
# df1 = df.dropna()
# df1.info()

# Датасет после заполнения пропусков нулями
# df1 = df.fillna(0)
# df1.info()

# Заполнение пропусков средними значениями по признаку
# df1 = df.fillna(df.mean())
# df1.info()

# Заполнение пропусков медианными значениями по признаку
# df1 = df.fillna(df.median())
# df1.info()


# Заполнение пропусков разными способами для разных
# столбцов
# df['imt'] = 10000*df['weight']/(df['height']*df['height'])
# df['weight'] = df['weight'].fillna(df['weight'].median())
# df['imt'] = df['imt'].fillna(df['imt'].mean())
# df.info()

# Код для построения диаграммы «ящик с усами» по признаку
# «размер прибыли»
# df_fem = df[df['Sex'] == 'Female']
# sns.boxplot(x=df_fem['capital-gain'])

# Вычисление крайних точек допустимого диапазона
# m = df_fem['capital-gain'].mean()
# s = df_fem['capital-gain'].std()
# l = m - 3 * s
# r = m + 3 * s
# df_fem1=df_fem[(df_fem['capital-gain'] > 1) & (df_fem['capital-gain']<r)]
# sns.boxplot(x=df_fem1['capital-gain'])

# 25-я и 75-я процентиль по размеру прибыли
# a = df_fem['capital-gain'].quantile(0.25)
# b = df_fem['capital-gain'].quantile(0.75)

# Левая и правая границы интервала
# l = a - 1.5 * (b-a)
# r = b + 1.5 * (b-a)
# # print(l,r)
#
# df_fem1=df_fem[(df_fem['capital-gain']>1) & (df_fem['capital-gain'] < r )]
# sns.boxplot(x=df_fem1['capital-gain'])

# Средние значения по росту и продолжительности рабочей недели
# для мужчин и женщин
# df1 = df[df['Sex'] == 'Male']
# df2 = df[df['Sex'] == 'Female']
# df1['hours-per-week'].mean(), df2['hours-per-week'].mean()
# print(df1['height'].mean(), df2['height'].mean())

# Формируем датасет
df3 = df[['Sex', 'hours-per-week', 'height']]
df3 = df3.dropna()

# Выполняем кластеризацию
kmeans = KMeans(n_clusters=2, random_state=0)
kmeans.fit(df3[['height', 'hours-per-week']])

# Столбец с результатом кластеризации
df3['label'] = kmeans.labels_

# Отображение последних 50 элементов
# df3.tail(50)

# Визуальное разделение на два кластера по реальному полу
# sns.scatterplot(df3, x='height', y='hours-per-week', hue='Sex')

# Визуальное разделение на два кластера по параметру ‘label’
# sns.scatterplot(df3, x='height', y='hours-per-week', hue='label')

# Подсчет количества мужчин
# print(sum(df3['Sex'] == 'Male'))

# Количество мужчин, которые предсказаны как женщины
print(sum(df3['Sex'] == 'Male') & (df3['label'] == 1))

# Выделяем три кластера на основе продолжительности образования
df1 = df[['education-num', 'capital-gain', 'education']]
df1 = df1.dropna()
kmeans = KMeans(n_clusters=3, random_state=0)
kmeans.fit(df1[['education-num']])
df1['label'] = kmeans.labels_

# Разбиение на кластеры по продолжительности образования
sns.scatterplot(data=df1, x='capital-gain', y='education-num', hue='label')

plt.show()
