### [Pandas](https://pandas.pydata.org/pandas-docs/stable/) `import pandas as pd`
* считывание файла
 * [read_csv()](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.read_csv.html) - прочитать csv файл `pd.read_csv('путь и название файла', sep='разделитель', decimal='. или ,')`
 * [read_excel()](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.read_excel.html) - прочитать файл excel `pd.read_excel('файл.xlsx', sheet_name='первый лист')`
* преобразование типов данных
  * [astype()](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.astype.html) - изменить тип данных `convert_dict = {'col1': float, 'col2': str} DataFrame.astype(convert_dict)`
  * [to_numeric()](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.to_numeric.html) - преобразовывает аргумент в числовую переменную `pd.to_numeric(arg=, errors='raise', downcast=None)`
  * [to_datetime()](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.to_datetime.html) - преобразовать в объект дататайм `pd.to_datetime(arg, format='%d.%m.%YZ%H:%M:%S')` [format](https://docs.python.org/3/library/datetime.html#strftime-and-strptime-behavior) 
   * [DatetimeIndex](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DatetimeIndex.html) - объект для расчета по заданному периоду например month `pd.DatetimeIndex(data['col']).month`
* соединение датафреймов
 * [pd.concat()](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.concat.html) - соединение датафреймов `pd.concat([df_1] + [df_2] * 4)`
 * [join()](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.join.html) - объединение данных по индексу `DataFrame.join(other, on=None, how='left')` проверка размера - shape
 * [merge()](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.merge.html) - соединение таблиц по общему столбцу `data.merge(subcat_dict, on='subcat_id', how='left')` проверка размера - shape
* работа с дубликатами
 * [duplicated()](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.duplicated.html) - возвращает логический ряд, обозначающий повторящиеся строки `DataFrame.duplicated()#.sum()`
 * [drop_duplicates()](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.drop_duplicates.html) - удаление повторяющихся строк `drop_duplicates()` inplace=False - удалять дубликаты на месте или возвращать копию
* работа с пропущенными значениями
 * [isna()](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.isna.html) - проверка датафрейма на пропущенные значения `DataFrame.isna()`
 * [dropna()](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.dropna.html) - удаляет пропущенные значения датафрейма `DataFrame.dropna(axis='columns')`
 * [fillna()](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.fillna.html) - заполняет значения Na и NAN заданным значением или специальным методом `DataFrame.fillna(value=None, method=None)`
 * [transform()](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.transform.html) - метод преобразования значений, который вернет туже длину (для заполнения пропусков средними значениями по группе) `df['total_income'].fillna(df.groupby('income_type_id')['total_income'].transform('mean'))`
* аггрегация данных
 * [pivottablejs](https://pypi.org/project/pivottablejs/) - библиотека визуального аггрегирования pivot table
 * [pivot_table()](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.pivot_table.html) - создать сводную таблицу `data.pivot_table(index=, columns='source', values=, aggfunc='sum')`
 * [agg()](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.agg.html) - группировка с использованием одной или нескольких функций `groupby('name_gr').agg({'name_col': ['count', 'sum']}) `
 * [describe()](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.describe.html) - вернуть таблицу из описательных статистик по датафрейму (чтобы получить статистики по категориальным столбцам необходимо выбрать только их) `DataFrame.select_dtypes(include=['object', 'datatime']).describe()`
 * [groupby()](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.groupby.html) - группирует датафрейм по выбранной переменной `DataFrame.groupby(by='переменная').func(функция группировки)`
* фильтрация датасета
 * [loc[]](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.loc.html) - получить доступ к строкам/столбцам по условиям или значениям `DataFrame.loc[строка, столбец]`
 * [query()](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.query.html) - метод сортировки `DataFrame.query('col1 == True or col2 > 20')`
 * [sample()](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.sample.html) - возвращает выборку заданного размера `DataFrame.sample(frac=1, replace=False, random_state=state)`
* работа со столбцами
 * [rename()](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.rename.html) - переименовать столбец `DataFrame.rename(columns={"A": "a", "B": "c"})`
 * [объединение столбцов через zip()](https://disk.yandex.ru/client/disk/%D0%BC%D0%B0%D1%82%D0%B5%D1%80%D0%B8%D0%B0%D0%BB%D1%8B%20%D1%8F%D0%BD%D0%B4%D0%B5%D0%BA%D1%81.%D0%BF%D1%80%D0%B0%D0%BA%D1%82%D0%B8%D0%BA%D1%83%D0%BC/%D0%9F%D0%BE%D0%BB%D0%B5%D0%B7%D0%BD%D1%8B%D0%B5%20%D1%81%D0%BA%D1%80%D0%B8%D0%BD%D1%8B%20(%D1%81%D1%81%D1%8B%D0%BB%D0%BA%D0%B8%20%D0%B8%D0%B7%20%D1%88%D0%BF%D0%B0%D1%80%D0%B3%D0%B0%D0%BB%D0%BA%D0%B8)?idApp=client&dialog=slider&idDialog=%2Fdisk%2F%D0%BC%D0%B0%D1%82%D0%B5%D1%80%D0%B8%D0%B0%D0%BB%D1%8B%20%D1%8F%D0%BD%D0%B4%D0%B5%D0%BA%D1%81.%D0%BF%D1%80%D0%B0%D0%BA%D1%82%D0%B8%D0%BA%D1%83%D0%BC%2F%D0%9F%D0%BE%D0%BB%D0%B5%D0%B7%D0%BD%D1%8B%D0%B5%20%D1%81%D0%BA%D1%80%D0%B8%D0%BD%D1%8B%20(%D1%81%D1%81%D1%8B%D0%BB%D0%BA%D0%B8%20%D0%B8%D0%B7%20%D1%88%D0%BF%D0%B0%D1%80%D0%B3%D0%B0%D0%BB%D0%BA%D0%B8)%2F%D0%BE%D0%B1%D1%8A%D0%B5%D0%B4%D0%B8%D0%BD%D0%B5%D0%BD%D0%B8%D0%B5%20%D1%81%D1%82%D0%BE%D0%BB%D0%B1%D1%86%D0%BE%D0%B2.png) - `data['new_col'] = list(zip(data['col1'], data['col2']))`
* другие полезные функции
 * [apply()](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.apply.html) - применяет функцию ко всем строкам/столбцам датафрейма `DataFrame.apply(func='')`
 * [reset_index()](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.reset_index.html) - сбросить значения индексов или уровней `reset_index()` drop=False - вставить столбец index или обновить текущий
 * [sort_values()](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.sort_values.html) - сортировка значений (по умолчанию по возрастанию ascending) `DataFrame.sort_values(ascending=True)`
 * [quantile()](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.quantile.html) - вычислить квантиль `quantile(0.99)`
 * [value_counts()](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.Series.value_counts.html) - возвращает серию, содержащую количество уникальных значений `Series.value_counts(normalize=False)`
 * [unique()](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.unique.html) - возвращает уникальные элементы в порядке их появления `DataFrame.unique()`


#machine_learning #preprocessing 