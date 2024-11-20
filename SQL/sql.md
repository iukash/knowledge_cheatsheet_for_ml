### <a class="anchor" id="sql"></a> SQL
```
SELECT CAST(DATE_PART('isodow', time) AS INTEGER) AS weekday_number, 
		TO_CHAR(time, 'Dy') AS weekday, 
        SUM(CASE WHEN action='create_order' THEN 1 ELSE 0 END) AS created_orders,
        SUM(CASE WHEN action='cancel_order' THEN 1 ELSE 0 END) AS canceled_orders
FROM user_actions
WHERE date_trunc('day', time) BETWEEN '2022-08-24' AND '2022-09-06'
GROUP BY weekday_number, weekday
ORDER BY weekday_number
```

* `SELECT` -- перечисление полей результирующей таблицы
* `FROM` -- указание источника данных
* `WHERE` -- фильтрация данных
* `GROUP BY` -- группировка данных
* `HAVING` -- фильтрация данных после группировки
* `ORDER BY` -- сортировка результирующей таблицы
* `LIMIT` -- ограничение количества выводимых записей

* `array_length(ARRAY, 1)` - размер массива данных
* `DISTINCT`  -- позволяет отбирать уникальные записи, т.е. избавляться от всех дубликатов в таблице
* `DATE_TRUNC('month', TIMESTAMP '2022-01-12 08:55:30')` -- используется для усечения дат и времени
* `CAST(поле AS тип данных)` изменение типа данных
* `BETWEEN значение_1 AND значение_2` срез промежутка данных
* `EXTRACT(часть_даты FROM поле) AS новое_поле_с_датой` извлечение части даты
* `DATE_TRUNC('часть_даты_до_которой_усекаем', поле) AS новое_поле_с_датой` усечение части даты
* `DATE_PART` -- ('year', DATE '2022-01-12')
* `INNER JOIN таблица_2 ON таблица_1.поле_1 = таблица_2.поле_2;` объединение таблиц (несовпадающие значения не войдут в финальную таблицу)
* `LEFT JOIN таблица_2 ON таблица_1.поле_1 = таблица_2.поле_2;` объединение таблиц (значения левой таблицы для которых отсутствуют правые будут равно NULL)

Практика SQL
http://localhost:8888/notebooks/knowledge_cheatsheet_for_ml%2FSQL%2Fsql_practice.ipynb



#machine_learning 
