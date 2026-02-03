# Benchmarks

Сравнение производительности `clustered-fast-trie` с `BTreeSet` из стандартной библиотеки.

## Запуск бенчмарков

### Все бенчмарки
```bash
cargo bench
```

### Конкретный файл бенчмарков
```bash
# Сравнение операций с полными датасетами
cargo bench --bench comparison

# Одиночные операции с разными размерами
cargo bench --bench single_ops
```

### Конкретный тест
```bash
cargo bench --bench comparison insert_sequential
cargo bench --bench comparison contains
cargo bench --bench single_ops single_insert
```

## Структура бенчмарков

### `comparison.rs` - Сравнение с BTreeSet

**Тестируемые сценарии:**

1. **insert_sequential** - вставка последовательных ключей (0, 1, 2, ...)
   - Размеры: 1K, 10K, 100K элементов
   - Показывает производительность на идеальных данных

2. **insert_clustered** - вставка кластерных данных (несколько диапазонов)
   - Паттерн: 0-1000, 10000-11000, 20000-21000, 30000-31000
   - Реальный сценарий для Kafka offsets, временных меток

3. **insert_random** - вставка псевдослучайных ключей
   - 10K элементов
   - Худший случай для Trie

4. **contains** - проверка наличия ключей
   - existing: поиск существующих ключей
   - missing: поиск отсутствующих ключей
   - 10K элементов в датасете

5. **contains_clustered** - поиск в кластерных данных
   - Несколько кластеров по 1K элементов
   - Показывает cache-friendly паттерны

6. **remove** - удаление ключей
   - sequential: удаление подряд
   - sparse: удаление через один
   - 10K элементов

7. **mixed_workload** - смешанная нагрузка
   - insert → contains → remove → insert
   - Реалистичный сценарий использования

### `single_ops.rs` - Одиночные операции

**Тестируемые сценарии:**

1. **single_insert** - одна вставка в датасет разного размера
   - Размеры: 100, 1K, 10K, 100K
   - Показывает как растет latency с размером

2. **single_contains** - один поиск в датасете разного размера
   - hit: ключ найден (в середине)
   - miss: ключ не найден
   - Размеры: 100, 1K, 10K, 100K

3. **single_remove** - одно удаление из датасета разного размера
   - Удаление из середины датасета
   - Размеры: 100, 1K, 10K, 100K

4. **sequential_pattern** - последовательные вставки
   - Прямой порядок (0→1000)
   - Обратный порядок (1000→0)
   - Показывает cache benefits

5. **worst_case_insert** - худший случай для вставки
   - Alternating pattern: чередование далеких ключей
   - Нет преимуществ от кеширования

## Ожидаемые результаты

### Где Trie должен быть быстрее:
- ✅ Sequential inserts (cache hot path)
- ✅ Clustered data (cache locality)
- ✅ Large datasets (O(log log U) vs O(log n))
- ✅ Contains on clustered data

### Где BTreeSet может быть быстрее:
- ⚠️ Random inserts (no cache benefits)
- ⚠️ Very small datasets (<100 elements)
- ⚠️ Sparse random keys

### O(log log U) vs O(log n):
- Trie: зависит от размера ключа (u64 = 8 уровней)
- BTreeSet: зависит от количества элементов
- Crossover point: ~256 элементов (log₂ 256 = 8)

## Интерпретация результатов

Criterion выводит:
- **time**: среднее время операции
- **thrpt**: throughput (операций/сек)
- **change**: изменение относительно предыдущего запуска

Смотрите на:
1. Абсолютные значения (ns/op)
2. Скейлинг с ростом размера
3. Variance (стабильность latency)

## Визуализация

После запуска бенчмарков Criterion создает HTML отчеты:

```bash
# Открыть отчет в браузере
start target/criterion/report/index.html  # Windows
open target/criterion/report/index.html   # macOS
xdg-open target/criterion/report/index.html  # Linux
```

## Примечания

- Используйте `--release` для реалистичных результатов (cargo bench делает это автоматически)
- Закройте другие приложения для стабильных измерений
- Первый запуск создает baseline для сравнения
- Последующие запуски сравниваются с baseline
