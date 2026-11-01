
# 1. ИМПОРТ БИБЛИОТЕК
# подключаем инструменты для HTTP-запросов, парсинга HTML,
# поиска счёта, подсчёта матчей и сохранения файлов

import requests
from bs4 import BeautifulSoup
import re
import csv
from pathlib import Path
from collections import Counter


# 2. ПОЛУЧЕНИЕ HTML-СТРАНИЦЫ
# отправляем GET-запрос к Wikipedia
# user-agent идентифицирует нашу программу
# raise_for_status() останавливает выполнение при HTTP-ошибке

url = "https://en.wikipedia.org/wiki/2025%E2%80%9326_EuroLeague_regular_season"

headers = {
    "User-Agent": "EuroLeagueStudy/0.1 (https://github.com/m3t4morph0sis)"
}

response = requests.get(url, headers=headers, timeout=15)
response.raise_for_status()


# 3. СОХРАНЕНИЕ ИСХОДНОГО HTML
# сохраняем полученную страницу в data/raw
# при необходимости ее можно загрузить из файла без интернета


project_root = Path(__file__).resolve().parent.parent
raw_path = project_root / "data" / "raw" / "euroleague.html"

raw_path.parent.mkdir(parents=True, exist_ok=True)
raw_path.write_text(response.text, encoding="utf-8")


# 4. ПОИСК HTML-ТАБЛИЦ
# BeautifulSoup разбирает HTML в дерево элементов
# находим все таблицы, среди которых есть результаты матчей

soup = BeautifulSoup(response.text, "html.parser")
tables = soup.find_all("table")


# 5. ИЗВЛЕЧЕНИЕ ДАННЫХ МАТЧЕЙ
# перебираем таблицы и ищем структуру с датой, двумя командами и счётом
# неподходящие таблицы пропускаем

matches = []

for table in tables:
    row = table.find("tr")

    if row is None:
        continue

    cells = row.find_all("td", recursive=False)

    if len(cells) < 4:
        continue

    home = cells[1].select_one(".fn.org")
    away = cells[3].select_one(".fn.org")

    if home is None or away is None:
        continue

    # регулярное выражение извлекает два числа из счёта
    score = cells[2].get_text(" ", strip=True)
    result = re.search(r"(\d+)\s*[-–—−]\s*(\d+)", score)

    if result is None:
        continue

    # формируем структурированную запись одного матча
    match = {
        "date": cells[0].get_text(" ", strip=True),
        "home_team": home.get_text(" ", strip=True),
        "away_team": away.get_text(" ", strip=True),
        "home_points": int(result.group(1)),
        "away_points": int(result.group(2))
    }

    matches.append(match)


# 6. СТАНДАРТИЗАЦИЯ НАЗВАНИЙ КОМАНД
# в сезоне Baskonia встречается под двумя названиями:
# Baskonia (12 матчей) и Kosner Baskonia (26 матчей)
# приводим оба названия к одному, чтобы не получить
# ошибочные 21 команду вместо 20.

for match in matches:
    for field in ("home_team", "away_team"):
        if match[field] == "Kosner Baskonia":
            match[field] = "Baskonia"


# 7. ПРОВЕРКА КОРРЕКТНОСТИ ДАННЫХ
# проверяем структуру регулярного чемпионата:
# 380 матчей, 20 команд, по 38 матчей у каждой
# каждая упорядоченная пара хозяева-гости уникальна

teams = set()
games_count = Counter()
match_pairs = []

for match in matches:
    home = match["home_team"]
    away = match["away_team"]

    teams.add(home)
    teams.add(away)

    games_count[home] += 1
    games_count[away] += 1

    match_pairs.append((home, away))

is_valid = (
    len(matches) == 380
    and len(teams) == 20
    and all(count == 38 for count in games_count.values())
    and len(match_pairs) == len(set(match_pairs))
)

if not is_valid:
    raise ValueError("данные не прошли проверку !")


# дополнительная проверка содержимого матчей
from datetime import datetime

for match in matches:
    # проверяем корректность даты
    datetime.strptime(match["date"], "%d %B %Y")

    # команда не может играть сама с собой
    if match["home_team"] == match["away_team"]:
        raise ValueError("команда играет сама с собой !")

    # очки должны быть положительными целыми числами
    if not all(
        type(match[field]) is int and match[field] > 0
        for field in ("home_points", "away_points")
    ):
        raise ValueError("некорректный счёт матча !")

    # в баскетболе итоговая ничья невозможна
    if match["home_points"] == match["away_points"]:
        raise ValueError("обнаружена ничья !")

print("дополнительные проверки: пройдены")



# 8. СОХРАНЕНИЕ РЕЗУЛЬТАТОВ В CSV
# сохраняем проверенные данные в data/processed
# UTF-8-SIG помогает корректно отображать Unicode в Excel

output_path = project_root / "data" / "processed" / "matches.csv"
output_path.parent.mkdir(parents=True, exist_ok=True)

fieldnames = [
    "date",
    "home_team",
    "away_team",
    "home_points",
    "away_points"
]

with open(output_path, "w", newline="", encoding="utf-8-sig") as file:
    writer = csv.DictWriter(file, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(matches)


# ИТОГОВЫЙ ОТЧЁТ
# выводим только основные результаты работы парсера

print("\n    EUROLEAGUE 2025/26    ")
print("HTTP-статус:", response.status_code)
print("HTML-таблиц найдено:", len(tables))
print("матчей обработано:", len(matches))
print("команд:", len(teams))
print("проверка данных: пройдена")
print("исходный HTML:", raw_path)
print("результат CSV:", output_path)



# # ДЕМОНСТРАЦИЯ ПРОБЛЕМЫ HTTP 403
#
# # Первоначальный вариант запроса без собственного User-Agent
# response = requests.get(url, timeout=15)
# print(response.status_code)
#
# # Исправленный вариант с идентифицирующим User-Agent
# headers = {
#     "User-Agent": "EuroLeagueStudy/0.1 (https://github.com/m3t4morph0sis)"
# }
#
# response = requests.get(
#     url,
#     headers=headers,
#     timeout=15
# )
# print(response.status_code)



# ДЕМОНСТРАЦИЯ ОШИБКИ С БАСКОНИЕЙ

# 1. Повторное извлечение исходных названий команд из HTML,
# чтобы увидеть данные ДО стандартизации.

# original_names = []
#
# for table in tables:
#     row = table.find("tr")
#
#     if row is None:
#         continue
#
#     cells = row.find_all("td", recursive=False)
#
#     if len(cells) < 4:
#         continue
#
#     home = cells[1].select_one(".fn.org")
#     away = cells[3].select_one(".fn.org")
#
#     if home is None or away is None:
#         continue
#
#     score = cells[2].get_text(" ", strip=True)
#
#     if re.search(r"(\d+)\s*[-–—−]\s*(\d+)", score) is None:
#         continue
#
#     original_names.extend([
#         home.get_text(" ", strip=True),
#         away.get_text(" ", strip=True)
#     ])
#
# before = Counter(original_names)
#
# print("\n ДО ИСПРАВЛЕНИЯ")
# print("Количество команд:", len(before))
#
# print("\nСПИСОК КОМАНД ДО ОЧИСТКИ:")
#
# for team, count in sorted(before.items()):
#     status = " проблема" if count != 38 else ""
#     print(f"{team:<35}  {count:>2} матчей{status}")
#
#
#
# # 2. Проверка данных ПОСЛЕ стандартизации.
# # Список matches уже содержит исправленные названия.
#
# after = Counter(
#     team
#     for match in matches
#     for team in (match["home_team"], match["away_team"])
# )
#
# print("\n ПОСЛЕ ИСПРАВЛЕНИЯ")
# print("Количество команд:", len(after))
#
# print("\nСПИСОК КОМАНД ПОСЛЕ ОЧИСТКИ:")
#
# for team, count in sorted(after.items()):
#     status = "OK" if count == 38 else "ПРОБЛЕМА"
#     print(f"{team:<35}  {count:>2} матчей  {status}")
#
# print("\nВсе команды сыграли 38 матчей:",
#       all(count == 38 for count in after.values()))


