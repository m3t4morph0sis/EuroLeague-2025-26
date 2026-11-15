# EUROLEAGUE 2025/26
# EXPLORATORY DATA ANALYSIS (EDA)


# 1. ИМПОРТ БИБЛИОТЕК

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib

# Сохраняем графики в PNG для Markdown и PyCharm
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import seaborn as sns

from scipy.stats import binomtest, pearsonr


# 2. НАСТРОЙКИ И ПУТИ

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_PATH = PROJECT_ROOT / "data" / "processed" / "matches.csv"

DOC_PATH = PROJECT_ROOT / "docs" / "02_eda.md"

CHARTS_PATH = PROJECT_ROOT / "docs" / "assets" / "eda"

CHARTS_PATH.mkdir(parents=True, exist_ok=True)

# Маркеры для автоматического обновления результатов в Markdown
START_MARKER = "<!-- EDA_RESULTS_START -->"
END_MARKER = "<!-- EDA_RESULTS_END -->"

# Месяцы на английском для независимого от локали разбора дат
MONTHS = {
    name: number
    for number, name in enumerate(
        [
            "January", "February", "March", "April",
            "May", "June", "July", "August",
            "September", "October", "November", "December"
        ],
        start=1
    )
}


# 3. ЗАГРУЗКА И ПОДГОТОВКА ДАННЫХ

def parse_date(value):
    """Преобразует дату вида '30 September 2025' в Timestamp."""

    parts = str(value).split()

    if len(parts) != 3:
        raise ValueError(f"Некорректная дата: {value}")

    day, month, year = parts

    return pd.Timestamp(
        year=int(year),
        month=MONTHS[month],
        day=int(day)
    )


def load_data():
    """Загружает CSV и проверяет исходную структуру."""

    if not DATA_PATH.exists():
        raise FileNotFoundError(f"CSV не найден: {DATA_PATH}")

    df = pd.read_csv(DATA_PATH, encoding="utf-8-sig")

    required_columns = [
        "date",
        "home_team",
        "away_team",
        "home_points",
        "away_points"
    ]

    # Проверяем наличие всех необходимых столбцов
    missing_columns = set(required_columns) - set(df.columns)

    if missing_columns:
        raise ValueError(
            f"Отсутствуют столбцы: {sorted(missing_columns)}"
        )

    # Проверяем пропущенные значения
    if df[required_columns].isna().any().any():
        raise ValueError("В исходных данных есть пропуски!")

    # Преобразуем строки дат в даты Python
    df["date"] = df["date"].apply(parse_date)

    # Приводим очки к целочисленному типу
    for column in ("home_points", "away_points"):
        values = pd.to_numeric(df[column], errors="raise")

        if not np.isfinite(values.to_numpy(dtype=float)).all():
            raise ValueError(f"Недопустимые значения в {column}")

        if (values % 1 != 0).any():
            raise ValueError(f"Дробное количество очков: {column}")

        df[column] = values.astype(int)

    return df


# 4. ПРОВЕРКА КАЧЕСТВА ДАННЫХ

def validate_data(df):
    """Проверяет структуру и базовую корректность датасета."""

    teams = set(df["home_team"]) | set(df["away_team"])

    games_per_team = pd.concat(
        [df["home_team"], df["away_team"]]
    ).value_counts()

    checks = {
        "Ровно 380 матчей": len(df) == 380,
        "Ровно 20 команд": len(teams) == 20,
        "По 38 матчей на команду": (
            len(games_per_team) == 20
            and games_per_team.eq(38).all()
        ),
        "Нет полных дубликатов": not df.duplicated().any(),
        "Нет повторяющихся пар хозяева-гости": (
            not df.duplicated(
                subset=["home_team", "away_team"]
            ).any()
        ),
        "Нет матчей команды с собой": (
            df["home_team"] != df["away_team"]
        ).all(),
        "Положительные очки": (
            (df["home_points"] > 0)
            & (df["away_points"] > 0)
        ).all(),
        "Нет ничьих": (
            df["home_points"] != df["away_points"]
        ).all()
    }

    print("\n    ПРОВЕРКА ДАННЫХ    ")

    for name, passed in checks.items():
        print(f"{name}: {'OK' if passed else 'ОШИБКА'}")

    if not all(checks.values()):
        raise ValueError("Датасет не прошёл проверку!")

    return df


# 5. СОЗДАНИЕ НОВЫХ ПРИЗНАКОВ

def create_features(df):
    """Добавляет признаки для статистического анализа."""

    df = df.copy()

    # Общее количество очков в матче
    df["total_points"] = (
        df["home_points"] + df["away_points"]
    )

    # Разница очков: хозяева минус гости
    df["point_diff"] = (
        df["home_points"] - df["away_points"]
    )

    # Победа хозяев: True или False
    df["home_win"] = (
        df["home_points"] > df["away_points"]
    )

    # Месяц матча для временного анализа
    df["month"] = df["date"].dt.strftime("%Y-%m")

    return df


# 6. ОПИСАТЕЛЬНАЯ СТАТИСТИКА

def descriptive_statistics(df):
    """Вычисляет основные характеристики распределения."""

    points = df["total_points"]

    mean = float(points.mean())
    median = float(points.median())

    # Выборочная дисперсия: знаменатель n - 1
    variance = float(points.var(ddof=1))
    std = float(points.std(ddof=1))

    q1 = float(points.quantile(0.25))
    q3 = float(points.quantile(0.75))

    iqr = q3 - q1

    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr

    # Находим матчи с нетипичным общим количеством очков
    outliers = df[
        (points < lower_bound)
        | (points > upper_bound)
    ].copy()

    results = {
        "count": len(df),
        "mean": mean,
        "median": median,
        "variance": variance,
        "std": std,
        "min": int(points.min()),
        "max": int(points.max()),
        "q1": q1,
        "q3": q3,
        "iqr": iqr,
        "lower_bound": lower_bound,
        "upper_bound": upper_bound,
        "outliers_count": len(outliers)
    }

    print("\n    ОПИСАТЕЛЬНАЯ СТАТИСТИКА    ")

    for name, value in results.items():
        print(f"{name}: {value}")

    return results, outliers


# 7. ПРОВЕРКА ГИПОТЕЗЫ О ДОМАШНЕМ ПРЕИМУЩЕСТВЕ

def home_advantage_analysis(df):
    """
    Проверяет гипотезу H0: вероятность победы хозяев = 0.5.
    Используется двусторонний биномиальный тест.
    """

    n = len(df)
    home_wins = int(df["home_win"].sum())
    away_wins = n - home_wins

    home_win_rate = home_wins / n

    # Биномиальный тест
    test = binomtest(
        k=home_wins,
        n=n,
        p=0.5,
        alternative="two-sided"
    )

    # Точный 95%-й доверительный интервал
    interval = test.proportion_ci(
        confidence_level=0.95,
        method="exact"
    )

    average_margin = float(df["point_diff"].mean())

    results = {
        "home_wins": home_wins,
        "away_wins": away_wins,
        "home_win_rate": home_win_rate,
        "average_margin": average_margin,
        "p_value": float(test.pvalue),
        "ci_low": float(interval.low),
        "ci_high": float(interval.high)
    }

    print("\n    ДОМАШНЕЕ ПРЕИМУЩЕСТВО    ")
    print("Побед хозяев:", home_wins)
    print("Побед гостей:", away_wins)
    print(f"Доля побед хозяев: {home_win_rate:.2%}")
    print(f"Средняя разница очков: {average_margin:.2f}")
    print(f"p-value: {test.pvalue:.6f}")
    print(
        f"95% доверительный интервал: "
        f"[{interval.low:.3f}; {interval.high:.3f}]"
    )

    return results


# 8. КОРРЕЛЯЦИОННЫЙ АНАЛИЗ

def correlation_analysis(df):
    """Вычисляет корреляцию Пирсона между очками команд."""

    correlation, p_value = pearsonr(
        df["home_points"],
        df["away_points"]
    )

    results = {
        "r": float(correlation),
        "p_value": float(p_value)
    }

    print("\n    КОРРЕЛЯЦИЯ    ")
    print(f"Коэффициент Пирсона: {correlation:.4f}")
    print(f"p-value: {p_value:.6f}")

    return results


# 9. АНАЛИЗ КОМАНД

def build_team_statistics(df):
    """Считает победы, поражения и средние очки команд."""

    teams = sorted(
        set(df["home_team"]) | set(df["away_team"])
    )

    records = []

    for team in teams:
        home = df[df["home_team"] == team]
        away = df[df["away_team"] == team]

        games = len(home) + len(away)

        wins_home = int(home["home_win"].sum())
        wins_away = int((~away["home_win"]).sum())

        wins = wins_home + wins_away
        losses = games - wins

        points_scored = (
            home["home_points"].sum()
            + away["away_points"].sum()
        )

        points_allowed = (
            home["away_points"].sum()
            + away["home_points"].sum()
        )

        records.append({
            "team": team,
            "games": games,
            "wins": wins,
            "losses": losses,
            "win_rate": wins / games,
            "avg_scored": points_scored / games,
            "avg_allowed": points_allowed / games
        })

    team_stats = pd.DataFrame(records)

    # Сортировка по победам, затем по разнице средних очков
    team_stats["avg_diff"] = (
        team_stats["avg_scored"] - team_stats["avg_allowed"]
    )

    team_stats = team_stats.sort_values(
        ["wins", "avg_diff"],
        ascending=[False, False]
    ).reset_index(drop=True)

    print("\n    СТАТИСТИКА КОМАНД ")
    print(team_stats.to_string(index=False))

    return team_stats


# 10. АНАЛИЗ ПО МЕСЯЦАМ

def monthly_analysis(df):
    """Считает средние очки и количество матчей по месяцам."""

    monthly = (
        df.groupby("month", as_index=False)
        .agg(
            avg_total_points=("total_points", "mean"),
            matches=("total_points", "size")
        )
        .sort_values("month")
    )

    print("\n    ПОМЕСЯЧНЫЙ АНАЛИЗ    ")
    print(monthly.to_string(index=False))

    return monthly


# 11. СОХРАНЕНИЕ ГРАФИКОВ

def save_chart(filename):
    """Сохраняет текущий график в PNG."""

    plt.tight_layout()
    plt.savefig(
        CHARTS_PATH / filename,
        dpi=160,
        bbox_inches="tight"
    )
    plt.close()


def build_charts(df, team_stats, monthly):
    """Создаёт графики для Markdown-доклада."""

    sns.set_theme(style="whitegrid")

    # ГРАФИК 1. Распределение общего количества очков
    plt.figure(figsize=(9, 5))

    sns.histplot(
        data=df,
        x="total_points",
        bins=25,
        kde=True
    )

    plt.axvline(
        df["total_points"].mean(),
        linestyle="--",
        label="Mean"
    )

    plt.title("Distribution of total points per match")
    plt.xlabel("Total points")
    plt.ylabel("Number of matches")
    plt.legend()

    save_chart("01_points_distribution.png")


    # ГРАФИК 2. Сравнение очков хозяев и гостей
    score_data = df[
        ["home_points", "away_points"]
    ].rename(
        columns={
            "home_points": "Home",
            "away_points": "Away"
        }
    ).melt(
        var_name="Side",
        value_name="Points"
    )

    plt.figure(figsize=(8, 5))

    sns.boxplot(
        data=score_data,
        x="Side",
        y="Points"
    )

    plt.title("Home vs Away: Points")
    plt.xlabel("")
    plt.ylabel("Points scored")

    save_chart("02_home_away_boxplot.png")


    # ГРАФИК 3. Распределение разницы очков
    plt.figure(figsize=(9, 5))

    sns.histplot(
        data=df,
        x="point_diff",
        bins=25
    )

    plt.axvline(0, linestyle="--")

    plt.title("Point difference: Home minus Away")
    plt.xlabel("Point difference")
    plt.ylabel("Number of matches")

    save_chart("03_point_difference.png")


    # ГРАФИК 4. Корреляция набранных очков
    plt.figure(figsize=(8, 6))

    sns.regplot(
        data=df,
        x="home_points",
        y="away_points",
        scatter_kws={"alpha": 0.5}
    )

    plt.title("Home and Away Points: Correlation")
    plt.xlabel("Home points")
    plt.ylabel("Away points")

    save_chart("04_correlation.png")


    # ГРАФИК 5. Средняя результативность по месяцам
    plt.figure(figsize=(10, 5))

    plt.plot(
        monthly["month"],
        monthly["avg_total_points"],
        marker="o"
    )

    plt.title("Average total points by month")
    plt.xlabel("Month")
    plt.ylabel("Average total points")
    plt.xticks(rotation=45)

    save_chart("05_monthly_trend.png")


    # ГРАФИК 6. Количество побед каждой команды
    sorted_teams = team_stats.sort_values("wins")

    plt.figure(figsize=(10, 9))

    sns.barplot(
        data=sorted_teams,
        x="wins",
        y="team"
    )

    plt.title("Wins by team")
    plt.xlabel("Wins")
    plt.ylabel("Team")
    plt.xlim(0, 38)

    save_chart("06_team_wins.png")

    print("\nГрафики сохранены в:", CHARTS_PATH)


# 12. ФОРМИРОВАНИЕ РЕЗУЛЬТАТОВ ДЛЯ MARKDOWN

def update_markdown(df, desc, home, corr, teams, monthly):
    """
    Подставляет реальные результаты между специальными
    маркерами в docs/02_eda.md.
    Остальной текст доклада остаётся без изменений.
    """

    if not DOC_PATH.exists():
        print("Markdown ещё не создан:", DOC_PATH)
        return

    content = DOC_PATH.read_text(encoding="utf-8")

    start = content.find(START_MARKER)
    end = content.find(END_MARKER)

    if start == -1 or end <= start:
        print("Не найдены маркеры результатов в Markdown.")
        return

    if home["p_value"] < 0.05:
        hypothesis_conclusion = (
            "Нулевая гипотеза отвергается на уровне 5% "
            "в рамках выбранной биномиальной модели."
        )
    else:
        hypothesis_conclusion = (
            "Недостаточно оснований отвергнуть нулевую "
            "гипотезу на уровне значимости 5%."
        )

    lines = [
        "### Основные показатели",
        "",
        "| Показатель | Результат |",
        "|---|---:|",
        f"| Количество матчей | {len(df)} |",
        f"| Среднее количество очков | {desc['mean']:.2f} |",
        f"| Медиана | {desc['median']:.2f} |",
        f"| Выборочная дисперсия | {desc['variance']:.2f} |",
        f"| Стандартное отклонение | {desc['std']:.2f} |",
        f"| Минимум | {desc['min']} |",
        f"| Максимум | {desc['max']} |",
        f"| Q1 | {desc['q1']:.2f} |",
        f"| Q3 | {desc['q3']:.2f} |",
        f"| IQR | {desc['iqr']:.2f} |",
        f"| Выбросы по правилу 1.5 IQR | {desc['outliers_count']} |",
        "",
        "### Домашнее преимущество",
        "",
        f"- Побед хозяев: **{home['home_wins']}**.",
        f"- Побед гостей: **{home['away_wins']}**.",
        f"- Доля побед хозяев: **{home['home_win_rate']:.2%}**.",
        f"- Средняя разница очков: **{home['average_margin']:.2f}**.",
        f"- p-value биномиального теста: **{home['p_value']:.6f}**.",
        (
            "- Точный 95% доверительный интервал: "
            f"**[{home['ci_low']:.3f}; {home['ci_high']:.3f}]**."
        ),
        "",
        f"**Вывод:** {hypothesis_conclusion}",
        "",
        "Проверка показывает статистическую связь с местом "
        "проведения матча в рамках выбранной модели, "
        "но сама по себе не доказывает причинный эффект.",
        "",
        "### Корреляция очков",
        "",
        f"- Коэффициент Пирсона: **{corr['r']:.4f}**.",
        f"- p-value: **{corr['p_value']:.6f}**.",
        "",
        "Корреляция описывает линейную связь двух числовых "
        "показателей и не доказывает причинность.",
        "",
        "### Пять команд с наибольшим числом побед",
        "",
        "| Команда | Победы | Поражения | Доля побед |",
        "|---|---:|---:|---:|"
    ]

    for _, team in teams.head(5).iterrows():
        lines.append(
            f"| {team['team']} | {int(team['wins'])} | "
            f"{int(team['losses'])} | {team['win_rate']:.1%} |"
        )

    lines.extend([
        "",
        "Таблица составлена по количеству побед в CSV, "
        "а не по официальным правилам ранжирования Евролиги.",
        "",
        "### Помесячные результаты",
        "",
        "| Месяц | Матчи | Средние суммарные очки |",
        "|---|---:|---:|"
    ])

    for _, month in monthly.iterrows():
        lines.append(
            f"| {month['month']} | {int(month['matches'])} | "
            f"{month['avg_total_points']:.2f} |"
        )

    lines.extend([
        "",
        "### Пять самых результативных матчей",
        "",
        "| Дата | Хозяева | Гости | Счёт | Всего очков |",
        "|---|---|---|---|---:|"
    ])

    for _, match in df.nlargest(5, "total_points").iterrows():
        date = match["date"].strftime("%d.%m.%Y")

        lines.append(
            f"| {date} | {match['home_team']} | "
            f"{match['away_team']} | "
            f"{match['home_points']}:{match['away_points']} | "
            f"{match['total_points']} |"
        )

    generated = "\n".join(lines)

    # Заменяем только автоматический раздел
    updated = (
        content[:start + len(START_MARKER)]
        + "\n\n"
        + generated
        + "\n\n"
        + content[end:]
    )

    DOC_PATH.write_text(updated, encoding="utf-8")

    print("Markdown обновлён:", DOC_PATH)


# 13. ПРИМЕР ПОДГОТОВКИ КОНТЕКСТА ДЛЯ LLM

def demonstrate_llm_context(df):
    """
    Показывает, как структурированные данные могут
    быть представлены в виде текстового контекста.
    Настоящая LLM здесь не запускается.
    """

    match = df.iloc[0]

    context = (
        f"Дата: {match['date'].strftime('%d.%m.%Y')}\n"
        f"Хозяева: {match['home_team']}\n"
        f"Гости: {match['away_team']}\n"
        f"Счёт: {match['home_points']}:"
        f"{match['away_points']}"
    )

    print("\n    ПРИМЕР КОНТЕКСТА ДЛЯ LLM    ")
    print(context)


# 14. ЗАПУСК АНАЛИЗА

def main():
    print("    EUROLEAGUE 2025/26: EDA    ")

    df = load_data()
    df = validate_data(df)
    df = create_features(df)

    desc, outliers = descriptive_statistics(df)
    home = home_advantage_analysis(df)
    corr = correlation_analysis(df)

    teams = build_team_statistics(df)
    monthly = monthly_analysis(df)

    build_charts(df, teams, monthly)

    update_markdown(df, desc, home, corr, teams, monthly)

    demonstrate_llm_context(df)

    print("\n    АНАЛИЗ ЗАВЕРШЁН    ")
    print("Всего матчей:", len(df))
    print("Выбросов обнаружено:", len(outliers))
    print("Графиков создано: 6")
    print("Путь к графикам:", CHARTS_PATH)


if __name__ == "__main__":
    main()
