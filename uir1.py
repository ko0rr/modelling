# вариант 7 + 8 = 15

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from math import ceil, sqrt, factorial


def read_data():
    df = pd.read_excel("!_УИР1_Варианты_с.xlsx", sheet_name="1-50", usecols="P")
    lst = df.iloc[:, 0].dropna().tolist()
    nums = [round(x, 3) for x in lst]
    return nums


def statistics(nums):
    n = len(nums)

    mean = np.mean(nums)                 # оценка математического ожидания m~
    disp = np.var(nums, ddof = 1)        # несмещенная оценка дисперсии D~
    std = np.std(nums, ddof = 1)         # оценка СКО sigma~
    cv = std / mean                      # коэффициент вариации nu

    confidence_intervals = {}

    t_values = {
        0.90: 1.643,
        0.95: 1.960,
        0.99: 2.576
    }

    sigma_mean = np.sqrt(disp / n)       # sigma~_m

    for p in [0.90, 0.95, 0.99]:
        t_p = t_values[p]
        error = t_p * sigma_mean         # epsilon_p

        confidence_intervals[p] = {
            "error": error,
            "lower": mean - error,
            "upper": mean + error
        }

    return {
        "n": n,
        "mean": mean,
        "disp": disp,
        "std": std,
        "cv": cv,
        "confidence_intervals": confidence_intervals
    }


def distribution_approximation(nums):
    mean = np.mean(nums)
    disp = np.var(nums, ddof = 1)
    std = np.sqrt(disp)
    v = std / mean

    print("\n" + "=" * 60)
    print("АППРОКСИМАЦИЯ ЗАКОНА РАСПРЕДЕЛЕНИЯ")
    print("=" * 60)

    print(f"m~ = {mean:.6f}")
    print(f"D~ = {disp:.6f}")
    print(f"sigma~ = {std:.6f}")
    print(f"nu = {v:.6f}")

    if v < 1 / np.sqrt(3):
        a = mean - np.sqrt(3) * std
        b = mean + np.sqrt(3) * std

        print("равномерное распределение")
        print(f"a = {a:.6f}")
        print(f"b = {b:.6f}")

        return {
            "distribution": "равномерное",
            "a": a,
            "b": b
        }

    elif np.isclose(v, 1.0, atol = 1e-3):
        alpha = 1 / mean

        print("экспоненциальное распределение")
        print(f"alpha = {alpha:.6f}")

        return {
            "distribution": "экспоненциальное",
            "alpha": alpha
        }

    elif v < 1:
        k = round(1 / (v ** 2))
        mean_tau = mean / k
        alpha = 1 / mean_tau

        print(f"распределение Эрланга {k}-го порядка")
        print(f"k = {k}")
        print(f"M[tau] = {mean_tau:.6f}")
        print(f"alpha = {alpha:.6f}")

        return {
            "distribution": "эрланг",
            "k": k,
            "alpha": alpha
        }

    else:
        q_max = 2 / (1 + v ** 2)

        if 0.3 <= q_max:
            q = 0.3
        else:
            q = q_max / 2

        t1 = (
            1
            + (1 - q) / (2 * q) * (v ** 2 - 1)
        ) * mean

        t2 = (
            1
            - q / (2 * (1 - q)) * (v ** 2 - 1)
        ) * mean

        print("гиперэкспоненциальное распределение")
        print(f"q <= {q_max:.6f}")
        print(f"q = {q:.6f}")
        print(f"t1 = {t1:.6f}")
        print(f"t2 = {t2:.6f}")

        return {
            "distribution": "гиперэкспоненциальное",
            "q": q,
            "t1": t1,
            "t2": t2
        }


def generate_nums(approximation, n = 300):
    generated = []

    if approximation["distribution"] == "равномерное":
        a = approximation["a"]
        b = approximation["b"]

        for i in range(n):
            x = np.random.uniform(a, b)
            generated.append(x)

    elif approximation["distribution"] == "экспоненциальное":
        alpha = approximation["alpha"]

        for i in range(n):
            x = np.random.exponential(1 / alpha)
            generated.append(x)

    elif approximation["distribution"] == "эрланг":
        k = approximation["k"]
        alpha = approximation["alpha"]

        for i in range(n):
            x = 0

            for j in range(k):
                x += np.random.exponential(1 / alpha)

            generated.append(x)

    else:
        q = approximation["q"]
        t1 = approximation["t1"]
        t2 = approximation["t2"]

        for i in range(n):
            r = np.random.random()

            if r < q:
                x = np.random.exponential(t1)
            else:
                x = np.random.exponential(t2)

            generated.append(x)

    return generated


def relative_deviation(value, reference):
    return (value - reference) / reference * 100


def find_period(nums):
    nums = np.array(nums)
    n = len(nums)

    for period in range(1, n // 2 + 1):
        if np.allclose(
            nums[:-period],
            nums[period:],
            rtol = 1e-3,
            atol = 1e-3
        ):
            return period

    return None


def plot_data(nums, title):
    nums = np.array(nums)

    plt.figure(figsize = (12, 5))

    plt.plot(
        range(1, len(nums) + 1),
        nums,
        marker = 'o',
        markersize = 3
    )

    plt.title(title)
    plt.xlabel('Номер значения')
    plt.ylabel('Значение')
    plt.grid()
    plt.show()

    diffs = np.diff(nums)

    if np.all(diffs > 0):
        print("строго возрастающая последовательность")
    elif np.all(diffs >= 0):
        print("неубывающая последовательность")
    elif np.all(diffs < 0):
        print("строго убывающая последовательность")
    elif np.all(diffs <= 0):
        print("невозрастающая последовательность")
    else:
        print("немонотонная последовательность")

    period = find_period(nums)

    if period is not None:
        print("последовательность периодическая")
        print(f"длина периода: {period}")
    else:
        print("периодичность не обнаружена")


def plot_histogram(nums, title):
    bins = ceil(sqrt(len(nums)))
    counts, bin_edges = np.histogram(nums, bins = bins)

    print("\nинтервалы гистограммы:")

    for i in range(len(bin_edges) - 1):
        print(
            f"{i + 1}: "
            f"[{bin_edges[i]:.3f}; {bin_edges[i + 1]:.3f}) "
            f"частота = {counts[i]}"
        )

    plt.figure(figsize = (12, 6))

    plt.hist(
        nums,
        bins = bin_edges,
        edgecolor = 'black',
        alpha = 0.7
    )

    bin_labels = [
        f"{bin_edges[i]:.1f}-{bin_edges[i + 1]:.1f}"
        for i in range(len(bin_edges) - 1)
    ]

    plt.xticks(
        (bin_edges[:-1] + bin_edges[1:]) / 2,
        bin_labels,
        rotation = 45,
        ha = 'right'
    )

    plt.title(title, fontsize = 15)
    plt.xlabel('Интервалы значений', fontsize = 12)
    plt.ylabel('Частота', fontsize = 12)
    plt.grid(True)

    plt.tight_layout()
    plt.show()


def plot_approximation_density(nums, approximation):
    x_max = max(nums) * 1.05
    x = np.linspace(0, x_max, 1000)

    if approximation["distribution"] == "равномерное":
        a = approximation["a"]
        b = approximation["b"]

        density = np.zeros_like(x)
        mask = (x >= a) & (x <= b)
        density[mask] = 1 / (b - a)

    elif approximation["distribution"] == "экспоненциальное":
        alpha = approximation["alpha"]

        density = alpha * np.exp(-alpha * x)

    elif approximation["distribution"] == "эрланг":
        k = approximation["k"]
        alpha = approximation["alpha"]

        density = (
            alpha ** k
            * x ** (k - 1)
            * np.exp(-alpha * x)
            / factorial(k - 1)
        )

    else:
        q = approximation["q"]
        t1 = approximation["t1"]
        t2 = approximation["t2"]

        alpha1 = 1 / t1
        alpha2 = 1 / t2

        density = (
            q * alpha1 * np.exp(-alpha1 * x)
            + (1 - q) * alpha2 * np.exp(-alpha2 * x)
        )

    bins = ceil(sqrt(len(nums)))

    plt.figure(figsize = (12, 6))

    plt.hist(
        nums,
        bins = bins,
        density = True,
        alpha = 0.6,
        edgecolor = 'black',
        label = 'заданная ЧП'
    )

    plt.plot(
        x,
        density,
        linewidth = 2,
        label = approximation["distribution"]
    )

    plt.title(
        'Гистограмма исходной ЧП и плотность аппроксимирующего закона'
    )
    plt.xlabel('Значение')
    plt.ylabel('Плотность')
    plt.grid()
    plt.legend()
    plt.show()


def autocorrelation(nums, lag):
    dd = pd.Series(nums)
    acf = [dd.autocorr(lag = k) for k in range(1, lag + 1)]
    return np.array(acf)


def autocorrelation_border(nums, lag = 10):
    alpha = 0.05
    z = norm.ppf(1 - alpha / (2 * lag))
    return z / np.sqrt(len(nums))


def plot_autocorrelation(nums, title, lag = 10):
    acf = autocorrelation(nums, lag)
    border = autocorrelation_border(nums, lag)

    print(acf)
    print(f"совместные 95% границы автокорреляции: ±{border:.3f}")

    plt.plot(
        range(1, lag + 1),
        acf,
        marker = 'o'
    )

    plt.axhline(
        border,
        linestyle = '--',
        label = 'совместная 95% граница'
    )

    plt.axhline(
        -border,
        linestyle = '--'
    )

    plt.axhline(
        0,
        linewidth = 0.8
    )

    plt.title(title)
    plt.xlabel('Сдвиг')
    plt.ylabel('Коэффициент автокорреляции')
    plt.xticks(range(1, lag + 1))
    plt.grid()
    plt.legend()
    plt.show()


def print_original_statistics(nums):
    sizes = [10, 20, 50, 100, 200, 300]
    results = {}

    for size in sizes:
        sample = nums[:size]
        results[size] = statistics(sample)

    reference = results[300]

    for size in sizes:
        result = results[size]

        result["mean_dev"] = relative_deviation(
            result["mean"],
            reference["mean"]
        )

        result["disp_dev"] = relative_deviation(
            result["disp"],
            reference["disp"]
        )

        result["std_dev"] = relative_deviation(
            result["std"],
            reference["std"]
        )

        result["cv_dev"] = relative_deviation(
            result["cv"],
            reference["cv"]
        )

        result["ci_dev"] = {}

        for p in [0.90, 0.95, 0.99]:
            result["ci_dev"][p] = relative_deviation(
                result["confidence_intervals"][p]["error"],
                reference["confidence_intervals"][p]["error"]
            )

    print("\n" + "=" * 60)
    print("ЗАДАННАЯ ПОСЛЕДОВАТЕЛЬНОСТЬ")
    print("=" * 60)

    for size in sizes:
        result = results[size]

        print("=" * 50)
        print(f"количество значений: {size}")
        print(f"матожидание: {result['mean']:.3f}")
        print(f"дисперсия: {result['disp']:.3f}")
        print(f"ско: {result['std']:.3f}")
        print(f"коэффициент вариации: {result['cv']:.5f}")

        print("\nдоверительные интервалы:")

        for p, interval in result["confidence_intervals"].items():
            print(
                f"p = {p}: "
                f"±{interval['error']:.3f} "
                f"[{interval['lower']:.3f}; {interval['upper']:.3f}]"
            )

        print("\nотклонения от выборки n = 300:")
        print(f"матожидание: {result['mean_dev']:.3f}%")
        print(f"дисперсия: {result['disp_dev']:.3f}%")
        print(f"ско: {result['std_dev']:.3f}%")
        print(f"коэффициент вариации: {result['cv_dev']:.3f}%")

        for p in [0.90, 0.95, 0.99]:
            print(
                f"дов. инт. {p}: "
                f"{result['ci_dev'][p]:.3f}%"
            )

    return results


def print_generated_statistics(nums, generated, original_results):
    sizes = [10, 20, 50, 100, 200, 300]
    results = {}

    for size in sizes:
        sample = generated[:size]
        results[size] = statistics(sample)

    print("\n" + "=" * 60)
    print("СГЕНЕРИРОВАННАЯ ПОСЛЕДОВАТЕЛЬНОСТЬ")
    print("=" * 60)

    for size in sizes:
        result = results[size]
        original = original_results[size]

        mean_dev = relative_deviation(
            result["mean"],
            original["mean"]
        )

        disp_dev = relative_deviation(
            result["disp"],
            original["disp"]
        )

        std_dev = relative_deviation(
            result["std"],
            original["std"]
        )

        cv_dev = relative_deviation(
            result["cv"],
            original["cv"]
        )

        print("=" * 50)
        print(f"количество значений: {size}")
        print(f"матожидание: {result['mean']:.3f}")
        print(f"дисперсия: {result['disp']:.3f}")
        print(f"ско: {result['std']:.3f}")
        print(f"коэффициент вариации: {result['cv']:.5f}")

        print("\nдоверительные интервалы:")

        for p, interval in result["confidence_intervals"].items():
            print(
                f"p = {p}: "
                f"±{interval['error']:.3f} "
                f"[{interval['lower']:.3f}; {interval['upper']:.3f}]"
            )

        print("\nотклонения от заданной последовательности:")
        print(f"матожидание: {mean_dev:.3f}%")
        print(f"дисперсия: {disp_dev:.3f}%")
        print(f"ско: {std_dev:.3f}%")
        print(f"коэффициент вариации: {cv_dev:.3f}%")

        for p in [0.90, 0.95, 0.99]:
            ci_dev = relative_deviation(
                result["confidence_intervals"][p]["error"],
                original["confidence_intervals"][p]["error"]
            )

            print(
                f"дов. инт. {p}: "
                f"{ci_dev:.3f}%"
            )

    return results


def print_autocorrelation_table(nums, generated, lag = 10):
    acf_nums = autocorrelation(nums, lag)
    acf_generated = autocorrelation(generated, lag)

    print("\n" + "=" * 60)
    print("КОЭФФИЦИЕНТЫ АВТОКОРРЕЛЯЦИИ")
    print("=" * 60)

    print("сдвиг:", list(range(1, lag + 1)))
    print("заданная ЧП:", np.round(acf_nums, 6))
    print("сгенерированная ЧП:", np.round(acf_generated, 6))

    deviations = []

    for i in range(lag):
        deviations.append(
            relative_deviation(
                acf_generated[i],
                acf_nums[i]
            )
        )

    print("отклонение, %:", np.round(deviations, 3))


def compare_plot(nums, generated):
    plt.figure(figsize = (12, 5))

    plt.plot(
        range(1, len(nums) + 1),
        nums,
        label = 'заданная ЧП'
    )

    plt.plot(
        range(1, len(generated) + 1),
        generated,
        label = 'сгенерированная ЧП'
    )

    plt.title('Сравнение числовых последовательностей')
    plt.xlabel('Номер значения')
    plt.ylabel('Значение')
    plt.grid()
    plt.legend()
    plt.show()


def compare_histogram(nums, generated):
    bins = ceil(sqrt(300))
    all_nums = np.concatenate((nums, generated))

    bin_edges = np.linspace(
        np.min(all_nums),
        np.max(all_nums),
        bins + 1
    )

    plt.figure(figsize = (12, 6))

    plt.hist(
        nums,
        bins = bin_edges,
        alpha = 0.6,
        edgecolor = 'black',
        label = 'заданная ЧП'
    )

    plt.hist(
        generated,
        bins = bin_edges,
        alpha = 0.6,
        edgecolor = 'black',
        label = 'сгенерированная ЧП'
    )

    plt.title('Сравнение гистограмм распределения частот')
    plt.xlabel('Интервалы значений')
    plt.ylabel('Частота')
    plt.grid()
    plt.legend()
    plt.show()


def compare_autocorrelation(nums, generated, lag = 10):
    acf_nums = autocorrelation(nums, lag)
    acf_generated = autocorrelation(generated, lag)

    border_nums = autocorrelation_border(nums, lag)
    border_generated = autocorrelation_border(generated, lag)

    plt.plot(
        range(1, lag + 1),
        acf_nums,
        marker = 'o',
        label = 'заданная ЧП'
    )

    plt.plot(
        range(1, lag + 1),
        acf_generated,
        marker = 'o',
        label = 'сгенерированная ЧП'
    )

    plt.axhline(
        border_nums,
        linestyle = '--',
        label = 'граница заданной ЧП'
    )

    plt.axhline(
        -border_nums,
        linestyle = '--'
    )

    plt.axhline(
        border_generated,
        linestyle = ':',
        label = 'граница сгенерированной ЧП'
    )

    plt.axhline(
        -border_generated,
        linestyle = ':'
    )

    plt.axhline(
        0,
        linewidth = 0.8
    )

    plt.title('Сравнение автокорреляции')
    plt.xlabel('Сдвиг')
    plt.ylabel('Коэффициент автокорреляции')
    plt.xticks(range(1, lag + 1))
    plt.grid()
    plt.legend()
    plt.show()


def correlation(nums, generated):
    corr = np.corrcoef(nums, generated)[0, 1]

    print("\nкоэффициент корреляции заданной и сгенерированной последовательностей:")
    print(f"{corr:.6f}")

    return corr


def main():
    nums = read_data()
    nums = nums[:300]

    original_results = print_original_statistics(nums)

    plot_data(
        nums,
        'График заданной числовой последовательности'
    )

    plot_histogram(
        nums,
        'Гистограмма распределения частот заданной последовательности'
    )

    plot_autocorrelation(
        nums,
        'Автокорреляционный анализ заданной последовательности'
    )

    approximation = distribution_approximation(nums)

    plot_approximation_density(
        nums,
        approximation
    )

    np.random.seed(15)

    generated = generate_nums(
        approximation,
        300
    )

    print("\nпервые 10 сгенерированных значений:")

    for i in range(10):
        print(f"{i + 1}: {generated[i]:.3f}")

    print_generated_statistics(
        nums,
        generated,
        original_results
    )

    plot_data(
        generated,
        'График сгенерированной числовой последовательности'
    )

    plot_histogram(
        generated,
        'Гистограмма распределения частот сгенерированной последовательности'
    )

    plot_autocorrelation(
        generated,
        'Автокорреляционный анализ сгенерированной последовательности'
    )

    print_autocorrelation_table(
        nums,
        generated
    )

    compare_plot(
        nums,
        generated
    )

    compare_histogram(
        nums,
        generated
    )

    compare_autocorrelation(
        nums,
        generated
    )

    correlation(
        nums,
        generated
    )


main()
