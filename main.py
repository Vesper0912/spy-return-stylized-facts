# -*- coding: utf-8 -*-
"""
SPY 日度收益率统计特征分析（Python 复现版）
================================================
复现并扩展 MATLAB 课程报告《SPY 日度收益率时间序列的统计特征分析》：
  1. 描述性统计 + Jarque-Bera 正态性检验
  2. 直方图与正态拟合对比（含对数纵轴尾部图）
  3. 收益率时序图（波动聚集）
  4. 均值显著性 t 检验
  5. 收益率 r_t 与 r_t^2 的 ACF + Ljung-Box 检验
  6. 【扩展】GARCH(1,1) 正态极大似然估计

数据：data/SPY_2021_2026.csv（Yahoo Finance 调整后收盘价，2021-10 ~ 2026-10）
运行：python main.py
输出：figures/ 下的图表 + 终端打印的统计结果
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from scipy import stats, optimize
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.tsa.stattools import acf

DATA_PATH = os.path.join(os.path.dirname(__file__), "data", "SPY_2021_2026.csv")
FIG_DIR = os.path.join(os.path.dirname(__file__), "figures")
os.makedirs(FIG_DIR, exist_ok=True)

# ------------------------------------------------- MATLAB 风格全局设置
C_BLUE   = "#0072BD"   # MATLAB 默认蓝
C_ORANGE = "#EDB120"   # MATLAB 置信线橙
C_RED    = "#D5000B"   # 正态拟合线红
C_HIST   = "#A9C7E8"   # MATLAB 直方图浅蓝

plt.rcParams.update({
    "font.sans-serif": ["Microsoft YaHei", "SimHei", "PingFang SC",
                        "Noto Sans CJK SC", "WenQuanYi Zen Hei", "DejaVu Sans"],
    "axes.unicode_minus": False,
    "font.size": 12,
    "axes.titlesize": 15,
    "axes.titleweight": "bold",
    "axes.labelsize": 13,
    "axes.labelweight": "bold",
    "axes.linewidth": 1.2,            # 加粗坐标框
    "axes.grid": True,
    "grid.color": "#D9D9D9",          # 浅灰网格
    "grid.linewidth": 0.8,
    "xtick.direction": "in",          # MATLAB 刻度朝内
    "ytick.direction": "in",
    "xtick.labelsize": 11,
    "ytick.labelsize": 11,
    "legend.fontsize": 11,
    "legend.framealpha": 1.0,
    "legend.edgecolor": "#666666",
    "figure.dpi": 110,
    "savefig.dpi": 300,               # 导出高清
    "savefig.bbox": "tight",
})


def _style_ax(ax):
    """让每个子图网格置于图形下方，避免压住数据。"""
    ax.set_axisbelow(True)
    for side in ("top", "right", "left", "bottom"):
        ax.spines[side].set_visible(True)


# 1. 数据读取与收益率构造
def load_returns(path=DATA_PATH):
    df = pd.read_csv(path, parse_dates=["Date"]).sort_values("Date")
    df = df.drop_duplicates(subset="Date").reset_index(drop=True)
    close = df["Close"].astype(float)
    df["r_simple"] = close.pct_change()
    df["r_log"] = np.log(close / close.shift(1))
    df = df.dropna(subset=["r_log"]).reset_index(drop=True)
    return df


# 2. 描述性统计 + JB 检验
def descriptive_stats(df):
    rows = []
    for col, name in [("r_simple", "简单收益率"), ("r_log", "对数收益率")]:
        x = df[col].values
        jb_stat, jb_p = stats.jarque_bera(x)
        rows.append({
            "口径": name,
            "均值": x.mean(),
            "标准差": x.std(ddof=1),
            "偏度": stats.skew(x),
            "原始峰度": stats.kurtosis(x, fisher=False),
            "超额峰度": stats.kurtosis(x, fisher=True),
            "JB统计量": jb_stat,
            "JB_p值": jb_p,
        })
    out = pd.DataFrame(rows).set_index("口径")
    ann_ret = df["r_log"].mean() * 252
    ann_vol = df["r_log"].std(ddof=1) * np.sqrt(252)
    print("=" * 60)
    print("表1  描述性统计（n = %d）" % len(df))
    print(out.round(6).to_string())
    print(f"\n年化收益 ≈ {ann_ret:.1%}，年化波动率 ≈ {ann_vol:.1%}（252 个交易日约定）")

    r = df["r_log"]
    z = (r - r.mean()) / r.std(ddof=1)
    for k in (3, 4):
        emp = (np.abs(z) > k).mean()
        theo = 2 * (1 - stats.norm.cdf(k))
        print(f"|z| > {k}σ：实际频率 {emp:.3%} vs 正态假设 {theo:.3%}（约 {emp/theo:.0f} 倍）")
    return out


# 3. 分布形态图
def plot_distribution(df):
    r = df["r_log"].values
    mu, sd = r.mean(), r.std(ddof=1)
    xs = np.linspace(r.min(), r.max(), 400)

    # 图1：直方图 vs 正态拟合
    fig, ax = plt.subplots(figsize=(10, 6.2))
    ax.hist(r, bins=100, density=True, color=C_HIST,
            edgecolor="white", linewidth=0.4, label="实际收益率直方图")
    ax.plot(xs, stats.norm.pdf(xs, mu, sd), color=C_RED, lw=2.5,
            label="正态分布拟合曲线")
    for s in (-2 * sd, 2 * sd):  # ±2σ 虚线
        ax.axvline(s, color="k", ls="--", lw=1.5)
    ax.set_title("SPY 日对数收益率分布与正态拟合对比")
    ax.set_xlabel("对数收益率")
    ax.set_ylabel("概率密度")
    ax.legend(loc="upper right")
    _style_ax(ax)
    fig.savefig(os.path.join(FIG_DIR, "fig1_hist_vs_normal.png"))
    plt.close(fig)

    # 图2：对数纵轴尾部对比
    fig, ax = plt.subplots(figsize=(10, 6.2))
    ax.hist(r, bins=120, density=True, color=C_HIST,
            edgecolor="white", linewidth=0.3, label="实际收益率直方图")
    ax.plot(xs, stats.norm.pdf(xs, mu, sd), color=C_RED, lw=2.5,
            label="正态分布拟合曲线")
    ax.set_yscale("log")
    ax.set_ylim(1e-4, None)
    ax.set_title("SPY 日对数收益率分布与正态拟合对比（对数纵轴）")
    ax.set_xlabel("对数收益率")
    ax.set_ylabel("概率密度（对数刻度）")
    ax.legend(loc="upper right")
    ax.grid(True, which="minor", axis="y", ls=":", alpha=0.5)  # 次刻度虚线网格
    _style_ax(ax)
    fig.savefig(os.path.join(FIG_DIR, "fig2_tail_logscale.png"))
    plt.close(fig)


#  4. 收益率时序图（波动聚集）
def plot_timeseries(df):
    fig, ax = plt.subplots(figsize=(11, 4.8))
    ax.plot(df["Date"], df["r_log"], lw=0.7, color=C_BLUE)
    ax.axhline(0, color="k", lw=0.6)
    ax.set_title("SPY 日对数收益率时序图（2021–2026）")
    ax.set_xlabel("日期")
    ax.set_ylabel("对数收益率")
    # 年份刻度：2022年 / 2023年 …
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y年"))
    _style_ax(ax)
    fig.savefig(os.path.join(FIG_DIR, "fig3_returns_ts.png"))
    plt.close(fig)


#  MATLAB 风格 ACF 火柴棍图
def _acf_stem(ax, series, nlags):
    acf_vals = acf(series, nlags=nlags, fft=True)
    lags = np.arange(nlags + 1)
    conf = 1.96 / np.sqrt(len(series))

    markerline, stemlines, _ = ax.stem(
        lags, acf_vals, linefmt=C_BLUE, markerfmt="o", basefmt=" ")
    plt.setp(markerline, markersize=6.5, color=C_BLUE,
             markerfacecolor=C_BLUE, markeredgecolor=C_BLUE)
    plt.setp(stemlines, color=C_BLUE, linewidth=1.1, alpha=0.85)

    ax.axhline(0, color="#666666", lw=0.8)
    ub, = ax.plot([-0.5, nlags + 0.5], [conf, conf], color=C_ORANGE, lw=1.6)
    ax.plot([-0.5, nlags + 0.5], [-conf, -conf], color=C_ORANGE, lw=1.6)
    ax.set_xlim(-0.5, nlags + 0.5)
    ax.set_xticks(np.arange(0, nlags + 1, 2))
    ax.set_xlabel("Lag")
    ax.set_ylabel("Sample Autocorrelation")
    ax.legend([markerline, ub], ["ACF", "Confidence Bound"], loc="upper right")
    _style_ax(ax)


#  5. 均值 t 检验 + ACF + Ljung-Box
def autocorr_analysis(df, nlags=20):
    r = df["r_log"].values
    t_stat, t_p = stats.ttest_1samp(r, 0.0)
    print("\n" + "=" * 60)
    print(f"均值 t 检验：t = {t_stat:.2f}，p = {t_p:.3f} → "
          + ("均值不显著异于零" if t_p > 0.05 else "均值显著异于零"))

    conf = 1.96 / np.sqrt(len(r))
    print(f"ACF 95% 置信界 ≈ ±{conf:.3f}")

    for series, name, fname, title in [
        (r, "r_t", "fig4_acf_r.png", r"对数收益率 $r_t$ 的自相关函数（ACF，滞后20阶）"),
        (r ** 2, "r_t^2", "fig5_acf_r2.png", r"对数收益率平方 $r_t^2$ 的自相关函数（ACF，滞后20阶）"),
    ]:
        fig, ax = plt.subplots(figsize=(10, 5.6))
        _acf_stem(ax, series, nlags)
        ax.set_title(title)
        fig.savefig(os.path.join(FIG_DIR, fname))
        plt.close(fig)

    lb_r = acorr_ljungbox(r, lags=[5, 10, 20], return_df=True)
    lb_r2 = acorr_ljungbox(r ** 2, lags=[5, 10, 20], return_df=True)
    table = pd.DataFrame({
        "滞后阶数": [5, 10, 20],
        "Q(r)": lb_r["lb_stat"].values,
        "p(r)": lb_r["lb_pvalue"].values,
        "Q(r²)": lb_r2["lb_stat"].values,
        "p(r²)": lb_r2["lb_pvalue"].values,
    }).set_index("滞后阶数")
    print("\n表2  Ljung-Box 检验（H0：前 m 阶自相关全为 0）")
    print(table.round(3).to_string())
    print("→ 收益率自身近似白噪声；收益率平方自相关显著且持久 → 波动率可预测")
    return table


#  6. GARCH(1,1) 手写 MLE
def fit_garch11(df):
    """GARCH(1,1) + 正态创新，scipy 手写极大似然（收益率×100 提高数值稳定性）"""
    r = df["r_log"].values * 100.0
    n = len(r)
    var0 = np.var(r)

    def neg_loglik(params):
        omega, alpha, beta = params
        if omega <= 0 or alpha < 0 or beta < 0 or alpha + beta >= 0.9999:
            return 1e10
        sig2 = np.empty(n)
        sig2[0] = var0
        for t in range(1, n):
            sig2[t] = omega + alpha * r[t - 1] ** 2 + beta * sig2[t - 1]
            if sig2[t] <= 0:
                return 1e10
        ll = -0.5 * np.sum(np.log(2 * np.pi) + np.log(sig2) + r ** 2 / sig2)
        return -ll

    res = optimize.minimize(neg_loglik, x0=[0.05, 0.1, 0.85],
                            method="Nelder-Mead",
                            options={"maxiter": 20000, "xatol": 1e-8, "fatol": 1e-8})
    omega, alpha, beta = res.x
    print("\n" + "=" * 60)
    print("GARCH(1,1) 估计（r_t 单位：%，正态创新）")
    print(f"  ω = {omega:.4f}, α = {alpha:.4f}, β = {beta:.4f}")
    print(f"  α + β = {alpha + beta:.4f}（越接近 1，波动持续性越强）")
    print(f"  无条件方差 = {omega / (1 - alpha - beta):.4f}（%/日）²，"
          f"对应年化波动 ≈ {np.sqrt(omega / (1 - alpha - beta) * 252):.1f}%")

    # 条件波动率时序图
    sig2 = np.empty(n)
    sig2[0] = var0
    for t in range(1, n):
        sig2[t] = omega + alpha * r[t - 1] ** 2 + beta * sig2[t - 1]
    cond_vol = np.sqrt(sig2) * np.sqrt(252) / 100
    fig, ax = plt.subplots(figsize=(11, 4.8))
    ax.plot(df["Date"].values, cond_vol, lw=0.9, color=C_RED)
    ax.set_title("GARCH(1,1) 条件波动率（年化）")
    ax.set_xlabel("日期")
    ax.set_ylabel("条件波动率（年化）")
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y年"))
    _style_ax(ax)
    fig.savefig(os.path.join(FIG_DIR, "fig6_garch_cond_vol.png"))
    plt.close(fig)
    return res.x


# main
if __name__ == "__main__":
    df = load_returns()
    print(f"样本：{df['Date'].iloc[0].date()} ~ {df['Date'].iloc[-1].date()}，n = {len(df)}")
    descriptive_stats(df)
    plot_distribution(df)
    plot_timeseries(df)
    autocorr_analysis(df)
    fit_garch11(df)
    print("\n全部图表已保存至 figures/ 目录。")

