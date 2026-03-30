#!/usr/bin/env python3
"""
Бэктестирование NASDAQ Intraday Strategy на исторических данных.
Адаптация Pine Script стратегии для дневных данных из CSV.
"""

import csv
import os
from dataclasses import dataclass, field
from typing import Optional

# ══════════════════════════════════════════════════════════════════════════════
# ПАРАМЕТРЫ СТРАТЕГИИ
# ══════════════════════════════════════════════════════════════════════════════

EMA_FAST = 9
EMA_SLOW = 21
EMA_TREND = 50  # Будет уменьшен из-за малого объёма данных
RSI_PERIOD = 14
RSI_OB = 70
RSI_OS = 30
ATR_PERIOD = 14
ATR_SL_MULT = 1.5
ATR_TP_MULT = 2.5
INITIAL_CAPITAL = 100_000.0
POSITION_SIZE_PCT = 0.10
COMMISSION_PCT = 0.02  # 0.02%


@dataclass
class Bar:
    date: str
    close: float
    open: float
    high: float
    low: float
    volume: float
    change_pct: float


@dataclass
class Trade:
    entry_date: str
    exit_date: str
    direction: str  # "LONG" / "SHORT"
    entry_price: float
    exit_price: float
    stop_loss: float
    take_profit: float
    exit_reason: str
    pnl: float
    pnl_pct: float
    position_size: float


@dataclass
class BacktestState:
    capital: float = INITIAL_CAPITAL
    position: Optional[str] = None  # "LONG" / "SHORT" / None
    entry_price: float = 0.0
    entry_date: str = ""
    stop_loss: float = 0.0
    take_profit: float = 0.0
    position_size: float = 0.0
    shares: float = 0.0
    trades: list = field(default_factory=list)
    equity_curve: list = field(default_factory=list)


# ══════════════════════════════════════════════════════════════════════════════
# ЗАГРУЗКА ДАННЫХ
# ══════════════════════════════════════════════════════════════════════════════

def parse_number(s: str) -> float:
    s = s.strip().strip('"').replace(',', '')
    if s.endswith('B'):
        return float(s[:-1]) * 1e9
    if s.endswith('M'):
        return float(s[:-1]) * 1e6
    if s.endswith('K'):
        return float(s[:-1]) * 1e3
    if s.endswith('%'):
        return float(s[:-1])
    return float(s)


def load_data(filepath: str) -> list[Bar]:
    bars = []
    with open(filepath, 'r', encoding='utf-8-sig') as f:
        reader = csv.DictReader(f)
        for row in reader:
            vals = list(row.values())
            bars.append(Bar(
                date=vals[0].strip('"'),
                close=parse_number(vals[1]),
                open=parse_number(vals[2]),
                high=parse_number(vals[3]),
                low=parse_number(vals[4]),
                volume=parse_number(vals[5]),
                change_pct=parse_number(vals[6]),
            ))
    # CSV идёт от новых к старым — разворачиваем
    bars.reverse()
    return bars


# ══════════════════════════════════════════════════════════════════════════════
# ИНДИКАТОРЫ
# ══════════════════════════════════════════════════════════════════════════════

def ema(values: list[float], period: int) -> list[Optional[float]]:
    result = [None] * len(values)
    if len(values) < period:
        # Недостаточно данных — используем SMA всех доступных
        if len(values) > 0:
            k = 2 / (min(period, len(values)) + 1)
            result[0] = values[0]
            for i in range(1, len(values)):
                result[i] = values[i] * k + result[i - 1] * (1 - k)
        return result
    # Первое значение — SMA за period баров
    sma = sum(values[:period]) / period
    result[period - 1] = sma
    k = 2 / (period + 1)
    for i in range(period, len(values)):
        result[i] = values[i] * k + result[i - 1] * (1 - k)
    return result


def rsi(closes: list[float], period: int) -> list[Optional[float]]:
    result = [None] * len(closes)
    if len(closes) < period + 1:
        return result
    gains = []
    losses = []
    for i in range(1, len(closes)):
        delta = closes[i] - closes[i - 1]
        gains.append(max(delta, 0))
        losses.append(max(-delta, 0))

    avg_gain = sum(gains[:period]) / period
    avg_loss = sum(losses[:period]) / period

    if avg_loss == 0:
        result[period] = 100.0
    else:
        rs = avg_gain / avg_loss
        result[period] = 100 - 100 / (1 + rs)

    for i in range(period, len(gains)):
        avg_gain = (avg_gain * (period - 1) + gains[i]) / period
        avg_loss = (avg_loss * (period - 1) + losses[i]) / period
        if avg_loss == 0:
            result[i + 1] = 100.0
        else:
            rs = avg_gain / avg_loss
            result[i + 1] = 100 - 100 / (1 + rs)
    return result


def atr(bars: list[Bar], period: int) -> list[Optional[float]]:
    trs = []
    for i, b in enumerate(bars):
        if i == 0:
            tr = b.high - b.low
        else:
            prev_close = bars[i - 1].close
            tr = max(b.high - b.low, abs(b.high - prev_close), abs(b.low - prev_close))
        trs.append(tr)
    return ema(trs, period)


def sma(values: list[float], period: int) -> list[Optional[float]]:
    result = [None] * len(values)
    for i in range(period - 1, len(values)):
        result[i] = sum(values[i - period + 1:i + 1]) / period
    return result


# ══════════════════════════════════════════════════════════════════════════════
# БЭКТЕСТ
# ══════════════════════════════════════════════════════════════════════════════

def run_backtest(bars: list[Bar]) -> BacktestState:
    n = len(bars)
    closes = [b.close for b in bars]
    volumes = [b.volume for b in bars]

    # Адаптируем периоды под количество данных
    ema_fast_period = min(EMA_FAST, max(2, n // 4))
    ema_slow_period = min(EMA_SLOW, max(3, n // 3))
    ema_trend_period = min(EMA_TREND, max(5, n // 2))
    rsi_period = min(RSI_PERIOD, max(2, n // 3))
    atr_period = min(ATR_PERIOD, max(2, n // 3))
    vol_ma_period = min(20, max(3, n // 3))

    print(f"{'═' * 70}")
    print(f"  ПАРАМЕТРЫ БЭКТЕСТА (адаптированы под {n} баров)")
    print(f"{'═' * 70}")
    print(f"  EMA Быстрая:  {ema_fast_period}")
    print(f"  EMA Медленная: {ema_slow_period}")
    print(f"  EMA Тренд:    {ema_trend_period}")
    print(f"  RSI Период:   {rsi_period}")
    print(f"  ATR Период:   {atr_period}")
    print(f"  Vol MA:       {vol_ma_period}")
    print(f"  Капитал:      ${INITIAL_CAPITAL:,.0f}")
    print(f"  Размер поз.:  {POSITION_SIZE_PCT * 100}%")
    print(f"  SL: {ATR_SL_MULT}x ATR  |  TP: {ATR_TP_MULT}x ATR")
    print(f"{'═' * 70}\n")

    # Рассчитываем индикаторы
    ema_f = ema(closes, ema_fast_period)
    ema_s = ema(closes, ema_slow_period)
    ema_t = ema(closes, ema_trend_period)
    rsi_vals = rsi(closes, rsi_period)
    atr_vals = atr(bars, atr_period)
    vol_ma = sma(volumes, vol_ma_period)

    state = BacktestState()

    for i in range(1, n):
        bar = bars[i]
        prev = bars[i - 1]

        # Пропускаем если индикаторы не готовы
        if any(v is None for v in [ema_f[i], ema_s[i], ema_f[i-1], ema_s[i-1], atr_vals[i]]):
            state.equity_curve.append((bar.date, state.capital))
            continue

        cur_atr = atr_vals[i]

        # ── Проверка стопов/тейков на текущем баре ──
        if state.position == "LONG":
            exit_reason = None
            exit_price = 0.0

            if bar.low <= state.stop_loss:
                exit_price = state.stop_loss
                exit_reason = "СТОП-ЛОСС"
            elif bar.high >= state.take_profit:
                exit_price = state.take_profit
                exit_reason = "ТЕЙК-ПРОФИТ"

            if exit_reason:
                pnl = (exit_price - state.entry_price) * state.shares
                commission = exit_price * state.shares * COMMISSION_PCT / 100
                pnl -= commission
                pnl_pct = pnl / state.position_size * 100
                state.capital += state.position_size + pnl
                state.trades.append(Trade(
                    entry_date=state.entry_date, exit_date=bar.date,
                    direction="LONG", entry_price=state.entry_price,
                    exit_price=exit_price, stop_loss=state.stop_loss,
                    take_profit=state.take_profit, exit_reason=exit_reason,
                    pnl=pnl, pnl_pct=pnl_pct, position_size=state.position_size
                ))
                state.position = None

        elif state.position == "SHORT":
            exit_reason = None
            exit_price = 0.0

            if bar.high >= state.stop_loss:
                exit_price = state.stop_loss
                exit_reason = "СТОП-ЛОСС"
            elif bar.low <= state.take_profit:
                exit_price = state.take_profit
                exit_reason = "ТЕЙК-ПРОФИТ"

            if exit_reason:
                pnl = (state.entry_price - exit_price) * state.shares
                commission = exit_price * state.shares * COMMISSION_PCT / 100
                pnl -= commission
                pnl_pct = pnl / state.position_size * 100
                state.capital += state.position_size + pnl
                state.trades.append(Trade(
                    entry_date=state.entry_date, exit_date=bar.date,
                    direction="SHORT", entry_price=state.entry_price,
                    exit_price=exit_price, stop_loss=state.stop_loss,
                    take_profit=state.take_profit, exit_reason=exit_reason,
                    pnl=pnl, pnl_pct=pnl_pct, position_size=state.position_size
                ))
                state.position = None

        # ── Сигналы входа ──
        if state.position is None and cur_atr > 0:
            # EMA пересечения
            ema_cross_up = ema_f[i-1] <= ema_s[i-1] and ema_f[i] > ema_s[i]
            ema_cross_down = ema_f[i-1] >= ema_s[i-1] and ema_f[i] < ema_s[i]

            # Тренд
            bull_trend = ema_t[i] is None or bar.close > ema_t[i]
            bear_trend = ema_t[i] is None or bar.close < ema_t[i]

            # RSI
            rsi_ok = rsi_vals[i] is None or (RSI_OS < rsi_vals[i] < RSI_OB)

            # Объём
            vol_ok = vol_ma[i] is None or bar.volume > vol_ma[i] * 1.2

            if ema_cross_up and bull_trend and rsi_ok:
                entry_price = bar.close
                sl = entry_price - cur_atr * ATR_SL_MULT
                tp = entry_price + cur_atr * ATR_TP_MULT
                pos_size = state.capital * POSITION_SIZE_PCT
                shares = pos_size / entry_price
                commission = pos_size * COMMISSION_PCT / 100
                state.capital -= pos_size
                state.position = "LONG"
                state.entry_price = entry_price
                state.entry_date = bar.date
                state.stop_loss = sl
                state.take_profit = tp
                state.position_size = pos_size - commission
                state.shares = shares

            elif ema_cross_down and bear_trend and rsi_ok:
                entry_price = bar.close
                sl = entry_price + cur_atr * ATR_SL_MULT
                tp = entry_price - cur_atr * ATR_TP_MULT
                pos_size = state.capital * POSITION_SIZE_PCT
                shares = pos_size / entry_price
                commission = pos_size * COMMISSION_PCT / 100
                state.capital -= pos_size
                state.position = "SHORT"
                state.entry_price = entry_price
                state.entry_date = bar.date
                state.stop_loss = sl
                state.take_profit = tp
                state.position_size = pos_size - commission
                state.shares = shares

        # Обновление equity curve
        unrealized = 0.0
        if state.position == "LONG":
            unrealized = (bar.close - state.entry_price) * state.shares
        elif state.position == "SHORT":
            unrealized = (state.entry_price - bar.close) * state.shares
        total_equity = state.capital + (state.position_size + unrealized if state.position else 0)
        state.equity_curve.append((bar.date, total_equity))

    # Закрываем открытую позицию по последней цене
    if state.position:
        last_bar = bars[-1]
        if state.position == "LONG":
            pnl = (last_bar.close - state.entry_price) * state.shares
        else:
            pnl = (state.entry_price - last_bar.close) * state.shares
        commission = last_bar.close * state.shares * COMMISSION_PCT / 100
        pnl -= commission
        pnl_pct = pnl / state.position_size * 100
        state.capital += state.position_size + pnl
        state.trades.append(Trade(
            entry_date=state.entry_date, exit_date=last_bar.date,
            direction=state.position, entry_price=state.entry_price,
            exit_price=last_bar.close, stop_loss=state.stop_loss,
            take_profit=state.take_profit, exit_reason="КОНЕЦ ДАННЫХ",
            pnl=pnl, pnl_pct=pnl_pct, position_size=state.position_size
        ))
        state.position = None

    return state


# ══════════════════════════════════════════════════════════════════════════════
# ОТЧЁТ
# ══════════════════════════════════════════════════════════════════════════════

def print_report(state: BacktestState, bars: list[Bar]):
    trades = state.trades
    n_trades = len(trades)

    print(f"\n{'═' * 70}")
    print(f"  РЕЗУЛЬТАТЫ БЭКТЕСТА")
    print(f"{'═' * 70}")
    print(f"  Период: {bars[0].date} — {bars[-1].date} ({len(bars)} баров)")
    print(f"  Начальный капитал:  ${INITIAL_CAPITAL:>12,.2f}")
    print(f"  Конечный капитал:   ${state.capital:>12,.2f}")

    total_return = (state.capital - INITIAL_CAPITAL) / INITIAL_CAPITAL * 100
    print(f"  Доходность:         {total_return:>12.2f}%")
    print(f"{'─' * 70}")

    if n_trades == 0:
        print("  Сделок не было.")
        print(f"{'═' * 70}")
        return

    wins = [t for t in trades if t.pnl > 0]
    losses = [t for t in trades if t.pnl <= 0]
    longs = [t for t in trades if t.direction == "LONG"]
    shorts = [t for t in trades if t.direction == "SHORT"]

    print(f"  Всего сделок:       {n_trades}")
    print(f"    Long:             {len(longs)}")
    print(f"    Short:            {len(shorts)}")
    print(f"  Прибыльных:         {len(wins)} ({len(wins)/n_trades*100:.1f}%)")
    print(f"  Убыточных:          {len(losses)} ({len(losses)/n_trades*100:.1f}%)")
    print(f"{'─' * 70}")

    total_profit = sum(t.pnl for t in wins) if wins else 0
    total_loss = sum(t.pnl for t in losses) if losses else 0
    print(f"  Общая прибыль:      ${total_profit:>12,.2f}")
    print(f"  Общий убыток:       ${total_loss:>12,.2f}")
    print(f"  Нетто P&L:          ${total_profit + total_loss:>12,.2f}")

    if wins:
        print(f"  Ср. прибыль:        ${total_profit/len(wins):>12,.2f}")
        print(f"  Макс. прибыль:      ${max(t.pnl for t in wins):>12,.2f}")
    if losses:
        print(f"  Ср. убыток:         ${total_loss/len(losses):>12,.2f}")
        print(f"  Макс. убыток:       ${min(t.pnl for t in losses):>12,.2f}")

    if total_loss != 0:
        profit_factor = abs(total_profit / total_loss)
        print(f"  Profit Factor:      {profit_factor:>12.2f}")

    # Max drawdown
    peak = INITIAL_CAPITAL
    max_dd = 0
    max_dd_pct = 0
    for _, equity in state.equity_curve:
        if equity > peak:
            peak = equity
        dd = peak - equity
        dd_pct = dd / peak * 100
        if dd > max_dd:
            max_dd = dd
            max_dd_pct = dd_pct
    print(f"  Макс. просадка:     ${max_dd:>12,.2f} ({max_dd_pct:.2f}%)")

    # Детали сделок
    print(f"\n{'═' * 70}")
    print(f"  ЖУРНАЛ СДЕЛОК")
    print(f"{'═' * 70}")
    print(f"  {'#':>2}  {'Напр.':>5}  {'Вход':>12}  {'Выход':>12}  {'P&L':>10}  {'%':>7}  {'Причина'}")
    print(f"  {'─'*2}  {'─'*5}  {'─'*12}  {'─'*12}  {'─'*10}  {'─'*7}  {'─'*15}")

    for i, t in enumerate(trades, 1):
        sign = "+" if t.pnl > 0 else ""
        print(f"  {i:>2}  {t.direction:>5}  {t.entry_price:>12,.2f}  {t.exit_price:>12,.2f}  {sign}{t.pnl:>9,.2f}  {sign}{t.pnl_pct:>6.2f}%  {t.exit_reason}")
        print(f"      {t.entry_date} → {t.exit_date}   SL: {t.stop_loss:,.2f}  TP: {t.take_profit:,.2f}")

    # Equity curve (текстовая)
    print(f"\n{'═' * 70}")
    print(f"  КРИВАЯ КАПИТАЛА")
    print(f"{'═' * 70}")
    if state.equity_curve:
        min_eq = min(e for _, e in state.equity_curve)
        max_eq = max(e for _, e in state.equity_curve)
        chart_width = 40
        for date, equity in state.equity_curve:
            if max_eq == min_eq:
                bar_len = chart_width
            else:
                bar_len = int((equity - min_eq) / (max_eq - min_eq) * chart_width)
            bar_char = "█" * bar_len
            marker = " ◄" if equity == max_eq else (" ◄ MIN" if equity == min_eq else "")
            print(f"  {date}  ${equity:>12,.2f}  {bar_char}{marker}")

    print(f"\n{'═' * 70}")
    print(f"  ПРЕДУПРЕЖДЕНИЕ: Данных всего {len(bars)} дневных баров.")
    print(f"  Результаты НЕ являются статистически значимыми.")
    print(f"  Для полноценного бэктеста необходимы:")
    print(f"    - минимум 6-12 месяцев дневных данных, ИЛИ")
    print(f"    - интрадей данные (5-15 мин) за 1-3 месяца")
    print(f"{'═' * 70}")


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    csv_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            "NASDAQ Composite Historical Data.csv")
    print(f"\n  Загрузка данных: {csv_path}\n")
    bars = load_data(csv_path)
    print(f"  Загружено {len(bars)} баров: {bars[0].date} — {bars[-1].date}")
    print(f"  Диапазон цен: {min(b.low for b in bars):,.2f} — {max(b.high for b in bars):,.2f}")
    print(f"  Средний дневной диапазон: {sum(b.high - b.low for b in bars) / len(bars):,.2f}")

    state = run_backtest(bars)
    print_report(state, bars)
