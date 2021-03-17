import datetime
import random
import time
import backtrader as bt
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from alpha_vantage.timeseries import TimeSeries
from deap import base
from deap import creator
from deap import tools
from tqdm import trange
from pathlib import Path


ALPHA_VANTAGE_DIR_PATH = Path('/data')
SECRET = "demo"


def get_alpha_vantage(key, ticker):
    """Given a key to Alpha Vantage and a valid ticker, this function will
    query alpha vantage and save the dataset into a csv in a predefined
    directory using ticker as the filename.
    """
    ts = TimeSeries(key=key, output_format="pandas", indexing_type="date")

    try:
        data, meta_data = ts.get_daily_adjusted(symbol=i, outputsize="full")
        data.to_csv(ALPHA_VANTAGE_DIR_PATH / f"{ticker}.csv")
        print(f"{ticker} has been downloaded to {ALPHA_VANTAGE_DIR_PATH}/{ticker}.csv")
    except:
        print(f"{ticker} Not found.")


def read_alpha_vantage(ticker):
    """If the ticker's csv has been downloaded with `get_alpha_vantage`,
    this function will return a pandas dataframe of adjusted open, adjusted
    high, adjusted low, adjusted close and volume rounded to 4 decimal places
    """
    if not (ALPHA_VANTAGE_DIR_PATH / f"{ticker}.csv").exists():
        return None

    df = pd.read_csv(
        ALPHA_VANTAGE_DIR_PATH / f"{ticker}.csv", index_col=0, parse_dates=True
    ).sort_index()
    df = df.rename(
        columns={
            "1. open": "Open",
            "2. high": "High",
            "3. low": "Low",
            "4. close": "Close",
            "5. adjusted close": "Adjusted Close",
            "6. volume": "Volume",
            "7. dividend amount": "Dividend",
            "8. split coefficient": "Split Coefficient",
        }
    )
    df["Unadjusted Open"] = df["Open"]
    df["Open"] = df["Close"] * df["Adjusted Close"] / df["Open"]
    df["High"] = df["High"] * df["Open"] / df["Unadjusted Open"]
    df["Low"] = df["Low"] * df["Open"] / df["Unadjusted Open"]
    df["Close"] = df["Adjusted Close"]
    return df[["Open", "High", "Low", "Close", "Volume"]].round(4)


get_alpha_vantage(key=SECRET, ticker="NVDA")
df = read_alpha_vantage(ticker="NVDA")

print(df.head())


class CrossoverStrategy(bt.Strategy):
    # list of parameters which are configurable for the strategy
    params = dict(fast_period=12, slow_period=26, signal_period=9)

    def __init__(self):

        self.fast_ma = bt.indicators.EMA(self.data.close, period=self.p.fast_period)
        self.slow_ma = bt.indicators.EMA(self.data.close, period=self.p.slow_period)
        self.macd_line = self.fast_ma - self.slow_ma
        self.signal_line = bt.indicators.EMA(self.macd_line, period=self.p.signal_period)
        self.macd_crossover = bt.indicators.CrossOver(self.macd_line, self.signal_line)

    def next(self):

        if self.macd_crossover > 0:
            self.close()  # close short position
            self.buy()  # enter long position
        elif self.macd_crossover < 0:
            self.close()  # close long position
            self.sell()  # enter short position


TICKER = "NVDA"
STRATEGY_PARAMS = dict(fast_period=12, slow_period=26, signal_period=9)


def run_backtest(plot=True, **strategy_params):

    cerebro = bt.Cerebro()
    cerebro.adddata(
        bt.feeds.PandasData(dataname=read_alpha_vantage(ticker=TICKER), name=TICKER)
    )

    # Remember to set it high enough or the strategy may not
    # be able to trade because of short of cash
    cerebro.broker.setcash(10000.0)

    # Print out the starting conditions
    print(f"Starting Portfolio Value: {cerebro.broker.getvalue():,.2f}")

    # Although we have defined some default params in the strategy,
    # we can override it by passing in keyword arguments here.
    cerebro.addstrategy(CrossoverStrategy, **strategy_params)

    cerebro.addobserver(bt.observers.Trades)
    cerebro.addobserver(bt.observers.DrawDown)
    cerebro.addanalyzer(bt.analyzers.SharpeRatio)

    # Let's say that we have 0.25% slippage and commission per trade,
    # that is 0.5% in total for a round trip.
    cerebro.broker.setcommission(commission=0.0025, margin=False)

    # Run over everything
    strats = cerebro.run()

    print(f"Final Portfolio Value:    {cerebro.broker.getvalue():,.2f}")

    if plot:
        cerebro.plot()


run_backtest(**STRATEGY_PARAMS)


# fix the seed so that we will get the same results
# feel free to change it or comment out the line
random.seed(1)

# GA parameters
PARAM_NAMES = ["fast_period", "slow_period", "signal_period"]
NGEN = 20
NPOP = 100
CXPB = 0.5
MUTPB = 0.3

data = bt.feeds.PandasData(dataname=read_alpha_vantage(ticker=TICKER), name=TICKER)


def evaluate(individual, plot=False, log=False):

    # convert list of parameter values into dictionary of kwargs
    strategy_params = {k: v for k, v in zip(PARAM_NAMES, individual)}

    # fast moving average by definition cannot be slower than the slow one
    if strategy_params["fast_period"] >= strategy_params["slow_period"]:
        return [-np.inf]

    # by setting stdstats to False, backtrader will not store the changes in
    # statistics like number of trades, buys & sells, etc.
    cerebro = bt.Cerebro(stdstats=False)
    cerebro.adddata(data)

    # Remember to set it high enough or the strategy may not
    # be able to trade because of short of cash
    initial_capital = 10_000.0
    cerebro.broker.setcash(initial_capital)

    # Pass in the genes of the individual as kwargs
    cerebro.addstrategy(CrossoverStrategy, **strategy_params)

    # This is needed for calculating our fitness score
    cerebro.addanalyzer(bt.analyzers.DrawDown)

    # Let's say that we have 0.25% slippage and commission per trade,
    # that is 0.5% in total for a round trip.
    cerebro.broker.setcommission(commission=0.0025, margin=False)

    # Run over everything
    strats = cerebro.run()

    profit = cerebro.broker.getvalue() - initial_capital
    max_dd = strats[0].analyzers.drawdown.get_analysis()["max"]["moneydown"]
    fitness = profit / (max_dd if max_dd > 0 else 1)

    if log:
        print(f"Starting Portfolio Value: {initial_capital:,.2f}")
        print(f"Final Portfolio Value:    {cerebro.broker.getvalue():,.2f}")
        print(f"Total Profit:             {profit:,.2f}")
        print(f"Maximum Drawdown:         {max_dd:,.2f}")
        print(f"Profit / Max DD:          {fitness}")

    if plot:
        cerebro.plot()

    return [fitness]


# our fitness score is supposed to be maximised and there is only 1 objective
creator.create("FitnessMax", base.Fitness, weights=(1.0,))

# our individual is a list of genes, with the fitness score the higher the better
creator.create("Individual", list, fitness=creator.FitnessMax)

# register some handy functions for calling
toolbox = base.Toolbox()
toolbox.register("indices", random.sample, range(NPOP), NPOP)
# crossover strategy
toolbox.register("mate", tools.cxUniform, indpb=CXPB)
# mutation strategy
toolbox.register("mutate", tools.mutUniformInt, low=1, up=151, indpb=0.2)
# selection strategy
toolbox.register("select", tools.selTournament, tournsize=3)
# fitness function
toolbox.register("evaluate", evaluate)

# definition of an individual & a population
toolbox.register("attr_fast_period", random.randint, 1, 51)
toolbox.register("attr_slow_period", random.randint, 10, 151)
toolbox.register("attr_signal_period", random.randint, 1, 101)
toolbox.register(
    "individual",
    tools.initCycle,
    creator.Individual,
    (
        toolbox.attr_fast_period,
        toolbox.attr_slow_period,
        toolbox.attr_signal_period,
    ),
)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

mean = np.ndarray(NGEN)
best = np.ndarray(NGEN)
hall_of_fame = tools.HallOfFame(maxsize=3)

t = time.perf_counter()
pop = toolbox.population(n=NPOP)
for g in trange(NGEN):
    # Select the next generation individuals
    offspring = toolbox.select(pop, len(pop))
    # Clone the selected individuals
    offspring = list(map(toolbox.clone, offspring))

    # Apply crossover on the offspring
    for child1, child2 in zip(offspring[::2], offspring[1::2]):
        if random.random() < CXPB:
            toolbox.mate(child1, child2)
            del child1.fitness.values
            del child2.fitness.values

    # Apply mutation on the offspring
    for mutant in offspring:
        if random.random() < MUTPB:
            toolbox.mutate(mutant)
            del mutant.fitness.values

    # Evaluate the individuals with an invalid fitness
    invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
    fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
    for ind, fit in zip(invalid_ind, fitnesses):
        ind.fitness.values = fit

    # The population is entirely replaced by the offspring
    pop[:] = offspring
    hall_of_fame.update(pop)
    print(
        "HALL OF FAME:\n"
        + "\n".join(
            [
                f"    {_}: {ind}, Fitness: {ind.fitness.values[0]}"
                for _, ind in enumerate(hall_of_fame)
            ]
        )
    )

    fitnesses = [
        ind.fitness.values[0] for ind in pop if not np.isinf(ind.fitness.values[0])
    ]
    mean[g] = np.mean(fitnesses)
    best[g] = np.max(fitnesses)

end_t = time.perf_counter()
print(f"Time Elapsed: {end_t - t:,.2f}")

fig, ax = plt.subplots(sharex=True, figsize=(16, 9))

sns.lineplot(x=range(NGEN), y=mean, ax=ax, label="Average Fitness Score")
sns.lineplot(x=range(NGEN), y=best, ax=ax, label="Best Fitness Score")
ax.set_title("Fitness Score")
ax.set_xticks(range(NGEN))
ax.set_xlabel("Iteration")

plt.tight_layout()
plt.show()

OPTIMISED_STRATEGY_PARAMS = {k: v for k, v in zip(PARAM_NAMES, hall_of_fame[0])}
run_backtest(**OPTIMISED_STRATEGY_PARAMS)
