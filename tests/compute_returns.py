from data.adaptors import YFinanceAdaptor
from data.visualisation import plot_returns_for_different_periods


def compute_returns():
    monthly = YFinanceAdaptor(
        interval=YFinanceAdaptor.Interval.MONTHLY.value
    ).training_set
    monthly["return"] = (monthly["stock price"] / monthly["stock price"].shift(1)) - 1
    weekly = YFinanceAdaptor(
        interval=YFinanceAdaptor.Interval.WEEKLY.value
    ).training_set
    weekly["return"] = (weekly["stock price"] / weekly["stock price"].shift(1)) - 1
    daily = YFinanceAdaptor(interval=YFinanceAdaptor.Interval.DAILY.value).training_set
    daily["return"] = (daily["stock price"] / daily["stock price"].shift(1)) - 1

    periodic_returns = [("Daily", daily), ("Weekly", weekly), ("Monthly", monthly)]

    return periodic_returns


if __name__ == "__main__":
    plot_returns_for_different_periods("Pfizer", compute_returns())
