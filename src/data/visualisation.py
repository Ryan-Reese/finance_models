import matplotlib.pyplot as plt
from pandas import DataFrame


def plot_security_prices(all_records: dict[str, DataFrame], security_type: str):
    plt.style.use("seaborn-v0_8")
    n_securities = len(all_records)
    if n_securities == 1:
        rows = 1
        cols = 1
    else:
        rows = int(n_securities / 2)
        cols = 2

    security_names = list(all_records.keys())
    fig, ax = plt.subplots(rows, cols)
    i, r = 0, 0

    def _axis_plot_security_prices(records, col, security_name: str):
        match n_securities:
            case 1:
                ax.set_title(security_name)
                records.plot(ax=ax, x="time", y=security_type)
            case 2:
                ax[col].set_title(security_name)
                records.plot(ax=ax[col], x="time", y=security_type)
            case _:
                ax[r, col].set_title(security_name)
                records.plot(ax=ax[r, col], x="time", y=security_type)

    while i < n_securities:
        _axis_plot_security_prices(all_records[security_names[i]], 0, security_names[i])
        i = i + 1
        if n_securities > 1:
            _axis_plot_security_prices(
                all_records[security_names[i]], 1, security_names[i]
            )
        i = i + 1
        r = r + 1

    fig.tight_layout()
    plt.show()


def plot_returns_for_different_periods(ticker, periodic_returns: list[tuple]):
    plt.style.use("seaborn")
    fig, ax = plt.subplots(len(periodic_returns), 1)

    for index, t in enumerate(periodic_returns):
        t[1].plot(ax=ax[index], x="time", y="Return")
        ax[index].set_title(ticker + " - " + t[0] + " Returns")

    fig.tight_layout()
    plt.show()
