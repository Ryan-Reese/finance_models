from data.adaptors import YahooFinancialsAdapter, MarketStackAdapter
from data.visualisation import plot_security_prices


def test_yahoo_financials_adapter():
    records = {
        "Apple": YahooFinancialsAdapter(
            ticker="MSFT", training_set_date_range=("2021-02-01", "2021-04-30")
        ).training_set,
        "Google": YahooFinancialsAdapter(
            ticker="MSFT", training_set_date_range=("2021-02-01", "2021-04-30")
        ).training_set,
    }
    plot_security_prices(records, "stock price")


def test_market_stack_adapter():
    records = {
        "Apple": MarketStackAdapter(
            ticker="AAPL", training_set_date_range=("2021-02-01", "2021-04-30")
        ).training_set,
        "Microsoft": MarketStackAdapter(
            ticker="MSFT", training_set_date_range=("2021-02-01", "2021-04-30")
        ).training_set,
    }
    plot_security_prices(records, "stock price")


if __name__ == "__main__":
    test_market_stack_adapter()
    # test_yahoo_financials_adapter()
