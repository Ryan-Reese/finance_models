import pandas as pd
import requests
import sys
import visualisation as vs
from abc import ABC, ABCMeta, abstractmethod
from enum import Enum
from pandas import DataFrame
from typing import Optional
from yahoofinancials import YahooFinancials


class Frequency(Enum):
    DAILY = "daily"
    WEEKLY = "weekly"
    MONTHLY = "monthly"


class StockPriceDatasetAdapter(metaclass=ABCMeta):

    DEFAULT_TICKER = "PFE"

    @property
    @abstractmethod
    def training_set(self, ticker=None) -> Optional[DataFrame]: ...

    @property
    @abstractmethod
    def validation_set(self, ticker=None) -> Optional[DataFrame]: ...


class BaseStockPriceDatasetAdapter(StockPriceDatasetAdapter, ABC):

    def __init__(self, ticker: Optional[str] = None) -> None:
        self._ticker = ticker
        self._training_set = None
        self._validation_set = None

    @abstractmethod
    def _connect_and_prepare(
        self, date_range: tuple[str, str]
    ) -> Optional[DataFrame]: ...

    @property
    def training_set(self, ticker: Optional[str] = None) -> Optional[DataFrame]:
        if isinstance(self._training_set, DataFrame):
            return self._training_set.copy()
        return None

    @property
    def validation_set(self, ticker: Optional[str] = None) -> Optional[DataFrame]:
        if isinstance(self._validation_set, DataFrame):
            return self._validation_set.copy()
        return None


class YahooFinancialsAdapter(BaseStockPriceDatasetAdapter):

    def __init__(
        self,
        ticker=StockPriceDatasetAdapter.DEFAULT_TICKER,
        frequency=Frequency.DAILY,
        training_set_date_range=("2020-01-01", "2021-12-31"),
        validation_set_date_range=("2013-07-01", "2013-08-31"),
    ):
        super().__init__(ticker=ticker)
        self._frequency = frequency
        self._yf = YahooFinancials(self._ticker)
        self._training_set = self._connect_and_prepare(training_set_date_range)
        self._validation_set = self._connect_and_prepare(validation_set_date_range)

    def _connect_and_prepare(self, date_range: tuple[str, str]) -> Optional[DataFrame]:
        stock_price_records = None
        records = self._yf.get_historical_price_data(
            date_range[0], date_range[1], self._frequency.value
        )[self._ticker]
        print(records)
        stock_price_records = DataFrame(data=records["prices"])[
            ["formatted_date", "close"]
        ]
        stock_price_records.rename(
            columns={"formatted_date": "time", "close": "stock price"}, inplace=True
        )
        return stock_price_records


if __name__ == "__main__":
    yf_adaptor = YahooFinancialsAdapter()
    records = yf_adaptor.training_set
    vs.plot_security_prices(records, "Pfizer")
