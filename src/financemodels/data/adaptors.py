import sys
import requests
import numpy as np
import pandas as pd
import yfinance as yf
from abc import ABC, ABCMeta, abstractmethod
from enum import Enum
from pandas import DataFrame
from typing import Optional, Any
from yahoofinancials import YahooFinancials
from pathlib import Path


class StockPriceDatasetAdapter(metaclass=ABCMeta):

    DEFAULT_TICKER = "AAPL"
    DEFAULT_TRAINING_SET_DATE_RANGE = ("2020-01-01", "2021-12-31")
    DEFAULT_VALIDATION_SET_DATE_RANGE = ("2013-07-01", "2013-08-31")

    @property
    @abstractmethod
    def training_set(self, ticker: Optional[str] = None) -> DataFrame: ...

    @property
    @abstractmethod
    def validation_set(self, ticker: Optional[str] = None) -> DataFrame: ...


class BaseStockPriceDatasetAdapter(StockPriceDatasetAdapter, ABC):

    def __init__(self, ticker: Optional[str] = None) -> None:
        self._ticker = ticker
        self._training_set = None
        self._validation_set = None

    @abstractmethod
    def _connect_and_prepare(self, date_range: tuple[str, str]) -> DataFrame: ...

    @property
    def training_set(self, ticker: Optional[str] = None) -> DataFrame:
        if isinstance(self._training_set, DataFrame):
            return self._training_set.copy()
        return DataFrame()

    @property
    def validation_set(self, ticker: Optional[str] = None) -> DataFrame:
        if isinstance(self._validation_set, DataFrame):
            return self._validation_set.copy()
        return DataFrame()


class YahooFinancialsAdapter(BaseStockPriceDatasetAdapter):

    class Frequency(Enum):
        DAILY = "daily"
        WEEKLY = "weekly"
        MONTHLY = "monthly"

    def __init__(
        self,
        ticker: str = StockPriceDatasetAdapter.DEFAULT_TICKER,
        frequency: Frequency = Frequency.DAILY,
        training_set_date_range: tuple[
            str, str
        ] = StockPriceDatasetAdapter.DEFAULT_TRAINING_SET_DATE_RANGE,
        validation_set_date_range: tuple[
            str, str
        ] = StockPriceDatasetAdapter.DEFAULT_VALIDATION_SET_DATE_RANGE,
    ):
        super().__init__(ticker=ticker)
        self._frequency = frequency
        self._yf = YahooFinancials(self._ticker)
        self._training_set = self._connect_and_prepare(training_set_date_range)
        self._validation_set = self._connect_and_prepare(validation_set_date_range)

    def _connect_and_prepare(self, date_range: tuple[str, str]) -> DataFrame:
        stock_price_records = None
        records = self._yf.get_historical_price_data(
            date_range[0], date_range[1], self._frequency.value
        )[self._ticker]
        stock_price_records = DataFrame(data=records["prices"])[
            ["formatted_date", "close"]
        ]
        stock_price_records.columns = ["time", "stock price"]
        return stock_price_records


class MarketStackAdapter(BaseStockPriceDatasetAdapter):

    _REQ_PARAMS = {"access_key": "85a88420013940af271a1003f8f42774", "limit": 500}
    _EOD_API_URL = "https://api.marketstack.com/v2/eod"
    _TICKER_API_URL = "https://api.marketstack.com/v2/tickers"

    class _PaginatedRecords:
        def __init__(self, api_url: str, req_params: dict[str, Any]) -> None:
            self._req_params = req_params
            self._offset = 0
            self._total_records = sys.maxsize
            self._api_url = api_url

        def __getitem__(self, index: int):
            if (self._offset + self._req_params["limit"]) >= self._total_records:
                raise StopIteration()
            self._req_params["offset"] = self._offset
            api_response = requests.get(self._api_url, self._req_params).json()
            print(api_response)
            self._total_records = api_response["pagination"]["total"]
            self._offset += self._req_params["limit"] + 1
            return api_response["data"]

    def __init__(
        self,
        ticker: Optional[str] = None,
        training_set_date_range: tuple[str, str] = ("2020-01-01", "2021-12-31"),
        validation_set_date_range: tuple[str, str] = ("2013-07-01", "2013-08-31"),
    ) -> None:
        super().__init__(ticker=ticker)
        self._training_set = self._connect_and_prepare(training_set_date_range)
        self._validation_set = self._connect_and_prepare(validation_set_date_range)

    def _connect_and_prepare(self, date_range: tuple[str, str]) -> DataFrame:
        def _extract_stock_price_details(
            stock_price_records: DataFrame, page: dict[str, Any]
        ) -> DataFrame:
            ticker_symbol = page["symbol"]
            stock_record_per_symbol = stock_price_records.get(ticker_symbol)
            if stock_record_per_symbol is None:
                stock_record_per_symbol = DataFrame()
            entry = {
                "stock price": [page["close"]],
                "time": [page["date"].split("T")[0]],
            }
            stock_price_records[ticker_symbol] = pd.concat(
                [stock_record_per_symbol, DataFrame(entry)], ignore_index=True
            )
            return stock_price_records

        if self._ticker is None:
            return DataFrame()
        req_params = MarketStackAdapter._REQ_PARAMS.copy()
        req_params["symbols"] = self._ticker
        req_params["date_from"] = date_range[0]
        req_params["date_to"] = date_range[1]
        stock_price_records = DataFrame()

        for records in MarketStackAdapter._PaginatedRecords(
            api_url=MarketStackAdapter._EOD_API_URL, req_params=req_params
        ):
            for page in records:
                stock_price_records = _extract_stock_price_details(
                    stock_price_records, page
                )

        print(stock_price_records)

        return stock_price_records


class YFinanceAdaptor(BaseStockPriceDatasetAdapter):

    class Interval(Enum):
        DAILY = "1d"
        WEEKLY = "1wk"
        MONTHLY = "1mo"

    def __init__(
        self,
        ticker: str = StockPriceDatasetAdapter.DEFAULT_TICKER,
        interval: str = Interval.DAILY.value,
        training_set_date_range: tuple[
            str, str
        ] = StockPriceDatasetAdapter.DEFAULT_TRAINING_SET_DATE_RANGE,
        validation_set_date_range: tuple[
            str, str
        ] = StockPriceDatasetAdapter.DEFAULT_VALIDATION_SET_DATE_RANGE,
    ):
        super().__init__(ticker=ticker)
        self._interval = interval
        self._yf = yf.Ticker(ticker)
        self._training_set = self._connect_and_prepare(training_set_date_range)
        self._validation_set = self._connect_and_prepare(validation_set_date_range)

    def _connect_and_prepare(self, date_range: tuple[str, str]) -> DataFrame:
        records = self._yf.history(
            start=date_range[0], end=date_range[1], interval=self._interval
        )
        records.reset_index(inplace=True)
        stock_price_records = DataFrame(data=records[["Date", "Close"]])
        stock_price_records.rename(
            columns={"Date": "time", "Close": "stock price"}, inplace=True
        )
        return stock_price_records


if __name__ == "__main__":
    adaptor = YFinanceAdaptor()
