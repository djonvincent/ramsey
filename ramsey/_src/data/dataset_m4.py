import os
from collections import namedtuple
from dataclasses import dataclass
from typing import Tuple, Optional
from urllib.parse import urlparse
from urllib.request import urlretrieve

import pandas as pd

dset = namedtuple("dset", ["name", "urls"])
URL__ = "https://github.com/Mcompetitions/M4-methods/raw/master/Dataset/"

__M4_HOURLY = dset(
    "hourly",
    [
        f"{URL__}/Train/Hourly-train.csv",
        f"{URL__}/Test/Hourly-test.csv",
    ],
)

__M4_DAILY = dset(
    "daily",
    [
        f"{URL__}/Train/Daily-train.csv",
        f"{URL__}/Test/Daily-test.csv",
    ],
)

__M4_WEEKLY = dset(
    "weekly",
    [
        f"{URL__}/Train/Weekly-train.csv",
        f"{URL__}/Test/Weekly-test.csv",
    ],
)

__M4_MONTHLY = dset(
    "monthly",
    [
        f"{URL__}/Train/Monthly-train.csv",
        f"{URL__}/Test/Monthly-test.csv",
    ],
)

__M4_QUARTERLY = dset(
    "quarterly",
    [
        f"{URL__}/Train/Quarterly-train.csv",
        f"{URL__}/Test/Quarterly-test.csv",
    ],
)

__M4_YEARLY = dset(
    "yearly",
    [
        f"{URL__}/Train/Yearly-train.csv",
        f"{URL__}/Test/Yearly-test.csv",
    ],
)


_M4_INFO = dset("info", [f"{URL__}/M4-info.csv"])


_M4_DATA_SETS = {
    "hourly": {
        "key": __M4_HOURLY,
        "n_observations": 700,
        "n_forecasts": 48,
        "series_prefix": "H",
    },
    "daily": {
        "key": __M4_DAILY,
        "n_observations": 93,
        "n_forecasts": 14,
        "series_prefix": "D",
    },
    "weekly": {
        "key": __M4_WEEKLY,
        "n_observations": 80,
        "n_forecasts": 13,
        "series_prefix": "W",
    },
    "monthly": {
        "key": __M4_MONTHLY,
        "n_observations": 42,
        "n_forecasts": 18,
        "series_prefix": "M",
    },
    "quarterly": {
        "key": __M4_QUARTERLY,
        "n_observations": 16,
        "n_forecasts": 8,
        "series_prefix": "Q",
    },
    "yearly": {
        "key": __M4_YEARLY,
        "n_observations": 13,
        "n_forecasts": 6,
        "series_prefix": "Y",
    },
}


# pylint: disable=too-few-public-methods
@dataclass
class M4Dataset:
    """A wrapper class to load M4 data."""

    __INTERVALS__ = [
        "hourly",
        "daily",
        "weekly",
        "monthly",
        "quarterly",
        "yearly",
    ]
    __DOMAINS__ = [
        "macro",
        "micro",
        "demographic",
        "industry",
        "finance",
        "other",
    ]
    data_dir: str = os.path.abspath(
        os.path.join(os.path.dirname(__file__), ".data", "m4")
    )

    def load(
            self,
            interval: Optional[str] = None,
            domain: Optional[str] = None
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Load a M4 data set.

        Parameters
        ----------
        interval: str
            either of "hourly", "daily", "weekly", "monthly", "yearly"
        domain: str
            either of "macro", "micro", "demographic", "industry", "finance",
            "other"

        Returns
        -------
        Tuple[pd.DataFrame, pd.DataFrame]
            a tuple of data frames where the first is the training data and
            the last the testing data used during the M4 competition
        """
        if interval is None and domain is None:
            raise ValueError(
                "at least of one 'interval' or 'domain' must be specified"
            )
        if interval is not None and interval not in self.__INTERVALS__:
            raise ValueError(
                f"'interval' should be one of: '{'/'.join(self.__INTERVALS__)}'"
            )
        if domain is not None and domain not in self.__DOMAINS__:
            raise ValueError(
                f"'domain' should be one of: '{'/'.join(self.__DOMAINS__)}'"
            )

        os.makedirs(self.data_dir, exist_ok=True)
        if domain is not None:
            self._download(_M4_INFO)
            info_csv_path = os.path.join(self.data_dir, f"M4-info.csv")
            m4_info = pd.read_csv(info_csv_path, sep=",", header=0, index_col=0)
        if interval is None:
            intervals = self.__INTERVALS__
        else:
            intervals = [interval]

        all_train = []
        all_test = []
        for interval in intervals:
            dataset = _M4_DATA_SETS[interval]["key"]
            train_csv_path = os.path.join(
                self.data_dir, f"{interval.capitalize()}-train.csv"
            )
            test_csv_path = os.path.join(
                self.data_dir, f"{interval.capitalize()}-test.csv"
            )
            train, test = self._load(dataset, train_csv_path, test_csv_path)
            if domain is not None:
                domain_idx = pd.DataFrame(
                    index = m4_info[
                        m4_info["category"].str.lower() == domain
                    ].index
                )
                train = train.join(domain_idx, how='inner')
                test = test.join(domain_idx, how='inner')
            all_train.append(train)
            all_test.append(test)

        all_train = pd.concat(all_train)
        all_test = pd.concat(all_test)
        return all_train, all_test

    def _download(self, dataset):
        for url in dataset.urls:
            file = os.path.basename(urlparse(url).path)
            out_path = os.path.join(self.data_dir, file)
            if url.lower().startswith("https"):
                if not os.path.exists(out_path):
                    urlretrieve(url, out_path)
            else:
                raise ValueError(f"{url} does not start with https")

    def _load(
        self, dataset, train_csv_path: str, test_csv_path: str
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        self._download(dataset)
        train_df = pd.read_csv(train_csv_path, sep=",", header=0, index_col=0)
        test_df = pd.read_csv(test_csv_path, sep=",", header=0, index_col=0)
        return train_df, test_df
