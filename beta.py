import typer
import pandas as pd
import requests
import datetime
import logging
import io
import numpy as np
from sklearn.linear_model import LinearRegression
from typing import Union, List

app = typer.Typer()

# Set up logger
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

class Historical_Beta:
    def __init__(self):
        self.base_url = 'https://derivalytics.com'
        self.endpoint = '/api/TimeSeries/query'

    def compute_beta(self,
                     independent_variable: str,
                     dependent_variable: Union[str, List[str]],
                     start_date: datetime.date = datetime.date.today() - datetime.timedelta(days=30),
                     end_date: datetime.date = datetime.date.today() - datetime.timedelta(days=1)) -> dict:
        
        # Normalize input
        dep_vars = dependent_variable if isinstance(dependent_variable, list) else [dependent_variable]
        all_symbols = [independent_variable] + dep_vars

        # Fetch & process data
        prices = self.fetch_time_series(symbols=all_symbols, start_date=start_date, end_date=end_date)
        returns = self.process_data(prices)

        results = {}
        for dep in dep_vars:
            try:
                X = returns[independent_variable].values.reshape(-1, 1)
                y = returns[dep].values

                if len(X) != len(y):
                    raise ValueError(f"Length mismatch: {independent_variable} has {len(X)} rows, {dep} has {len(y)}")

                Xy = pd.DataFrame(np.hstack([X, y.reshape(-1, 1)]), columns=[independent_variable, dep])
                Xy = Xy.replace([np.inf, -np.inf], np.nan).dropna()

                X_clean = Xy[independent_variable].values.reshape(-1, 1)
                y_clean = Xy[dep].values

                if len(X_clean) < 3:
                    raise ValueError(f"Too few clean observations to compute beta for {dep}.")

                model = LinearRegression().fit(X_clean, y_clean)
                beta = model.coef_[0]
                logger.info(f"Beta for {dep} vs {independent_variable}: {beta:.4f}")
                results[dep] = beta
            except Exception as e:
                logger.warning(f"Failed to compute beta for {dep}: {e}")
                results[dep] = None
        return results

    def fetch_time_series(self,
                          symbols: List[str],
                          start_date: datetime.date = datetime.date.today() - datetime.timedelta(days=30),
                          end_date: datetime.date = datetime.date.today() - datetime.timedelta(days=1)) -> pd.DataFrame:
        if not all(isinstance(s, str) for s in symbols):
            raise ValueError("All symbols must be strings.")

        if not isinstance(start_date, datetime.date) or not isinstance(end_date, datetime.date):
            raise ValueError("start_date and end_date must be datetime.date instances.")

        url = f"{self.base_url}{self.endpoint}"
        params = {
            "start_date": start_date.strftime('%d%b%y').lower(),
            "end_date": end_date.strftime('%d%b%y').lower(),
            "symbols": ','.join(symbols),
            "fmt": 'json',
            "periodicity": '1d'
        }

        try:
            response = requests.get(url, params=params)
            response.raise_for_status()
            raw_csv = response.text
            df = pd.read_csv(io.StringIO(raw_csv))
        except Exception as e:
            logger.error(f"Failed to fetch or parse data: {e}")
            raise

        try:
            df['DateFMT'] = pd.to_datetime(df.iloc[:, 0], utc=True)
            df.set_index('DateFMT', inplace=True, drop=True)
            df.drop(columns=df.columns[0:2], inplace=True, errors='ignore')
        except Exception as e:
            logger.error(f"Failed to process time series data: {e}")
            raise

        if df.empty or df.shape[1] < 2:
            raise ValueError("Returned data is empty or missing expected columns.")

        return df

    def process_data(self, data: pd.DataFrame) -> pd.DataFrame:
        try:
            log_data = np.log(data.replace(0, np.nan))
            returns = log_data.diff().dropna()
            returns = returns.replace([np.inf, -np.inf], np.nan).dropna()
        except Exception as e:
            logger.error(f"Failed to compute log returns: {e}")
            raise
        return returns

@app.command()
def compute(
    tickers: List[str] = typer.Argument(..., help="First ticker is independent; rest are dependents"),
    lookback: int = typer.Option(30, help="Days of lookback window")
):
    if len(tickers) < 2:
        typer.echo("Provide at least 2 tickers (1 independent, 1+ dependents).", err=True)
        raise typer.Exit(code=1)

    independent = tickers[1]
    dependent = tickers[2:]
    end_date = datetime.date.today() - datetime.timedelta(days=1)
    start_date = end_date - datetime.timedelta(days=lookback)

    beta_calculator = Historical_Beta()
    betas = beta_calculator.compute_beta(independent, dependent, start_date, end_date)

    for symbol, beta in betas.items():
        if beta is not None:
            typer.echo(f"{symbol} beta vs {independent}: {beta:.4f}")
        else:
            typer.echo(f"{symbol} beta vs {independent}: Failed")


if __name__ == "__main__":
    app()