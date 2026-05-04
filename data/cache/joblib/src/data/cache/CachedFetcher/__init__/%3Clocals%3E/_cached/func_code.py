# first line: 36
        @self._memory.cache
        def _cached(ticker: str, start: date, end: date, interval: str) -> pd.DataFrame:
            return inner_fetch(ticker, start, end, interval)
