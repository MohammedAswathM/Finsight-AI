"""Database setup — downloads price data via yfinance and stores in SQLite.

Tables:
- prices: ticker, date, open, high, low, close, volume, adjusted_close
- fundamentals: ticker, date, revenue, net_income, eps, pe_ratio, dividend_yield

Run: python -m data.db_setup
"""
from __future__ import annotations

import sqlite3
from pathlib import Path

import yfinance as yf
import pandas as pd

# Database path
DB_PATH = Path(__file__).parent.parent / "outputs" / "finsight.db"
DB_PATH.parent.mkdir(parents=True, exist_ok=True)


def init_db() -> None:
    """Initialize SQLite database with required tables."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    # Prices table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS prices (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            ticker TEXT NOT NULL,
            date TEXT NOT NULL,
            open REAL,
            high REAL,
            low REAL,
            close REAL,
            volume INTEGER,
            adjusted_close REAL,
            UNIQUE(ticker, date)
        )
    """)

    # Fundamentals table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS fundamentals (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            ticker TEXT NOT NULL,
            date TEXT NOT NULL,
            revenue REAL,
            net_income REAL,
            eps REAL,
            pe_ratio REAL,
            dividend_yield REAL,
            book_value REAL,
            UNIQUE(ticker, date)
        )
    """)

    # Create indexes for faster queries
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_prices_ticker_date ON prices(ticker, date)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_fundamentals_ticker_date ON fundamentals(ticker, date)")

    conn.commit()
    conn.close()
    print(f"Database initialized at: {DB_PATH}")


def fetch_and_store_prices(tickers: list[str], period: str = "2y") -> None:
    """Fetch price data from yfinance and store in SQLite.
    
    Args:
        tickers: List of stock tickers (e.g., ["AAPL", "MSFT", "GOOGL"])
        period: Data period to fetch (default: 2 years)
    """
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    for ticker in tickers:
        print(f"Fetching {ticker}...")
        try:
            # Download data from yfinance
            stock = yf.Ticker(ticker)
            df = stock.history(period=period)
            
            if df.empty:
                print(f"  Warning: No data for {ticker}")
                continue
            
            # Reset index to get date as column
            df = df.reset_index()
            df["ticker"] = ticker
            
            # Rename columns to match schema - handle different yfinance versions
            column_mapping = {
                "Date": "date",
                "Open": "open",
                "High": "high",
                "Low": "low",
                "Close": "close",
                "Volume": "volume",
                "Adj Close": "adjusted_close"
            }
            
            # Apply column mapping only for columns that exist
            for old_col, new_col in column_mapping.items():
                if old_col in df.columns:
                    df = df.rename(columns={old_col: new_col})
            
            # Ensure required columns exist with defaults
            for col in ["open", "high", "low", "close", "volume", "adjusted_close"]:
                if col not in df.columns:
                    df[col] = None
            
            # Convert date to string format
            if "date" in df.columns:
                df["date"] = pd.to_datetime(df["date"]).dt.strftime("%Y-%m-%d")
            else:
                print(f"  Warning: No date column for {ticker}")
                continue
            
            # Select only the columns we need
            df = df[["ticker", "date", "open", "high", "low", "close", "volume", "adjusted_close"]]

            # Idempotent insert (safe to re-run).
            rows = df.to_records(index=False).tolist()
            cursor.executemany(
                """
                INSERT OR REPLACE INTO prices
                (ticker, date, open, high, low, close, volume, adjusted_close)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                rows,
            )
            conn.commit()
            print(f"  Stored {len(rows)} rows for {ticker}")
            
        except Exception as e:
            print(f"  Error fetching {ticker}: {e}")
    
    conn.close()
    print("Price data fetch complete!")


def fetch_and_store_fundamentals(tickers: list[str]) -> None:
    """Fetch fundamental data from yfinance and store in SQLite.
    
    Args:
        tickers: List of stock tickers
    """
    conn = sqlite3.connect(DB_PATH)
    
    for ticker in tickers:
        print(f"Fetching fundamentals for {ticker}...")
        try:
            stock = yf.Ticker(ticker)
            info = stock.info
            
            # Get the most recent quarterly data
            financials = stock.quarterly_financials
            
            if financials.empty:
                print(f"  Warning: No fundamentals for {ticker}")
                continue
            
            # Get latest date - handle both string and datetime formats
            latest_date_raw = financials.index[0]
            if hasattr(latest_date_raw, 'strftime'):
                latest_date = latest_date_raw.strftime("%Y-%m-%d")
            else:
                latest_date = str(latest_date_raw)[:10]  # Take first 10 chars if it's a string
            
            # Extract key metrics - handle different pandas versions
            try:
                revenue = financials.loc["Total Revenue"].iloc[0] if "Total Revenue" in financials.index else None
            except:
                revenue = None
            
            try:
                net_income = financials.loc["Net Income"].iloc[0] if "Net Income" in financials.index else None
            except:
                net_income = None
            
            # Get info metrics
            eps = info.get("trailingEps")
            pe_ratio = info.get("trailingPE")
            dividend_yield = info.get("dividendYield")
            book_value = info.get("bookValue")
            
            # Insert into database
            cursor = conn.cursor()
            cursor.execute("""
                INSERT OR REPLACE INTO fundamentals 
                (ticker, date, revenue, net_income, eps, pe_ratio, dividend_yield, book_value)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (ticker, latest_date, revenue, net_income, eps, pe_ratio, dividend_yield, book_value))
            
            conn.commit()
            print(f"  Stored fundamentals for {ticker}")
            
        except Exception as e:
            print(f"  Error fetching fundamentals for {ticker}: {e}")
    
    conn.close()
    print("Fundamentals fetch complete!")


def get_sample_tickers() -> list[str]:
    """Return default list of tickers to fetch."""
    return [
        "AAPL",    # Apple
        "MSFT",    # Microsoft
        "GOOGL",   # Alphabet
        "AMZN",    # Amazon
        "NVDA",    # NVIDIA
        "META",    # Meta
        "TSLA",    # Tesla
        "JPM",     # JPMorgan
        "V",       # Visa
        "JNJ",     # Johnson & Johnson
    ]


def main():
    """Main entry point for db_setup."""
    print("=" * 50)
    print("FinSight AI - Database Setup")
    print("=" * 50)
    
    # Initialize database
    init_db()
    
    # Get tickers
    tickers = get_sample_tickers()
    print(f"\nFetching data for: {', '.join(tickers)}")
    
    # Fetch and store prices
    print("\n--- Fetching Price Data ---")
    fetch_and_store_prices(tickers, period="2y")
    
    # Fetch and store fundamentals
    print("\n--- Fetching Fundamentals ---")
    fetch_and_store_fundamentals(tickers)
    
    print("\n" + "=" * 50)
    print("Database setup complete!")
    print(f"Database location: {DB_PATH}")
    print("=" * 50)


if __name__ == "__main__":
    main()