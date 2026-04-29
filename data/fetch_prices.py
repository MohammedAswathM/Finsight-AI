"""Price fetching utilities for FinSight AI.

Helper functions to query price data from SQLite database.
"""
from __future__ import annotations

import sqlite3
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

import pandas as pd

DB_PATH = Path(__file__).parent.parent / "outputs" / "finsight.db"


def get_connection() -> sqlite3.Connection:
    """Get database connection."""
    return sqlite3.connect(DB_PATH)


def get_available_tickers() -> list[str]:
    """Get list of tickers in the database."""
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT DISTINCT ticker FROM prices ORDER BY ticker")
    tickers = [row[0] for row in cursor.fetchall()]
    conn.close()
    return tickers


def get_price_data(
    ticker: str,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    period: Optional[str] = None
) -> pd.DataFrame:
    """Get price data for a ticker.
    
    Args:
        ticker: Stock ticker symbol
        start_date: Start date (YYYY-MM-DD)
        end_date: End date (YYYY-MM-DD)
        period: Alternative to dates - e.g., "6mo", "1y", "2y"
    
    Returns:
        DataFrame with price data
    """
    conn = get_connection()
    
    # Build query
    query = "SELECT * FROM prices WHERE ticker = ?"
    params = [ticker.upper()]
    
    if period:
        # Calculate start date from period
        end_date = datetime.now().strftime("%Y-%m-%d")
        if period == "6mo":
            start_date = (datetime.now() - timedelta(days=180)).strftime("%Y-%m-%d")
        elif period == "1y":
            start_date = (datetime.now() - timedelta(days=365)).strftime("%Y-%m-%d")
        elif period == "2y":
            start_date = (datetime.now() - timedelta(days=730)).strftime("%Y-%m-%d")
        elif period == "1mo":
            start_date = (datetime.now() - timedelta(days=30)).strftime("%Y-%m-%d")
        elif period == "3mo":
            start_date = (datetime.now() - timedelta(days=90)).strftime("%Y-%m-%d")
        else:
            start_date = (datetime.now() - timedelta(days=180)).strftime("%Y-%m-%d")
    
    if start_date:
        query += " AND date >= ?"
        params.append(start_date)
    if end_date:
        query += " AND date <= ?"
        params.append(end_date)
    
    query += " ORDER BY date ASC"
    
    df = pd.read_sql_query(query, conn, params=params)
    conn.close()
    return df


def get_latest_price(ticker: str) -> Optional[dict]:
    """Get the most recent price for a ticker."""
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("""
        SELECT * FROM prices 
        WHERE ticker = ? 
        ORDER BY date DESC 
        LIMIT 1
    """, (ticker.upper(),))
    
    row = cursor.fetchone()
    conn.close()
    
    if row:
        columns = ["id", "ticker", "date", "open", "high", "low", "close", "volume", "adjusted_close"]
        return dict(zip(columns, row))
    return None


def get_fundamentals(ticker: str) -> Optional[dict]:
    """Get fundamental data for a ticker."""
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("""
        SELECT * FROM fundamentals 
        WHERE ticker = ? 
        ORDER BY date DESC 
        LIMIT 1
    """, (ticker.upper(),))
    
    row = cursor.fetchone()
    conn.close()
    
    if row:
        columns = ["id", "ticker", "date", "revenue", "net_income", "eps", "pe_ratio", "dividend_yield", "book_value"]
        return dict(zip(columns, row))
    return None


def get_date_range(ticker: str) -> tuple[str, str]:
    """Get the date range of available data for a ticker."""
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("""
        SELECT MIN(date), MAX(date) FROM prices WHERE ticker = ?
    """, (ticker.upper(),))
    
    row = cursor.fetchone()
    conn.close()
    return row[0] or "", row[1] or ""


if __name__ == "__main__":
    # Test the module
    print("Available tickers:", get_available_tickers())
    
    if get_available_tickers():
        ticker = "AAPL"
        print(f"\nPrice data for {ticker}:")
        df = get_price_data(ticker, period="6mo")
        print(df.head())
        
        print(f"\nLatest price for {ticker}:")
        print(get_latest_price(ticker))
        
        print(f"\nFundamentals for {ticker}:")
        print(get_fundamentals(ticker))