"""Chart Agent — generates matplotlib charts from SQL query results.

This agent takes the data from the SQL agent and generates a matplotlib
chart, saving it as a PNG file.

Owned by Member 2 (feature/sql-chart branch).
"""
from __future__ import annotations

import sys
import re
import sqlite3
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, Optional

import pandas as pd

# Allow running as a script: `python agents/chart_agent.py`
if __package__ is None and str(Path(__file__).resolve().parents[1]) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from agents.base_agent import append_trace
from state import AgentState

# Database path
DB_PATH = Path(__file__).parent.parent / "outputs" / "finsight.db"
OUTPUT_DIR = Path(__file__).parent.parent / "outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


class ChartAgent:
    """Generates matplotlib charts from financial data."""
    
    def __init__(self):
        """Initialize the chart agent."""
        # No LLM needed — we generate charts deterministically from DB data.
        pass
    
    def _extract_ticker_from_query(self, query: str) -> Optional[str]:
        """Extract ticker symbol from the query."""
        # Common patterns: "AAPL", "Apple", "for AAPL", "of MSFT"
        patterns = [
            r'\b([A-Z]{1,5})\b',  # Uppercase ticker
            r'(?:for|of|on)\s+([A-Za-z]+)',  # "for Apple"
        ]
        
        # Known company names
        company_map = {
            'apple': 'AAPL', 'microsoft': 'MSFT', 'google': 'GOOGL',
            'alphabet': 'GOOGL', 'amazon': 'AMZN', 'nvidia': 'NVDA',
            'meta': 'META', 'facebook': 'META', 'tesla': 'TSLA',
            'jpmorgan': 'JPM', 'visa': 'V', 'johnson': 'JNJ',
        }
        
        query_lower = query.lower()
        for company, ticker in company_map.items():
            if company in query_lower:
                return ticker
        
        # Try to find uppercase ticker
        match = re.search(r'\b([A-Z]{2,5})\b', query)
        if match:
            return match.group(1)
        
        return None
    
    def _extract_period_from_query(self, query: str) -> str:
        """Extract time period from the query."""
        query_lower = query.lower()
        
        if '6 month' in query_lower or '6mo' in query_lower or 'last 6 month' in query_lower:
            return "6mo"
        elif '1 year' in query_lower or '1y' in query_lower or 'last year' in query_lower:
            return "1y"
        elif '2 year' in query_lower or '2y' in query_lower:
            return "2y"
        elif '3 month' in query_lower or '3mo' in query_lower:
            return "3mo"
        elif '1 month' in query_lower or '1mo' in query_lower:
            return "1mo"
        else:
            return "6mo"  # Default
    
    def _determine_chart_type(self, query: str) -> str:
        """Determine what type of chart to generate."""
        query_lower = query.lower()
        
        if 'volume' in query_lower:
            return "volume"
        elif 'candle' in query_lower or 'candlestick' in query_lower:
            return "candlestick"
        elif 'high' in query_lower and 'low' in query_lower:
            return "highlow"
        elif 'moving average' in query_lower or 'ma' in query_lower:
            return "moving_average"
        else:
            return "line"  # Default
    
    def _get_data_for_chart(self, ticker: str, period: str) -> pd.DataFrame:
        """Get price data for chart generation."""
        # Calculate date range
        end_date = datetime.now()
        if period == "6mo":
            start_date = end_date - timedelta(days=180)
        elif period == "1y":
            start_date = end_date - timedelta(days=365)
        elif period == "2y":
            start_date = end_date - timedelta(days=730)
        elif period == "3mo":
            start_date = end_date - timedelta(days=90)
        elif period == "1mo":
            start_date = end_date - timedelta(days=30)
        else:
            start_date = end_date - timedelta(days=180)
        
        conn = sqlite3.connect(DB_PATH)
        df = pd.read_sql_query("""
            SELECT date, open, high, low, close, volume
            FROM prices
            WHERE ticker = ? AND date >= ? AND date <= ?
            ORDER BY date ASC
        """, conn, params=(ticker, start_date.strftime("%Y-%m-%d"), end_date.strftime("%Y-%m-%d")))
        conn.close()
        
        return df
    
    def _save_chart(self, df: pd.DataFrame, chart_type: str, ticker: str) -> str:
        """Generate and save a chart, returning the relative output path."""
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.dates as mdates
        import matplotlib.pyplot as plt

        has_volume = 'volume' in df.columns
        has_ohlc = all(col in df.columns for col in ['open', 'high', 'low', 'close'])

        chart_path = OUTPUT_DIR / f"chart_{ticker}_{int(time.time())}.png"
        dates = pd.to_datetime(df['date'])

        if chart_type == "volume" and has_volume:
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
            ax1.plot(dates, df['close'], color='#2196F3', linewidth=1.5)
            ax1.set_ylabel('Price ($)', fontsize=12)
            ax1.set_title(f'{ticker} - Closing Price', fontsize=14, fontweight='bold')
            ax1.grid(True, alpha=0.3)
            ax1.fill_between(dates, df['close'], alpha=0.2, color='#2196F3')

            ax2.bar(dates, df['volume'] / 1e6, color='#FF9800', alpha=0.7, width=1)
            ax2.set_ylabel('Volume (Millions)', fontsize=12)
            ax2.set_xlabel('Date', fontsize=12)
            ax2.grid(True, alpha=0.3)
            ax2.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
            ax2.xaxis.set_major_locator(mdates.MonthLocator(interval=1))

        elif chart_type == "candlestick" and has_ohlc:
            fig, ax = plt.subplots(figsize=(14, 7))
            opens = df['open'].values
            highs = df['high'].values
            lows = df['low'].values
            closes = df['close'].values
            colors = ['#26A69A' if close >= open_price else '#EF5350' for close, open_price in zip(closes, opens)]

            for i in range(len(df)):
                ax.plot([dates[i], dates[i]], [lows[i], highs[i]], color=colors[i], linewidth=0.5)
                ax.add_patch(plt.Rectangle(
                    (mdates.date2num(dates[i]) - 0.3, min(opens[i], closes[i])),
                    0.6,
                    abs(closes[i] - opens[i]),
                    facecolor=colors[i],
                    edgecolor=colors[i],
                    linewidth=0.5,
                ))

            ax.set_xlabel('Date', fontsize=12)
            ax.set_ylabel('Price ($)', fontsize=12)
            ax.set_title(f'{ticker} - Candlestick Chart', fontsize=14, fontweight='bold')
            ax.grid(True, alpha=0.3)
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
            ax.xaxis.set_major_locator(mdates.MonthLocator(interval=1))

        else:
            fig, ax = plt.subplots(figsize=(12, 6))
            ax.plot(dates, df['close'], color='#2196F3', linewidth=2, label='Close Price')
            ax.fill_between(dates, df['close'], alpha=0.2, color='#2196F3')
            ax.set_xlabel('Date', fontsize=12)
            ax.set_ylabel('Price ($)', fontsize=12)
            ax.set_title(f'{ticker} - Closing Price History', fontsize=14, fontweight='bold')
            ax.grid(True, alpha=0.3)
            ax.legend(loc='upper left')
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
            ax.xaxis.set_major_locator(mdates.MonthLocator(interval=1))

        plt.xticks(rotation=45)
        plt.tight_layout()
        fig.savefig(chart_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        return f"outputs/{chart_path.name}"
    
    def run(self, query: str, sql_result: str = "") -> str:
        """Generate a chart based on the query.
        
        Args:
            query: Natural language query
            sql_result: Result from SQL agent (optional, for context)
            
        Returns:
            Path to saved chart
        """
        try:
            # Extract ticker and period from query
            ticker = self._extract_ticker_from_query(query)
            if not ticker:
                # Default to AAPL if not found
                ticker = "AAPL"
            
            period = self._extract_period_from_query(query)
            chart_type = self._determine_chart_type(query)
            
            # Get data
            df = self._get_data_for_chart(ticker, period)
            
            if df.empty:
                return "Error: No data available for the specified query."
            
            return self._save_chart(df, chart_type, ticker)
            
        except Exception as e:
            return f"Error generating chart: {str(e)}"


# Global instance (lazy initialization)
_chart_agent: Optional[ChartAgent] = None


def get_chart_agent() -> ChartAgent:
    """Get or create the chart agent instance."""
    global _chart_agent
    if _chart_agent is None:
        _chart_agent = ChartAgent()
    return _chart_agent


def run_chart(state: AgentState) -> Dict[str, Any]:
    """Generate a chart based on SQL query results.
    
    Required signature for orchestrator integration.
    
    Args:
        state: AgentState containing sql_result
        
    Returns:
        Updated state with chart_path
    """
    query = state.get("query", "")
    sql_result = state.get("sql_result", "")
    
    if not query:
        return {
            "chart_path": None,
            "trace_log": append_trace("Chart agent: No query provided"),
        }
    
    try:
        agent = get_chart_agent()
        chart_path = agent.run(query, sql_result)
        return {
            "chart_path": chart_path,
            "trace_log": append_trace(f"Chart agent: Generated chart - {chart_path}"),
        }
    except Exception as e:
        return {
            "chart_path": None,
            "trace_log": append_trace(f"Chart agent: Error - {str(e)[:80]}"),
        }


if __name__ == "__main__":
    # Standalone test
    print("Testing Chart Agent...")
    print("=" * 50)
    
    test_query = "Show AAPL closing prices for last 6 months"
    print(f"Query: {test_query}")
    print("-" * 50)
    
    result = run_chart({"query": test_query, "sql_result": "sample"})
    print(f"Chart path: {result.get('chart_path')}")
