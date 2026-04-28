"""SQL Agent — converts natural language to SQL queries using Groq.

This agent takes a user's financial query, generates the appropriate SQL,
executes it against the SQLite database, and returns the results.

Owned by Member 2 (feature/sql-chart branch).
"""
from __future__ import annotations

import sys
import sqlite3
from pathlib import Path
from typing import Any, Dict, Optional

import pandas as pd

# Allow running as a script: `python agents/sql_agent.py`
if __package__ is None and str(Path(__file__).resolve().parents[1]) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from agents.base_agent import append_trace, get_llm, strip_code_fence
from state import AgentState

# Database path
DB_PATH = Path(__file__).parent.parent / "outputs" / "finsight.db"

# Schema description for the LLM
SCHEMA = """
CREATE TABLE prices (
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
);

CREATE TABLE fundamentals (
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
);

Indexes:
- prices(ticker, date)
- fundamentals(ticker, date)

Sample data:
- tickers: AAPL, MSFT, GOOGL, AMZN, NVDA, META, TSLA, JPM, V, JNJ
- date range: 2024-01-01 to present
- prices columns: open, high, low, close, volume, adjusted_close
- fundamentals columns: revenue, net_income, eps, pe_ratio, dividend_yield, book_value
"""


class SQLAgent:
    """LangChain SQL agent using Groq LLM."""
    
    def __init__(self, db_path: Path = DB_PATH):
        """Initialize the SQL agent.
        
        Args:
            db_path: Path to SQLite database
        """
        self.db_path = db_path
        self.llm = get_llm(temperature=0.1)
        
        if db_path.exists():
            # Keep a connection path only; execution is via sqlite3 for reliability.
            pass
        else:
            raise FileNotFoundError(f"Database not found at {db_path}. Run data/db_setup.py first.")
        
        # Get available tickers for context
        self.available_tickers = self._get_tickers()
    
    def _get_tickers(self) -> list[str]:
        """Get list of available tickers from database."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT DISTINCT ticker FROM prices ORDER BY ticker")
        tickers = [row[0] for row in cursor.fetchall()]
        conn.close()
        return tickers
    
    def _build_prompt(self, query: str) -> str:
        """Build the prompt for SQL generation."""
        return f"""You are a financial data analyst. Given a user's natural language query, 
generate a valid SQLite SQL query to answer it.

Database Schema:
{SCHEMA}

Available Tickers: {', '.join(self.available_tickers)}

Rules:
1. Always use UPPER CASE for ticker symbols
2. Return only the SQL query, no explanations
3. Use proper SQL syntax for SQLite
4. For date filters, use YYYY-MM-DD format
5. If the query asks for "last X months", calculate the appropriate date range
6. Always include ORDER BY date ASC for time series data
7. Limit results to 100 rows unless specified otherwise

User Query: {query}

SQL Query:"""
    
    def run(self, query: str) -> str:
        """Execute a natural language query.
        
        Args:
            query: Natural language financial query
            
        Returns:
            Formatted result string with data and metadata
        """
        try:
            # Generate SQL from natural language
            prompt = self._build_prompt(query)
            response = self.llm.invoke(prompt)
            sql_query = strip_code_fence(getattr(response, "content", str(response))).strip()

            # Some models return `sql\nSELECT ...` after fence stripping.
            if sql_query.lower().startswith("sql"):
                sql_query = sql_query[3:].lstrip()
            
            # Execute the SQL query
            conn = sqlite3.connect(self.db_path)
            df = pd.read_sql_query(sql_query, conn)
            conn.close()
            
            if df.empty:
                return "No data found for the given query."
            
            # Format the result
            result = self._format_result(df, sql_query)
            return result
            
        except Exception as e:
            return f"Error executing query: {str(e)}"
    
    def _format_result(self, df: pd.DataFrame, sql: str) -> str:
        """Format the query result for display."""
        result_lines = []
        # Keep output ASCII-only for Windows terminals (avoid UnicodeEncodeError).
        result_lines.append(f"Query Result ({len(df)} rows)")
        result_lines.append("=" * 40)
        result_lines.append(f"SQL: {sql}")
        result_lines.append("")
        
        # Add data preview (first 10 rows)
        preview = df.head(10).to_string(index=False)
        result_lines.append(preview)
        
        if len(df) > 10:
            result_lines.append(f"\n... and {len(df) - 10} more rows")
        
        # Add summary statistics for numeric columns
        numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
        if len(numeric_cols) > 0:
            result_lines.append("\nSummary Statistics:")
            summary = df[numeric_cols].describe().to_string()
            result_lines.append(summary)
        
        return "\n".join(result_lines)
    
    def get_raw_data(self, query: str) -> pd.DataFrame:
        """Get raw DataFrame result (for chart agent).
        
        Args:
            query: Natural language query
            
        Returns:
            DataFrame with query results
        """
        try:
            prompt = self._build_prompt(query)
            response = self.llm.invoke(prompt)
            sql_query = strip_code_fence(getattr(response, "content", str(response))).strip()
            if sql_query.lower().startswith("sql"):
                sql_query = sql_query[3:].lstrip()
            
            conn = sqlite3.connect(self.db_path)
            df = pd.read_sql_query(sql_query, conn)
            conn.close()
            return df
            
        except Exception as e:
            print(f"Error: {e}")
            return pd.DataFrame()


# Global instance (lazy initialization)
_sql_agent: Optional[SQLAgent] = None


def get_sql_agent() -> SQLAgent:
    """Get or create the SQL agent instance."""
    global _sql_agent
    if _sql_agent is None:
        _sql_agent = SQLAgent()
    return _sql_agent


def run_sql(state: AgentState) -> Dict[str, Any]:
    """Execute SQL query based on user's natural language input.
    
    Required signature for orchestrator integration.
    
    Args:
        state: AgentState containing the query
        
    Returns:
        Updated state with sql_result
    """
    query = state.get("query", "")

    # IMPORTANT (LangGraph): return ONLY updated keys to avoid parallel-write
    # collisions on unrelated state keys (e.g., `forecast`).
    if not query:
        return {
            "sql_result": "No query provided.",
            "trace_log": append_trace("SQL agent: No query provided"),
        }

    try:
        agent = get_sql_agent()
        result = agent.run(query)
        return {
            "sql_result": result,
            "trace_log": append_trace(f"SQL agent: Executed query - '{query[:50]}...'"),
        }
    except FileNotFoundError as e:
        return {
            "sql_result": f"Database not initialized. Run data/db_setup.py first. Error: {e}",
            "trace_log": append_trace("SQL agent: Database not found"),
        }
    except Exception as e:
        return {
            "sql_result": f"Error: {str(e)}",
            "trace_log": append_trace(f"SQL agent: Error - {str(e)[:80]}"),
        }


if __name__ == "__main__":
    # Standalone test
    print("Testing SQL Agent...")
    print("=" * 50)
    
    test_query = "Show AAPL closing prices for last 6 months"
    print(f"Query: {test_query}")
    print("-" * 50)
    
    result = run_sql({"query": test_query})
    print(result.get("sql_result", "No result"))
