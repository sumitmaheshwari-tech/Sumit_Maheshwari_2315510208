import duckdb
from pathlib import Path
from typing import Tuple, Any, Optional, List, Dict
import json

class DBExecutor:
    """
    In-memory analytical query executor using DuckDB.
    Sets up views for sales_data, targets, and enriched sales analytics view.
    """
    def __init__(self, dataset_dir: Path):
        self.dataset_dir = Path(dataset_dir)
        self.sales_csv = (self.dataset_dir / "sales_data.csv").resolve().as_posix()
        self.targets_csv = (self.dataset_dir / "targets.csv").resolve().as_posix()
        self.conn = duckdb.connect(database=":memory:")
        self._init_database()

    def _init_database(self):
        # 1. Base table: sales_data
        self.conn.execute(f"""
            CREATE TABLE sales_data AS 
            SELECT * FROM read_csv_auto('{self.sales_csv}', header=True);
        """)

        # 2. Base table: targets
        self.conn.execute(f"""
            CREATE TABLE targets AS 
            SELECT * FROM read_csv_auto('{self.targets_csv}', header=True);
        """)

        # 3. Enriched analytical view: sales
        self.conn.execute("""
            CREATE VIEW sales AS
            SELECT 
                order_id,
                order_date,
                strftime(order_date::DATE, '%Y-%m') AS month,
                EXTRACT(YEAR FROM order_date::DATE)::INTEGER AS year,
                'Q' || EXTRACT(QUARTER FROM order_date::DATE)::VARCHAR AS quarter,
                region,
                country,
                city,
                customer_id,
                customer_segment,
                product_category,
                product_subcategory,
                product_name,
                quantity,
                unit_price,
                discount,
                shipping_cost,
                profit,
                ROUND(quantity * unit_price * (1.0 - discount), 4) AS revenue
            FROM sales_data;
        """)

    def validate_query(self, sql: str) -> Tuple[bool, Optional[str]]:
        """
        Validates SQL syntax and schema without executing side effects.
        """
        clean_sql = sql.strip().rstrip(";")
        try:
            self.conn.execute(f"EXPLAIN {clean_sql}")
            return True, None
        except Exception as e:
            return False, str(e)

    def execute_query(self, sql: str) -> Tuple[bool, Any, Optional[str]]:
        """
        Executes a SQL query in-memory and returns:
        (success, serializable_result, error_message)
        """
        clean_sql = sql.strip().rstrip(";")
        try:
            rel = self.conn.execute(clean_sql)
            columns = [desc[0] for desc in rel.description] if rel.description else []
            rows = rel.fetchall()

            if not rows:
                return True, [], None

            # If scalar (1 row, 1 column), simplify
            if len(rows) == 1 and len(columns) == 1:
                val = rows[0][0]
                if isinstance(val, (int, float)):
                    return True, round(val, 2) if isinstance(val, float) else val, None
                return True, val, None

            # Return list of dictionaries
            result = []
            for row in rows:
                row_dict = {}
                for col_name, val in zip(columns, row):
                    if isinstance(val, float):
                        row_dict[col_name] = round(val, 2)
                    else:
                        row_dict[col_name] = val
                result.append(row_dict)

            return True, result, None

        except Exception as e:
            return False, None, str(e)

    def close(self):
        try:
            self.conn.close()
        except Exception:
            pass
