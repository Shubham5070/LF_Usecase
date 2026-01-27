# db_migration.py
"""
Database Migration Tool - PostgreSQL to SQLite
Exports data from PostgreSQL and imports into SQLite
"""

import psycopg2
import sqlite3
from pathlib import Path
from typing import List, Dict, Any
import json
from datetime import datetime, date, time
from decimal import Decimal
from dotenv import load_dotenv
import os

load_dotenv()

# PostgreSQL Configuration
PG_CONFIG = {
    "host": os.getenv("DB_HOST", "localhost"),
    "port": int(os.getenv("DB_PORT", "5432")),
    "dbname": os.getenv("DB_NAME", "load_forecasting"),
    "user": os.getenv("DB_USER", "postgres"),
    "password": os.getenv("DB_PASSWORD", "")
}

# SQLite Configuration
SQLITE_DB_PATH = Path("./data/load_forecasting.db")
SQLITE_DB_PATH.parent.mkdir(exist_ok=True)


class DatabaseMigrator:
    """Migrates data from PostgreSQL to SQLite"""
    
    def __init__(self):
        self.pg_conn = None
        self.sqlite_conn = None
        
    def connect_postgresql(self):
        """Connect to PostgreSQL database"""
        print("[MIGRATION] Connecting to PostgreSQL...")
        self.pg_conn = psycopg2.connect(**PG_CONFIG)
        print("[MIGRATION] ✅ PostgreSQL connected")
        
    def connect_sqlite(self):
        """Connect to SQLite database"""
        print(f"[MIGRATION] Connecting to SQLite at {SQLITE_DB_PATH}...")
        self.sqlite_conn = sqlite3.connect(str(SQLITE_DB_PATH))
        print("[MIGRATION] ✅ SQLite connected")
        
    def get_postgresql_tables(self) -> List[str]:
        """Get all tables in lf schema from PostgreSQL (excluding temp tables)"""
        with self.pg_conn.cursor() as cur:
            cur.execute("""
                SELECT table_name 
                FROM information_schema.tables 
                WHERE table_schema = 'lf'
                  AND table_name NOT LIKE 'tmp_%'
                ORDER BY table_name;
            """)
            tables = [row[0] for row in cur.fetchall()]
        
        print(f"[MIGRATION] Found {len(tables)} tables in PostgreSQL: {tables}")
        return tables
    
    def get_table_schema(self, table_name: str) -> List[Dict[str, Any]]:
        """Get column definitions from PostgreSQL table"""
        with self.pg_conn.cursor() as cur:
            cur.execute(f"""
                SELECT 
                    column_name, 
                    data_type, 
                    is_nullable,
                    column_default
                FROM information_schema.columns 
                WHERE table_schema = 'lf' 
                  AND table_name = '{table_name}'
                ORDER BY ordinal_position;
            """)
            
            columns = []
            for row in cur.fetchall():
                columns.append({
                    'name': row[0],
                    'type': self._convert_pg_type_to_sqlite(row[1]),
                    'nullable': row[2] == 'YES',
                    'default': row[3]
                })
        
        return columns
    
    def _convert_pg_type_to_sqlite(self, pg_type: str) -> str:
        """Convert PostgreSQL data type to SQLite type"""
        type_mapping = {
            'integer': 'INTEGER',
            'bigint': 'INTEGER',
            'smallint': 'INTEGER',
            'serial': 'INTEGER',
            'bigserial': 'INTEGER',
            'double precision': 'REAL',
            'real': 'REAL',
            'numeric': 'REAL',
            'decimal': 'REAL',
            'character varying': 'TEXT',
            'varchar': 'TEXT',
            'character': 'TEXT',
            'char': 'TEXT',
            'text': 'TEXT',
            'timestamp without time zone': 'TEXT',
            'timestamp with time zone': 'TEXT',
            'date': 'TEXT',
            'time': 'TEXT',
            'time without time zone': 'TEXT',
            'time with time zone': 'TEXT',
            'boolean': 'INTEGER',
            'json': 'TEXT',
            'jsonb': 'TEXT',
            'uuid': 'TEXT'
        }
        
        return type_mapping.get(pg_type.lower(), 'TEXT')
    
    def _convert_value_for_sqlite(self, value: Any) -> Any:
        """
        Convert PostgreSQL value to SQLite-compatible type
        
        SQLite only supports: NULL, INTEGER, REAL, TEXT, BLOB
        """
        if value is None:
            return None
        
        # Handle datetime types
        if isinstance(value, datetime):
            return value.isoformat()
        
        if isinstance(value, date):
            return value.isoformat()
        
        if isinstance(value, time):
            return value.isoformat()
        
        # Handle numeric types
        if isinstance(value, Decimal):
            return float(value)
        
        # Handle boolean
        if isinstance(value, bool):
            return 1 if value else 0
        
        # Handle bytes/memoryview
        if isinstance(value, (bytes, memoryview)):
            return bytes(value)
        
        # Handle lists/dicts (JSON)
        if isinstance(value, (list, dict)):
            return json.dumps(value)
        
        # Everything else
        return value
    
    def create_sqlite_table(self, table_name: str, columns: List[Dict[str, Any]]):
        """Create table in SQLite"""
        col_defs = []
        for col in columns:
            col_def = f"{col['name']} {col['type']}"
            if not col['nullable']:
                col_def += " NOT NULL"
            col_defs.append(col_def)
        
        create_sql = f"""
        CREATE TABLE IF NOT EXISTS {table_name} (
            {', '.join(col_defs)}
        );
        """
        
        print(f"[MIGRATION] Creating table: {table_name}")
        self.sqlite_conn.execute(create_sql)
        self.sqlite_conn.commit()
    
    def migrate_table_data(self, table_name: str, batch_size: int = 1000):
        """Migrate data from PostgreSQL to SQLite"""
        print(f"[MIGRATION] Migrating data for table: {table_name}")
        
        # Get total row count
        with self.pg_conn.cursor() as cur:
            cur.execute(f"SELECT COUNT(*) FROM lf.{table_name}")
            total_rows = cur.fetchone()[0]
        
        if total_rows == 0:
            print(f"[MIGRATION] ⚠️  Table {table_name} is empty, skipping data migration")
            return
        
        print(f"[MIGRATION] Total rows to migrate: {total_rows:,}")
        
        # Get column names
        with self.pg_conn.cursor() as cur:
            cur.execute(f"SELECT * FROM lf.{table_name} LIMIT 0")
            columns = [desc[0] for desc in cur.description]
        
        # Migrate in batches
        offset = 0
        migrated = 0
        errors = 0
        
        while offset < total_rows:
            with self.pg_conn.cursor() as cur:
                # Order by a safe column if exists
                order_clause = "ORDER BY entrydatetime" if "entrydatetime" in columns else ""
                cur.execute(f"""
                    SELECT * FROM lf.{table_name} 
                    {order_clause}
                    LIMIT {batch_size} OFFSET {offset}
                """)
                rows = cur.fetchall()
            
            if not rows:
                break
            
            # Insert into SQLite
            placeholders = ', '.join(['?' for _ in columns])
            insert_sql = f"""
                INSERT INTO {table_name} ({', '.join(columns)}) 
                VALUES ({placeholders})
            """
            
            # Convert data types for SQLite
            converted_rows = []
            for row in rows:
                try:
                    converted_row = tuple(self._convert_value_for_sqlite(value) for value in row)
                    converted_rows.append(converted_row)
                except Exception as e:
                    print(f"[MIGRATION] ⚠️  Error converting row: {e}")
                    errors += 1
                    continue
            
            # Execute batch insert
            try:
                self.sqlite_conn.executemany(insert_sql, converted_rows)
                self.sqlite_conn.commit()
                
                migrated += len(converted_rows)
                offset += batch_size
                
                print(f"[MIGRATION] Progress: {migrated:,}/{total_rows:,} ({migrated/total_rows*100:.1f}%)")
                
            except Exception as e:
                print(f"[MIGRATION] ⚠️  Batch insert error: {e}")
                print(f"[MIGRATION] Trying row-by-row insert for this batch...")
                
                # Fallback: insert row by row
                for converted_row in converted_rows:
                    try:
                        self.sqlite_conn.execute(insert_sql, converted_row)
                        self.sqlite_conn.commit()
                        migrated += 1
                    except Exception as row_error:
                        print(f"[MIGRATION] ⚠️  Failed to insert row: {row_error}")
                        errors += 1
                
                offset += batch_size
        
        if errors > 0:
            print(f"[MIGRATION] ⚠️  Completed with {errors} errors: {migrated:,}/{total_rows:,} rows migrated for {table_name}")
        else:
            print(f"[MIGRATION] ✅ Completed: {migrated:,} rows migrated for {table_name}")
    
    def create_indexes(self, table_name: str):
        """Create indexes in SQLite for better performance"""
        index_configs = {
            't_actual_demand': [
                ('idx_actual_date', 'date'),
                ('idx_actual_datetime', 'datetime'),
                ('idx_actual_block', 'block')
            ],
            't_forecasted_demand': [
                ('idx_forecast_date', 'date'),
                ('idx_forecast_datetime', 'datetime'),
                ('idx_forecast_block', 'block'),
                ('idx_forecast_model', 'model_id')
            ],
            't_holidays': [
                ('idx_holidays_date', 'date')
            ],
            't_metrics': [
                ('idx_metrics_date', 'date'),
                ('idx_metrics_model', 'model_id')
            ],
            't_actual_weather': [
                ('idx_actual_weather_date', 'date'),
                ('idx_actual_weather_datetime', 'datetime')
            ],
            't_forecasted_weather': [
                ('idx_forecast_weather_date', 'date'),
                ('idx_forecast_weather_datetime', 'datetime')
            ]
        }
        
        if table_name in index_configs:
            print(f"[MIGRATION] Creating indexes for {table_name}...")
            for idx_name, column in index_configs[table_name]:
                try:
                    self.sqlite_conn.execute(f"""
                        CREATE INDEX IF NOT EXISTS {idx_name} 
                        ON {table_name}({column})
                    """)
                    print(f"[MIGRATION]   ✅ Created index: {idx_name}")
                except Exception as e:
                    print(f"[MIGRATION]   ⚠️  Failed to create index {idx_name}: {e}")
            
            self.sqlite_conn.commit()
    
    def verify_migration(self, table_name: str) -> Dict[str, int]:
        """Verify row counts match between PostgreSQL and SQLite"""
        # PostgreSQL count
        with self.pg_conn.cursor() as cur:
            cur.execute(f"SELECT COUNT(*) FROM lf.{table_name}")
            pg_count = cur.fetchone()[0]
        
        # SQLite count
        sqlite_cur = self.sqlite_conn.execute(f"SELECT COUNT(*) FROM {table_name}")
        sqlite_count = sqlite_cur.fetchone()[0]
        
        match = pg_count == sqlite_count
        status = "✅" if match else "❌"
        
        print(f"[VERIFICATION] {status} {table_name}: PG={pg_count:,}, SQLite={sqlite_count:,}")
        
        return {
            'table': table_name,
            'postgresql': pg_count,
            'sqlite': sqlite_count,
            'match': match
        }
    
    def migrate_all(self):
        """Complete migration process"""
        print("\n" + "="*60)
        print("DATABASE MIGRATION: PostgreSQL → SQLite")
        print("="*60 + "\n")
        
        try:
            # Connect to databases
            self.connect_postgresql()
            self.connect_sqlite()
            
            # Get tables (exclude temp tables)
            tables = self.get_postgresql_tables()
            
            if not tables:
                print("[MIGRATION] ⚠️  No tables found in lf schema")
                return
            
            verification_results = []
            
            # Migrate each table
            for table_name in tables:
                print(f"\n{'-'*60}")
                print(f"Processing: {table_name}")
                print(f"{'-'*60}")
                
                try:
                    # Get schema
                    columns = self.get_table_schema(table_name)
                    
                    # Create table
                    self.create_sqlite_table(table_name, columns)
                    
                    # Migrate data
                    self.migrate_table_data(table_name)
                    
                    # Create indexes
                    self.create_indexes(table_name)
                    
                    # Verify
                    result = self.verify_migration(table_name)
                    verification_results.append(result)
                    
                except Exception as e:
                    print(f"[MIGRATION] ❌ Failed to migrate {table_name}: {e}")
                    import traceback
                    traceback.print_exc()
            
            # Final summary
            print("\n" + "="*60)
            print("MIGRATION SUMMARY")
            print("="*60)
            
            all_match = all(r['match'] for r in verification_results)
            
            for result in verification_results:
                status = "✅" if result['match'] else "❌"
                print(f"{status} {result['table']}: {result['sqlite']:,} rows")
            
            if all_match:
                print("\n✅ Migration completed successfully!")
                print(f"📁 SQLite database: {SQLITE_DB_PATH}")
                print(f"\nTo use SQLite, update your .env file:")
                print(f"   DB_TYPE=sqlite")
                print(f"   SQLITE_DB_PATH={SQLITE_DB_PATH}")
            else:
                print("\n⚠️  Migration completed with mismatches!")
                print("Check the logs above for details.")
            
        except Exception as e:
            print(f"\n❌ Migration failed: {e}")
            import traceback
            traceback.print_exc()
            
        finally:
            if self.pg_conn:
                self.pg_conn.close()
            if self.sqlite_conn:
                self.sqlite_conn.close()


def main():
    """Run the migration"""
    migrator = DatabaseMigrator()
    migrator.migrate_all()


if __name__ == "__main__":
    main()