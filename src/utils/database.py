import os
import pandas as pd
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker
from dotenv import load_dotenv

load_dotenv()

class Database:
    def __init__(self):
        self.db_url = (
            f"postgresql://{os.getenv('DB_USER')}:{os.getenv('DB_PASSWORD')}"
            f"@{os.getenv('DB_HOST')}:{os.getenv('DB_PORT')}/{os.getenv('DB_NAME')}"
        )
        self.engine = create_engine(self.db_url)
        self.SessionLocal = sessionmaker(bind=self.engine)
    
    def get_session(self):
        return self.SessionLocal()
    
    def execute_query(self, query):
        with self.engine.connect() as conn:
            result = conn.execute(text(query))
            return result.fetchall()
    
    def insert_dataframe(self, df, table_name, if_exists='append'):
        """Insert pandas DataFrame into database table"""
        df.to_sql(table_name, self.engine, if_exists=if_exists, index=False)
        print(f"✅ Inserted {len(df)} rows into {table_name}")
    
    def read_table(self, table_name, limit=None):
        """Read table into pandas DataFrame"""
        query = f"SELECT * FROM {table_name}"
        if limit:
            query += f" LIMIT {limit}"
        return pd.read_sql(query, self.engine)
    
    def test_connection(self):
        """Test database connection"""
        try:
            result = self.execute_query("SELECT version()")
            print("✅ Database connection successful")
            print(f"PostgreSQL version: {result[0][0][:50]}...")
            return True
        except Exception as e:
            print(f"❌ Database connection failed: {e}")
            return False

# Singleton instance
db = Database()

if __name__ == "__main__":
    db.test_connection()
