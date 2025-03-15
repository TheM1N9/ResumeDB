from app import app, db
from datetime import datetime
from sqlalchemy import text


def create_tables():
    with app.app_context():
        # Create all tables
        db.create_all()

        # Verify tables were created and show their structure
        with db.engine.connect() as conn:
            # List all tables
            tables = db.inspect(db.engine).get_table_names()
            print("\nCreated tables:", tables)

            # Show structure of each table
            for table in tables:
                columns = conn.execute(text(f"PRAGMA table_info({table})")).fetchall()
                print(f"\nStructure of {table} table:")
                for col in columns:
                    print(f"  {col[1]}: {col[2]} ({'NOT NULL' if col[3] else 'NULL'})")


if __name__ == "__main__":
    create_tables()
    print("\nDatabase tables created successfully!")
