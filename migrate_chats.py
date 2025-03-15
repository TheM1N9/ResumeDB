from app import app, db, Chat, ConversationHistory
from sqlalchemy import text


def migrate_existing_messages():
    with app.app_context():
        # First, ensure all tables exist
        db.create_all()

        try:
            # Check if we need to add chat_id column
            with db.engine.connect() as conn:
                # Get column info
                columns = conn.execute(
                    text("PRAGMA table_info(conversation_history)")
                ).fetchall()
                column_names = [col[1] for col in columns]

                # Add chat_id column if it doesn't exist
                if "chat_id" not in column_names:
                    conn.execute(
                        text(
                            """
                        ALTER TABLE conversation_history 
                        ADD COLUMN chat_id INTEGER REFERENCES chat(id)
                    """
                        )
                    )
                    print("Added chat_id column to conversation_history table")

            # Get all users with messages
            result = db.session.execute(
                text(
                    """
                SELECT DISTINCT user_id 
                FROM conversation_history 
                WHERE chat_id IS NULL
            """
                )
            )
            user_ids = [row[0] for row in result]
            print(f"Found {len(user_ids)} users with messages to migrate")

            for user_id in user_ids:
                # Create a new chat for existing messages
                chat = Chat(user_id=user_id, title="Previous Conversations")
                db.session.add(chat)
                db.session.commit()
                print(f"Created new chat for user {user_id}")

                # Update existing messages to belong to the new chat
                db.session.execute(
                    text(
                        """
                    UPDATE conversation_history 
                    SET chat_id = :chat_id 
                    WHERE user_id = :user_id AND chat_id IS NULL
                """
                    ),
                    {"chat_id": chat.id, "user_id": user_id},
                )
                print(f"Updated messages for user {user_id} to chat {chat.id}")

            db.session.commit()
            print("Migration completed successfully!")

        except Exception as e:
            print(f"Error during migration: {e}")
            db.session.rollback()
            raise


if __name__ == "__main__":
    migrate_existing_messages()
