from sqlalchemy import create_engine

from alembic import context
from musictracker.config import settings
from musictracker.models import Base

config = context.config
connection = config.attributes.get("connection")


def migrate(conn):
    context.configure(connection=conn, target_metadata=Base.metadata, compare_type=True)
    with context.begin_transaction():
        context.run_migrations()


if context.is_offline_mode():
    context.configure(url=settings.database_url, target_metadata=Base.metadata, literal_binds=True)
    with context.begin_transaction():
        context.run_migrations()
elif connection is not None:
    migrate(connection)
else:
    with create_engine(settings.database_url).connect() as conn:
        migrate(conn)
