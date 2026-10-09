"""Add conversations and public agent decision journals, without replacing existing data."""
import sqlalchemy as sa

from alembic import op

revision = "0002"
down_revision = "0001"
branch_labels = None
depends_on = None


def upgrade():
    op.create_table("conversations", sa.Column("id", sa.String(32), primary_key=True),
        sa.Column("title", sa.String(100), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False))
    with op.batch_alter_table("queries") as batch:
        batch.add_column(sa.Column("conversation_id", sa.String(32), nullable=True))
        batch.create_foreign_key("fk_queries_conversation", "conversations", ["conversation_id"], ["id"])
        batch.create_index("ix_queries_conversation_id", ["conversation_id"])
    op.create_table("agent_steps", sa.Column("id", sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column("query_id", sa.String(32), sa.ForeignKey("queries.id"), nullable=False),
        sa.Column("kind", sa.String(30), nullable=False), sa.Column("payload", sa.JSON(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False))
    op.create_index("ix_agent_steps_query_id", "agent_steps", ["query_id"])


def downgrade():
    raise RuntimeError("Destructive downgrade is disabled; preserve conversation/research history.")
