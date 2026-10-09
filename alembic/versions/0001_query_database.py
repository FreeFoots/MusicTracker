"""Initial query/metadata/observation schema; preserve existing validated initial tables."""
import sqlalchemy as sa

from alembic import op

revision = "0001"
down_revision = None
branch_labels = None
depends_on = None


def upgrade():
    definitions = {
        "queries": [sa.Column("id", sa.String(32), primary_key=True), sa.Column("input", sa.Text(), nullable=False),
            sa.Column("request", sa.JSON(), nullable=False), sa.Column("request_hash", sa.String(64), nullable=False),
            sa.Column("idempotency_key", sa.String(128)), sa.Column("parent_id", sa.String(32), sa.ForeignKey("queries.id")),
            sa.Column("status", sa.String(30), nullable=False), sa.Column("persistence_status", sa.String(20), nullable=False),
            sa.Column("plan", sa.JSON(), nullable=False), sa.Column("checkpoint", sa.JSON(), nullable=False),
            sa.Column("report", sa.JSON(), nullable=False), sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
            sa.Column("finished_at", sa.DateTime(timezone=True)), sa.Column("lease_until", sa.DateTime(timezone=True)),
            sa.Column("lease_owner", sa.String(32)), sa.Column("error", sa.Text()), sa.UniqueConstraint("idempotency_key")],
        "entities": [sa.Column("id", sa.String(32), primary_key=True), sa.Column("platform", sa.String(20), nullable=False),
            sa.Column("platform_id", sa.String(40), nullable=False), sa.Column("title", sa.Text(), nullable=False),
            sa.Column("artists", sa.JSON(), nullable=False), sa.Column("metadata_json", sa.JSON(), nullable=False),
            sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False), sa.UniqueConstraint("platform", "platform_id")],
        "tool_calls": [sa.Column("id", sa.String(32), primary_key=True), sa.Column("query_id", sa.String(32), sa.ForeignKey("queries.id"), nullable=False),
            sa.Column("tool", sa.String(80), nullable=False), sa.Column("arguments", sa.JSON(), nullable=False),
            sa.Column("status", sa.String(30), nullable=False), sa.Column("error_code", sa.String(80)),
            sa.Column("duration_ms", sa.Integer(), nullable=False), sa.Column("created_at", sa.DateTime(timezone=True), nullable=False)],
        "events": [sa.Column("id", sa.Integer(), primary_key=True, autoincrement=True),
            sa.Column("query_id", sa.String(32), sa.ForeignKey("queries.id"), nullable=False), sa.Column("type", sa.String(50), nullable=False),
            sa.Column("payload", sa.JSON(), nullable=False), sa.Column("created_at", sa.DateTime(timezone=True), nullable=False)],
        "query_results": [sa.Column("id", sa.String(32), primary_key=True), sa.Column("query_id", sa.String(32), sa.ForeignKey("queries.id"), nullable=False),
            sa.Column("entity_id", sa.String(32), sa.ForeignKey("entities.id"), nullable=False), sa.Column("snapshot", sa.JSON(), nullable=False),
            sa.Column("origin", sa.String(20), nullable=False), sa.Column("reused_from_query_id", sa.String(32), sa.ForeignKey("queries.id")),
            sa.UniqueConstraint("query_id", "entity_id")],
        "observations": [sa.Column("id", sa.String(32), primary_key=True), sa.Column("query_id", sa.String(32), sa.ForeignKey("queries.id"), nullable=False),
            sa.Column("entity_id", sa.String(32), sa.ForeignKey("entities.id"), nullable=False), sa.Column("attempt_id", sa.String(32), sa.ForeignKey("tool_calls.id")),
            sa.Column("metric", sa.String(50), nullable=False), sa.Column("value", sa.Float()), sa.Column("status", sa.String(20), nullable=False),
            sa.Column("data", sa.JSON(), nullable=False), sa.Column("collected_at", sa.DateTime(timezone=True), nullable=False),
            sa.UniqueConstraint("query_id", "entity_id", "metric")],
        "report_versions": [sa.Column("id", sa.String(32), primary_key=True), sa.Column("query_id", sa.String(32), sa.ForeignKey("queries.id"), nullable=False),
            sa.Column("report", sa.JSON(), nullable=False), sa.Column("created_at", sa.DateTime(timezone=True), nullable=False)],
    }
    inspector = sa.inspect(op.get_bind())
    for name, columns in definitions.items():
        if inspector.has_table(name):
            actual = {c["name"] for c in inspector.get_columns(name)}
            expected = {c.name for c in columns if isinstance(c, sa.Column)}
            if actual != expected:
                raise RuntimeError(f"Existing table {name} differs from initial schema; manual migration required")
            continue
        op.create_table(name, *columns)
        for column in ["query_id", "entity_id"]:
            if any(isinstance(c, sa.Column) and c.name == column for c in columns):
                op.create_index(f"ix_{name}_{column}", name, [column])
        if name == "queries":
            op.create_index("ix_queries_status", name, ["status"])


def downgrade():
    raise RuntimeError("Downgrade deletes research data; intentionally disabled. Use a reviewed migration instead.")
