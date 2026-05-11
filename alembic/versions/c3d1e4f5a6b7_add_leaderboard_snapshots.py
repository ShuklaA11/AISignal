"""add leaderboard_snapshots table

Revision ID: c3d1e4f5a6b7
Revises: b2e7f1a3c4d5
Create Date: 2026-05-11 00:00:00.000000

"""
from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op


revision: str = "c3d1e4f5a6b7"
down_revision: Union[str, Sequence[str], None] = "b2e7f1a3c4d5"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Create leaderboard_snapshots for time-series ranking data.

    One row = one provider's full ranking at one point in time, for one
    metric (Elo, throughput, $/Mtok, etc.). The rankings list is stored as
    JSON to keep the schema flexible across heterogeneous providers.
    """
    op.create_table(
        "leaderboard_snapshots",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column("provider", sa.String(), nullable=False),
        sa.Column("metric", sa.String(), nullable=False),
        sa.Column("captured_at", sa.DateTime(), nullable=False),
        sa.Column("rankings_json", sa.Text(), nullable=False, server_default="[]"),
        sa.Column("source_url", sa.String(), nullable=True),
        sa.Column("notes", sa.String(), nullable=True),
    )
    with op.batch_alter_table("leaderboard_snapshots", schema=None) as batch_op:
        batch_op.create_index(
            "ix_leaderboard_snapshots_provider_metric_captured",
            ["provider", "metric", "captured_at"],
        )
        batch_op.create_index(
            "ix_leaderboard_snapshots_captured_at", ["captured_at"]
        )


def downgrade() -> None:
    with op.batch_alter_table("leaderboard_snapshots", schema=None) as batch_op:
        batch_op.drop_index("ix_leaderboard_snapshots_captured_at")
        batch_op.drop_index("ix_leaderboard_snapshots_provider_metric_captured")
    op.drop_table("leaderboard_snapshots")
