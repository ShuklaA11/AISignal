"""add section, audience_tags, quality_weight to articles and sources

Revision ID: b2e7f1a3c4d5
Revises: 65cd4d5a9a77
Create Date: 2026-05-07 00:00:00.000000

"""
from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op


revision: str = "b2e7f1a3c4d5"
down_revision: Union[str, Sequence[str], None] = "65cd4d5a9a77"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Add section/audience_tags/quality_weight to articles and sources."""
    with op.batch_alter_table("articles", schema=None) as batch_op:
        batch_op.add_column(sa.Column("section", sa.String(), nullable=True))
        batch_op.add_column(
            sa.Column(
                "audience_tags_json",
                sa.String(),
                nullable=False,
                server_default="[]",
            )
        )
        batch_op.add_column(
            sa.Column(
                "quality_weight",
                sa.Float(),
                nullable=False,
                server_default="1.0",
            )
        )
        batch_op.create_index("ix_articles_section", ["section"])

    with op.batch_alter_table("sources", schema=None) as batch_op:
        batch_op.add_column(sa.Column("section", sa.String(), nullable=True))
        batch_op.add_column(
            sa.Column(
                "audience_tags_json",
                sa.String(),
                nullable=False,
                server_default="[]",
            )
        )
        batch_op.add_column(
            sa.Column(
                "quality_weight",
                sa.Float(),
                nullable=False,
                server_default="1.0",
            )
        )


def downgrade() -> None:
    """Drop section/audience_tags/quality_weight columns."""
    with op.batch_alter_table("sources", schema=None) as batch_op:
        batch_op.drop_column("quality_weight")
        batch_op.drop_column("audience_tags_json")
        batch_op.drop_column("section")

    with op.batch_alter_table("articles", schema=None) as batch_op:
        batch_op.drop_index("ix_articles_section")
        batch_op.drop_column("quality_weight")
        batch_op.drop_column("audience_tags_json")
        batch_op.drop_column("section")
