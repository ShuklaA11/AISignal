"""add why_it_matters to articles

Revision ID: d4a2b8c6e1f9
Revises: c3d1e4f5a6b7
Create Date: 2026-08-01 00:00:00.000000

"""

from typing import Sequence, Union

import sqlalchemy as sa

from alembic import op

revision: str = "d4a2b8c6e1f9"
down_revision: Union[str, Sequence[str], None] = "c3d1e4f5a6b7"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Add the one-line why_it_matters column to articles.

    Nullable with no default: articles processed before this field existed
    keep NULL, and the digest template falls back to the role summary.
    """
    with op.batch_alter_table("articles", schema=None) as batch_op:
        batch_op.add_column(sa.Column("why_it_matters", sa.String(), nullable=True))


def downgrade() -> None:
    """Drop the why_it_matters column."""
    with op.batch_alter_table("articles", schema=None) as batch_op:
        batch_op.drop_column("why_it_matters")
