"""add signal_counts_json to user_ml_profiles

Revision ID: 65cd4d5a9a77
Revises: a1b2c3d4e5f6
Create Date: 2026-02-23 19:00:17.113871

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = '65cd4d5a9a77'
down_revision: Union[str, Sequence[str], None] = 'a1b2c3d4e5f6'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Add signal_counts_json for confidence-aware weight decay."""
    with op.batch_alter_table('user_ml_profiles', schema=None) as batch_op:
        batch_op.add_column(
            sa.Column('signal_counts_json', sa.String(), nullable=False, server_default='{}')
        )


def downgrade() -> None:
    """Remove signal_counts_json."""
    with op.batch_alter_table('user_ml_profiles', schema=None) as batch_op:
        batch_op.drop_column('signal_counts_json')
