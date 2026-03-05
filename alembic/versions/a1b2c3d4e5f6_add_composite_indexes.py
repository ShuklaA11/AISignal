"""add composite indexes

Revision ID: a1b2c3d4e5f6
Revises: 4f386ad98446
Create Date: 2026-02-22 00:00:00.000000

"""
from typing import Sequence, Union

from alembic import op


# revision identifiers, used by Alembic.
revision: str = 'a1b2c3d4e5f6'
down_revision: Union[str, Sequence[str], None] = '4f386ad98446'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Add composite indexes for frequently queried column pairs."""
    with op.batch_alter_table('feed_impressions', schema=None) as batch_op:
        batch_op.create_index('ix_feed_impressions_user_article', ['user_id', 'article_id'])

    with op.batch_alter_table('digests', schema=None) as batch_op:
        batch_op.create_index('ix_digests_user_date', ['user_id', 'digest_date'])

    with op.batch_alter_table('scoring_metrics', schema=None) as batch_op:
        batch_op.create_index('ix_scoring_metrics_user_date', ['user_id', 'metric_date'])

    with op.batch_alter_table('digest_clicks', schema=None) as batch_op:
        batch_op.create_index('ix_digest_clicks_user_digest', ['user_id', 'digest_id'])


def downgrade() -> None:
    """Drop composite indexes."""
    with op.batch_alter_table('digest_clicks', schema=None) as batch_op:
        batch_op.drop_index('ix_digest_clicks_user_digest')

    with op.batch_alter_table('scoring_metrics', schema=None) as batch_op:
        batch_op.drop_index('ix_scoring_metrics_user_date')

    with op.batch_alter_table('digests', schema=None) as batch_op:
        batch_op.drop_index('ix_digests_user_date')

    with op.batch_alter_table('feed_impressions', schema=None) as batch_op:
        batch_op.drop_index('ix_feed_impressions_user_article')
