"""Add income gain return to Stock model

Revision ID: 5b37c7d779db
Revises: 9aa84f0ef442
Create Date: 2024-09-17 19:21:59.035380

"""
from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision = '5b37c7d779db'
down_revision = '9aa84f0ef442'
branch_labels = None
depends_on = None


def upgrade():
    # ### commands auto generated by Alembic - please adjust! ###
    with op.batch_alter_table('stock', schema=None) as batch_op:
        batch_op.add_column(sa.Column('income_gain_pct', sa.Float(), nullable=True))

    # ### end Alembic commands ###


def downgrade():
    # ### commands auto generated by Alembic - please adjust! ###
    with op.batch_alter_table('stock', schema=None) as batch_op:
        batch_op.drop_column('income_gain_pct')

    # ### end Alembic commands ###
