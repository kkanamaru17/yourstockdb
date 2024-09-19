"""Add purchase_date to Stock model

Revision ID: fde6d94e06c7
Revises: d31e26c15440
Create Date: 2024-09-16 19:12:04.198108

"""
from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision = 'fde6d94e06c7'
down_revision = 'd31e26c15440'
branch_labels = None
depends_on = None


def upgrade():
    # ### commands auto generated by Alembic - please adjust! ###
    with op.batch_alter_table('stock', schema=None) as batch_op:
        batch_op.add_column(sa.Column('purchase_date', sa.Date(), nullable=False))

    # ### end Alembic commands ###


def downgrade():
    # ### commands auto generated by Alembic - please adjust! ###
    with op.batch_alter_table('stock', schema=None) as batch_op:
        batch_op.drop_column('purchase_date')

    # ### end Alembic commands ###