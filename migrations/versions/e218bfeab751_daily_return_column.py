"""daily return column

Revision ID: e218bfeab751
Revises: a62075b3aca3
Create Date: 2024-09-08 21:38:08.811901

"""
from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision = 'e218bfeab751'
down_revision = 'a62075b3aca3'
branch_labels = None
depends_on = None


def upgrade():
    # ### commands auto generated by Alembic - please adjust! ###
    with op.batch_alter_table('stock', schema=None) as batch_op:
        batch_op.add_column(sa.Column('daily_return', sa.Float(), nullable=True))

    # ### end Alembic commands ###


def downgrade():
    # ### commands auto generated by Alembic - please adjust! ###
    with op.batch_alter_table('stock', schema=None) as batch_op:
        batch_op.drop_column('daily_return')

    # ### end Alembic commands ###