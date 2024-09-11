"""added stockmemo model

Revision ID: 9f04fd4a8406
Revises: e218bfeab751
Create Date: 2024-09-11 21:23:06.744503

"""
from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision = '9f04fd4a8406'
down_revision = 'e218bfeab751'
branch_labels = None
depends_on = None


def upgrade():
    # ### commands auto generated by Alembic - please adjust! ###
    op.create_table('stock_memo',
    sa.Column('id', sa.Integer(), nullable=False),
    sa.Column('ticker', sa.String(length=10), nullable=False),
    sa.Column('memo', sa.Text(), nullable=True),
    sa.PrimaryKeyConstraint('id'),
    sa.UniqueConstraint('ticker')
    )
    # ### end Alembic commands ###


def downgrade():
    # ### commands auto generated by Alembic - please adjust! ###
    op.drop_table('stock_memo')
    # ### end Alembic commands ###
