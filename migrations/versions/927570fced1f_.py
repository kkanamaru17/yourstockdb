"""empty message

Revision ID: 927570fced1f
Revises: 5b37c7d779db
Create Date: 2024-09-21 22:24:26.563369

"""
from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision = '927570fced1f'
down_revision = '5b37c7d779db'
branch_labels = None
depends_on = None


def upgrade():
    # ### commands auto generated by Alembic - please adjust! ###
    op.create_table('o_auth',
    sa.Column('id', sa.Integer(), nullable=False),
    sa.Column('provider', sa.String(length=50), nullable=False),
    sa.Column('provider_user_id', sa.String(length=256), nullable=False),
    sa.Column('user_id', sa.Integer(), nullable=False),
    sa.ForeignKeyConstraint(['user_id'], ['user.id'], ),
    sa.PrimaryKeyConstraint('id'),
    sa.UniqueConstraint('provider_user_id')
    )
    # ### end Alembic commands ###


def downgrade():
    # ### commands auto generated by Alembic - please adjust! ###
    op.drop_table('o_auth')
    # ### end Alembic commands ###