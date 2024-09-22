"""Add email to User model

Revision ID: 117d40a7f415
Revises: 927570fced1f
Create Date: 2024-09-22 13:36:53.682679

"""
from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision = '117d40a7f415'
down_revision = '927570fced1f'
branch_labels = None
depends_on = None


def upgrade():
    # ### commands auto generated by Alembic - please adjust! ###
    with op.batch_alter_table('user', schema=None) as batch_op:
        batch_op.add_column(sa.Column('email', sa.String(length=120), nullable=True))
        batch_op.create_unique_constraint(None, ['email'])

    # ### end Alembic commands ###


def downgrade():
    # ### commands auto generated by Alembic - please adjust! ###
    with op.batch_alter_table('user', schema=None) as batch_op:
        batch_op.drop_constraint(None, type_='unique')
        batch_op.drop_column('email')

    # ### end Alembic commands ###
