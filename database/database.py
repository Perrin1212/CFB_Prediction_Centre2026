from sqlalchemy import create_engine
from sqlalchemy.orm import DeclarativeBase, sessionmaker

from config.settings import DATABASE_PATH, DATABASE_URL


# Make sure the database directory exists.
DATABASE_PATH.parent.mkdir(
    parents=True,
    exist_ok=True,
)


class Base(DeclarativeBase):
    """Base class for all database models."""

    pass


engine = create_engine(
    DATABASE_URL,
    echo=False,
)


SessionLocal = sessionmaker(
    bind=engine,
    autoflush=False,
    autocommit=False,
)


def init_db() -> None:
    """Create all database tables if they do not already exist."""

    # Import models here so SQLAlchemy registers every model
    # with Base.metadata before create_all() runs.
    from database import models  # noqa: F401

    Base.metadata.create_all(
        bind=engine,
    )