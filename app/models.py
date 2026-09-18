from datetime import datetime
from sqlalchemy import Column, Integer, String, Text, DateTime, ForeignKey, ARRAY
from sqlalchemy.orm import relationship
from pgvector.sqlalchemy import Vector

from app.database import Base

EMBED_DIM = 384  # all-MiniLM-L6-v2 output size


class Category(Base):
    __tablename__ = "categories"

    id = Column(Integer, primary_key=True)
    slug = Column(String, unique=True, nullable=False)
    name = Column(String, nullable=False)
    description = Column(Text, default="")

    # interview state for this category (single-user tool, so state lives on the row)
    starter_index = Column(Integer, default=0)
    phase = Column(String, default="starter")  # 'starter' or 'adaptive'
    current_question = Column(Text, default="")

    entries = relationship("KnowledgeEntry", back_populates="category")


class StarterQuestion(Base):
    __tablename__ = "starter_questions"

    id = Column(Integer, primary_key=True)
    category_id = Column(Integer, ForeignKey("categories.id"), nullable=False)
    order = Column(Integer, default=0)
    text = Column(Text, nullable=False)


class KnowledgeEntry(Base):
    __tablename__ = "knowledge_entries"

    id = Column(Integer, primary_key=True)
    category_id = Column(Integer, ForeignKey("categories.id"), nullable=False)

    question_text = Column(Text, nullable=False)
    raw_answer = Column(Text, nullable=False)

    title = Column(String, default="")
    structured_summary = Column(Text, default="")
    tags = Column(ARRAY(String), default=list)

    embedding = Column(Vector(EMBED_DIM), nullable=True)

    created_at = Column(DateTime, default=datetime.utcnow)

    category = relationship("Category", back_populates="entries")
    images = relationship("EntryImage", back_populates="entry", cascade="all, delete-orphan")


class EntryImage(Base):
    __tablename__ = "entry_images"

    id = Column(Integer, primary_key=True)
    entry_id = Column(Integer, ForeignKey("knowledge_entries.id"), nullable=False)
    filename = Column(String, nullable=False)  # stored under /app/data/uploads
    created_at = Column(DateTime, default=datetime.utcnow)

    entry = relationship("KnowledgeEntry", back_populates="images")


class Scenario(Base):
    """
    A single real-world case (e.g. one tricky vehicle repair) captured as a
    running interview: an initial description, then AI-generated follow-ups
    that dig into that specific incident, ending in a structured write-up.
    """
    __tablename__ = "scenarios"

    id = Column(Integer, primary_key=True)
    category_id = Column(Integer, ForeignKey("categories.id"), nullable=False)

    status = Column(String, default="open")  # 'open' or 'resolved' (written up)
    current_question = Column(Text, default="")

    # Filled in once the case is finished and AI writes it up
    title = Column(String, default="")
    summary = Column(Text, default="")       # full narrative: symptom -> dead ends -> fix
    root_cause = Column(Text, default="")
    tags = Column(ARRAY(String), default=list)
    embedding = Column(Vector(EMBED_DIM), nullable=True)

    created_at = Column(DateTime, default=datetime.utcnow)

    category = relationship("Category")
    turns = relationship(
        "ScenarioTurn", back_populates="scenario",
        cascade="all, delete-orphan", order_by="ScenarioTurn.order",
    )
    images = relationship("ScenarioImage", back_populates="scenario", cascade="all, delete-orphan")


class ScenarioTurn(Base):
    __tablename__ = "scenario_turns"

    id = Column(Integer, primary_key=True)
    scenario_id = Column(Integer, ForeignKey("scenarios.id"), nullable=False)
    order = Column(Integer, default=0)
    question = Column(Text, nullable=False)
    answer = Column(Text, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)

    scenario = relationship("Scenario", back_populates="turns")


class ScenarioImage(Base):
    __tablename__ = "scenario_images"

    id = Column(Integer, primary_key=True)
    scenario_id = Column(Integer, ForeignKey("scenarios.id"), nullable=False)
    filename = Column(String, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)

    scenario = relationship("Scenario", back_populates="images")
