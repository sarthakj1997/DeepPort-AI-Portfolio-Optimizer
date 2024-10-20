# src/database.py

import os
from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from sqlalchemy import Column, Integer, String, Float, DateTime
import datetime

Base = declarative_base()

class AssetData(Base):
    __tablename__ = 'asset_data'
    id = Column(Integer, primary_key=True)
    ticker = Column(String)
    date = Column(DateTime)
    open = Column(Float)
    high = Column(Float)
    low = Column(Float)
    close = Column(Float)
    adj_close = Column(Float)
    volume = Column(Integer)

class SentimentData(Base):
    __tablename__ = 'sentiment_data'
    id = Column(Integer, primary_key=True)
    ticker = Column(String)
    date = Column(DateTime)
    sentiment_score = Column(Float)

class OptimizationResult(Base):
    __tablename__ = 'optimization_results'
    id = Column(Integer, primary_key=True)
    method = Column(String)
    ticker = Column(String)
    weight = Column(Float)
    date = Column(DateTime, default=datetime.datetime.utcnow)

def get_engine(db_path='data/database.db'):
    engine = create_engine(f'sqlite:///{db_path}')
    return engine

def create_tables(engine):
    Base.metadata.create_all(engine)

def get_session(engine):
    Session = sessionmaker(bind=engine)
    session = Session()
    return session
