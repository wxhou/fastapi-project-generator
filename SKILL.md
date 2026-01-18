---
name: fastapi-project-generator
description: Generate complete FastAPI project skeleton with SQLAlchemy, MongoDB, JWT auth, Redis cache, Celery tasks, Docker, and CRUD generators. Use when user wants to create or scaffold a FastAPI project.
---

This skill guides the creation of a production-ready FastAPI project with all necessary components.

## Project Structure

Generate this standardized structure:

```
project/
├── app/
│   ├── api/                    # API modules
│   │   ├── router.py           # Central route registration
│   │   └── [module]/           # Feature module (e.g., user, form)
│   │       ├── __init__.py
│   │       ├── models.py       # SQLAlchemy models
│   │       ├── schemas.py      # Pydantic schemas
│   │       ├── router.py       # APIRouter definitions
│   │       ├── views.py        # View functions
│   │       └── tasks.py        # Celery tasks
│   ├── core/                   # Core components
│   │   ├── __init__.py
│   │   ├── config.py           # Settings (from pydantic-settings)
│   │   ├── exceptions.py       # Custom exceptions
│   │   ├── middleware.py       # CORS, compression, etc.
│   │   └── security.py         # JWT, password hashing (pwdlib)
│   ├── db/                     # Database layer
│   │   ├── __init__.py
│   │   ├── base.py             # Base class for models
│   │   ├── session.py          # Session management
│   │   └── repositories.py     # Generic repository pattern
│   ├── schemas/                # Shared schemas
│   │   └── __init__.py
│   ├── tasks/                  # Celery configuration
│   │   ├── __init__.py
│   │   ├── celery_app.py       # Celery instance
│   │   └── worker.py           # Task definitions
│   ├── cache/                  # Redis cache layer
│   │   ├── __init__.py
│   │   └── redis.py            # Redis client
│   ├── mongo/                  # MongoDB layer (if needed)
│   │   ├── __init__.py
│   │   └── client.py           # Motor client
│   ├── utils/                  # Utilities
│   │   ├── __init__.py
│   │   ├── logger.py           # Logging setup
│   │   └── helpers.py          # Helper functions
│   └── main.py                 # Application entry point
├── tests/                      # Test suite
│   ├── __init__.py
│   ├── conftest.py
│   ├── test_api/
│   └── test_db/
├── alembic/                    # Database migrations
│   ├── versions/
│   └── env.py
├── docker/                     # Docker configuration
│   ├── Dockerfile
│   ├── docker-compose.yml
│   └── .dockerignore
├── scripts/                    # Utility scripts
│   └── init_db.py
├── requirements.txt            # Python dependencies
├── requirements-dev.txt        # Development dependencies
├── requirements-test.txt       # Test dependencies
├── .env                        # Environment variables
├── .env.example                # Environment template
├── .gitignore
├── alembic.ini
├── pyproject.toml              # Poetry/pip config
└── README.md
```

## Core Dependencies

Use these versions as baseline:

```txt
# Core
fastapi>=0.109.0
uvicorn[standard]>=0.27.0
pydantic>=2.5.0
pydantic-settings>=2.1.0

# Database
sqlalchemy>=2.0.25
alembic>=1.13.0
psycopg2-binary>=2.9.9  # or asyncmy for async
redis>=5.0.1
motor>=3.3.0  # MongoDB async

# Auth
python-jose[cryptography]>=3.3.0
passlib[bcrypt]>=1.7.4  # or pwdlib[argon2]

# Tasks
celery>=5.3.0
redis>=5.0.1

# Utilities
python-multipart>=0.0.6
python-dotenv>=1.0.0
loguru>=0.7.0
httpx>=0.26.0
```

## Configuration Management

Use Pydantic Settings for environment-based config:

```python
# app/core/config.py
from pydantic_settings import SettingsConfigDict
from pydantic import BaseSettings, Field


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file='.env',
        env_file_encoding='utf-8',
        extra='ignore'
    )

    # App
    PROJECT_NAME: str = "FastAPI Project"
    VERSION: str = "1.0.0"
    API_V1_PREFIX: str = "/api/v1"

    # Environment
    ENVIRONMENT: str = Field(default="development", validation_alias="ENV")

    # Database
    DATABASE_URL: str  # postgresql+asyncpg://user:pass@host/db
    DATABASE_URL_SYNC: str = ""  # for alembic

    # Redis
    REDIS_URL: str = "redis://localhost:6379/0"

    # JWT
    SECRET_KEY: str
    ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 30
    REFRESH_TOKEN_EXPIRE_DAYS: int = 7

    # MongoDB (optional)
    MONGODB_URL: str = ""
    MONGODB_DB: str = ""

    # Celery
    CELERY_BROKER_URL: str = "redis://localhost:6379/1"
    CELERY_RESULT_URL: str = "redis://localhost:6379/2"

    # CORS
    BACKEND_CORS_ORIGINS: list[str] = ["http://localhost:3000"]
```

## Database Setup

### Async SQLAlchemy with Repository Pattern

```python
# app/db/session.py
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, async_sessionmaker
from app.core.config import settings

engine = create_async_engine(settings.DATABASE_URL, echo=False)
async_session_maker = async_sessionmaker(engine, expire_on_commit=False)


async def get_db() -> AsyncSession:
    async with async_session_maker() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise
        finally:
            await session.close()
```

### Repository Pattern

```python
# app/db/repositories/base.py
from typing import Generic, TypeVar, Type, Optional
from sqlalchemy import select, update, delete
from sqlalchemy.ext.asyncio import AsyncSession
from app.db.base import Base

ModelType = TypeVar("ModelType", bound=Base)


class Repository(Generic[ModelType]):
    def __init__(self, model: Type[ModelType], db: AsyncSession):
        self.model = model
        self.db = db

    async def get(self, id: int) -> Optional[ModelType]:
        return await self.db.get(self.model, id)

    async def get_all(self, skip: int = 0, limit: int = 100) -> list[ModelType]:
        result = await self.db.execute(select(self.model).offset(skip).limit(limit))
        return result.scalars().all()

    async def create(self, obj: dict) -> ModelType:
        db_obj = self.model(**obj)
        self.db.add(db_obj)
        await self.db.flush()
        await self.db.refresh(db_obj)
        return db_obj

    async def update(self, id: int, obj: dict) -> Optional[ModelType]:
        await self.db.execute(update(self.model).where(self.model.id == id).values(**obj))
        return await self.get(id)

    async def delete(self, id: int) -> bool:
        await self.db.execute(delete(self.model).where(self.model.id == id))
        return True
```

## Security & Auth

### Password Hashing (use pwdlib)

```python
# app/core/security.py
from pwdlib import PasswordHash

password_hash = PasswordHash.recommended()


def get_password_hash(password: str) -> str:
    return password_hash.hash(password)


def verify_password(plain_password: str, hashed_password: str) -> bool:
    return password_hash.verify(plain_password, hashed_password)
```

### JWT Token Creation

```python
# app/core/security.py (continued)
from datetime import datetime, timedelta
from jose import jwt
from app.core.config import settings


def create_access_token(data: dict, expires_delta: Optional[timedelta] = None) -> str:
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=settings.ACCESS_TOKEN_EXPIRE_MINUTES)
    to_encode.update({"exp": expire})
    return jwt.encode(to_encode, settings.SECRET_KEY, algorithm=settings.ALGORITHM)
```

### OAuth2 Password Flow

```python
# app/core/oauth2.py
from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer
from jose import JWTError, jwt
from app.core.config import settings

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/api/v1/auth/login")


async def get_current_user(token: str = Depends(oauth2_scheme)):
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        payload = jwt.decode(token, settings.SECRET_KEY, algorithms=[settings.ALGORITHM])
        username: str = payload.get("sub")
        if username is None:
            raise credentials_exception
    except JWTError:
        raise credentials_exception
    return username
```

## Celery Tasks

```python
# app/tasks/celery_app.py
from celery import Celery
from app.core.config import settings

celery_app = Celery(
    "worker",
    broker=settings.CELERY_BROKER_URL,
    backend=settings.CELERY_RESULT_URL,
    include=["app.tasks.worker"]
)

celery_app.conf.update(
    task_serializer="json",
    accept_content=["json"],
    result_serializer="json",
    timezone="UTC",
    enable_utc=True,
    task_track_started=True,
    task_time_limit=30 * 60,  # 30 minutes
)


# app/tasks/worker.py
from app.tasks.celery_app import celery_app


@celery_app.task(bind=True)
def send_email_task(self, email: str, subject: str, body: str):
    """Example background task"""
    # Your email sending logic here
    return {"status": "sent", "email": email}
```

## API Module Template

```python
# app/api/user/router.py
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession
from typing import list

from app.db.session import get_db
from app.db.repositories.base import Repository
from app.api.user.models import User
from app.api.user.schemas import UserCreate, UserResponse, UserUpdate
from app.core.security import get_password_hash
from app.core.oauth2 import get_current_user

router = APIRouter()


@router.get("/", response_model=list[UserResponse])
async def get_users(
    skip: int = 0,
    limit: int = 100,
    db: AsyncSession = Depends(get_db),
    current_user = Depends(get_current_user)
):
    repo = Repository(User, db)
    return await repo.get_all(skip=skip, limit=limit)


@router.post("/", response_model=UserResponse, status_code=status.HTTP_201_CREATED)
async def create_user(
    user_in: UserCreate,
    db: AsyncSession = Depends(get_db)
):
    repo = Repository(User, db)
    # Check if user exists
    existing = await repo.get_by_field("email", user_in.email)
    if existing:
        raise HTTPException(status_code=400, detail="Email already registered")

    user_data = user_in.model_dump()
    user_data["hashed_password"] = get_password_hash(user_data.pop("password"))
    return await repo.create(user_data)
```

## Main Application Entry

```python
# app/main.py
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.core.config import settings
from app.api.router import api_router


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: Connect to databases, start Celery
    yield
    # Shutdown: Close connections


app = FastAPI(
    title=settings.PROJECT_NAME,
    version=settings.VERSION,
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.BACKEND_CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Register routes
app.include_router(api_router, prefix=settings.API_V1_PREFIX)
```

## Docker Configuration

```dockerfile
# docker/Dockerfile
FROM python:3.12-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application
COPY . .

# Expose port
EXPOSE 8000

# Start command
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

```yaml
# docker/docker-compose.yml
version: '3.8'

services:
  app:
    build: .
    ports:
      - "8000:8000"
    environment:
      - ENVIRONMENT=production
      - DATABASE_URL=postgresql+asyncpg://user:pass@db:5432/app
      - REDIS_URL=redis://redis:6379/0
    depends_on:
      - db
      - redis
    volumes:
      - .:/app

  db:
    image: postgres:16
    environment:
      POSTGRES_DB: app
      POSTGRES_USER: user
      POSTGRES_PASSWORD: pass
    volumes:
      - postgres_data:/var/lib/postgresql/data

  redis:
    image: redis:7-alpine
    volumes:
      - redis_data:/data

volumes:
  postgres_data:
  redis_data:
```

## Implementation Workflow

When user requests a FastAPI project:

1. **Gather Requirements**: Confirm project name, database choice, auth requirements
2. **Generate Structure**: Create the directory structure
3. **Create Core Files**: Generate main.py, config.py, base models
4. **Setup Auth**: Implement JWT auth, password hashing
5. **Add Database**: SQLAlchemy models with repository pattern
6. **Add Cache**: Redis integration
7. **Add Tasks**: Celery configuration
8. **Add Docker**: Containerization files
9. **Write README**: Documentation

Generate production-grade, async-first code with proper error handling, logging, and type hints.
