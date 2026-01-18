# FastAPI Project Generator

A Claude Code skill for generating production-ready FastAPI project skeletons.

## Features

- **Complete Project Structure** - Standardized FastAPI project layout
- **Multiple Databases** - SQLAlchemy (PostgreSQL/MySQL) and MongoDB support
- **Authentication** - JWT with pwdlib password hashing (Argon2id)
- **Caching** - Redis integration
- **Background Tasks** - Celery task queue support
- **Docker** - Containerization with docker-compose
- **Repository Pattern** - Clean database access layer
- **Multi-environment** - Development/Testing/Production configs

## Installation

### Via /plugin Command (Recommended)

```bash
/plugin install https://github.com/wxhou/fastapi-project-generator
```

### From Source

```bash
cd ~/.claude/skills
git clone https://github.com/wxhou/fastapi-project-generator.git
```

### Manual

1. Copy `fastapi-project-generator` folder to `~/.claude/skills/`
2. Restart Claude Code

## Usage

In Claude Code, simply describe your FastAPI project:

```
Create a FastAPI project with PostgreSQL, JWT auth, and Redis caching
```

The skill will generate:
- Project structure
- Configuration files (`.env`, `settings.py`)
- Database models with repository pattern
- JWT authentication
- API routes with CRUD operations
- Celery task configuration
- Docker files

## Project Structure Generated

```
project/
├── app/
│   ├── api/
│   ├── core/
│   ├── db/
│   ├── tasks/
│   ├── cache/
│   └── main.py
├── tests/
├── docker/
├── requirements.txt
└── README.md
```

## Requirements

- Python 3.12+
- PostgreSQL (or MySQL)
- Redis
- Celery (optional, for background tasks)

## License

MIT
