# FastAPI Project Generator

A Claude Code skill for generating production-ready FastAPI project skeletons.

## Features

- **Complete Project Structure** - Standardized FastAPI project layout
- **Multiple Databases** - SQLAlchemy (MySQL) and MongoDB support
- **Authentication** - JWT with pwdlib password hashing (Argon2id)
- **Caching** - Redis integration
- **Background Tasks** - Celery task queue support
- **Docker** - Containerization with docker-compose
- **Multi-environment** - Development/Testing/Production configs
- **Unified Response** - Response wrappers with error codes

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
Create a FastAPI project with MySQL, JWT auth, and Redis caching
```

The skill will generate:
- Project structure
- Configuration files (`settings/` with multi-environment)
- Database models
- JWT authentication
- API routes with CRUD operations
- Redis cache integration
- Celery task configuration
- Docker files

## Project Structure Generated

```
project/
├── app/
│   ├── api/
│   │   ├── router.py           # Central route registration
│   │   └── [module]/           # Feature modules
│   │       ├── models.py
│   │       ├── schemas.py
│   │       ├── router.py
│   │       ├── views.py
│   │       └── tasks.py
│   ├── core/
│   │   ├── exceptions.py
│   │   ├── middleware.py
│   │   └── oauth2.py
│   ├── extensions/
│   │   ├── db.py
│   │   ├── cache.py
│   │   └── mongo.py
│   ├── settings/
│   │   ├── __init__.py
│   │   ├── development.py
│   │   ├── testing.py
│   │   └── production.py
│   └── utils/
├── common/
│   ├── response.py
│   ├── security.py
│   └── pagination.py
├── tests/
├── alembic/
│   ├── versions/
│   └── env.py
├── docker/
│   ├── Dockerfile
│   └── docker-compose.yml
├── requirements.txt
├── main.py                     # Application entry point (root)
├── .env
├── .env.example
└── alembic.ini
```

## Requirements

- Python 3.12+
- MySQL 8+
- Redis
- Celery (optional, for background tasks)

## License

MIT
