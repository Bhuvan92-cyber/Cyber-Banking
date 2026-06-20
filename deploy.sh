#!/bin/bash
# Cyber-Banking Deployment Helper Script
# Usage: chmod +x deploy.sh && ./deploy.sh [option]

set -e

echo "🚀 Cyber-Banking Deployment Helper"
echo "===================================="

case "$1" in
    docker)
        echo "📦 Starting Docker deployment..."
        docker-compose down
        docker-compose up -d
        echo "✅ Docker deployment started"
        echo "📍 App running at: http://localhost:8000"
        ;;
    
    local)
        echo "🏠 Setting up local development environment..."
        
        # Create virtual environment
        python3 -m venv venv
        source venv/bin/activate
        
        # Install dependencies
        pip install --upgrade pip
        pip install -r requirements.txt
        
        # Migrations
        python manage.py migrate
        
        # Collect static
        python manage.py collectstatic --noinput
        
        echo "✅ Local setup complete"
        echo "🏃 Run: python manage.py runserver"
        ;;
    
    migrate)
        echo "📊 Running database migrations..."
        python manage.py migrate
        echo "✅ Migrations complete"
        ;;
    
    static)
        echo "📁 Collecting static files..."
        python manage.py collectstatic --clear --noinput
        echo "✅ Static files collected"
        ;;
    
    superuser)
        echo "👤 Creating superuser..."
        python manage.py createsuperuser
        echo "✅ Superuser created"
        ;;
    
    backup-db)
        echo "💾 Backing up database..."
        TIMESTAMP=$(date +%Y%m%d_%H%M%S)
        pg_dump cyber_banking > backups/backup_$TIMESTAMP.sql
        echo "✅ Database backed up to: backups/backup_$TIMESTAMP.sql"
        ;;
    
    backup-media)
        echo "💾 Backing up media files..."
        TIMESTAMP=$(date +%Y%m%d_%H%M%S)
        tar -czf backups/media_$TIMESTAMP.tar.gz media/
        echo "✅ Media backed up to: backups/media_$TIMESTAMP.tar.gz"
        ;;
    
    test)
        echo "🧪 Running tests..."
        python manage.py test
        echo "✅ Tests completed"
        ;;
    
    check)
        echo "🔍 Running Django security check..."
        python manage.py check --deploy
        echo "✅ Security checks passed"
        ;;
    
    clean)
        echo "🧹 Cleaning up..."
        find . -type d -name __pycache__ -exec rm -r {} + 2>/dev/null || true
        find . -type f -name "*.pyc" -delete
        echo "✅ Cleanup complete"
        ;;
    
    help|"")
        echo ""
        echo "Available commands:"
        echo "  ./deploy.sh docker       - Deploy with Docker"
        echo "  ./deploy.sh local        - Setup local development"
        echo "  ./deploy.sh migrate      - Run database migrations"
        echo "  ./deploy.sh static       - Collect static files"
        echo "  ./deploy.sh superuser    - Create admin user"
        echo "  ./deploy.sh backup-db    - Backup PostgreSQL database"
        echo "  ./deploy.sh backup-media - Backup media files"
        echo "  ./deploy.sh test         - Run tests"
        echo "  ./deploy.sh check        - Run security checks"
        echo "  ./deploy.sh clean        - Clean cache files"
        echo "  ./deploy.sh help         - Show this help"
        echo ""
        ;;
    
    *)
        echo "❌ Unknown command: $1"
        echo "Run './deploy.sh help' for available commands"
        exit 1
        ;;
esac
