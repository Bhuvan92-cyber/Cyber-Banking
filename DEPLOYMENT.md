# 🚀 Cyber-Banking Deployment Guide

## Quick Start: Choose Your Deployment Method

### 1. Docker Deployment (Easiest - Recommended for Testing)

```bash
# Clone repository
git clone https://github.com/Bhuvan92-cyber/Cyber-Banking.git
cd Cyber-Banking

# Start with Docker Compose
docker-compose up -d

# Create superuser
docker-compose exec web python manage.py createsuperuser

# Access the app
# Open: http://localhost:8000
```

**Stop Services:**
```bash
docker-compose down
```

---

### 2. Heroku Deployment (Easiest - Free/Paid Tiers)

#### Prerequisites:
- Heroku account (sign up at heroku.com)
- Heroku CLI installed

#### Steps:

```bash
# Login to Heroku
heroku login

# Create app
heroku create your-app-name

# Add PostgreSQL addon
heroku addons:create heroku-postgresql:hobby-dev

# Set environment variables
heroku config:set SECRET_KEY=$(python -c 'from django.core.management.utils import get_random_secret_key; print(get_random_secret_key())')
heroku config:set DEBUG=False
heroku config:set ALLOWED_HOSTS=your-app-name.herokuapp.com

# Deploy
git push heroku main

# Run migrations
heroku run python manage.py migrate

# Create superuser
heroku run python manage.py createsuperuser

# View logs
heroku logs --tail
```

**Access App:** `https://your-app-name.herokuapp.com`

---

### 3. Manual Server Deployment (VPS/Dedicated Server)

#### Prerequisites:
- Linux server (Ubuntu 20.04+ recommended)
- SSH access
- Domain name (optional, but recommended)

#### Step 1: Server Setup

```bash
# Update system
sudo apt-get update && sudo apt-get upgrade -y

# Install dependencies
sudo apt-get install -y python3.12 python3.12-venv python3-pip \
    postgresql postgresql-contrib nginx git curl

# Create app user
sudo useradd -m cyberbanking
sudo usermod -aG sudo cyberbanking
```

#### Step 2: Clone & Setup Application

```bash
# Switch to app user
sudo su - cyberbanking

# Clone repository
git clone https://github.com/Bhuvan92-cyber/Cyber-Banking.git
cd Cyber-Banking

# Create virtual environment
python3.12 -m venv venv
source venv/bin/activate

# Install Python dependencies
pip install --upgrade pip
pip install -r requirements.txt

# Copy settings
cp .env.example .env
nano .env  # Edit with your configuration
```

#### Step 3: Database Setup

```bash
# Create PostgreSQL user and database
sudo -u postgres psql
```

```sql
CREATE DATABASE cyber_banking;
CREATE USER cyber_user WITH PASSWORD 'strong_password_here';
ALTER ROLE cyber_user SET client_encoding TO 'utf8';
ALTER ROLE cyber_user SET default_transaction_isolation TO 'read committed';
ALTER ROLE cyber_user SET default_transaction_deferrable TO on;
ALTER ROLE cyber_user SET timezone TO 'UTC';
GRANT ALL PRIVILEGES ON DATABASE cyber_banking TO cyber_user;
\q
```

#### Step 4: Django Setup

```bash
# Apply migrations
python manage.py migrate

# Create superuser
python manage.py createsuperuser

# Collect static files
python manage.py collectstatic --noinput

# Test with Gunicorn
gunicorn cyber_physical_banking.wsgi:application --bind 0.0.0.0:8000
```

#### Step 5: Systemd Service Setup

```bash
# Copy service file
sudo cp cyber-banking.service /etc/systemd/system/

# Edit paths in service file if needed
sudo nano /etc/systemd/system/cyber-banking.service

# Create log directory
sudo mkdir -p /var/log/cyber-banking
sudo chown cyberbanking:www-data /var/log/cyber-banking

# Enable and start service
sudo systemctl daemon-reload
sudo systemctl enable cyber-banking
sudo systemctl start cyber-banking
sudo systemctl status cyber-banking
```

#### Step 6: Nginx Configuration

```bash
# Copy Nginx config
sudo cp nginx.conf /etc/nginx/sites-available/cyber-banking

# Edit domain name in config
sudo nano /etc/nginx/sites-available/cyber-banking

# Enable site
sudo ln -s /etc/nginx/sites-available/cyber-banking /etc/nginx/sites-enabled/

# Disable default site
sudo rm /etc/nginx/sites-enabled/default

# Test Nginx
sudo nginx -t

# Restart Nginx
sudo systemctl restart nginx
```

#### Step 7: SSL Certificate (Let's Encrypt)

```bash
# Install Certbot
sudo apt-get install -y certbot python3-certbot-nginx

# Get certificate
sudo certbot --nginx -d yourdomain.com -d www.yourdomain.com

# Auto-renewal test
sudo certbot renew --dry-run
```

#### Step 8: Firewall Setup

```bash
sudo ufw allow 22/tcp
sudo ufw allow 80/tcp
sudo ufw allow 443/tcp
sudo ufw enable
```

---

### 4. AWS Deployment (Scalable - Recommended for Production)

#### Using Elastic Beanstalk (Easiest on AWS):

```bash
# Install AWS CLI and EB CLI
pip install awsebcli

# Initialize Elastic Beanstalk
eb init -p python-3.12 cyber-banking --region us-east-1

# Create environment
eb create cyber-banking-env

# Deploy
eb deploy

# View logs
eb logs

# Monitor
eb open
```

#### Using EC2 + RDS (More Control):

1. **Launch EC2 Instance:**
   - Choose Ubuntu 20.04 LTS AMI
   - t3.micro or t3.small instance type
   - Configure security group (HTTP, HTTPS, SSH)

2. **Create RDS Database:**
   - Engine: PostgreSQL 15
   - Multi-AZ: No (for production, enable)
   - Database name: cyber_banking

3. **Connect and Deploy:**
   - SSH to EC2 instance
   - Follow "Manual Server Deployment" steps above
   - Update DATABASE_URL with RDS endpoint

---

### 5. DigitalOcean Deployment

```bash
# Create Droplet:
# - OS: Ubuntu 20.04 x64
# - Size: $5-6/month (Basic)
# - Region: Closest to users

# SSH to Droplet
ssh root@your_droplet_ip

# Follow "Manual Server Deployment" steps above
```

---

## Post-Deployment Checklist

- [ ] Set `DEBUG=False` in production
- [ ] Update `SECRET_KEY` with strong random value
- [ ] Configure `ALLOWED_HOSTS` with your domain
- [ ] Enable HTTPS/SSL certificate
- [ ] Set up automated database backups
- [ ] Configure email service (for notifications)
- [ ] Monitor server logs and performance
- [ ] Set up error tracking (Sentry)
- [ ] Test all ML model endpoints
- [ ] Test user registration and login
- [ ] Test file uploads and CSV processing
- [ ] Verify static files are served correctly

---

## Production Essentials

### Environment Variables Template

```
SECRET_KEY=your-secret-key
DEBUG=False
ALLOWED_HOSTS=yourdomain.com,www.yourdomain.com
CSRF_TRUSTED_ORIGINS=https://yourdomain.com,https://www.yourdomain.com

DATABASE_URL=postgresql://user:password@host:5432/cyber_banking

SECURE_SSL_REDIRECT=True
SESSION_COOKIE_SECURE=True
CSRF_COOKIE_SECURE=True

GUNICORN_WORKERS=4
```

### Monitoring & Logging

```bash
# View app logs (systemd)
sudo journalctl -u cyber-banking -f

# View Nginx logs
sudo tail -f /var/log/nginx/error.log
sudo tail -f /var/log/nginx/access.log

# Monitor disk space
df -h

# Monitor memory usage
free -h
```

### Backup Strategy

```bash
# Backup PostgreSQL database
pg_dump cyber_banking > backup_$(date +%Y%m%d_%H%M%S).sql

# Backup media files
tar -czf media_backup_$(date +%Y%m%d).tar.gz media/
```

---

## Troubleshooting

### "500 Internal Server Error"
```bash
# Check Gunicorn logs
sudo journalctl -u cyber-banking -n 50

# Check Nginx error log
sudo tail -20 /var/log/nginx/error.log
```

### Database Connection Issues
```bash
# Test PostgreSQL connection
psql -U cyber_user -d cyber_banking -h localhost
```

### Static Files Not Loading
```bash
# Collect static files again
python manage.py collectstatic --clear --noinput

# Check permissions
ls -la staticfiles/
```

---

## Recommended Hosting Providers

| Provider | Cost | Ease | Scalability | Recommendation |
|----------|------|------|-------------|---|
| **Heroku** | Free-$50+/mo | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ | Best for beginners |
| **Docker** | Free (local) | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ | Best for testing |
| **DigitalOcean** | $5-40/mo | ⭐⭐⭐⭐ | ⭐⭐⭐ | Best budget option |
| **AWS EC2** | $5-200+/mo | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ | Best for scaling |
| **VPS** | $5-50/mo | ⭐⭐⭐ | ⭐⭐⭐ | Cost effective |

---

**For support and updates, visit: https://github.com/Bhuvan92-cyber/Cyber-Banking**
