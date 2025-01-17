#!/bin/bash
# Main setup script for ML/DevOps infrastructure

# Set up environment variables
echo "Setting up environment variables..."
cat > .env << EOL
# GitHub Configuration
GITHUB_TOKEN=${GITHUB_TOKEN}
GITHUB_OWNER=${GITHUB_OWNER}
GITHUB_REPO=${GITHUB_REPO}

# Docker Registry
DOCKER_REGISTRY=${DOCKER_REGISTRY}
DOCKER_USERNAME=${DOCKER_USERNAME}
DOCKER_PASSWORD=${DOCKER_PASSWORD}

# ML Tools
MLFLOW_TRACKING_URI=http://localhost:5000
WANDB_API_KEY=${WANDB_API_KEY}

# Monitoring
GRAFANA_ADMIN_PASSWORD=${GRAFANA_ADMIN_PASSWORD}
PROMETHEUS_RETENTION=15d

# Security
SONAR_TOKEN=${SONAR_TOKEN}
SNYK_TOKEN=${SNYK_TOKEN}

# Logging
ELASTIC_PASSWORD=${ELASTIC_PASSWORD}
KIBANA_PASSWORD=${KIBANA_PASSWORD}
EOL

# Create necessary directories
echo "Creating directory structure..."
mkdir -p {.github,config,scripts,models,logs,data}/{workflows,monitoring,mlops,security}

# Set up pre-commit hooks
echo "Setting up pre-commit hooks..."
pre-commit install

# Install dependencies
echo "Installing dependencies..."
pip install -r requirements.txt
npm install

# Initialize monitoring
echo "Setting up monitoring..."
./scripts/setup-monitoring.sh

# Initialize ML tracking
echo "Setting up ML infrastructure..."
./scripts/setup-mlops.sh

# Set up security scanning
echo "Setting up security tools..."
./scripts/setup-security.sh

echo "Setup complete! Check setup-log.txt for details"

