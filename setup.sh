#!/bin/bash

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Error handling
set -e
trap 'last_command=$current_command; current_command=$BASH_COMMAND' DEBUG
trap 'echo -e "${RED}\"${last_command}\" command failed with exit code $?.${NC}"' EXIT

# Function to check if a command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Function to print status messages
print_status() {
    echo -e "${GREEN}==>${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}Warning:${NC} $1"
}

# Check for Homebrew (needed for installing other tools)
if ! command_exists brew; then
    print_status "Installing Homebrew..."
    /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
else
    print_status "Homebrew already installed"
fi

# Create virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
    print_status "Creating virtual environment..."
    python3 -m venv venv
else
    print_status "Virtual environment already exists"
fi

# Activate virtual environment
print_status "Activating virtual environment..."
source venv/bin/activate

# Install Python requirements
print_status "Installing Python requirements..."
pip install -r requirements.txt

# Install kubectl if not present
if ! command_exists kubectl; then
    print_status "Installing kubectl..."
    brew install kubectl
else
    print_status "kubectl already installed"
fi

# Install Helm if not present
if ! command_exists helm; then
    print_status "Installing Helm..."
    brew install helm
else
    print_status "Helm already installed"
fi

# Install Docker if not present
if ! command_exists docker; then
    print_status "Installing Docker..."
    brew install --cask docker
    print_warning "Please start Docker Desktop manually after installation"
else
    print_status "Docker already installed"
fi

# Install minikube if not present
if ! command_exists minikube; then
    print_status "Installing minikube..."
    brew install minikube
else
    print_status "minikube already installed"
fi

# Download NLTK data
print_status "Downloading NLTK data..."
python -c "import nltk; nltk.download('punkt')"

# Start minikube if it's not running
if ! minikube status | grep -q "Running"; then
    print_status "Starting minikube..."
    minikube start
else
    print_status "minikube is already running"
fi

# Install NGINX Ingress Controller
print_status "Setting up NGINX Ingress Controller..."
helm repo add ingress-nginx https://kubernetes.github.io/ingress-nginx
helm repo update
if ! kubectl get namespace ingress-nginx >/dev/null 2>&1; then
    kubectl create namespace ingress-nginx
fi
helm upgrade --install nginx-ingress ingress-nginx/ingress-nginx \
    --namespace ingress-nginx \
    --set controller.publishService.enabled=true

# Create resume-summarizer namespace
print_status "Creating Kubernetes namespace..."
kubectl create namespace resume-summarizer --dry-run=client -o yaml | kubectl apply -f -

# Set up local development environment
print_status "Setting up local development environment..."
if [ ! -d "data/models" ]; then
    mkdir -p data/models
fi

if [ ! -d "data/output" ]; then
    mkdir -p data/output
fi

# Update hosts file for local development
if ! grep -q "resume-summarizer.local" /etc/hosts; then
    print_status "Updating /etc/hosts file..."
    echo "127.0.0.1 resume-summarizer.local" | sudo tee -a /etc/hosts
fi

# Final instructions
echo -e "\n${GREEN}Setup complete!${NC}"
echo -e "\nNext steps:"
echo -e "1. Start developing: ${YELLOW}source venv/bin/activate${NC}"
echo -e "2. Deploy to Kubernetes: ${YELLOW}kubectl apply -k k8s/${NC}"
echo -e "3. Access services at: ${YELLOW}http://resume-summarizer.local${NC}"
echo -e "\nFor more information, check the README.md file"

# Remove error handling trap
trap - EXIT