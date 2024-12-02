# Requires -RunAsAdministrator

# Color definitions for output
$colors = @{
    Red = [System.ConsoleColor]::Red
    Green = [System.ConsoleColor]::Green
    Yellow = [System.ConsoleColor]::Yellow
}

# Function to print status messages
function Write-Status {
    param([string]$Message)
    Write-Host "==> $Message" -ForegroundColor $colors.Green
}

function Write-Warning {
    param([string]$Message)
    Write-Host "Warning: $Message" -ForegroundColor $colors.Yellow
}

# Function to check if a command exists
function Test-Command {
    param([string]$Command)
    return [bool](Get-Command -Name $Command -ErrorAction SilentlyContinue)
}

# Error handling
$ErrorActionPreference = "Stop"

try {
    # Check if running as Administrator
    $currentPrincipal = New-Object Security.Principal.WindowsPrincipal([Security.Principal.WindowsIdentity]::GetCurrent())
    if (-not $currentPrincipal.IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)) {
        throw "This script must be run as Administrator. Right-click PowerShell and select 'Run as Administrator'."
    }

    # Install Chocolatey if not present
    if (-not (Test-Command -Command "choco")) {
        Write-Status "Installing Chocolatey..."
        Set-ExecutionPolicy Bypass -Scope Process -Force
        [System.Net.ServicePointManager]::SecurityProtocol = [System.Net.ServicePointManager]::SecurityProtocol -bor 3072
        Invoke-Expression ((New-Object System.Net.WebClient).DownloadString('https://chocolatey.org/install.ps1'))
    } else {
        Write-Status "Chocolatey already installed"
    }

    # Refresh environment variables
    $env:Path = [System.Environment]::GetEnvironmentVariable("Path", "Machine") + ";" + [System.Environment]::GetEnvironmentVariable("Path", "User")

    # Install Python if not present
    if (-not (Test-Command -Command "python")) {
        Write-Status "Installing Python..."
        choco install python -y
        $env:Path = [System.Environment]::GetEnvironmentVariable("Path", "Machine") + ";" + [System.Environment]::GetEnvironmentVariable("Path", "User")
    } else {
        Write-Status "Python already installed"
    }

    # Create virtual environment if it doesn't exist
    if (-not (Test-Path "venv")) {
        Write-Status "Creating virtual environment..."
        python -m venv venv
    } else {
        Write-Status "Virtual environment already exists"
    }

    # Activate virtual environment
    Write-Status "Activating virtual environment..."
    .\venv\Scripts\Activate.ps1

    # Install Python requirements
    Write-Status "Installing Python requirements..."
    pip install -r requirements.txt

    # Install Docker Desktop if not present
    if (-not (Test-Command -Command "docker")) {
        Write-Status "Installing Docker Desktop..."
        choco install docker-desktop -y
        Write-Warning "Please start Docker Desktop manually after installation"
    } else {
        Write-Status "Docker Desktop already installed"
    }

    # Install kubectl if not present
    if (-not (Test-Command -Command "kubectl")) {
        Write-Status "Installing kubectl..."
        choco install kubernetes-cli -y
    } else {
        Write-Status "kubectl already installed"
    }

    # Install Helm if not present
    if (-not (Test-Command -Command "helm")) {
        Write-Status "Installing Helm..."
        choco install kubernetes-helm -y
    } else {
        Write-Status "Helm already installed"
    }

    # Install minikube if not present
    if (-not (Test-Command -Command "minikube")) {
        Write-Status "Installing minikube..."
        choco install minikube -y
    } else {
        Write-Status "minikube already installed"
    }

    # Download NLTK data
    Write-Status "Downloading NLTK data..."
    python -c "import nltk; nltk.download('punkt')"

    # Start minikube if it's not running
    $minikubeStatus = minikube status
    if (-not ($minikubeStatus -match "Running")) {
        Write-Status "Starting minikube..."
        minikube start
    } else {
        Write-Status "minikube is already running"
    }

    # Install NGINX Ingress Controller
    Write-Status "Setting up NGINX Ingress Controller..."
    helm repo add ingress-nginx https://kubernetes.github.io/ingress-nginx
    helm repo update

    $ingressNamespace = kubectl get namespace ingress-nginx --no-headers --ignore-not-found
    if (-not $ingressNamespace) {
        kubectl create namespace ingress-nginx
    }

    helm upgrade --install nginx-ingress ingress-nginx/ingress-nginx `
        --namespace ingress-nginx `
        --set controller.publishService.enabled=true

    # Create resume-summarizer namespace
    Write-Status "Creating Kubernetes namespace..."
    kubectl create namespace resume-summarizer --dry-run=client -o yaml | kubectl apply -f -

    # Set up local development environment
    Write-Status "Setting up local development environment..."
    if (-not (Test-Path "data\models")) {
        New-Item -ItemType Directory -Force -Path "data\models"
    }

    if (-not (Test-Path "data\output")) {
        New-Item -ItemType Directory -Force -Path "data\output"
    }

    # Update hosts file for local development
    $hostsFile = "$env:SystemRoot\System32\drivers\etc\hosts"
    $hostsContent = Get-Content $hostsFile
    if (-not ($hostsContent -match "resume-summarizer.local")) {
        Write-Status "Updating hosts file..."
        Add-Content -Path $hostsFile -Value "`n127.0.0.1 resume-summarizer.local" -Force
    }

    # Final instructions
    Write-Host "`nSetup complete!" -ForegroundColor $colors.Green
    Write-Host "`nNext steps:"
    Write-Host "1. Start developing: " -NoNewline
    Write-Host ".\venv\Scripts\Activate.ps1" -ForegroundColor $colors.Yellow
    Write-Host "2. Deploy to Kubernetes: " -NoNewline
    Write-Host "kubectl apply -k k8s/" -ForegroundColor $colors.Yellow
    Write-Host "3. Access services at: " -NoNewline
    Write-Host "http://resume-summarizer.local" -ForegroundColor $colors.Yellow
    Write-Host "`nFor more information, check the README.md file"

} catch {
    Write-Host "Error: $_" -ForegroundColor $colors.Red
    exit 1
}
