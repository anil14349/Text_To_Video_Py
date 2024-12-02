# Text to Video Generator

An intelligent system that generates video content from text descriptions using advanced natural language processing and video generation techniques. The system includes text-to-speech capabilities, monitoring, and both Gradio and Streamlit web interfaces.

## Features

- **Text-to-Video Generation**
  - Natural language text input
  - Advanced video generation models
  - Text-to-Speech integration
  - Multiple style options

- **Web Interfaces**
  - Gradio interface for simple, focused processing
  - Streamlit dashboard for advanced features
  - Batch processing capabilities
  - Interactive model analytics

- **Monitoring & Analytics**
  - Prometheus metrics collection
  - Grafana dashboard integration
  - System resource monitoring
  - Model performance tracking

## Important Note About Model Files

This repository does not include large model files. You will need to download them separately. The following files are required:

- `asset/gpt/model.safetensors`
- `models_cache/distilgpt2/model.safetensors`
- `asset/Embed.safetensors`

These files can be downloaded from their respective model hubs or generated using the provided scripts.

## Installation

### Prerequisites

Before installation, ensure you have:
- Python 3.11 or higher
- Git
- Sufficient disk space for model files (approximately 2GB)

### Windows Setup

1. Open PowerShell as Administrator and run:
```powershell
# Enable script execution
Set-ExecutionPolicy Bypass -Scope Process -Force

# Run setup script
.\setup.ps1
```

The setup script will:
- Install Chocolatey (Windows package manager)
- Install Python if not present
- Set up virtual environment
- Install project dependencies
- Install Docker Desktop
- Install Kubernetes tools (kubectl, helm, minikube)
- Configure local development environment

### macOS Setup

Run the bash setup script:
```bash
chmod +x setup.sh
./setup.sh
```

1. Clone the repository:
```bash
git clone https://github.com/anil14349/Text_To_Video_Py.git
cd t2v
```

2. Create and activate a virtual environment:
```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Download model files:
```bash
# Script will be provided to download necessary model files
python scripts/download_models.py  # Note: This is a placeholder. Actual script needs to be implemented.
```

## Model Files Management

This repository uses `.gitignore` to exclude large model files. These files need to be managed separately from the Git repository. The following directories are ignored:

```
# Model files
*.safetensors
*.ckpt
*.pth
*.pt

# Model directories
models/
models_cache/
asset/
```

When sharing this project, ensure you provide a way for others to obtain the necessary model files, either through:
- A model download script
- Links to the original model sources
- A separate file storage solution

## Configuration

The system is configured through `config.yaml`. Key configuration sections:

- Model settings
- Text-to-Speech parameters
- Monitoring configuration
- Web interface settings

Customize the configuration file according to your needs before running the application.

## Usage

### Command Line Interface

Run the main application:
```bash
python src/main.py
```

### Gradio Interface

Start the simple web interface:
```bash
python src/web/gradio_app.py
```
Access at: http://localhost:7860

### Streamlit Dashboard

Launch the full-featured dashboard:
```bash
streamlit run src/web/streamlit_app.py
```
Access at: http://localhost:8501

### Monitoring Setup

1. Start the Prometheus metrics server:
```bash
# Metrics will be available at localhost:8000
python -c "from src.utils.monitoring import MetricsCollector; MetricsCollector()"
```

2. Install and configure Grafana:
```bash
# On macOS
brew install grafana
brew services start grafana
```
Access Grafana at: http://localhost:3000

3. Configure Prometheus data source in Grafana:
   - Add data source: http://localhost:8000
   - Import the provided dashboard configurations

## Kubernetes Deployment

### Prerequisites

- Kubernetes cluster (local or cloud)
- kubectl installed and configured
- Helm (optional, for installing ingress-nginx)
- Docker registry access

### Deployment Steps

1. Build and push the Docker image:
```bash
docker build -t your-registry/resume-summarizer:latest .
docker push your-registry/resume-summarizer:latest
```

2. Create the namespace and resources:
```bash
kubectl apply -k k8s/
```

3. Install NGINX Ingress Controller (if not already installed):
```bash
helm repo add ingress-nginx https://kubernetes.github.io/ingress-nginx
helm repo update
helm install nginx-ingress ingress-nginx/ingress-nginx
```

4. Update your hosts file:
```bash
echo "127.0.0.1 resume-summarizer.local" | sudo tee -a /etc/hosts
```

### Accessing Services

After deployment, the services will be available at:

- Gradio Interface: http://resume-summarizer.local/gradio
- Streamlit Dashboard: http://resume-summarizer.local/streamlit
- Grafana Dashboard: http://resume-summarizer.local/grafana

### Kubernetes Resources

The application is deployed with the following resources:

- **Deployments**:
  - Main application (2 replicas)
  - Gradio interface (2 replicas)
  - Streamlit dashboard (2 replicas)
  - Prometheus
  - Grafana

- **Services**:
  - LoadBalancer services for web interfaces
  - ClusterIP service for metrics collection

- **Storage**:
  - PersistentVolumeClaims for models and monitoring data

- **Configuration**:
  - ConfigMaps for application settings
  - Secrets for sensitive data

### Scaling

Scale deployments as needed:
```bash
kubectl scale deployment gradio -n resume-summarizer --replicas=3
kubectl scale deployment streamlit -n resume-summarizer --replicas=3
```

### Monitoring

Monitor the deployment:
```bash
# Get pod status
kubectl get pods -n resume-summarizer

# View logs
kubectl logs -f deployment/resume-summarizer -n resume-summarizer

# Get service status
kubectl get svc -n resume-summarizer
```

### Cleanup

Remove all resources:
```bash
kubectl delete namespace resume-summarizer
```

## CI/CD

The project includes GitHub Actions workflows for Continuous Integration and Deployment:

### Continuous Integration (CI)

The CI workflow (`ci.yml`) runs on every push and pull request to `main` and `develop` branches:

- Runs tests on Python 3.8 and 3.9
- Performs linting with flake8
- Generates test coverage reports
- Builds and pushes Docker image (on push to main/develop)

### Continuous Deployment (CD)

The CD workflow (`cd.yml`) runs when a new tag is pushed:

- Creates a new GitHub release
- Builds and pushes Docker image with version tag
- Updates the 'latest' Docker image

### GitHub Secrets Required

Set up the following secrets in your GitHub repository:

- `DOCKERHUB_USERNAME`: Your Docker Hub username
- `DOCKERHUB_TOKEN`: Your Docker Hub access token

## Project Structure

```
t2v/
├── config.yaml           # Centralized configuration
├── requirements.txt      # Project dependencies
└── src/
    ├── main.py          # Main application entry point
    ├── data/
    │   └── dataset.py   # Data handling and loading
    ├── model/
    │   └── pipeline.py  # Core model processing
    ├── web/
    │   ├── gradio_app.py    # Gradio interface
    │   └── streamlit_app.py # Streamlit dashboard
    └── utils/
        ├── config.py        # Configuration management
        ├── text_processor.py # Text extraction
        ├── tts_generator.py # Text-to-Speech
        ├── model_manager.py # Model management
        ├── monitoring.py    # Prometheus metrics
        └── evaluator.py     # Performance evaluation
```

## Features in Detail

### Text-to-Video Generation
- Supports multiple text formats
- Advanced video generation models
- Text-to-Speech integration
- Configurable video parameters

### Text-to-Speech
- Multiple voice options (male/female)
- Adjustable voice characteristics
- High-quality audio generation

### Model Management
- Version tracking
- Performance metrics storage
- Easy model switching
- Training history

### Monitoring
- Request tracking
- Performance metrics
- System resource usage
- Model version tracking
- Generation time metrics

### Evaluation
- BLEU score calculation
- ROUGE metrics
- Readability assessment
- Diversity analysis

## Web Interface Features

### Gradio Interface
- Simple text input
- Real-time processing
- Audio generation options
- Example text processing

### Streamlit Dashboard
- Multiple pages/tabs
- Batch processing
- Model analytics
- System monitoring
- Interactive visualizations
- Downloadable results

## Contributing

1. Fork the repository
2. Create your feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## License

[Specify your license here]

## Support

For support, please [specify contact method or raise an issue in the repository].