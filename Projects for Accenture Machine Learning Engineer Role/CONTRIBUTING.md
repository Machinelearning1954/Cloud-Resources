# Contributing to Federal ML Portfolio

Thank you for your interest in contributing to this federal machine learning portfolio! This document provides guidelines for contributing to the project.

## Code of Conduct

This project adheres to professional standards of conduct. All contributors are expected to:

- Be respectful and inclusive in all interactions
- Focus on constructive feedback and collaboration
- Prioritize security and compliance in all contributions
- Maintain confidentiality of any sensitive information

## How to Contribute

### Reporting Issues

If you find a bug, security vulnerability, or have a feature request:

1. Check existing issues to avoid duplicates
2. Create a new issue with a clear title and description
3. Include steps to reproduce (for bugs)
4. Add relevant labels (bug, enhancement, security, etc.)

### Submitting Changes

1. **Fork the repository**
   ```bash
   git clone https://github.com/yourusername/federal-ml-portfolio.git
   cd federal-ml-portfolio
   ```

2. **Create a feature branch**
   ```bash
   git checkout -b feature/your-feature-name
   ```

3. **Make your changes**
   - Follow the coding standards below
   - Add tests for new functionality
   - Update documentation as needed

4. **Test your changes**
   ```bash
   # Run tests
   pytest tests/ -v
   
   # Run linting
   black src/
   flake8 src/
   
   # Run security scan
   bandit -r src/
   ```

5. **Commit your changes**
   ```bash
   git add .
   git commit -m "feat: add new feature description"
   ```
   
   Use conventional commit messages:
   - `feat:` for new features
   - `fix:` for bug fixes
   - `docs:` for documentation changes
   - `test:` for test additions/changes
   - `refactor:` for code refactoring
   - `security:` for security improvements

6. **Push to your fork**
   ```bash
   git push origin feature/your-feature-name
   ```

7. **Create a Pull Request**
   - Provide a clear description of changes
   - Reference any related issues
   - Ensure all CI checks pass

## Coding Standards

### Python Style Guide

- Follow PEP 8 style guidelines
- Use Black for code formatting (line length: 100)
- Use type hints for function parameters and returns
- Write docstrings for all public functions and classes

Example:
```python
def predict_failure(
    vehicle_data: Dict[str, float],
    model: XGBClassifier
) -> Dict[str, Union[int, float, str]]:
    """
    Predict vehicle failure risk.
    
    Args:
        vehicle_data: Dictionary with sensor readings
        model: Trained XGBoost model
        
    Returns:
        Dictionary with risk score, level, and recommendation
    """
    # Implementation
    pass
```

### Security Guidelines

- Never commit secrets, API keys, or credentials
- Use environment variables for sensitive configuration
- Sanitize all user inputs
- Follow OWASP security best practices
- Run security scans before submitting PRs

### Testing Requirements

- Maintain minimum 80% code coverage
- Write unit tests for all new functions
- Include integration tests for API endpoints
- Add edge case and error handling tests

### Documentation Standards

- Update README.md for user-facing changes
- Add inline comments for complex logic
- Update API documentation for endpoint changes
- Include usage examples for new features

## Federal Compliance Considerations

When contributing to federal-focused projects:

- **Data Privacy**: Ensure no PHI or PII in code or tests
- **Security**: Follow NIST 800-171 security controls
- **Explainability**: Maintain model interpretability features
- **Audit Trails**: Log all significant operations
- **Compliance**: Document regulatory considerations

## Review Process

1. **Automated Checks**: CI/CD pipeline runs automatically
   - Code quality (linting, formatting)
   - Tests (unit, integration)
   - Security scans (Bandit, Snyk, Trivy)

2. **Manual Review**: Maintainers review code for:
   - Correctness and functionality
   - Code quality and maintainability
   - Security and compliance
   - Documentation completeness

3. **Approval**: At least one maintainer approval required

4. **Merge**: Squash and merge to main branch

## Development Setup

### Prerequisites

- Python 3.8+
- Docker 20.10+
- Git 2.30+

### Local Environment

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
pip install -r requirements-dev.txt  # Development dependencies

# Install pre-commit hooks
pre-commit install
```

### Running Tests Locally

```bash
# All tests
pytest tests/ -v

# With coverage
pytest tests/ --cov=src --cov-report=html

# Specific test file
pytest tests/test_models.py -v

# Specific test
pytest tests/test_models.py::test_xgboost_training -v
```

### Building Docker Images

```bash
# Build image
docker build -t project-name:dev .

# Run container
docker run -p 8000:8000 project-name:dev

# Run tests in container
docker run project-name:dev pytest tests/
```

## Project Structure

When adding new features, follow the existing structure:

```
project-name/
├── src/
│   ├── data/          # Data processing
│   ├── models/        # ML models
│   ├── api/           # API endpoints
│   └── utils/         # Utilities
├── tests/
│   ├── test_data.py
│   ├── test_models.py
│   └── test_api.py
├── docs/              # Documentation
├── notebooks/         # Jupyter notebooks
└── README.md
```

## Questions?

If you have questions about contributing:

- Open a discussion in GitHub Discussions
- Review existing issues and PRs
- Contact maintainers via email

Thank you for contributing to federal machine learning excellence!
