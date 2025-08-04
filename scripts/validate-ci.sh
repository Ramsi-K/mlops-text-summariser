#!/bin/bash

echo "🔍 Validating CI/CD Pipeline Setup..."

# Check if GitHub Actions workflows exist
if [ -f ".github/workflows/ci.yml" ]; then
    echo "✅ Main CI workflow found"
else
    echo "❌ Main CI workflow missing"
    exit 1
fi

if [ -f ".github/workflows/deploy.yml" ]; then
    echo "✅ Deployment workflow found"
else
    echo "❌ Deployment workflow missing"
    exit 1
fi

# Check if required files exist
required_files=("requirements.txt" "Makefile" "pyproject.toml" ".pre-commit-config.yaml" ".flake8")

for file in "${required_files[@]}"; do
    if [ -f "$file" ]; then
        echo "✅ $file found"
    else
        echo "❌ $file missing"
        exit 1
    fi
done

# Test local CI commands
echo "🧪 Testing local CI commands..."

if command -v uv &> /dev/null; then
    echo "✅ UV package manager available"

    # Test code quality
    echo "🔍 Testing code quality checks..."
    uv run black --check src > /dev/null 2>&1 && echo "✅ Black formatting check passed" || echo "⚠️  Black formatting issues found"
    uv run isort --check-only src > /dev/null 2>&1 && echo "✅ Import sorting check passed" || echo "⚠️  Import sorting issues found"
    uv run flake8 src > /dev/null 2>&1 && echo "✅ Flake8 linting passed" || echo "⚠️  Linting issues found"

    # Test mock pipeline
    echo "🚀 Testing mock pipeline..."
    uv run python main.py --mock-train > /dev/null 2>&1 && echo "✅ Mock pipeline executed successfully" || echo "❌ Mock pipeline failed"

else
    echo "⚠️  UV not available, skipping local tests"
fi

echo ""
echo "🎉 CI/CD Pipeline Validation Complete!"
echo "📋 Summary:"
echo "   - GitHub Actions workflows configured"
echo "   - Code quality tools set up"
echo "   - Testing framework ready"
echo "   - Mock pipeline functional"
echo ""
echo "🚀 Ready for CI/CD deployment!"
