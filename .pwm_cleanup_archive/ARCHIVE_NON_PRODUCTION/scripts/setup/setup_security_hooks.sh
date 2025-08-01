#!/bin/bash

# LUKHAS AGI Security Setup Script
# Sets up pre-commit hooks and secret scanning tools

echo "🔒 Setting up LUKHAS AGI Security Tools..."

# Check if pre-commit is installed
if ! command -v pre-commit &> /dev/null; then
    echo "📦 Installing pre-commit..."
    pip install pre-commit
fi

# Install pre-commit hooks
echo "🔧 Installing pre-commit hooks..."
pre-commit install

# Install detect-secrets if not present
if ! command -v detect-secrets &> /dev/null; then
    echo "📦 Installing detect-secrets..."
    pip install detect-secrets
fi

# Create secrets baseline (ignore existing secrets for now)
echo "📋 Creating secrets baseline..."
detect-secrets scan --baseline .secrets.baseline --exclude-files '\.env$|\.env\..*|venv/|\.venv/|__pycache__/|\.git/'

# Create additional security tools
echo "🛡️ Setting up additional security tools..."

# Create a simple secret scanner script
cat > scripts/scan_secrets.sh << 'EOF'
#!/bin/bash
echo "🔍 Running comprehensive secret scan..."

echo "📋 Using detect-secrets..."
detect-secrets scan --exclude-files '\.env$|\.env\..*|venv/|\.venv/|__pycache__/|\.git/' --force-use-all-plugins

echo "🔍 Checking for common patterns..."
echo "API Keys:"
grep -r "api_key\s*=\s*['\"][^'\"\$]" --include="*.py" . --exclude-dir=.venv --exclude-dir=venv || echo "✅ No hardcoded API keys found"

echo "Passwords:"
grep -r "password\s*=\s*['\"][^'\"\$]" --include="*.py" . --exclude-dir=.venv --exclude-dir=venv | grep -v "placeholder\|example\|test" || echo "✅ No hardcoded passwords found"

echo "Secrets:"
grep -r "secret.*=.*['\"][^'\"\$]" --include="*.py" . --exclude-dir=.venv --exclude-dir=venv | grep -v "placeholder\|example\|test" || echo "✅ No hardcoded secrets found"

echo "🎯 Scan complete!"
EOF

# Make scripts directory and set permissions
mkdir -p scripts
chmod +x scripts/scan_secrets.sh

# Create a git hook for push
cat > .git/hooks/pre-push << 'EOF'
#!/bin/bash
echo "🔒 Running security checks before push..."

# Run secret scan
if command -v detect-secrets &> /dev/null; then
    detect-secrets scan --baseline .secrets.baseline --exclude-files '\.env$|\.env\..*|venv/|\.venv/|__pycache__/|\.git/'
    if [ $? -ne 0 ]; then
        echo "❌ Secret scan failed! Push aborted."
        exit 1
    fi
fi

echo "✅ Security checks passed!"
EOF

chmod +x .git/hooks/pre-push

# Update .env.example with the new required variables
echo "📝 Updating .env.example with new security variables..."

# Add the new SMTP and JWT variables to .env.example if not already present
if ! grep -q "SMTP_USERNAME" .env.example; then
    cat >> .env.example << 'EOF'

# ===== Email Configuration =====
# SMTP server credentials
SMTP_USERNAME=your_smtp_username_here
SMTP_PASSWORD=your_smtp_password_here
SMTP_SENDER=hello@your-domain.com
SMTP_SERVER=smtp-relay.brevo.com
SMTP_PORT=587

# ===== JWT Security =====
# JWT secret key for token signing (generate with: openssl rand -base64 64)
JWT_SECRET_KEY=generate_secure_jwt_secret_here
EOF
fi

# Update the .env file too if it exists
if [ -f .env ]; then
    if ! grep -q "SMTP_USERNAME" .env; then
        echo "" >> .env
        echo "# Email Configuration" >> .env
        echo "SMTP_USERNAME=your_smtp_username_here" >> .env
        echo "SMTP_PASSWORD=your_smtp_password_here" >> .env
        echo "SMTP_SENDER=hello@your-domain.com" >> .env
        echo "SMTP_SERVER=smtp-relay.brevo.com" >> .env
        echo "SMTP_PORT=587" >> .env
        echo "" >> .env
        echo "# JWT Security" >> .env
        echo "JWT_SECRET_KEY=generate_secure_jwt_secret_here" >> .env
    fi
fi

echo "🎉 Security setup complete!"
echo ""
echo "📋 Next steps:"
echo "1. Update your .env file with real values"
echo "2. Generate a secure JWT secret: openssl rand -base64 64"
echo "3. Test the setup: pre-commit run --all-files"
echo "4. Run security scan: ./scripts/scan_secrets.sh"
echo ""
echo "🔒 Security tools installed:"
echo "  ✅ Pre-commit hooks"
echo "  ✅ Secret scanning (detect-secrets)"
echo "  ✅ Custom LUKHAS secret patterns"
echo "  ✅ Git pre-push security checks"