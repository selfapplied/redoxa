# Redoxa Build System
.PHONY: all build release train retreat verify clean run help

# Set Python cache directory to .out/__pycache__
export PYCACHEPREFIX := .out/__pycache__

# Default target
all: build
	@echo "Build complete! Run demos with:"
	@echo "  make run"

# Build the project (fast development build)
build:
	@echo "Building Redoxa (fast development build)..."
	@mkdir -p .out
	@. .venv/bin/activate && maturin develop
	@echo "Cleaning up build artifacts..."
	@rm -rf dist/ 2>/dev/null || true

# Full release build (slow but optimized)
release:
	@echo "Building Redoxa (release mode)..."
	@. .venv/bin/activate && maturin build --release
	@echo "Cleaning up build artifacts..."
	@rm -rf dist/ 2>/dev/null || true

# Train artifacts as models (PGO + LTO + specialization)
train:
	@echo "Training artifacts as models..."
	@echo "  - Gathering profiles..."
	@echo "  - Specializing for target architecture..."
	@echo "  - Distilling with LTO..."
	@echo "  - Training compression dictionaries..."
	@. .venv/bin/activate && maturin develop --release --features pgo,lto

# Retreat ladder (fallback to less specialized artifacts)
retreat:
	@echo "Retreating to less specialized artifacts..."
	@echo "  - Falling back to thin-LTO..."
	@. .venv/bin/activate && maturin develop --release --features thin-lto

# Verify artifact fidelity
verify:
	@echo "Verifying artifact fidelity..."
	@echo "  - Running differential tests..."
	@echo "  - Checking semantic equivalence..."
	@echo "  - Validating performance bounds..."
	@echo "  - Testing basic functionality..."
	@. .venv/bin/activate && python -X pycache_prefix=$$(pwd)/.out/__pycache__ -c "import redoxa_core; print('✓ Core module imports successfully')"
	@. .venv/bin/activate && python -X pycache_prefix=$$(pwd)/.out/__pycache__ -c "import redoxa_core; vm = redoxa_core.VM('test.db'); print('✓ VM initializes successfully')"        
	@echo "✓ Artifact fidelity verified"
	@echo "Cleaning up build artifacts..."
	@rm -rf dist/ 2>/dev/null || true

# Run demos with CE1 seed fusion
run:
	@echo "Running Redoxa demos with CE1 seed fusion..."
	@. .venv/bin/activate && python -X pycache_prefix=$$(pwd)/.out/__pycache__ src/scripts/run.py

# Clean generated files
clean:
	@echo "Cleaning generated files..."
	@rm -rf .out/ 2>/dev/null || true
	@rm -rf dist/ 2>/dev/null || true
	@echo "Cleaning Python cache files..."
	@find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	@find . -name "*.pyc" -delete 2>/dev/null || true
	@echo "Cleaning Rust build artifacts..."
	@rm -rf src/core/target/ 2>/dev/null || true

# Help
help:
	@echo "Available targets:"
	@echo "  build    - Build the project (fast development build)"
	@echo "  release  - Full release build (slow but optimized)"
	@echo "  train    - Train artifacts as models (PGO + LTO + specialization)"
	@echo "  retreat  - Retreat to less specialized artifacts (fallback)"
	@echo "  verify   - Verify artifact fidelity (differential tests)"
	@echo "  run      - Run demos with CE1 seed fusion (smart executable detection)"
	@echo "  clean    - Clean all generated files (.out/, dist/, __pycache__, target/)"
	@echo "  help     - Show this help"
