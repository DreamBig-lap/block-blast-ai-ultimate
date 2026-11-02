# Contributing to Block Blast AI

Thank you for your interest in contributing to **Block Blast AI**! We welcome community contributions, bug reports, and enhancements.

To ensure stability and performance, please review this guide before submitting pull requests.

---

## Code of Conduct

Be welcoming, polite, and constructive. We value safe, reproducible, and verifiable engineering.

---

## Development Environment Setup

### 1. Prerequisites
- **Python**: 3.9 or higher (tested up to 3.13)
- **ADB (Android Debug Bridge)**: Installed and added to system `PATH`
- **Git**

### 2. Installation
Clone the repository and install the project in editable mode with development dependencies:

```bash
git clone https://github.com/<your-username>/block-blast-ai.git
cd block-blast-ai

# Create virtual environment
python -m venv .venv

# Activate virtual environment
# Windows (PowerShell):
.venv\Scripts\Activate.ps1
# Linux / macOS:
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
pip install -e .
```

---

## Architecture Principles

Block Blast AI follows a strictly modular architecture with separated domains:

1. **`game_engine/`**: Pure mathematical domain logic.
   - 64-bit unsigned integer Bitboards (`uint64`).
   - Bitwise line clearing (`check_lines_bitboard`, `clear_lines_bitboard`).
   - Immutable piece representation (`Piece`, `CANONICAL_41_CATALOG`).
   - Zero I/O or external dependencies in this layer.

2. **`ai_engine/`**: High-performance decision-making.
   - 3-ply Beam Search across piece permutations.
   - Multi-metric evaluation (aggregate height, bumpiness, hole penalty, line clears, tray clearance).
   - Microsecond-level placement simulation.

3. **`computer_vision/`**: Screen perception.
   - Screen capture and dynamic board localization (`BoardDetector`).
   - HSV color segmentation & multi-metric confidence cell classifier (`CellClassifier`).
   - Piece identification via template matching against canonical shapes.

4. **`action_engine/`**: Hardware interaction and validation.
   - ADB screencap and raw socket streaming.
   - Human-like Bezier curve touch gestures via `adb shell sendevent` or `adb shell input swipe`.
   - Fast-Path verification and closed-loop visual tracking.
   - Dynamic device discovery (no hardcoded device serials).

---

## Testing Guidelines

Every change must maintain the existing test suite integrity:

### Running Tests
Execute the entire test suite using Python's built-in `unittest`:

```bash
python -m unittest discover tests
```

Or using `pytest`:

```bash
pytest tests/
```

### Pull Request Requirements
1. **100% Pass Rate**: All 246+ tests must pass without failures.
2. **Type Hints**: All functions and methods must include standard Python type annotations.
3. **No Hardcoded Serials/Paths**: Never hardcode device serial numbers (e.g., `R58N62...`) or absolute filesystem paths (`C:\Users\...`). Always use dynamic resolution (`get_default_device_id()`, `PROJECT_ROOT`, or configuration objects).
4. **Safety Verification**: Code changes impacting the physical drag controller must be tested in `dry_run=True` mode before testing on real hardware.

---

## Submitting Pull Requests

1. Fork the repository and create a new feature branch:
   ```bash
   git checkout -b feature/your-feature-name
   ```
2. Write clean code and add corresponding unit tests in `tests/`.
3. Verify all tests pass:
   ```bash
   python -m unittest discover tests
   ```
4. Commit your changes with clear, descriptive commit messages:
   ```bash
   git commit -m "feat(action_engine): improve fast-path verification threshold"
   ```
5. Push to your branch and open a Pull Request against `main`.

Thank you for helping make Block Blast AI faster, safer, and smarter!
