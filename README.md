# Shock-Based Model Retraining

> Learn to detect distribution shocks in production ML models and trigger retraining based on data drift, not arbitrary schedules—through hands-on Python code.

## What You'll Learn

- Why calendar-based retraining fails and how distribution shocks actually degrade model performance
- Implement statistical drift detection methods: Kolmogorov-Smirnov test, Population Stability Index, Jensen-Shannon divergence
- Build shock-based monitoring systems that trigger retraining only when data distributions actually change
- Design production-ready MLOps pipelines with logging, error handling, and configuration management
- Create end-to-end model monitoring systems that save compute costs and catch real degradation faster

## Prerequisites

- Basic Python syntax (functions, classes, loops)
- Familiarity with NumPy and Pandas
- Understanding of basic ML concepts (training, prediction, features)
- No MLOps experience required—we build from scratch

## How to Use This Course

This is a **code-first course**. Every lesson is a runnable Python script with inline comments explaining concepts. No slide decks, no theory walls—just working programs.

Start with `basics/01_hello_world.py` and progress sequentially. Each file ends with a challenge to extend what you learned.

## Course Roadmap

| # | File | Level | Concepts | Time |
|---|------|-------|----------|------|
| 01 | `basics/01_hello_world.py` | Basics | Simulate model drift, visualize performance decay | ~10 min |
| 02 | `basics/02_core_concepts.py` | Basics | Distribution comparison, KS test, statistical significance | ~15 min |
| 03 | `basics/03_first_real_program.py` | Basics | Detect drift in real feature data, trigger alerts | ~15 min |
| 04 | `intermediate/04_patterns_and_tools.py` | Intermediate | PSI, Jensen-Shannon divergence, multiple drift metrics | ~20 min |
| 05 | `intermediate/05_real_world_usage.py` | Intermediate | Multi-feature monitoring, aggregated drift scores | ~20 min |
| 06 | `intermediate/06_challenge.py` | Intermediate | Build complete shock detector with windowing | ~25 min |
| 07 | `advanced/07_deep_dive.py` | Advanced | Adaptive thresholds, statistical power, edge cases | ~30 min |
| 08 | `advanced/08_production_patterns.py` | Advanced | Logging, config, error handling, CLI interface | ~30 min |
| 09 | `advanced/09_capstone_project.py` | Advanced | End-to-end monitoring system with retraining pipeline | ~40 min |

**Total time:** ~3.5 hours of hands-on coding

## Quick Start

```bash
# Clone the repository
git clone https://github.com/vk032503/shock-based-model-retraining.git
cd shock-based-model-retraining

# Install dependencies
pip install -r requirements.txt

# Run the first lesson
python basics/01_hello_world.py

# Run tests to validate your understanding
pytest tests/
```

## Repository Structure

```
shock-based-model-retraining/
├── basics/
│   ├── 01_hello_world.py          # Simulate and visualize model drift
│   ├── 02_core_concepts.py        # Statistical drift detection fundamentals
│   └── 03_first_real_program.py   # First working drift detector
├── intermediate/
│   ├── 04_patterns_and_tools.py   # Multiple drift metrics (PSI, JS)
│   ├── 05_real_world_usage.py     # Multi-feature monitoring
│   └── 06_challenge.py            # Complete shock detector
├── advanced/
│   ├── 07_deep_dive.py            # Adaptive thresholds and edge cases
│   ├── 08_production_patterns.py  # Production-ready implementation
│   └── 09_capstone_project.py     # Full MLOps monitoring system
├── tests/
│   └── test_course.py             # Pytest validation suite
├── requirements.txt
├── .env.example
└── README.md
```

## Key Insights

**The Problem:** Research on 555K fraud transactions showed the Ebbinghaus forgetting curve has R² = -0.31 for model performance—worse than a flat line. Models don't gradually forget; they get shocked by sudden distribution shifts.

**The Solution:** Monitor data distributions in real-time and trigger retraining only when statistical tests detect significant drift. This approach:
- Saves compute costs (no unnecessary retraining)
- Catches real degradation faster (responds to actual changes)
- Provides interpretable signals (statistical significance)

## Source / Credits

- Original research article: [Why MLOps Retraining Schedules Fail — Models Don't Forget, They Get Shocked](https://towardsdatascience.com/why-mlops-retraining-schedules-fail-models-dont-forget-they-get-shocked/)
- Course designed for hands-on learning by running real code

## Contributing

Found a bug or have a suggestion? Open an issue or submit a PR. This course is designed to be practical and battle-tested.

## License

MIT License - Learn freely, build confidently.