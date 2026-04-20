# ML Internship Project Template

Welcome! This repository serves as a starting point for your project. The structure here is minimal, designed to give you a foundation to build upon. Feel free to modify, expand, and customize it as needed.

## Repository Structure

- `.gitignore` and `.dockerignore`: Prevents unnecessary files from being pushed to version control.
- `.pre-commit-config.yaml`: Configures `pre-commit` hooks to enforce code style and linting checks automatically before each commit.
- `.ruff.toml`: Enforces Python linting & code style.
- `uv.lock`: Lists essential Python libraries. Add to it as your project grows.
- `pyproject.toml`: Basic setup script if you turn your project into a package.
- `README.md`: This file. Update it as your project evolves!
- `Dockerfile`, `compose.yaml`: Required for containarizatino of your app, leave it to next steps for now.
- `notebooks/`: Store your Jupyter notebooks here.
- `scripts/`: Place your project scripts here (e.g., for data processing or model training).
- `data/`: A global folder where your data can be stored in different formats (e.g. raw and processed)

## Further Development
This template provides a basic structure. You should develop it further based on the specific requirements of your project. Here are some ideas:
- Add configurations or hyperparameters in a `config.yaml`.
- Write tests for your code in a `tests/` folder.
- Document your work here in the `README.md` as your project progresses.
- Convert your `.ipynb` notebooks into `.py` files and add to your commits, this will simplify collaboration and PR review.

Good luck with your internship!
