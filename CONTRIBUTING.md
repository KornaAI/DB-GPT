# Contribution 

First of all, thank you for considering contributing to this project. 
It's people like you that make it a reality for the community. There are many ways to contribute, and we appreciate all of them.

This guide will help you get started with contributing to this project.

## Fork The Repository

1. Fork the repository you want to contribute to by clicking the "Fork" button on the project page.

2. Clone the repository to your local machine using the following command:

```
git clone https://github.com/<YOUR-GITHUB-USERNAME>/DB-GPT
```
Please replace `<YOUR-GITHUB-USERNAME>` with your GitHub username.


## Create A New Development Environment

1. Create a new virtual environment using the following command:
```
# Make sure python >= 3.10
conda create -n dbgpt_env python=3.10
conda activate dbgpt_env
```

2. Change to the project directory using the following command:
```
cd DB-GPT
```

3. Install uv package manager:

There are several ways to install uv:

**Option 1: Using pipx (Recommended)**
```bash
python -m pip install --upgrade pip
python -m pip install --upgrade pipx
python -m pipx ensurepath
pipx install uv
```

**Option 2: Using curl (macOS/Linux)**
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Then verify uv installation:
```bash
uv --version
```

4. Install the project and all dependencies using uv:
```bash
# This will install all packages and development dependencies
# it will take some minutes
uv sync --all-packages \
--extra "base" \
--extra "proxy_openai" \
--extra "rag" \
--extra "storage_chromadb" \
--extra "dbgpts"
```

**Note for users in China:** If you encounter network issues, you can configure uv to use Chinese mirrors:
```bash
# Set environment variable for Tsinghua mirror
export UV_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple

# Or add --index-url parameter to the sync command
uv sync --all-packages \
--extra "base" \
--extra "proxy_openai" \
--extra "rag" \
--extra "storage_chromadb" \
--extra "dbgpts" \
--index-url=https://pypi.tuna.tsinghua.edu.cn/simple
```

5. Install pre-commit hooks
```
uv run pre-commit install
```

**Important Note:** After using `uv sync`, you should use `uv run` to execute commands in the virtual environment, or you can activate the environment with:
```bash
# Activate the virtual environment
source .venv/bin/activate  # On macOS/Linux
# or
.venv\Scripts\activate     # On Windows
```

6. Install `make` command
The `make` command has been installed by default on most Unix-based systems. If you not 
have it, you can install it by searching on the internet.

## New Branch And Make Changes

1. Create a new branch for your changes using the following command:
```
git checkout -b <branch-name>
```
Please replace `<branch-name>` with a descriptive name for your branch.

2. Make your changes to the code or documentation.

3. Add tests for your changes if necessary.

4. Format your code using the following command:
```
make fmt
```

5. Run the tests using the following command:
```
make test
```

6. Check types using the following command:
```
make mypy
```

7. Check lint using the following command:
```
make fmt-check
```

8. If all checks pass, you can add and commit your changes using the following commands:
```
git add xxxx
```
make sure to replace `xxxx` with the files you want to commit.

then commit your changes using the following command:
```
git commit -m "your commit message"
```
Please replace `your commit message` with a meaningful commit message.

It will take some time to get used to the process, but it's worth it. And it will run 
all git hooks and checks before you commit. If it fails, you need to fix the issues 
then re-commit it.

9. Push the changes to your forked repository using the following command:
```
git push origin <branch-name>
```

## Create A Pull Request

1. Go to the GitHub website and navigate to your forked repository.

2. Click the "New pull request" button.

3. Select the branch you just pushed to and the branch you want to merge into on the original repository.
Write necessary information about your changes and click "Create pull request".

4. Wait for the project maintainer to review your changes and provide feedback.

That's it you made it 🐣⭐⭐

# Developing inside a Container

If you are using VS Code as your IDE for development, you can refer to the [configuration here](.devcontainer/README.md) to set up the Dev Containers development environment.
