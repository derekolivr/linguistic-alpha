# Linguistic Alpha

This project aims to find alpha in financial markets by analyzing the linguistic properties of corporate earnings calls and SEC filings. The core hypothesis is that the way executives communicate can predict future stock price volatility.

## Getting Started

Follow these instructions to set up your local development environment and run the analysis notebook.

### Prerequisites

- Python 3.9+
- [Visual Studio Code](https://code.visualstudio.com/)
- [Python Extension for VS Code](https://marketplace.visualstudio.com/items?itemName=ms-python.python)

### Setup Instructions

1.  **Clone the Repository**

    ```bash
    git clone <your-repository-url>
    cd linguistic-alpha
    ```

2.  **Create and Activate the Virtual Environment**

    We use a `venv` to manage project dependencies.

    - **Create the environment:**
      ```bash
      python3 -m venv venv
      ```
    - **Activate it:** - On macOS/Linux:
      `bash
  source venv/bin/activate
  ` - On Windows:
      `bash
  .\\venv\\Scripts\\activate
  `
      You'll know it's activated when you see `(venv)` at the beginning of your terminal prompt.

3.  **Install Dependencies**
    ```bash
    pip install -r requirements.txt
    ```

### Running the Analysis Notebook

The primary analysis is done in a Jupyter Notebook, which allows for interactive exploration and visualization of the data.

1.  **Open the Project in VS Code**

    ```bash
    code .
    ```

2.  **Open the Notebook**
    Navigate to and open the `analysis/backtesting.ipynb` file in VS Code.

3.  **Select the Python Kernel**

    - When you open the notebook, VS Code may prompt you to select a kernel. Click on "Select Kernel" in the top right.
    - Choose the Python interpreter that is located inside your `venv` directory (e.g., `venv/bin/python`). If it's not listed, you can select "Python Environments" and browse to find it.
    - Once selected, the top right of the notebook interface should show `(venv)`.

4.  **Run the Cells**
    You can now run the cells in the notebook one by one to execute the analysis pipeline, see the dataframes, and view the correlation results.
