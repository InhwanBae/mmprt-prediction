<h2 align="center">Deep-Learning Based MMPRT Prediction</h2>
<!-- <p align="center">
  <a href="https://InhwanBae.github.io/"><strong>Inhwan Bae</strong></a>
  ·  
  <a href="#"><strong>Author 1</strong></a>
  ·
  <a href="#"><strong>Author 2</strong></a>
  <br>
  Journal Name, Year
</p> -->

<p align="center">
  <a href="https://inhwanbae.github.io/mmprt-prediction/"><strong><code>Web Application</code></strong></a>
  <a href="#"><strong><code>Publication</code></strong></a>
  <a href="https://github.com/InhwanBae/mmprt-prediction"><strong><code>Source Code</code></strong></a>
</p>

<div align='center'>
  <br><img src="assets/img/figure-web-application.png" width=100%>
  <br><em>Screenshot of the online MMPRT prediction service.</em>
</div>

<br>This repository provides the source code for training and testing machine learning and deep learning models to predict Medial Meniscus Posterior Root Tear (MMPRT) using patient demographic data and measurements.

<br>

## Model Inference
For a live demonstration, you can directly use the pre-trained deep-learning models on our [**Web Application**](https://inhwanbae.github.io/mmprt-prediction/)

<br>

## Training your own models
You can train and test your own models using the provided code. Follow the instructions below.
  - [**`Environment Setup (Windows)`**](#environment-setup-windows): Instructions to set up environment.
  - [**`Data Preparation and Model Training`**](#data-preparation-and-model-training): Instructions to evaluate models with your own data.

<br>

### Environment Setup

1. Install Python 3.11
  - Download the ["Python 3.11 installer"](https://www.python.org/ftp/python/3.11.9/python-3.11.9-amd64.exe) from the official website: https://python.org
  - Run the installer. Crucially, check the box for "Add python.exe to PATH" and then choose "Install Now".

2. Download the source code
  - In your browser, go to: https://github.com/InhwanBae/mmprt-prediction
  - Click the green "Code" button → "Download ZIP".
    - Alternatively, you can use this direct [ZIP link](https://github.com/InhwanBae/mmprt-prediction/archive/refs/heads/main.zip).
  - Open File Explorer, right-click the downloaded ZIP → "Extract All...", choose a destination folder (e.g., `C:\Users\User\Documents`) and click "Extract".

3. Create a virtual environment & install requirements
  - In File Explorer, navigate to the extracted folder (e.g., `C:\Users\User\Documents\mmprt-prediction-main`).
  - To open Command Prompt in this folder, click the address bar of File Explorer, type `cmd`, and press Enter.
  - In the opened Command Prompt window, run the following commands one by one by one:
    ```cmd
    :: Create a virtual environment named .venv
    python -m venv .venv
    
    :: Activate the virtual environment
    .venv\Scripts\activate

    :: Install the required packages
    pip install -r utils/requirements.txt
    ```

<br>

### Data Preparation and Model Training

4. Prepare your data
  - Place your CSV data file(s) in the `data` folder.
  - Check the configuration files in the `config` folder to match your dataset's schema.
    - `config/input_output_cols.yaml`: Define the model's input and output columns.
    - `config/group_cols.yaml`: Specify column groupings used by the deep learning model.

5. Run Model Training and Evaluation
  - Once your environment is set up and data is prepared, you can run the training and evaluation scripts.
  - To train and test Machine Learning models: Double-click `Run_Machine_Learning.bat`.
  - To train and test Deep Learning models: Double-click `Run_Deep_Learning.bat`.
    - Alternatively, you can run the following command in Command Prompt:
        ```cmd
        :: Activate the virtual environment
        .venv\Scripts\activate

        :: Run the training and testing scripts
        python train_test_ML_models.py
        python train_test_DL_models.py
        ```
  - The trained models and evaluation results will be saved in the `results` folder. You can repeat this process (4 and 5) with different data or configuration settings.
