# Cotton Plant Disease Classification using CNN

# Steps

## 1. Install Python 3.11.4 and Add to PATH

### Windows:
* Download Python 3.11.4 from [python.org](https://www.python.org/downloads/release/python-3114/).
* During installation, check **"Add Python to PATH"**.
* Verify installation using:
```sh
    python --version
    python3 --version
```
### macOS/Linux
* `brew install python@3.11` or `sudo apt update && sudo apt install python3.11`
* Verify installation using:
```sh
python3 --version
```
a
## 2. Clone the GitHub Repository
* `git clone https://github.com/InkolluvenkatBhargav/CNN.git`
* `cd CNN.git`
* Download Dataset from [Here](https://drive.google.com/file/d/177pPHpf9H9wbR0df9vw-M6v_omxJf5BF/view?usp=sharing)

## 3. Create a Virtual Environment
* Windows `python -m venv venv`
* macOS/Linux `python3 -m venv venv`

## 4. Activate the Virtual Environment
* Windows CMD `venv\Scripts\activate`
* Windows Powershell `venv\Scripts\Activate.ps1`
* macOS/Linux `source venv/bin/activate`
   
## 5. Install Dependencies from requirements.txt
* In terminal `pip install -r requirements.txt`

## 6. Run CNN Model
* In Terminal run `python model.py`
* After successful execution you should see a file named
    * pred_cotton.h5 
* For Visualization execute each cell in `model.ipynb` by selecting venv kernel

## 7. Run Flask Application
* `python index.py`

## 8. Open Browser
* Go to `http://127.0.0.1:5000/`
