echo [$(date)]:   "START"
echo [$(date)]:  "Install python virtual environment"
conda create --prefix ./venv python=3.11 -y
echo [$(date)]: "Activate python virtual environment"
source activate ./venv
echo [$(date)]: "Install project requirements "
pip install -r requirements.txt
echo [$(date)]: "END"