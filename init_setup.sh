set -e

echo [$(date)]:   "START"
echo [$(date)]:  "Install python virtual environment"
python -m venv ./venv
echo [$(date)]: "Activate python virtual environment"
source ./venv/bin/activate
echo [$(date)]: "Install project requirements "
python -m pip install -r requirements.txt
python -m pip install .
echo [$(date)]: "END"
