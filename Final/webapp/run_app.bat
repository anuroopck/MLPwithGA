@echo off
echo Installing dependencies...
pip install -r requirements.txt

echo Starting Streamlit app...
streamlit run app.py

pause

