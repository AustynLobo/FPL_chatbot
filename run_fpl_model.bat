@echo off
SETLOCAL
cd /d "C:\Users\austy\Downloads\fantasy-chatbot"

:: Set encoding to handle the fancy characters without crashing
set PYTHONIOENCODING=utf-8

:: Log the Start Time
echo [%date% %time%] START: FPL model run >> logs\task_scheduler.log
echo [%date% %time%] Export TOTW visualisations to S3 bucket >> logs\task_scheduler.log

call venv\Scripts\activate

:: Run the script. 
:: "> nul 2>&1" hides all output and errors from the console/log.
python fpl_predictor.py --export --s3-bucket my-fpl-predictions > nul 2>&1
python .\fpl_team_of_the_week.py --mode actual --export --s3-bucket my-fpl-predictions > nul 2>&1
python .\fpl_team_of_the_week.py --mode predict --export --s3-bucket my-fpl-predictions > nul 2>&1

echo [%date% %time%] FINISHED >> logs\task_scheduler.log
echo ------------------------------------------------ >> logs\task_scheduler.log