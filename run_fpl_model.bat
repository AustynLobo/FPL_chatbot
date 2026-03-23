@echo off
SETLOCAL
cd /d "C:\Users\austy\Downloads\fantasy-chatbot"

set PYTHONIOENCODING=utf-8

echo [%date% %time%] START: FPL model run >> logs\task_scheduler.log 2>&1

call venv\Scripts\activate >> logs\task_scheduler.log 2>&1

echo [%date% %time%] Running fpl_predictor.py >> logs\task_scheduler.log 2>&1
python fpl_predictor.py --export --s3-bucket my-fpl-predictions >> logs\task_scheduler.log 2>&1
echo [%date% %time%] fpl_predictor.py exit code: %ERRORLEVEL% >> logs\task_scheduler.log 2>&1

echo [%date% %time%] Running TOTW actual >> logs\task_scheduler.log 2>&1
python fpl_team_of_the_week.py --mode actual --save --export --s3-bucket my-fpl-predictions>> logs\task_scheduler.log 2>&1
echo [%date% %time%] TOTW actual exit code: %ERRORLEVEL% >> logs\task_scheduler.log 2>&1

echo [%date% %time%] Running TOTW predict >> logs\task_scheduler.log 2>&1
python fpl_team_of_the_week.py --mode predict --save --export --s3-bucket my-fpl-predictions >> logs\task_scheduler.log 2>&1
echo [%date% %time%] TOTW predict exit code: %ERRORLEVEL% >> logs\task_scheduler.log 2>&1

echo [%date% %time%] Uploading charts to S3 >> logs\task_scheduler.log 2>&1
aws s3 cp data\predictions\ s3://my-fpl-predictions/charts/ --recursive --exclude "*" --include "*.png" >> logs\task_scheduler.log 2>&1
echo [%date% %time%] S3 upload exit code: %ERRORLEVEL% >> logs\task_scheduler.log 2>&1

echo [%date% %time%] FINISHED >> logs\task_scheduler.log 2>&1
echo ------------------------------------------------ >> logs\task_scheduler.log 2>&1