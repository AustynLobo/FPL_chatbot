@echo off
cd C:\Users\austy\Downloads\fantasy-chatbot
call venv\Scripts\activate
echo Running FPL model at %date% %time% >> logs\task_scheduler.log
python fpl_predictor.py --export --s3-bucket my-fpl-predictions >> logs\task_scheduler.log 2>&1
echo Finished at %date% %time% >> logs\task_scheduler.log