CLS
@echo off
TITLE FPGA LLM Client V1 localhost

::IP
for /f "tokens=16" %%i in ('ipconfig ^|find /i "ipv4"') do (
set ip=%%i
goto out
)
:out

cd /d C:\Users\david\Desktop\FPGA_Demo_Interface\client
python client_v1.py