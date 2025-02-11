Set COUNTER=1
:x

echo %Counter%
if "%Counter%"=="3" (
    echo "END!"
) else (
    timeout /t 1
    start cmd.exe /c "python intro_DP_WA.py %Counter%"
    set /A COUNTER+=1
    goto x
)