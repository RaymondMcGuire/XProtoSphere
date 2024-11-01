echo off
setlocal enabledelayedexpansion

set DIR="%cd%"

SET CLANG_FORMAT=clang-format

SET /a index=0
for /R %DIR% %%f in (*.h, *.c, *.hpp, *.cpp, *.cuh, *.cu) do (
    %CLANG_FORMAT% -style=file -i %%f
    set /a index+=1
    Set "NUM=0000!index!"&Set "NUM=!NUM:~-4!"
    echo [!NUM!]Format file: %%f
)
echo.
echo Total format !index! files.
echo.
pause
