@echo off
set "SCRIPT_PATH=%~dp0run_hidden.vbs"
set "STARTUP_FOLDER=%APPDATA%\Microsoft\Windows\Start Menu\Programs\Startup"
set "SHORTCUT_NAME=%STARTUP_FOLDER%\AI_Video_Summarizer.lnk"

echo [1/1] Adding AI Video Summarizer to Windows Startup...

powershell -Command "$s=(New-Object -COM WScript.Shell).CreateShortcut('%SHORTCUT_NAME%');$s.TargetPath='wscript.exe';$s.Arguments='\"%SCRIPT_PATH%\"';$s.WorkingDirectory='%~dp0';$s.Save()"

echo SUCCESS! The AI engine will now start automatically whenever you turn on your PC.
echo Refresh your browser to start using it now.
pause
