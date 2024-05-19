@echo off
setlocal ENABLEDELAYEDEXPANSION

REM Specify the path to the zip file
set "zipfile=C:\Users\me\Documents\biomotivate\ebooks.zip"
set "zipfile_new=C:\Users\me\Documents\biomotivate\ebooks_renamed.zip"

set/a fileNum = 1

REM Specify the temporary directory to extract the files
set "tempdir=%temp%\extracted_files"

REM Extract the files to a temporary directory
mkdir "%tempdir%"
powershell -Command "Expand-Archive -Path '%zipfile%' -DestinationPath '%tempdir%'"

REM Change directory to the temporary directory
cd /d "%tempdir%\DATASET 17 Recovery Books"

REM Delete files that don't end with the .epub extension
for %%f in (*) do (if not "%%~xf"==".epub" del "%%~f")

(for %%f in (*.epub) do (
    echo !fileNum!%%~xf	%%~nf%%~xf
    set/a fileNum += 1
)) > C:\Users\me\Documents\biomotivate\ebook_filenames.tsv

set/a fileNum = 1

for %%F in (*) do (
  ren "%%F" "!fileNum!%%~xF"
  set/a fileNum += 1
)

REM Re-zip the renamed files
powershell -Command "Compress-Archive -Path '*' -DestinationPath '%zipfile_new%' -Force"

REM Cleanup: Delete the temporary directory
cd /d "%~dp0" 
rd /s /q "%tempdir%"

echo Ok