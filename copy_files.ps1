$sourceDir = "."
$targetDir = "../Deep-Learning-Image-Matching-Clean"

# Create necessary directories
$directories = @("src", "tests", "configs", "scripts")
foreach ($dir in $directories) {
    New-Item -ItemType Directory -Path "$targetDir/$dir" -Force
}

# Copy essential files
Copy-Item "README.md" "$targetDir/"
Copy-Item "requirements.txt" "$targetDir/"
Copy-Item ".gitignore" "$targetDir/"
Copy-Item "Project_Report.txt" "$targetDir/"
Copy-Item "Hedeflerimiz.txt" "$targetDir/"
Copy-Item "Objectives.txt" "$targetDir/"
Copy-Item "run_tests.py" "$targetDir/"

# Copy directories
Copy-Item "src/*" "$targetDir/src/" -Recurse
Copy-Item "tests/*" "$targetDir/tests/" -Recurse
Copy-Item "configs/*" "$targetDir/configs/" -Recurse
Copy-Item "scripts/*" "$targetDir/scripts/" -Recurse 
