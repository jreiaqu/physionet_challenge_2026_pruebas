# Para lanzar el docker directamente sin tener que copiar y pegar siempre
# run_docker.ps1
Write-Host "Iniciando contenedor Docker con los volúmenes montados..." -ForegroundColor Green

docker run -it --rm `
   -v "${PWD}\..\training_data:/training_data" `
   -v "${PWD}\..\test_data:/test_data" `
   -v "${PWD}\..\models:/models" `
   -v "${PWD}\..\output_data:/output_data" `
   -v "${PWD}:/challenge" `
   image bash