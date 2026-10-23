<#
Crea el ambiente virtual en la raiz del repo e instala las dependencias
de todos los scripts de la Etapa 2 (limpieza, descarga, dummy y evaluacion).

Uso, desde la raiz del repo:
  .\setup_env.ps1             crea .venv si no existe e instala
  .\setup_env.ps1 -Recreate   borra .venv y lo crea de cero

Si PowerShell bloquea el script:
  powershell -ExecutionPolicy Bypass -File .\setup_env.ps1
#>
param(
    [string]$VenvDir = ".venv",
    [switch]$Recreate
)

$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot

function Invoke-Checked {
    param([string]$Exe, [string[]]$Arguments)
    & $Exe @Arguments
    if ($LASTEXITCODE -ne 0) {
        throw "Fallo el comando: $Exe $($Arguments -join ' ')"
    }
}

# 1. Python del sistema
if (-not (Get-Command python -ErrorAction SilentlyContinue)) {
    throw "No encuentro 'python' en el PATH. Instala Python 3.10 o superior."
}
$ver = (& python -c "import sys; print('%d.%d' % sys.version_info[:2])").Trim()
if ([version]$ver -lt [version]"3.10") {
    throw "Se requiere Python 3.10 o superior y encontre $ver."
}
Write-Host "Python $ver detectado."

# 2. Ambiente virtual en la raiz
if ($Recreate -and (Test-Path $VenvDir)) {
    Write-Host "Borrando $VenvDir ..."
    Remove-Item -Recurse -Force $VenvDir
}
if (-not (Test-Path $VenvDir)) {
    Write-Host "Creando $VenvDir ..."
    Invoke-Checked "python" @("-m", "venv", $VenvDir)
}
$venvPy = Join-Path $VenvDir "Scripts\python.exe"
if (-not (Test-Path $venvPy)) {
    throw "El venv quedo incompleto. Corre de nuevo con -Recreate."
}

# 3. Dependencias
Invoke-Checked $venvPy @("-m", "pip", "install", "--upgrade", "pip")

# requirements.txt tiene todas las dependencias de la Etapa 2:
#   numpy, pandas, scipy, scikit-learn -> scripts/dataset_cleaner.py y los de evaluacion
#   kagglehub                          -> scripts/download_data.py
# Se busca en la raiz y en IA-Proyecto-Etapa2.
$req = @("requirements.txt", "IA-Proyecto-Etapa2\requirements.txt") |
    Where-Object { Test-Path $_ } | Select-Object -First 1
if (-not $req) {
    throw "No encontre requirements.txt ni en la raiz ni en IA-Proyecto-Etapa2."
}
Write-Host "Instalando desde $req ..."
Invoke-Checked $venvPy @("-m", "pip", "install", "--no-cache-dir", "-r", $req)

# 4. Verificacion (atrapa instalaciones incompletas, como la de sklearn)
Invoke-Checked $venvPy @("-c",
    "import numpy, pandas, scipy, sklearn, kagglehub; from sklearn.utils.validation import check_array; print('Imports OK. scikit-learn', sklearn.__version__)")

Write-Host ""
Write-Host "Listo. Activa el ambiente con:  .\$VenvDir\Scripts\Activate.ps1"
Write-Host "Antes de correr el limpiador:   `$env:PYTHONHASHSEED = '0'"
