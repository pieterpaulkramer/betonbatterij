<#
.SYNOPSIS
    Helper CLI for managing the local Docker Home Assistant test environment
    for BetonBatterij.

.EXAMPLE
    .\local_ha_test\run_test.ps1 start
    .\local_ha_test\run_test.ps1 test
    .\local_ha_test\run_test.ps1 seed -Scenario dhw_tank
    .\local_ha_test\run_test.ps1 simulate -Seconds 600
    .\local_ha_test\run_test.ps1 logs
    .\local_ha_test\run_test.ps1 reset
#>
[CmdletBinding()]
param(
    [Parameter(Position = 0)]
    [ValidateSet("start", "stop", "restart", "status", "logs", "seed", "simulate", "test", "reset")]
    [string]$Action = "test",

    [string]$Scenario = "cv_heating",
    [int]$Seconds = 600,
    [string]$Url = "http://127.0.0.1:8123",
    [switch]$Follow,
    [switch]$Full,
    [switch]$Rpi4
)

$ErrorActionPreference = "Stop"

$LocalDir = $PSScriptRoot
$ComposeFile = Join-Path $LocalDir "docker-compose.yml"
$ConfigDir = Join-Path $LocalDir "config"
$ShareDir = Join-Path $LocalDir "share"
$ScriptsDir = Join-Path $LocalDir "scripts"

function Invoke-Compose {
    param([string[]]$ComposeArgs)
    $baseArgs = @("compose", "-f", $ComposeFile)
    if ($Rpi4) {
        $baseArgs += @("--profile", "rpi4")
    }
    & docker @baseArgs @ComposeArgs
    if ($LASTEXITCODE -ne 0) {
        throw "docker compose failed with exit code $LASTEXITCODE"
    }
}

switch ($Action) {
    "start" {
        if ($Rpi4) {
            Invoke-Compose @("up", "-d", "homeassistant-rpi4")
        } else {
            Invoke-Compose @("up", "-d", "homeassistant")
        }
        Write-Host "Docker Home Assistant gestart. UI: $Url" -ForegroundColor Green
    }

    "stop" {
        Invoke-Compose @("down")
        Write-Host "Docker Home Assistant gestopt." -ForegroundColor Yellow
    }

    "restart" {
        if ($Rpi4) {
            Invoke-Compose @("restart", "homeassistant-rpi4")
        } else {
            Invoke-Compose @("restart", "homeassistant")
        }
        Write-Host "Docker Home Assistant herstart." -ForegroundColor Green
    }

    "status" {
        Invoke-Compose @("ps")
    }

    "logs" {
        $service = if ($Rpi4) { "homeassistant-rpi4" } else { "homeassistant" }
        $logArgs = @("logs", "--tail=120")
        if ($Follow) {
            $logArgs += "-f"
        }
        $logArgs += $service
        Invoke-Compose $logArgs
    }

    "seed" {
        & python (Join-Path $ScriptsDir "mock_house.py") --url $Url --scenario $Scenario
        exit $LASTEXITCODE
    }

    "simulate" {
        & python (Join-Path $ScriptsDir "mock_house.py") --url $Url --simulate-seconds $Seconds
        exit $LASTEXITCODE
    }

    "test" {
        $service = if ($Rpi4) { "homeassistant-rpi4" } else { "homeassistant" }
        Invoke-Compose @("up", "-d", $service)
        $env:HA_BASE_URL = $Url
        $env:HA_COMPOSE_SERVICE = $service
        $marker = if ($Full) { "e2e or e2e_full" } else { "e2e" }
        & python -m pytest -m $marker -v
        exit $LASTEXITCODE
    }

    "reset" {
        Write-Host "Stoppen van Docker Home Assistant en opschonen van runtime-state..." -ForegroundColor Yellow
        Invoke-Compose @("down")

        $pathsToRemove = @(
            (Join-Path $ConfigDir ".storage"),
            (Join-Path $ConfigDir "deps"),
            (Join-Path $ConfigDir "tts"),
            (Join-Path $ConfigDir ".cloud"),
            (Join-Path $ConfigDir ".HA_VERSION"),
            (Join-Path $ConfigDir "home-assistant_v2.db"),
            (Join-Path $ConfigDir "home-assistant_v2.db-wal"),
            (Join-Path $ConfigDir "home-assistant_v2.db-shm"),
            (Join-Path $ConfigDir "home-assistant.log"),
            (Join-Path $ConfigDir "home-assistant.log.1"),
            (Join-Path $ConfigDir "home-assistant.log.fault"),
            (Join-Path $ShareDir "betonbatterij_export"),
            (Join-Path $LocalDir ".ha_auth.json")
        )
        foreach ($item in $pathsToRemove) {
            if (Test-Path $item) {
                Remove-Item -Recurse -Force $item
                Write-Host "  Verwijderd: $item"
            }
        }
        Write-Host "Reset klaar. configuration.yaml en packages/ zijn behouden." -ForegroundColor Green
    }
}
