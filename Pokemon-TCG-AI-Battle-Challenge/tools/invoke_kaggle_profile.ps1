param(
    [Parameter(Mandatory = $true)]
    [ValidateSet("bolt", "uza")]
    [string]$Profile,

    [Parameter(ValueFromRemainingArguments = $true)]
    [string[]]$KaggleArgs
)

$ErrorActionPreference = "Stop"

if ($Profile -eq "uza") {
    $token = [Environment]::GetEnvironmentVariable("KAGGLE_UZA_TOKEN", "User")
    if (-not $token) {
        throw "KAGGLE_UZA_TOKEN is not configured in the user environment."
    }
    $env:KAGGLE_API_TOKEN = $token
    Remove-Item Env:KAGGLE_USERNAME -ErrorAction SilentlyContinue
    Remove-Item Env:KAGGLE_KEY -ErrorAction SilentlyContinue
}
else {
    Remove-Item Env:KAGGLE_API_TOKEN -ErrorAction SilentlyContinue
}

& python -m kaggle @KaggleArgs
exit $LASTEXITCODE
