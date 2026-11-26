# Builds the WebAssembly version into dist/ (index.html, JS loader, .wasm).
# Serve that folder with any static file server, e.g.
#   python -m http.server 8080 -d dist
# (Exit codes are checked by hand: cargo writes progress to stderr, which
# Windows PowerShell 5.1 would treat as an error under ErrorAction Stop.)
Set-Location $PSScriptRoot

cargo build --release --target wasm32-unknown-unknown
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

# The JS loader must match the macroquad version in Cargo.lock, so take it
# from that crate's source instead of keeping a copy in the repo.
$meta = cargo metadata --format-version 1 | ConvertFrom-Json
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
$macroquad = $meta.packages | Where-Object { $_.name -eq "macroquad" }
$bundle = Join-Path (Split-Path $macroquad.manifest_path) "js\mq_js_bundle.js"

New-Item -ItemType Directory -Force dist | Out-Null
Copy-Item web\index.html dist\ -ErrorAction Stop
Copy-Item $bundle dist\ -ErrorAction Stop
Copy-Item target\wasm32-unknown-unknown\release\gravity-sim.wasm dist\ -ErrorAction Stop
Get-ChildItem dist | Format-Table Name, Length -AutoSize
