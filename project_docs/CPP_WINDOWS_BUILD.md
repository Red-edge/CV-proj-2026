# C++ Windows Build

## Summary

The repository now includes a new C++ runtime under `cpp/` and a root `CMakeLists.txt`.

## Current dependency targets

- OpenCV
- Hikrobot MVS SDK
- ONNX Runtime CUDA or OpenCV DNN fallback
- PEAK PCAN-Basic

## Presets

Configure:

```powershell
cmake --preset windows-msvc-release
```

Build:

```powershell
cmake --build --preset windows-msvc-release
```

## Important note

This workstation currently did not expose `cmake` / `cl` in the active terminal during implementation, so you may need to open a Visual Studio Developer Command Prompt first or install the Visual Studio Build Tools + CMake.

## Example run

```powershell
.\build\windows-msvc-release\Release\central_control.exe `
  --config .\configs\windows_default.yaml `
  --record-rendered `
  --headless `
  --duration-sec 30 `
  --backend cuda
```

## Runtime switches

- `--headless`: disable live GUI window
- `--record-rendered`: save processed output video
- `--no-record`: force no recording even if config enables recording
