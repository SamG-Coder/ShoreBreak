# CUDA WebShader snapshot

Source: https://github.com/SamG-Coder/cuda-webshader

Commit: `f0f3699b498cfe6fe5419e072a4f4e2faa63b781`

The `compiler/` and `runtime/` directories are an unmodified snapshot of `src/compiler/`
and `src/runtime/` at this revision. The repository had an unrelated untracked showcase;
that directory is not included. The upstream MIT license is preserved in `LICENSE`.

ShoreBreak compiles its CUDA at build time and loads the resulting artifact/ABI pairs
through `GpuRuntime.kernel`. No absolute paths or external compiler checkout are needed.
