# CUDA WebShader snapshot

Source: https://github.com/SamG-Coder/cuda-webshader

Commit: `f0f3699b498cfe6fe5419e072a4f4e2faa63b781`

The `compiler/` and `runtime/` directories were copied from `src/compiler/`
and `src/runtime/` at this revision. The upstream MIT license is preserved in `LICENSE`.
The vendored `compiler/compiler.js` adds graphics derivative/texture intrinsics,
additional scalar math built-ins, and float-vector indexing for this port. The
runtime and other compiler modules retain their upstream implementation.

ShoreBreak compiles its CUDA at build time. The production graphics adapter binds
the emitted programs to native WebGPU vertex/fragment stages. Diagnostic compute
tests load artifact/ABI pairs through `GpuRuntime.kernel`. No external compiler
checkout is needed.
