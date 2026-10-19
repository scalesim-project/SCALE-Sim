# Ramulator 2.1 portability patch

`0001-clang-dependent-template.patch` applies to upstream commit
`c6c7a5de0cef35935d6fad7369e6271406798865`.

Apple Clang 17 rejects `config[name].as<T>()` inside the templated
`ParamReader<T>` with “use 'template' keyword to treat 'as' as a dependent
template name”. The patch adds the standard C++ `template` disambiguator. It
changes no simulation logic or timing parameters.

`../setup_ramulator2.sh` applies this patch and accepts it on subsequent runs;
it refuses additional tracked or staged vendor changes. The vendor checkout is
therefore intentionally modified by exactly this patch, rather than pristine.

The library is built in C++20 mode without Python bindings. On Apple Silicon the
setup script detects the hardware using `sysctl`, because an Intel-installed
Bash or CMake may otherwise select x86_64. Set `RAMULATOR2_ARCH` to override that
choice.
