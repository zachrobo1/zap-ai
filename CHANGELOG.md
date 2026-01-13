# Changelog

## [0.3.1](https://github.com/zachrobo1/zap-ai/compare/v0.3.0...v0.3.1) (2026-01-13)


### Bug Fixes

* **ci:** trigger PyPI deploy from release-please workflow ([33f4139](https://github.com/zachrobo1/zap-ai/commit/33f413948f129a3e25eeb13ed892b6eb99e4322a))

## [0.3.0](https://github.com/zachrobo1/zap-ai/compare/v0.2.1...v0.3.0) (2026-01-13)


### Features

* add coverage and PyPI badges (v0.3.0) ([4c8485c](https://github.com/zachrobo1/zap-ai/commit/4c8485ca72227b2cb879822d19f5fd53b40a4d2d))
* add human-in-the-loop approval workflows ([#11](https://github.com/zachrobo1/zap-ai/issues/11)) ([3547b8f](https://github.com/zachrobo1/zap-ai/commit/3547b8f4b6569fffabce11f49aa8c4be84285458))
* **tracing:** add BaseTracingProvider ABC for extensible tracing ([#10](https://github.com/zachrobo1/zap-ai/issues/10)) ([a47cf49](https://github.com/zachrobo1/zap-ai/commit/a47cf49742d88d3e7f98214c440799d5c5355232))


### Bug Fixes

* improve CI coverage and versioning config ([193de3b](https://github.com/zachrobo1/zap-ai/commit/193de3b0dfeceaa06bdd4265982e10fc5d3784db))
* trigger release-please on push to main ([ec1f2c1](https://github.com/zachrobo1/zap-ai/commit/ec1f2c153173d051f2b08d01cd5833f6d74a03ad))


### Documentation

* add MkDocs documentation site with GitHub Pages deployment ([#9](https://github.com/zachrobo1/zap-ai/issues/9)) ([8e8ec6e](https://github.com/zachrobo1/zap-ai/commit/8e8ec6e4df2612c452dfdc1d3975c979c89e19d8))

## [0.2.1](https://github.com/zachrobo1/zap-ai/compare/v0.2.0...v0.2.1) (2026-01-08)


### Features

* add context support for dynamic prompts ([#4](https://github.com/zachrobo1/zap-ai/issues/4)) ([7751951](https://github.com/zachrobo1/zap-ai/commit/775195151644c681a982c58bae956c706d4318f5))
* add GitHub releases with changelog tracking ([#3](https://github.com/zachrobo1/zap-ai/issues/3)) ([d02ae20](https://github.com/zachrobo1/zap-ai/commit/d02ae20972f7e174d22c100c120fcfc64d9f2e1a))

## [0.2.0](https://github.com/zachrobo1/zap-ai/compare/v0.1.0...v0.2.0) (2025-01-04)

### Features

* add tracing module with Langfuse integration ([#2](https://github.com/zachrobo1/zap-ai/pull/2))
* add Temporal integration tests to CI ([#1](https://github.com/zachrobo1/zap-ai/pull/1))

## 0.1.0 (2024-12-29)

### Features

* Library MVP - initial release with core Zap functionality
* Temporal-based agent orchestration
* MCP tool integration via FastMCP
* LiteLLM provider abstraction
