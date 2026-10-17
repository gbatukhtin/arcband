# arcband

[![Crates.io](https://img.shields.io/crates/v/arcband.svg)](https://crates.io/crates/arcband)
[![Documentation](https://docs.rs/arcband/badge.svg)](https://docs.rs/arcband)
[![License](https://img.shields.io/badge/license-MIT%20OR%20Apache--2.0-blue.svg)](LICENSE)

A high-performance GPU vector text and glyph renderer built on top of `wgpu`. 

`arcband` decomposes Bézier curves into circular arcs and line segments, partitions them into vertical horizontal spatial bands, and resolves the winding numbers and signed distances per pixel directly in the fragment shader.

---

## Features

- **Direct GPU Evaluation:** No CPU rasterization or texture atlas baking. Glyphs are rendered at arbitrary scales and zoom levels without resolution loss or atlas bloat.
- **Band Partitioning:** Glyphs are split into 32 bands per unit cell, reducing the search space to only segments overlapping the active scanlines.
- **Sub-Pixel Anti-Aliasing:** Analytic distance-to-curve approximation provides smooth, artifact-free edges.
- **Zero-Overhead Clipping:** Rectangular bounds clipping is resolved entirely in the vertex stage without fragment discard branches.
- **Layered Composition:** Built-in multi-layer rendering support (`LayeredTextRenderer`) for UI and complex multi-pass text stacking.
- **Raw Throughput:** Designed for low frame latency and massive glyph instance counts (~45,000+ glyphs rendered at 1500+ FPS on modern GPUs).

---

## Installation

Add `arcband` to your `Cargo.toml`:

```toml
[dependencies]
arcband = "0.2.1"
wgpu = "30.0"