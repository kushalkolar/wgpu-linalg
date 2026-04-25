// sparse matvec using scatter-add
//
//  y = (A @ c) * scale_factor + scale_add
//
//  A is [p, k] sparse, supplied as CSR (indptr, indices, values).
//  c is C[:, t], a single column of the dense [k, T] matrix C (row-major).
//  scale_factor and scale_add are dense [p,] vectors applied per row.
//  y has length p and is written row-major into an [m, n] r32float texture
//  with m * n = p, so y[i] -> tex[i % n, i / n].
//
//  One thread per output row.

// A, sparse CSR
@group(0) @binding(0) var<storage, read> indptr:  array<u32>;
@group(0) @binding(1) var<storage, read> indices: array<u32>;
@group(0) @binding(2) var<storage, read> values:  array<f32>;

// C
@group(0) @binding(3) var<storage, read> C:       array<f32>;

// t index
@group(0) @binding(4) var<uniform> t: u32;

// texture to visualize
@group(0) @binding(5) var out_tex: texture_storage_2d<r32float, write>;

@group(0) @binding(6) var<storage, read> scale_factor: array<f32>;
@group(0) @binding(7) var<storage, read> scale_add:    array<f32>;

override wg_size: u32;
override T: u32;
override n_cols: u32;

@compute @workgroup_size(wg_size)
fn spmv(@builtin(global_invocation_id) gid: vec3u) {
    let row = gid.x;
    let p = arrayLength(&indptr) - 1u;
    if (row >= p) {
        return;
    }

    let row_start = indptr[row];
    let row_end   = indptr[row + 1u];

    var sum: f32 = 0.0;
    for (var j: u32 = row_start; j < row_end; j = j + 1u) {
        let col = indices[j];
        sum = sum + values[j] * C[col * T + t];
    }

    let val = fma(scale_factor[row], sum, scale_add[row]);
    textureStore(out_tex, vec2u(row % n_cols, row / n_cols), vec4f(val, 0.0, 0.0, 0.0));
}
