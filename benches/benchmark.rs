extern crate dlfs_rust as dlfs;
extern crate nalgebra as na;

use criterion::{criterion_group, criterion_main, Criterion};
use dlfs::utils::activation_function::*;
use na::DMatrix;

fn softmax_f32(c: &mut Criterion) {
    let mut m: DMatrix<f32> = na::DMatrix::from_element(1000, 10000, 1.123f32);
    c.bench_function("softmax_f32", |b| b.iter(|| m.softmax_mut()));
}

fn softmax_f64(c: &mut Criterion) {
    let mut m: DMatrix<f64> = na::DMatrix::from_element(1000, 10000, 1.123f64);
    c.bench_function("softmax_f64", |b| b.iter(|| m.softmax_mut()));
}

criterion_group!(
    benches,
    softmax_f32,
    softmax_f64
);
criterion_main!(benches);
