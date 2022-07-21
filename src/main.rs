extern crate dlfs_rust as dlfs;

use dlfs::utils::mnist::Mnist;
use anyhow::Result;

const BATCH_SIZE: usize = 100;

#[tokio::main]
async fn main() -> Result<()> {
    let mnist: Mnist<f32> = Mnist::new().await?.normalize();

    for i in (0..mnist.train_images.nrows()).step_by(BATCH_SIZE) {
        let input_batch = mnist.train_images.fixed_rows::<BATCH_SIZE>(i);
        println!("{}", input_batch.len());
    }

    Ok(())
}
