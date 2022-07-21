use std::path::Path;
use std::fs::{self, File};
use std::io::Read;
use na::DMatrix;
use anyhow::{Result, Context};
use futures;
use flate2::read::GzDecoder;

const URL: &str = "http://yann.lecun.com/exdb/mnist/";
const DATA_PATH: &str = "./dataset/";
const TRAINING_DATA: &str = "train-images-idx3-ubyte.gz";
const TRAINING_LABEL: &str = "train-labels-idx1-ubyte.gz";
const TEST_DATA: &str = "t10k-images-idx3-ubyte.gz";
const TEST_LABEL: &str = "t10k-labels-idx1-ubyte.gz";

pub struct Mnist<T> {
    pub train_images: DMatrix<T>,
    pub train_labels: DMatrix<T>,
    pub test_images: DMatrix<T>,
    pub test_labels: DMatrix<T>,
}

impl<T> Mnist<T>
where
    T: num::Float + num::FromPrimitive + na::Scalar,
{
    pub async fn new() -> Result<Self> {
        if !Path::new(format!("{}{}", DATA_PATH, TRAINING_DATA).as_str()).exists()
            || !Path::new(format!("{}{}", DATA_PATH, TRAINING_LABEL).as_str()).exists()
            || !Path::new(format!("{}{}", DATA_PATH, TEST_DATA).as_str()).exists()
            || !Path::new(format!("{}{}", DATA_PATH, TEST_LABEL).as_str()).exists()
        {
            download_dataset().await.context("failed to download dataset")?;
        }

        let train_images = load_train_image::<T>().context("failed to load train images")?;
        let train_labels = load_train_label::<T>().context("failed to load train labels")?;
        let test_images = load_test_image::<T>().context("failed to load test images")?;
        let test_labels = load_test_label::<T>().context("failed to load test labels")?;

        Ok(Mnist {
            train_images,
            train_labels,
            test_images,
            test_labels,
        })
    }

    pub fn normalize(self) -> Self {
        Mnist {
            train_images: self.train_images.map(|p| p / T::from_u8(255).unwrap()),
            train_labels: self.train_labels,
            test_images: self.test_images.map(|p| p / T::from_u8(255).unwrap()),
            test_labels: self.test_labels,
        }
    }
}

async fn download_dataset() -> Result<()> {
    if !Path::new(DATA_PATH).exists() {
        fs::create_dir(DATA_PATH).context("failed to create dataset directory")?;
    }
    let train_data_response = reqwest::get(format!("{}{}", URL, TRAINING_DATA));
    let train_label_response = reqwest::get(format!("{}{}", URL, TRAINING_LABEL));
    let test_data_response = reqwest::get(format!("{}{}", URL, TEST_DATA));
    let test_label_response = reqwest::get(format!("{}{}", URL, TEST_LABEL));

    let (train_data_body, train_label_body, test_data_body, test_label_body) = futures::join!(
        train_data_response,
        train_label_response,
        test_data_response,
        test_label_response,
    );

    let (train_data_bytes, train_label_bytes, test_data_bytes, test_label_bytes) = futures::join!(
        train_data_body?.bytes(),
        train_label_body?.bytes(),
        test_data_body?.bytes(),
        test_label_body?.bytes(),
    );

    let mut train_data_file = File::create(format!("{}{}", DATA_PATH, TRAINING_DATA))?;
    let mut train_label_file = File::create(format!("{}{}", DATA_PATH, TRAINING_LABEL))?;
    let mut test_data_file = File::create(format!("{}{}", DATA_PATH, TEST_DATA))?;
    let mut test_label_file = File::create(format!("{}{}", DATA_PATH, TEST_LABEL))?;

    std::io::copy(&mut train_data_bytes?.as_ref(), &mut train_data_file)?;
    std::io::copy(&mut train_label_bytes?.as_ref(), &mut train_label_file)?;
    std::io::copy(&mut test_data_bytes?.as_ref(), &mut test_data_file)?;
    std::io::copy(&mut test_label_bytes?.as_ref(), &mut test_label_file)?;

    Ok(())
}

fn load_train_image<T>() -> Result<DMatrix<T>>
where
    T: num::Float + num::FromPrimitive + na::Scalar,
{
    let mut gz = GzDecoder::new(File::open(format!("{}{}", DATA_PATH, TRAINING_DATA))?);
    let mut data: Vec<u8> = Vec::with_capacity(47040016);
    gz.read_to_end(&mut data)?;
    Ok(DMatrix::from_row_slice(60000, 784, &data[16..]).map(|p| T::from_u8(p).unwrap()))
}

fn load_train_label<T>() -> Result<DMatrix<T>>
where
    T: num::Float + num::FromPrimitive + na::Scalar,
{
    let mut gz = GzDecoder::new(File::open(format!("{}{}", DATA_PATH, TRAINING_LABEL))?);
    let mut data: Vec<u8> = Vec::with_capacity(60008);
    let mut onehot_data = Vec::with_capacity(600000);
    gz.read_to_end(&mut data)?;
    data[8..].into_iter().for_each(|&l| {
        let mut onehot: [T; 10] = [T::zero(); 10];
        onehot[l as usize] = T::one();
        onehot.into_iter().for_each(|p| onehot_data.push(p));
    });
    Ok(DMatrix::from_row_slice(60000, 10, &onehot_data))
}

fn load_test_image<T>() -> Result<DMatrix<T>>
where
    T: num::Float + num::FromPrimitive + na::Scalar,
{
    let mut gz = GzDecoder::new(File::open(format!("{}{}", DATA_PATH, TEST_DATA))?);
    let mut data: Vec<u8> = Vec::with_capacity(7840016);
    gz.read_to_end(&mut data)?;
    Ok(DMatrix::from_row_slice(10000, 784, &data[16..]).map(|p| T::from_u8(p).unwrap()))
}

fn load_test_label<T>() -> Result<DMatrix<T>>
where
    T: num::Float + num::FromPrimitive + na::Scalar,
{
    let mut gz = GzDecoder::new(File::open(format!("{}{}", DATA_PATH, TEST_LABEL))?);
    let mut data: Vec<u8> = Vec::with_capacity(10008);
    let mut onehot_data = Vec::with_capacity(100000);
    gz.read_to_end(&mut data)?;
    data[8..].into_iter().for_each(|&l| {
        let mut onehot: [T; 10] = [T::zero(); 10];
        onehot[l as usize] = T::one();
        onehot.into_iter().for_each(|p| onehot_data.push(p));
    });
    Ok(DMatrix::from_row_slice(10000, 10, &onehot_data))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn mnist_data_shape() -> Result<()> {
        let mnist: Mnist<f64> = Mnist::new().await?.normalize();

        assert_eq!(mnist.train_images.shape(), (60000, 784));
        assert_eq!(mnist.train_labels.shape(), (60000, 10));
        assert_eq!(mnist.test_images.shape(), (10000, 784));
        assert_eq!(mnist.test_labels.shape(), (10000, 10));

        Ok(())
    }
}
