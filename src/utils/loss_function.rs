use na::DMatrix;

pub trait LossFunction {
    type Item: na::Scalar + num::Float + num::FromPrimitive;
    fn sum_squared_error(&self, label: Self) -> Self::Item;
    fn cross_entropy_error(&self, label: Self) -> Self::Item;
}

impl<T> LossFunction for DMatrix<T>
where
    T: na::Scalar + num::Float + num::FromPrimitive,
{
    type Item = T;

    fn sum_squared_error(&self, label: Self) -> Self::Item {
        self.iter()
            .zip(label.iter())
            .fold(T::zero(), |acc, (&p, &l)| acc + (p - l) * (p - l))
            / T::from_u8(2).unwrap()
    }

    fn cross_entropy_error(&self, label: Self) -> Self::Item {
        let loss = self
            .iter()
            .zip(label.iter())
            .fold(T::zero(), |acc, (&p, &l)| {
                if l == T::zero() {
                    acc
                } else {
                    if p == T::zero() {
                        acc + T::from_u8(20).unwrap()
                    } else {
                        acc + -(p.ln())
                    }
                }
            });
        loss / T::from_usize(self.nrows()).unwrap()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use na::DMatrix;

    const LABEL_F64: [f64; 20] = [
        0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0,
        0.0, 0.0,
    ];
    const PREDICTION_F64: [f64; 20] = [
        0.1, 0.05, 0.6, 0.0, 0.05, 0.1, 0.0, 0.1, 0.0, 0.0, 0.1, 0.05, 0.1, 0.0, 0.05, 0.1, 0.0,
        0.6, 0.0, 0.0,
    ];

    #[test]
    fn sum_squared_error() {
        let label_f64 = DMatrix::from_row_slice(2, 10, &LABEL_F64);
        let prediction_f64 = DMatrix::from_row_slice(2, 10, &PREDICTION_F64);

        assert_eq!(
            prediction_f64.sum_squared_error(label_f64),
            0.6950000000000001
        );
    }

    #[test]
    fn cross_entropy_error() {
        let label_f64 = DMatrix::from_row_slice(2, 10, &LABEL_F64);
        let prediction_f64 = DMatrix::from_row_slice(2, 10, &PREDICTION_F64);

        assert_eq!(
            prediction_f64.cross_entropy_error(label_f64),
            1.4067053583800182
        );
    }
}
