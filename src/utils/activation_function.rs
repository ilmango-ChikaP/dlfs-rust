use na::DMatrix;

pub trait ActivationFunction {
    fn sigmoid(&self) -> Self;
    fn softmax(&self) -> Self;
    fn relu(&self) -> Self;
    fn tanh(&self) -> Self;
    fn sigmoid_mut(&mut self) -> ();
    fn softmax_mut(&mut self) -> ();
    fn relu_mut(&mut self) -> ();
    fn tanh_mut(&mut self) -> ();
}

impl<T> ActivationFunction for DMatrix<T>
where
    T: na::Scalar + num::Float,
{
    fn sigmoid(&self) -> Self {
        self.map(|x| _sigmoid(x))
    }

    fn softmax(&self) -> Self {
        let mut result = Vec::with_capacity(self.len());
        self.row_iter().for_each(|row| {
            let max = row.iter().fold(T::zero(), |acc, &x| acc.max(x));
            let exp = row.map(|x| (x - max).exp());
            let sum = exp.iter().fold(T::zero(), |acc, &x| acc + x);
            exp.map(|x| x / sum).iter().for_each(|&x| result.push(x));
        });
        DMatrix::from_row_slice(self.nrows(), self.ncols(), &result)
    }

    fn relu(&self) -> Self {
        self.map(|x| x.max(T::zero()))
    }

    fn tanh(&self) -> Self {
        self.map(|x| x.tanh())
    }

    fn sigmoid_mut(&mut self) -> () {
        *self = self.sigmoid();
    }

    fn softmax_mut(&mut self) -> () {
        *self = self.softmax();
    }

    fn relu_mut(&mut self) -> () {
        *self = self.relu();
    }

    fn tanh_mut(&mut self) -> () {
        *self = self.tanh();
    }
}

fn _sigmoid<T>(x: T) -> T
where
    T: num::Float,
{
    T::one() / (T::one() + (-x).exp())
}

#[cfg(test)]
mod tests {
    use super::*;
    use na::{dmatrix, DMatrix};

    const ARRAY_F64: [f64; 6] = [-2.0, -1.0, 0.0, 1.0, 2.0, 3.0];

    #[test]
    fn sigmoid() {
        let mut m_f64 = DMatrix::from_row_slice(2, 3, &ARRAY_F64);

        m_f64.sigmoid_mut();

        assert_eq!(
            m_f64,
            dmatrix![
                0.11920292202211755, 0.2689414213699951, 0.5;
                0.7310585786300049, 0.8807970779778823, 0.9525741268224334
            ]
        );
    }

    #[test]
    fn softmax() {
        let mut m_f64 = DMatrix::from_row_slice(2, 3, &ARRAY_F64);

        m_f64.softmax_mut();

        assert_eq!(
            m_f64,
            dmatrix![
                0.09003057317038046, 0.24472847105479764, 0.6652409557748218;
                0.09003057317038046, 0.24472847105479764, 0.6652409557748218
            ]
        );
    }

    #[test]
    fn relu() {
        let mut m_f64 = DMatrix::from_row_slice(2, 3, &ARRAY_F64);

        m_f64.relu_mut();

        assert_eq!(m_f64, dmatrix![0., 0., 0.; 1., 2., 3.]);
    }

    #[test]
    fn tanh() {
        let mut m_f64 = DMatrix::from_row_slice(2, 3, &ARRAY_F64);

        m_f64.tanh_mut();

        assert_eq!(
            m_f64,
            dmatrix![
                -0.9640275800758169, -0.7615941559557649, 0.0;
                0.7615941559557649, 0.9640275800758169, 0.9950547536867305
            ]
        );
    }
}
