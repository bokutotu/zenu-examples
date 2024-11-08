use zenu::{autograd::Variable, dataset::Dataset};

pub struct MnistDataset {
    data: Vec<(Vec<u8>, u8)>,
}

impl MnistDataset {
    pub fn new(data: Vec<(Vec<u8>, u8)>) -> Self {
        Self { data }
    }
}

impl Dataset<f32> for MnistDataset {
    type Item = (Vec<u8>, u8);

    fn item(&self, item: usize) -> Vec<Variable<f32, Cpu>> {
        let (x, y) = &self.data[item];
        let x_f32 = x.iter().map(|&x| f32::from(x)).collect::<Vec<_>>();
        let x = from_vec::<f32, _, Cpu>(x_f32, [784]);
        x.get_data_mut().to_ref_mut().div_scalar_assign(127.5);
        x.get_data_mut().to_ref_mut().sub_scalar_assign(1.0);
        vec![x]
    }

    fn len(&self) -> usize {
        self.data.len()
    }

    fn all_data(&mut self) -> &mut [Self::Item] {
        &mut self.data as &mut [Self::Item]
    }
}
