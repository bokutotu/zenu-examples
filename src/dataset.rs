use std::{fs::File, io::BufRead};

use zenu::{
    autograd::{creator::from_vec::from_vec, Variable},
    dataset::Dataset,
    matrix::device::cpu::Cpu,
};

#[allow(clippy::module_name_repetitions)]
#[derive(Clone)]
pub struct MnistDataset {
    data: Vec<Vec<usize>>,
}

impl MnistDataset {
    #[must_use]
    #[expect(clippy::missing_panics_doc)]
    pub fn new(path: &str) -> Self {
        let file = File::open(path).unwrap();
        // split by line
        let data = std::io::BufReader::new(file)
            .lines()
            .map(|line| {
                line.unwrap()
                    .split(',')
                    .map(|x| x.parse::<usize>().unwrap())
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>();
        Self { data }
    }
}

impl Dataset<f32> for MnistDataset {
    type Item = Vec<usize>;

    #[expect(clippy::cast_precision_loss)]
    fn item(&self, item: usize) -> Vec<Variable<f32, Cpu>> {
        let x = &self.data[item];
        let x_f32 = x.iter().map(|&x| x as f32).collect::<Vec<_>>();
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
