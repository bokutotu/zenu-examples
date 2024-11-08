use zenu::{
    autograd::Variable,
    layer::{layers::linear::Linear, Module},
    macros::Parameters,
    matrix::{device::Device, num::Num},
};

#[derive(Parameters)]
#[parameters(num = T, device = D)]
pub struct Generator<T: Num, D: Device> {
    pub linear1: Linear<T, D>,
    pub linear2: Linear<T, D>,
    pub linear3: Linear<T, D>,
}

#[derive(Parameters)]
#[parameters(num = T, device = D)]
pub struct Discriminator<T: Num, D: Device> {
    pub linear1: Linear<T, D>,
    pub linear2: Linear<T, D>,
    pub linear3: Linear<T, D>,
}

impl<T: Num, D: Device> Generator<T, D> {
    pub fn new(input_size: usize, hidden_size: usize, output_size: usize) -> Self {
        Self {
            linear1: Linear::new(input_size, hidden_size, true),
            linear2: Linear::new(hidden_size, hidden_size, true),
            linear3: Linear::new(hidden_size, output_size, true),
        }
    }
}

impl<T: Num, D: Device> Discriminator<T, D> {
    pub fn new(input_size: usize, hidden_size: usize, output_size: usize) -> Self {
        Self {
            linear1: Linear::new(input_size, hidden_size, true),
            linear2: Linear::new(hidden_size, hidden_size, true),
            linear3: Linear::new(hidden_size, output_size, true),
        }
    }
}

impl<T: Num, D: Device> Module<T, D> for Generator<T, D> {
    type Input = Variable<T, D>;
    type Output = Variable<T, D>;
    fn call(&self, x: Self::Input) -> Self::Output {
        let x = self.linear1.forward(x);
        let x = x.relu();
        let x = self.linear2.forward(&x);
        let x = x.relu();
        let x = self.linear3.forward(&x);
        x.sigmoid()
    }
}

impl<T: Num, D: Device> Module<T, D> for Discriminator<T, D> {
    type Input = Variable<T, D>;
    type Output = Variable<T, D>;
    fn call(&self, x: Self::Input) -> Self::Output {
        let x = self.linear1.forward(x);
        let x = x.relu();
        let x = self.linear2.forward(&x);
        let x = x.relu();
        let x = self.linear3.forward(&x);
        x.sigmoid()
    }
}