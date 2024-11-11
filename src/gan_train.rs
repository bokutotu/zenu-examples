use zenu::{
    autograd::{
        creator::{ones::ones_like, rand::uniform, zeros::zeros_like},
        loss::mse::mean_squared_error,
        Variable,
    },
    layer::Module,
    matrix::{device::Device, dim::DimDyn, num::Num},
    optimizer::{adam::Adam, Optimizer},
};

use crate::gan_model::{Discriminator, Generator};

pub struct GanTrainer<T: Num, D: Device> {
    generator: Generator<T, D>,
    discriminator: Discriminator<T, D>,
    generator_optimizer: Adam<T, D>,
    discriminator_optimizer: Adam<T, D>,
    batch_size: usize,
    hidden_size: usize,
}

impl<T: Num, D: Device> GanTrainer<T, D> {
    pub fn new(
        generator: Generator<T, D>,
        discriminator: Discriminator<T, D>,
        generator_optimizer: Adam<T, D>,
        discriminator_optimizer: Adam<T, D>,
        batch_size: usize,
        hidden_size: usize,
    ) -> Self {
        Self {
            generator,
            discriminator,
            generator_optimizer,
            discriminator_optimizer,
            batch_size,
            hidden_size,
        }
    }

    fn train_generator(&self) -> T {
        let shape = DimDyn::from([self.batch_size, self.hidden_size]);
        let input = uniform(T::one(), T::minus_one(), None, shape);

        let gen_out = self.generator.call(input);
        let dis_out = self.discriminator.call(gen_out);

        let ans = ones_like(&dis_out);

        let loss = mean_squared_error(ans, dis_out);
        loss.backward();
        self.generator_optimizer.update(&self.generator);
        loss.clear_grad();
        loss.get_as_ref().asum()
    }

    fn train_discriminator(&self, input: Variable<T, D>) -> T {
        let dis_out = self.discriminator.call(input);
        let ans = zeros_like(&dis_out);
        let loss = mean_squared_error(ans, dis_out);
        loss.backward();
        self.discriminator_optimizer.update(&self.discriminator);
        loss.clear_grad();
        loss.get_as_ref().asum()
    }

    pub fn train_one_step(&self, input: Variable<T, D>) -> (T, T) {
        let gen_loss = self.train_generator();
        let disc_loss = self.train_discriminator(input);

        (gen_loss, disc_loss)
    }
}
