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
        let input = uniform(T::minus_one(), T::one(), None, shape);

        let gen_out = self.generator.call(input);
        let dis_out = self.discriminator.call(gen_out);

        let ans = ones_like(&dis_out);

        let loss = mean_squared_error(ans, dis_out);
        loss.backward();
        self.generator_optimizer.update(&self.generator);
        loss.clear_grad();
        loss.get_as_ref().asum()
    }

    #[expect(clippy::needless_pass_by_value)]
    fn train_discriminator(&self, input: Variable<T, D>) -> T {
        let real_out = self.discriminator.call(input.clone());
        let real_ans = ones_like(&real_out);
        let real_loss = mean_squared_error(real_ans, real_out);
        real_loss.backward();
        self.discriminator_optimizer.update(&self.discriminator);
        real_loss.clear_grad();

        let shape = DimDyn::from([self.batch_size, self.hidden_size]);
        let noise = uniform(T::minus_one(), T::one(), None, shape);
        let fake_data = self.generator.call(noise);
        let fake_out = self.discriminator.call(fake_data);
        let fake_ans = zeros_like(&fake_out);
        let fake_loss = mean_squared_error(fake_ans, fake_out);
        fake_loss.backward();

        self.discriminator_optimizer.update(&self.discriminator);
        fake_loss.clear_grad();

        real_loss.get_as_ref().asum() + fake_loss.get_as_ref().asum()
    }

    pub fn generate(&self) -> Variable<T, D> {
        let shape = DimDyn::from([1, self.hidden_size]);
        let noise = uniform(T::minus_one(), T::one(), None, shape);
        self.generator.call(noise)
    }

    pub fn train_one_step(&self, input: Variable<T, D>) -> (T, T) {
        let gen_loss = self.train_generator();
        let disc_loss = self.train_discriminator(input);

        (gen_loss, disc_loss)
    }
}
