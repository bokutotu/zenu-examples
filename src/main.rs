use zenu::{dataset::DataLoader, dataset_loader::mnist_dataset, optimizer::adam::Adam};
use zenu_test_proj::{dataset::MnistDataset, gan_model::Generator, gan_train::GanTrainer};

fn main() {
    let batch_size = 64;
    let hidden_size = 128;
    let num_epochs = 100;

    let mnist_data = mnist_dataset().unwrap();
    let dataset = MnistDataset::new(mnist_data.0);

    let generator = Generator::new(hidden_size, hidden_size, 28 * 28);
    let discriminator = Discriminator::new(28 * 28, hidden_size, 1);

    let generator_optimizer = Adam::new(0.0002, 0.5, 0.999, 1e-8, &generator.parameters());
    let discriminator_optimizer = Adam::new(0.0002, 0.5, 0.999, 1e-8, &discriminator.parameters());

    let trainer = GanTrainer::new(
        generator,
        discriminator,
        generator_optimizer,
        discriminator_optimizer,
        batch_size,
        128,
    );

    for epoch in 0..num_epochs {
        let mut gen_loss = 0.0;
        let mut disc_loss = 0.0;
        let mut dataloader = DataLoader::new(dataset.clone(), 64);
        dataloader.shuffle();

        for (i, data) in dataloader.iter().enumerate() {
            let real_images = data[0].clone();
            let (g_loss, d_loss) = trainer.train_step(real_images);
            gen_loss += g_loss;
            disc_loss += d_loss;

            if i % 100 == 0 {
                println!(
                    "[Epoch {}][Batch {}/{}] [D loss: {:.6}] [G loss: {:.6}]",
                    epoch,
                    i,
                    dataloader.len(),
                    disc_loss / (i + 1),
                    gen_loss / (i + 1)
                );
            }
        }
    }
}
