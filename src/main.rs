use image::{ImageBuffer, Luma};
use zenu::matrix::device::cpu::Cpu;
use zenu::{dataset::DataLoader, optimizer::adam::Adam};
use zenu_example::gan_model::Discriminator;
use zenu_example::{dataset::MnistDataset, gan_model::Generator, gan_train::GanTrainer};

fn main() {
    let batch_size = 64;
    let hidden_size = 128;
    let num_epochs = 100;

    let generator = Generator::new(hidden_size, hidden_size, 28 * 28);
    let discriminator = Discriminator::new(28 * 28, hidden_size, 1);

    let generator_optimizer = Adam::new(0.0002, 0.5, 0.999, 1e-8, &generator);
    let discriminator_optimizer = Adam::new(0.0002, 0.5, 0.999, 1e-8, &discriminator);

    let trainer = GanTrainer::new(
        generator,
        discriminator,
        generator_optimizer,
        discriminator_optimizer,
        batch_size,
        hidden_size,
    );
    let dataset = MnistDataset::new("mnist_train_flattened.txt");

    for epoch in 0..num_epochs {
        let mut gen_loss = 0.0;
        let mut disc_loss = 0.0;
        let mut dataloader = DataLoader::new(dataset.clone(), 64);
        dataloader.shuffle();
        let len = dataloader.len();

        for (i, data) in dataloader.enumerate() {
            let real_images = data[0].clone();
            let (g_loss, d_loss) = trainer.train_one_step(real_images);
            gen_loss += g_loss;
            disc_loss += d_loss;

            if i % 100 == 0 {
                println!(
                    "[Epoch {}][Batch {}/{}] [D loss: {:.6}] [G loss: {:.6}]",
                    epoch,
                    i,
                    len,
                    disc_loss / (i + 1) as f32,
                    gen_loss / (i + 1) as f32
                );
            }
        }

        if epoch % 2 == 0 {
            let image = trainer.generate();
            let image = image.to::<Cpu>();
            let data = image.get_as_ref().reshape_new_matrix([28 * 28]);
            let data = data.as_slice().to_vec();
            let data = data
                .into_iter()
                .map(|x| ((x + 1.) * 127.5) as u8)
                .collect::<Vec<u8>>();
            let img_buffer: ImageBuffer<Luma<u8>, Vec<u8>> =
                ImageBuffer::from_vec(28, 28, data).unwrap();
            // if output is not exists
            // create output directory
            std::fs::create_dir_all("output").unwrap();
            img_buffer.save(format!("output/{epoch}.png",)).unwrap();
        }
    }
}
