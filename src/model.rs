use zenu::{
    layer::layers::rnn::{GRU, LSTM, RNN},
    macros::Parameters,
    matrix::{device::Device, num::Num},
};

#[derive(Parameters)]
#[parameters(num = T, device = D)]
pub struct RnnModel<T: Num, D: Device> {
    pub layer: RNN<T, D>,
}

#[derive(Parameters)]
#[parameters(num = T, device = D)]
pub struct LSTMModel<T: Num, D: Device> {
    pub layer: LSTM<T, D>,
}

#[derive(Parameters)]
#[parameters(num = T, device = D)]
pub struct GRUModel<T: Num, D: Device> {
    pub layer: GRU<T, D>,
}
