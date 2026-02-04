use arrow_quiver::selection::sel_intersect::train_model_if_needed;
use itertools::Itertools;

fn main() {
    env_logger::init();
    train_model_if_needed();

    for density in [100, 1_000, 10_000, 100_000, 1_000_000] {
        let _ = (0..density)
            .map(|_| fastrand::i32(0..1_000_000))
            .sorted()
            .dedup()
            .collect_vec();
    }
}
