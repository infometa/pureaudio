use std::process::exit;

#[path = "../audio/mod.rs"]
mod audio;
#[path = "../capture.rs"]
mod capture;

fn main() {
    if let Err(err) = capture::main() {
        eprintln!("{}", err);
        exit(1);
    }
}
