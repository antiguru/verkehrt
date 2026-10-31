//! Per-kernel totals of one acceptance query on the corgi data.
//!
//! Usage: `VERKEHRT_CLICKBENCH=DIR cargo run --release --example acceptance_stats -- QUERY [ISA]`
//! where `QUERY` is a program name such as `q40` and `ISA` is an instruction set name as
//! `verkehrt run --isa` takes it. The columns load as the acceptance bench loads them and are
//! bound as interpreter globals, so `Interpreter::run` collects the statistics that
//! `Prepared::execute` does not. The query runs five times on fresh interpreters, and the run
//! with the least total kernel time prints. The times still differ from the criterion
//! time of the same query, most for the short ones.

use std::io::sink;
use std::path::Path;

use verkehrt::columns::{CLICKBENCH_SIGNED, load_inputs};
use verkehrt::kernels::Isa;
use verkehrt::{Interpreter, Options};

fn main() {
    let mut args = std::env::args().skip(1);
    let query = args.next().expect("usage: acceptance_stats QUERY [ISA]");
    let isa = args
        .next()
        .map(|n| Isa::from_name(&n).unwrap_or_else(|| panic!("unknown instruction set {n}")));
    let dir = std::env::var_os("VERKEHRT_CLICKBENCH").expect("set VERKEHRT_CLICKBENCH");
    let path = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests/acceptance/clickbench")
        .join(format!("{query}.vk"));
    let src = std::fs::read_to_string(&path).unwrap();
    let options = || Options {
        isa,
        stats: true,
        ..Options::default()
    };
    let prepared = Interpreter::new(options()).unwrap().prepare(&src).unwrap();
    let inputs = load_inputs(Path::new(&dir), &prepared, CLICKBENCH_SIGNED).unwrap();
    let mut best: Option<(f64, Vec<u8>)> = None;
    for _ in 0..5 {
        let mut interp = Interpreter::new(options()).unwrap();
        for (name, value) in &inputs {
            interp.bind(name, value.clone());
        }
        interp.run(&src, &mut sink(), &mut sink()).unwrap();
        let mut text = Vec::new();
        interp.write_stats(&mut text);
        let total: f64 = String::from_utf8_lossy(&text)
            .lines()
            .filter_map(|l| l.rsplit_once("time=")?.1.parse::<f64>().ok())
            .sum();
        if best.as_ref().is_none_or(|(t, _)| total < *t) {
            best = Some((total, text));
        }
    }
    let (total, text) = best.unwrap();
    println!("query {query} total kernel time {total:.1} us");
    print!("{}", String::from_utf8_lossy(&text));
}
