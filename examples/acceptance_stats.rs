//! Per-kernel totals of one acceptance query on the corgi data.
//!
//! Usage: `VERKEHRT_CLICKBENCH=DIR cargo run --release --example acceptance_stats -- QUERY [ISA]`
//! where `QUERY` is a program name such as `q40` and `ISA` is an instruction set name as
//! `verkehrt run --isa` takes it. `QUERY` names a ClickBench program, or a TPC-H program as
//! `tpch/q06`, which reads `VERKEHRT_TPCH` instead. `VERKEHRT_TILE_ROWS` is `off` or a tile row
//! count and replaces the default tiling, `VERKEHRT_SHARING=off` turns shared evaluation off,
//! `VERKEHRT_SELECTIVE=off` turns selective evaluation off, and `VERKEHRT_REDUCTION_FUSION=off`
//! turns fused reductions off. The columns load as the
//! acceptance bench loads them and are bound as interpreter globals, so `Interpreter::run`
//! collects the statistics that `Prepared::execute` does not. The query runs five times on fresh interpreters, and the run
//! with the least total kernel time prints, with the number of region runs that completed tiled
//! (`Interpreter::tiled_regions`), the guard evaluations that narrowed
//! (`Interpreter::narrowed_tiles`), and the fused reduction runs per root kernel
//! (`Interpreter::fused_reductions`). The times still differ from the criterion
//! time of the same query, most for the short ones.

use std::io::sink;
use std::path::Path;

use verkehrt::columns::{
    CLICKBENCH_SIGNED, load_inputs, reduction_fusion_from_env, selective_from_env,
    sharing_from_env, tiling_from_env,
};
use verkehrt::kernels::Isa;
use verkehrt::{Interpreter, Options};

fn main() {
    let mut args = std::env::args().skip(1);
    let query = args.next().expect("usage: acceptance_stats QUERY [ISA]");
    let isa = args
        .next()
        .map(|n| Isa::from_name(&n).unwrap_or_else(|| panic!("unknown instruction set {n}")));
    let (suite, program, var, signed) = match query.split_once('/') {
        Some(("tpch", program)) => ("tpch", program, "VERKEHRT_TPCH", &[][..]),
        Some((suite, _)) => panic!("unknown suite {suite}"),
        None => (
            "clickbench",
            &*query,
            "VERKEHRT_CLICKBENCH",
            CLICKBENCH_SIGNED,
        ),
    };
    let dir = std::env::var_os(var).unwrap_or_else(|| panic!("set {var}"));
    let path = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests/acceptance")
        .join(suite)
        .join(format!("{program}.vk"));
    let src = std::fs::read_to_string(&path).unwrap();
    let options = || Options {
        isa,
        stats: true,
        tiling: tiling_from_env(),
        no_sharing: !sharing_from_env(),
        no_selective: !selective_from_env(),
        no_reduction_fusion: !reduction_fusion_from_env(),
        ..Options::default()
    };
    let prepared = Interpreter::new(options()).unwrap().prepare(&src).unwrap();
    let inputs = load_inputs(Path::new(&dir), &prepared, signed).unwrap();
    let mut best: Option<(f64, Vec<u8>, u64, u64, String)> = None;
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
        if best.as_ref().is_none_or(|(t, ..)| total < *t) {
            let fused = interp
                .fused_reductions()
                .iter()
                .map(|(k, n)| format!(" {k}={n}"))
                .collect::<String>();
            best = Some((
                total,
                text,
                interp.tiled_regions(),
                interp.narrowed_tiles(),
                fused,
            ));
        }
    }
    let (total, text, tiled, narrowed, fused) = best.unwrap();
    println!("query {query} total kernel time {total:.1} us");
    println!("tiled_regions {tiled}");
    println!("narrowed_tiles {narrowed}");
    println!("fused_reductions{fused}");
    print!("{}", String::from_utf8_lossy(&text));
}
