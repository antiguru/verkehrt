//! Shape of the two q32 sort columns: extents, composite width, record, and active radix passes.
use std::fs;

fn load(dir: &str, name: &str) -> Vec<u64> {
    let b = fs::read(format!("{dir}/{name}.u64")).unwrap();
    // The file word is the radix image: `load_inputs` flips bit 63 to get the i64 and `image` flips it back.
    b.chunks_exact(8)
        .map(|c| u64::from_le_bytes(c.try_into().unwrap()))
        .collect()
}

fn bits(x: u64) -> u32 {
    u64::BITS - x.leading_zeros()
}

fn active_bytes(keys: impl Iterator<Item = u128> + Clone, digits: u32) -> Vec<u32> {
    let first = keys.clone().next().unwrap();
    (0..digits)
        .filter(|d| keys.clone().any(|k| (k >> (8 * d)) as u8 != (first >> (8 * d)) as u8))
        .collect()
}

fn main() {
    let dir = std::env::args().nth(1).unwrap();
    let a = load(&dir, "WatchID");
    let b = load(&dir, "ClientIP");
    println!("rows {}", a.len());
    // Optional second argument: write uniform columns with the same extents, for the sort timing.
    if let Some(out) = std::env::args().nth(2) {
        let ext = |v: &[u64]| (*v.iter().min().unwrap(), *v.iter().max().unwrap());
        let mut s = 0x9E3779B97F4A7C15u64;
        let mut next = move || {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            s
        };
        for (name, col) in [("WatchID", &a), ("ClientIP", &b)] {
            let (lo, hi) = ext(col);
            let span = hi - lo + 1;
            let bytes: Vec<u8> = col.iter().flat_map(|_| (lo + next() % span).to_le_bytes()).collect();
            fs::write(format!("{out}/{name}.u64"), bytes).unwrap();
        }
        // Optional third argument: the real columns, each shuffled independently.
        if let Some(out) = std::env::args().nth(3) {
            for (name, col) in [("WatchID", &a), ("ClientIP", &b)] {
                let mut v = col.clone();
                for i in (1..v.len()).rev() {
                    let j = usize::try_from(next() % (i as u64 + 1)).unwrap();
                    v.swap(i, j);
                }
                let bytes: Vec<u8> = v.iter().flat_map(|x| x.to_le_bytes()).collect();
                fs::write(format!("{out}/{name}.u64"), bytes).unwrap();
            }
        }
    }
    let (amin, amax) = (*a.iter().min().unwrap(), *a.iter().max().unwrap());
    let (bmin, bmax) = (*b.iter().min().unwrap(), *b.iter().max().unwrap());
    println!("WatchID (major) image min {amin} max {amax} range bits {}", bits(amax - amin));
    println!("ClientIP (minor) image min {bmin} max {bmax} range bits {}", bits(bmax - bmin));
    let bb = bits(bmax - bmin);
    let width = bits(amax - amin) + bb;
    println!(
        "composite width {width} bits, record {} (16-byte holds up to 96)",
        if width <= 96 { "R16, 16 bytes" } else { "R24<u32>, 24 bytes" }
    );
    let comp = a
        .iter()
        .zip(&b)
        .map(|(&x, &y)| (u128::from(x - amin) << bb) | u128::from(y - bmin));
    let act = active_bytes(comp, width.div_ceil(8));
    println!("composite: {} digits, {} active passes {:?}", width.div_ceil(8), act.len(), act);
    let sa = active_bytes(a.iter().map(|&x| u128::from(x)), 8);
    let sb = active_bytes(b.iter().map(|&x| u128::from(x)), 8);
    println!("argsort(ClientIP): active passes {} {:?}, 16-byte (u64,u32) pairs", sb.len(), sb);
    println!("argsort(WatchID): active passes {} {:?}, 16-byte (u64,u32) pairs", sa.len(), sa);
    println!("two single sorts: {} passes in total, composite: {} passes", sa.len() + sb.len(), act.len());
    let mut ds = a.clone();
    ds.sort_unstable();
    ds.dedup();
    println!("distinct WatchID {}", ds.len());
    let mut ds = b.clone();
    ds.sort_unstable();
    ds.dedup();
    println!("distinct ClientIP {}", ds.len());
}
