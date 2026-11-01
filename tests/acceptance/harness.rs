//! Runs a query program on a prepared data directory and prints its answer in the canonical
//! form `prepare.py` writes DuckDB's answers in.

use std::fmt::Write as _;
use std::path::Path;

use verkehrt::columns::load_inputs;
use verkehrt::value::{Array, Scalar, Value};
use verkehrt::{Interpreter, Options};

/// The kind letters of the program's `// output: [k1,k2,...]` comment.
pub fn parse_kinds(program: &str) -> Vec<char> {
    program
        .lines()
        .find_map(|l| l.trim().strip_prefix("// output:"))
        .map(|rest| rest.chars().filter(char::is_ascii_alphabetic).collect())
        .unwrap_or_default()
}

/// Rows of one export: arrays have one per element, scalars one.
fn rows(v: &Value) -> Result<usize, String> {
    match v {
        Value::Array(a) => Ok(a.len()),
        Value::Scalar(_) => Ok(1),
        Value::Str(_) => Err("string export".to_string()),
    }
}

fn int_field(a: &Array, row: usize) -> Result<String, String> {
    Ok(match a.get(row) {
        Scalar::I8(v) => v.to_string(),
        Scalar::I16(v) => v.to_string(),
        Scalar::I32(v) => v.to_string(),
        Scalar::I64(v) => v.to_string(),
        Scalar::U8(v) => v.to_string(),
        Scalar::U16(v) => v.to_string(),
        Scalar::U32(v) => v.to_string(),
        Scalar::U64(v) => v.to_string(),
        s => return Err(format!("not an integer: {s}")),
    })
}

/// One field of `row` in an export, as `prepare.py`'s `field` writes it for `kind`.
fn field(kind: char, v: &Value, row: usize) -> Result<String, String> {
    // A scalar is a one-element array.
    let one;
    let a = match v {
        Value::Array(a) => a,
        Value::Scalar(s) => {
            one = scalar_array(s);
            &one
        }
        Value::Str(_) => return Err("string export".to_string()),
    };
    match kind {
        'u' | 'i' => int_field(a, row),
        'f' => match a.get(row) {
            Scalar::F64(x) => Ok(format!("{x:.4}")),
            Scalar::F32(x) => Ok(format!("{:.4}", f64::from(x))),
            s => Err(format!("not a float: {s}")),
        },
        'g' => match a.get(row) {
            Scalar::F64(x) => format_general(x),
            s => Err(format!("not a float: {s}")),
        },
        's' => match a.get(row) {
            Scalar::U8(b) => Ok(format!("{b:02x}")),
            s => Err(format!("not a byte: {s}")),
        },
        k => Err(format!("unknown output kind '{k}'")),
    }
}

/// Formats like Python's `f'{x:.11e}'`: eleven fractional digits, then a signed exponent of at
/// least two digits. Rust's `{:e}` writes `e18` where Python writes `e+18`. Infinity and NaN
/// have no exponent and are an error.
pub fn format_general(x: f64) -> Result<String, String> {
    if !x.is_finite() {
        return Err(format!("cannot format {x} as a general float"));
    }
    let s = format!("{x:.11e}");
    let (mantissa, exp) = s
        .split_once('e')
        .ok_or_else(|| format!("no exponent in {s}"))?;
    let exp: i32 = exp.parse().map_err(|e| format!("exponent of {s}: {e}"))?;
    Ok(format!(
        "{mantissa}e{}{:02}",
        if exp < 0 { '-' } else { '+' },
        exp.abs()
    ))
}

fn scalar_array(s: &Scalar) -> Array {
    match s {
        Scalar::I8(v) => Array::from_vec(vec![*v]),
        Scalar::I16(v) => Array::from_vec(vec![*v]),
        Scalar::I32(v) => Array::from_vec(vec![*v]),
        Scalar::I64(v) => Array::from_vec(vec![*v]),
        Scalar::U8(v) => Array::from_vec(vec![*v]),
        Scalar::U16(v) => Array::from_vec(vec![*v]),
        Scalar::U32(v) => Array::from_vec(vec![*v]),
        Scalar::U64(v) => Array::from_vec(vec![*v]),
        Scalar::F32(v) => Array::from_vec(vec![*v]),
        Scalar::F64(v) => Array::from_vec(vec![*v]),
        Scalar::Bool(v) => Array::from_bools(&[*v]),
    }
}

/// One line per row, tab separated, each row ending in a newline.
pub fn format_rows(kinds: &[char], exports: &[(String, Value)]) -> Result<String, String> {
    if kinds.len() != exports.len() {
        return Err(format!(
            "{} output kinds but {} exports",
            kinds.len(),
            exports.len()
        ));
    }
    let Some((first, _)) = exports.first() else {
        return Ok(String::new());
    };
    let n = rows(&exports[0].1).map_err(|e| format!("export '{first}': {e}"))?;
    for (name, v) in exports {
        let m = rows(v).map_err(|e| format!("export '{name}': {e}"))?;
        if m != n {
            return Err(format!(
                "export '{name}' has {m} rows, export '{first}' has {n}"
            ));
        }
    }
    let mut out = String::new();
    for row in 0..n {
        for (i, (kind, (name, v))) in kinds.iter().zip(exports).enumerate() {
            if i > 0 {
                out.push('\t');
            }
            let f = field(*kind, v, row).map_err(|e| format!("export '{name}': {e}"))?;
            write!(out, "{f}").unwrap();
        }
        out.push('\n');
    }
    Ok(out)
}

/// Prepares the program, loads its columns from `dir` (flipping back the `signed` columns),
/// executes, and formats the exports.
pub fn run_query(dir: &Path, program_path: &Path, signed: &[&str]) -> Result<String, String> {
    let program = std::fs::read_to_string(program_path)
        .map_err(|e| format!("{}: {e}", program_path.display()))?;
    let kinds = parse_kinds(&program);
    let interp = Interpreter::new(Options::default()).map_err(|e| e.to_string())?;
    let prepared = interp.prepare(&program).map_err(|e| e.to_string())?;
    let inputs = load_inputs(dir, &prepared, signed)?;
    let named: Vec<(&str, Value)> = inputs
        .iter()
        .map(|(n, v)| (n.as_str(), v.clone()))
        .collect();
    let exports = prepared
        .execute(&interp, &named, &mut std::io::sink())
        .map_err(|e| e.to_string())?;
    format_rows(&kinds, &exports)
}
