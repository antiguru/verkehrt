//! Loads the column files of the acceptance data into program inputs.

use std::fs;
use std::path::Path;

use crate::interp::Prepared;
use crate::value::{Array, ScalarType, Value, ValueType};

/// Loads one value per declared input of `prepared` from the column files in `dir`, in
/// declaration order.
///
/// An input of type `u64[]` or `i64[]` named `N` is read from `dir/N.u64`, which holds the
/// rows as little-endian 64-bit words. An input of type `u8[]` named `C_bytes` is read from
/// `dir/C.bytes`, one byte per row, and `dir/C.ends` (little-endian `u64` cumulative end
/// offsets) must show that every row is exactly one byte. Any other input is an error naming
/// it.
pub fn load_inputs(dir: &Path, prepared: &Prepared) -> Result<Vec<(String, Value)>, String> {
    let mut out = Vec::new();
    for (name, ty) in prepared.inputs() {
        let array = match ty {
            ValueType::Array(elem @ (ScalarType::U64 | ScalarType::I64)) => {
                load_words(&dir.join(format!("{name}.u64")), elem)?
            }
            // Strip the suffix once, so `x_bytes_bytes` names the column `x_bytes`.
            ValueType::Array(ScalarType::U8) if name.ends_with("_bytes") => {
                load_single_bytes(dir, name.strip_suffix("_bytes").unwrap_or(name))?
            }
            _ => {
                return Err(format!(
                    "input '{name}' has type {ty}, which no column file provides"
                ));
            }
        };
        out.push((name.to_string(), Value::Array(array)));
    }
    Ok(out)
}

/// Reads `path` as little-endian 64-bit words into an array of `elem`.
fn load_words(path: &Path, elem: ScalarType) -> Result<Array, String> {
    let bytes = fs::read(path).map_err(|e| format!("{}: {e}", path.display()))?;
    if bytes.len() % 8 != 0 {
        return Err(format!(
            "{}: {} bytes is not a whole number of 8-byte words",
            path.display(),
            bytes.len()
        ));
    }
    // Decode each word from little-endian and store it in the target's byte order, so the
    // load is correct on any endianness. Both `u64` and `i64` use the same bit pattern.
    Array::build_bytes(bytes.len() / 8, elem, |out| {
        for (dst, src) in out
            .as_chunks_mut::<8>()
            .0
            .iter_mut()
            .zip(bytes.as_chunks::<8>().0)
        {
            *dst = u64::from_le_bytes(*src).to_ne_bytes();
        }
        Ok(())
    })
}

/// Reads `dir/column.bytes` as one byte per row after checking `dir/column.ends`.
fn load_single_bytes(dir: &Path, column: &str) -> Result<Array, String> {
    let ends_path = dir.join(format!("{column}.ends"));
    let bytes_path = dir.join(format!("{column}.bytes"));
    let ends = fs::read(&ends_path).map_err(|e| format!("{}: {e}", ends_path.display()))?;
    let bytes = fs::read(&bytes_path).map_err(|e| format!("{}: {e}", bytes_path.display()))?;
    if ends.len() % 8 != 0 {
        return Err(format!(
            "{}: {} bytes is not a whole number of 8-byte words",
            ends_path.display(),
            ends.len()
        ));
    }
    let rows = ends.len() / 8;
    if bytes.len() != rows {
        return Err(format!(
            "column '{column}' has {rows} rows but {} bytes, so it is not one byte per row",
            bytes.len()
        ));
    }
    for (i, w) in ends.as_chunks::<8>().0.iter().enumerate() {
        let end = u64::from_le_bytes(*w);
        if usize::try_from(end) != Ok(i + 1) {
            return Err(format!(
                "column '{column}' row {i} ends at byte {end}, so it is not one byte per row"
            ));
        }
    }
    Array::build_bytes(rows, ScalarType::U8, |out| {
        out.copy_from_slice(&bytes);
        Ok(())
    })
}

#[cfg(test)]
mod tests {
    use std::path::PathBuf;

    use super::*;
    use crate::{Interpreter, Options};

    /// A fresh directory named after the test and the process.
    fn scratch(test: &str) -> PathBuf {
        let dir = std::env::temp_dir().join(format!("verkehrt-{test}-{}", std::process::id()));
        let _ = fs::remove_dir_all(&dir);
        fs::create_dir_all(&dir).unwrap();
        dir
    }

    fn words(v: &[u64]) -> Vec<u8> {
        v.iter().flat_map(|x| x.to_le_bytes()).collect()
    }

    #[test]
    fn load_inputs_reads_u64_and_bytes() {
        let dir = scratch("load-inputs");
        fs::write(dir.join("a.u64"), words(&[1, 2, u64::MAX])).unwrap();
        fs::write(dir.join("b.u64"), words(&[u64::MAX, 7, 0])).unwrap();
        fs::write(dir.join("s.ends"), words(&[1, 2, 3])).unwrap();
        fs::write(dir.join("s.bytes"), b"xyz").unwrap();
        let interp = Interpreter::new(Options::default()).unwrap();
        let p = interp
            .prepare(
                "input a: u64[]; input b: i64[]; input s_bytes: u8[];\n\
                 export n = length(a);",
            )
            .unwrap();
        let got = load_inputs(&dir, &p).unwrap();
        let names: Vec<&str> = got.iter().map(|(n, _)| n.as_str()).collect();
        assert_eq!(names, ["a", "b", "s_bytes"]);
        assert_eq!(
            got[0].1,
            Value::Array(Array::from_vec(vec![1u64, 2, u64::MAX]))
        );
        assert_eq!(got[1].1, Value::Array(Array::from_vec(vec![-1i64, 7, 0])));
        assert_eq!(got[2].1, Value::Array(Array::from_vec(b"xyz".to_vec())));
    }

    #[test]
    fn load_inputs_strips_the_bytes_suffix_once() {
        let dir = scratch("bytes-suffix");
        fs::write(dir.join("x_bytes.ends"), words(&[1, 2])).unwrap();
        fs::write(dir.join("x_bytes.bytes"), b"ab").unwrap();
        // A column named `x` must not be read for the input `x_bytes_bytes`.
        fs::write(dir.join("x.ends"), words(&[1])).unwrap();
        fs::write(dir.join("x.bytes"), b"z").unwrap();
        let interp = Interpreter::new(Options::default()).unwrap();
        let p = interp
            .prepare("input x_bytes_bytes: u8[]; export n = length(x_bytes_bytes);")
            .unwrap();
        let got = load_inputs(&dir, &p).unwrap();
        assert_eq!(got[0].1, Value::Array(Array::from_vec(b"ab".to_vec())));
    }

    #[test]
    fn load_inputs_rejects_multibyte_strings() {
        let dir = scratch("multibyte");
        fs::write(dir.join("s.ends"), words(&[1, 3])).unwrap();
        fs::write(dir.join("s.bytes"), b"xyz").unwrap();
        let interp = Interpreter::new(Options::default()).unwrap();
        let p = interp
            .prepare("input s_bytes: u8[]; export n = length(s_bytes);")
            .unwrap();
        let err = load_inputs(&dir, &p).unwrap_err();
        assert!(err.contains("not one byte per row"), "{err}");
    }

    #[test]
    fn load_inputs_rejects_other_input_types() {
        let dir = scratch("other-types");
        let interp = Interpreter::new(Options::default()).unwrap();
        let p = interp
            .prepare("input x: f64[]; export n = length(x);")
            .unwrap();
        let err = load_inputs(&dir, &p).unwrap_err();
        assert!(err.contains("input 'x'"), "{err}");
    }
}
