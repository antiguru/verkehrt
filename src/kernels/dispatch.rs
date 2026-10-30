//! The dispatch table, resolved once per interpreter, and the whole-array driver that runs a
//! kernel entry over complete arguments.
//!
//! `unsafe` lives here and in the x86 kernel modules only. Here it is the call of a resolved
//! entry, which is sound because `Registry::resolve` installs an ISA entry only for an ISA
//! whose `is_available` returned true.

use std::sync::Arc;

use crate::kernels::scalar::elementwise::arity_error;
use crate::kernels::{InputSpec, KernelDef, KernelId, KernelImpl, Registry, Variant};
use crate::value::{Arg, Array, ArrayBuilder, Scalar, Value, ValueType};

#[cfg(test)]
mod tests;

/// One entry per kernel, indexed by `KernelId`, with the variant each was picked as.
/// Resolving needs no feature detection at call time, so a call is a table lookup.
///
/// A dispatch is tied to the registry it was resolved from by a token that the registry shares with it. `call`
/// takes the registry for the kernel definitions and rejects any other registry with an
/// error, so entries are never run against another registry's argument checks.
pub struct Dispatch {
    registry: Arc<()>,
    entries: Vec<(KernelImpl, Variant)>,
}

impl Dispatch {
    pub(crate) fn new(registry: Arc<()>, entries: Vec<(KernelImpl, Variant)>) -> Dispatch {
        Dispatch { registry, entries }
    }

    fn slot(&self, id: KernelId) -> (KernelImpl, Variant) {
        self.entries[usize::try_from(id.0).expect("u32 fits in usize")]
    }

    /// The variant that `call` runs for the kernel.
    pub fn variant_used(&self, id: KernelId) -> Variant {
        self.slot(id).1
    }

    /// The entry that `call` runs for the kernel.
    pub fn entry(&self, id: KernelId) -> KernelImpl {
        self.slot(id).0
    }

    /// Runs a kernel over whole arguments. It checks the arity and argument kinds against
    /// the kernel's inputs and unifies array lengths first. Then an elementwise kernel gets
    /// its output allocated and is called once, a reduction is updated once with all rows
    /// (never when there are none), a selection is called once with base 0, and a whole
    /// kernel is called as is. Fails when `reg` is not the registry this dispatch was resolved
    /// from, or when kernels were added to it since.
    pub fn call(&self, reg: &Registry, id: KernelId, args: &[Value]) -> Result<Value, String> {
        if !Arc::ptr_eq(&reg.token, &self.registry) || reg.defs.len() != self.entries.len() {
            return Err("dispatch was resolved from a different registry".into());
        }
        let def = reg.get(id);
        match self.entry(id) {
            // Whole kernels check their own arguments.
            KernelImpl::Whole(f) => f(args),
            KernelImpl::Elementwise(f) => {
                let (call_args, rows) = prepare(def, args)?;
                match def.output {
                    ValueType::Array(t) => {
                        let n = rows.ok_or("an array kernel needs an array operand")?;
                        Array::build_bytes(n, t, |out| {
                            // SAFETY: resolve installs an entry only for an ISA whose
                            // is_available() returned true. `out` holds n rows of the output
                            // type and prepare unified every `Rows` argument to n rows.
                            unsafe { f(&call_args, out) }
                        })
                        .map(Value::Array)
                    }
                    ValueType::Scalar(t) => {
                        if rows.is_some() {
                            return Err("a scalar kernel got an array operand".into());
                        }
                        // A u64 gives the one-row destination the alignment of any element.
                        let mut one = 0u64;
                        let out = &mut bytemuck::bytes_of_mut(&mut one)[..t.width()];
                        // SAFETY: resolve installs an entry only for an ISA whose
                        // is_available() returned true. `out` is one row of the output type
                        // and every argument is a `Splat`.
                        unsafe { f(&call_args, out) }?;
                        Ok(Value::Scalar(Scalar::from_bytes(t, out)))
                    }
                    ValueType::Str => Err("an elementwise kernel cannot produce a string".into()),
                }
            }
            KernelImpl::Reduction(make) => {
                let (call_args, rows) = prepare(def, args)?;
                let n = rows.ok_or("a reduction needs an array operand")?;
                // SAFETY: resolve installs an entry only for an ISA whose is_available()
                // returned true.
                let mut reducer = unsafe { make() };
                if n > 0 {
                    reducer.update(&call_args)?;
                }
                reducer.finish().map(Value::Scalar)
            }
            KernelImpl::Selection(f) => {
                let (call_args, _) = prepare(def, args)?;
                let ValueType::Array(t) = def.output else {
                    return Err("a selection kernel must produce an array".into());
                };
                let mut out = ArrayBuilder::new(t);
                // SAFETY: resolve installs an entry only for an ISA whose is_available()
                // returned true. The one tile is the whole array, so its base is 0.
                unsafe { f(&call_args, 0, &mut out) }?;
                Ok(Value::Array(out.finish()))
            }
        }
    }
}

/// Checks `args` against the kernel's inputs, and returns them as entry arguments with the
/// common row count of the array arguments, `None` when all are scalars.
fn prepare(def: &KernelDef, args: &[Value]) -> Result<(Vec<Arg>, Option<usize>), String> {
    if def.inputs.len() != args.len() {
        return Err(arity_error(def.inputs.len(), args.len()));
    }
    let mut rows: Option<usize> = None;
    let mut call_args = Vec::with_capacity(args.len());
    for (spec, value) in def.inputs.iter().zip(args) {
        if !spec.accepts(value.ty()) {
            return Err(kind_error(*spec, value.ty()));
        }
        match value {
            Value::Array(a) => {
                match rows {
                    Some(n) if n != a.len() => {
                        return Err(format!("length mismatch: {n} vs {}", a.len()));
                    }
                    _ => rows = Some(a.len()),
                }
                call_args.push(Arg::Rows(a.clone()));
            }
            Value::Scalar(s) => call_args.push(Arg::Splat(s.clone())),
            Value::Str(_) => return Err("a string argument needs a whole kernel".into()),
        }
    }
    Ok((call_args, rows))
}

fn kind_error(spec: InputSpec, got: ValueType) -> String {
    match spec {
        InputSpec::Array(t) => format!("expected {}[], got {got}", t.name()),
        InputSpec::Scalar(t) => format!("expected {}, got {got}", t.name()),
        InputSpec::Either(t) => format!("expected {0} or {0}[], got {got}", t.name()),
        InputSpec::Str => format!("expected str, got {got}"),
    }
}
