//! The operations lowering builds around a lifted lambda body. The checker builds the same
//! operations on lifted types, so both resolve the same kernels.

use crate::error::Error;
use crate::error::Span;
use crate::ops::Op;
use crate::value::ValueType;

/// A higher-order call that applies a lambda.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Higher {
    Map,
    ZipMap,
    Filter,
}

impl Higher {
    pub(crate) fn name(self) -> &'static str {
        match self {
            Higher::Map => "map",
            Higher::ZipMap => "zip_map",
            Higher::Filter => "filter",
        }
    }
}

/// Builds operations over values of type `V`, terms for lowering and lifted types for the
/// checker.
pub(crate) trait Build {
    type V: Clone;

    fn ty(v: &Self::V) -> ValueType;

    /// Applies `op`, broadcasting scalars as `Registry::lookup_lifted` decides.
    fn apply(&self, op: Op, args: Vec<Self::V>, span: Span) -> Result<Self::V, Error>;

    /// Applies `op` to the operands exactly as given.
    fn apply_exact(&self, op: Op, args: Vec<Self::V>, span: Span) -> Result<Self::V, Error>;
}

/// The arrays that the lambda's parameters are bound to, given the call's array arguments.
pub(crate) fn params<B: Build>(
    b: &B,
    kind: Higher,
    xs: Vec<B::V>,
    span: Span,
) -> Result<Vec<B::V>, Error> {
    if kind != Higher::ZipMap {
        return Ok(xs);
    }
    let mut xs = xs.into_iter();
    let (x, y) = (
        xs.next().expect("zip_map takes two arrays"),
        xs.next().expect("zip_map takes two arrays"),
    );
    let ctx = |e: Error| {
        context(
            e,
            format!(
                "'zip_map' compares the lengths of its {} and {} arrays",
                B::ty(&x),
                B::ty(&y)
            ),
        )
    };
    // Each parameter reads its array through a check against the other's
    // length, so a mismatch fails even when the body ignores a parameter. The
    // checks are arrays to the planner, so fused kernels still match through them.
    let (lx, ly) = (
        b.apply(Op::Length, vec![x.clone()], span).map_err(ctx)?,
        b.apply(Op::Length, vec![y.clone()], span).map_err(ctx)?,
    );
    let px = b
        .apply_exact(Op::SameLen, vec![x.clone(), ly], span)
        .map_err(ctx)?;
    let py = b
        .apply_exact(Op::SameLen, vec![y.clone(), lx], span)
        .map_err(ctx)?;
    Ok(vec![px, py])
}

/// The result of the call from its array arguments `xs`, the parameter arrays `params`, and
/// the lifted lambda body.
pub(crate) fn result<B: Build>(
    b: &B,
    kind: Higher,
    xs: &[B::V],
    params: &[B::V],
    body: B::V,
    span: Span,
) -> Result<B::V, Error> {
    let first = &params[0];
    // A body that does not depend on its parameters stays scalar, so it is filled to the
    // length of the first array.
    let body = if super::is_array(B::ty(&body)) {
        body
    } else {
        let ctx = |e: Error| {
            context(
                e,
                format!(
                    "the lambda body does not use its parameters, so '{}' fills its {} value \
                     to the length of the {} array",
                    kind.name(),
                    B::ty(&body),
                    B::ty(first)
                ),
            )
        };
        let len = b
            .apply(Op::Length, vec![first.clone()], span)
            .map_err(ctx)?;
        b.apply(Op::Fill, vec![len, body.clone()], span)
            .map_err(ctx)?
    };
    if kind != Higher::Filter {
        return Ok(body);
    }
    let x = xs[0].clone();
    let ctx = |e: Error| {
        context(
            e,
            format!(
                "'filter' keeps the elements of its {} array where the predicate holds",
                B::ty(&x)
            ),
        )
    };
    b.apply(Op::Compress, vec![x.clone(), body], span)
        .map_err(ctx)
}

/// Prefixes the message of `e` with what the call was doing, keeping its span.
fn context(e: Error, doing: String) -> Error {
    Error {
        message: format!("{doing}: {}", e.message),
        ..e
    }
}
