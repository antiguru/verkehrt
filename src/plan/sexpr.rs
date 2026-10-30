//! A minimal s-expression reader and printer for egglog program text and extraction output.

use std::fmt;

/// An atom or a parenthesized list.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Sexp {
    Atom(String),
    List(Vec<Sexp>),
}

impl Sexp {
    /// Rewrites every list head with `f`. Atoms in argument position are left alone.
    pub fn map_heads(&self, f: &impl Fn(&str) -> String) -> Sexp {
        match self {
            Sexp::Atom(_) => self.clone(),
            Sexp::List(items) => {
                let mut out = Vec::with_capacity(items.len());
                for (i, item) in items.iter().enumerate() {
                    out.push(match item {
                        Sexp::Atom(a) if i == 0 => Sexp::Atom(f(a)),
                        _ => item.map_heads(f),
                    });
                }
                Sexp::List(out)
            }
        }
    }
}

impl fmt::Display for Sexp {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Sexp::Atom(a) => f.write_str(a),
            Sexp::List(items) => {
                f.write_str("(")?;
                for (i, item) in items.iter().enumerate() {
                    if i > 0 {
                        f.write_str(" ")?;
                    }
                    write!(f, "{item}")?;
                }
                f.write_str(")")
            }
        }
    }
}

/// Parses exactly one s-expression. Atoms are maximal runs of characters other than
/// whitespace and parentheses. Quoted strings are not supported because neither the
/// patterns nor the extracted terms contain them.
pub fn parse(text: &str) -> Result<Sexp, String> {
    let tokens = tokenize(text);
    let mut pos = 0;
    let sexp = parse_at(&tokens, &mut pos, text)?;
    match tokens.get(pos) {
        None => Ok(sexp),
        Some(extra) => Err(format!("trailing `{extra}` in s-expression `{text}`")),
    }
}

fn tokenize(text: &str) -> Vec<&str> {
    let mut tokens = Vec::new();
    let mut start = None;
    for (i, c) in text.char_indices() {
        if c.is_whitespace() || c == '(' || c == ')' {
            if let Some(s) = start.take() {
                tokens.push(&text[s..i]);
            }
            if !c.is_whitespace() {
                tokens.push(&text[i..i + 1]);
            }
        } else if start.is_none() {
            start = Some(i);
        }
    }
    if let Some(s) = start {
        tokens.push(&text[s..]);
    }
    tokens
}

fn parse_at(tokens: &[&str], pos: &mut usize, text: &str) -> Result<Sexp, String> {
    let token = *tokens
        .get(*pos)
        .ok_or_else(|| format!("unexpected end of s-expression `{text}`"))?;
    *pos += 1;
    match token {
        ")" => Err(format!("unexpected `)` in s-expression `{text}`")),
        "(" => {
            let mut items = Vec::new();
            loop {
                match tokens.get(*pos) {
                    None => return Err(format!("missing `)` in s-expression `{text}`")),
                    Some(&")") => {
                        *pos += 1;
                        return Ok(Sexp::List(items));
                    }
                    Some(_) => items.push(parse_at(tokens, pos, text)?),
                }
            }
        }
        atom => Ok(Sexp::Atom(atom.to_string())),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn round_trips() {
        let text = "(K_dot_f64 (Leaf 0) (Bcast (Leaf 12)))";
        assert_eq!(parse(text).unwrap().to_string(), text);
        assert_eq!(parse("  atom ").unwrap(), Sexp::Atom("atom".into()));
    }

    #[test]
    fn rejects_malformed() {
        assert!(parse("(a b").is_err());
        assert!(parse("a)").is_err());
        assert!(parse(")").is_err());
        assert!(parse("").is_err());
    }

    #[test]
    fn maps_heads_only() {
        let s = parse("(count (gt_f64 ?x (Bcast ?c)))").unwrap();
        let m = s.map_heads(&|h| if h == "Bcast" { h.into() } else { format!("K_{h}") });
        assert_eq!(m.to_string(), "(K_count (K_gt_f64 ?x (Bcast ?c)))");
    }
}
