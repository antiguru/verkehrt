use crate::error::{Error, Span};

#[derive(Clone, Debug, PartialEq)]
pub enum Tok {
    Ident(String),
    /// A numeric literal. `float` is set when the text has a fractional part.
    Num { text: String, float: bool, suffix: Option<String> },
    Str(String),
    LParen,
    RParen,
    LBrace,
    RBrace,
    LBracket,
    RBracket,
    Comma,
    Semi,
    Colon,
    Arrow,
    Pipe,
    Assign,
    Plus,
    Minus,
    Star,
    Slash,
    EqEq,
    Ne,
    Lt,
    Le,
    Gt,
    Ge,
    Eof,
}

#[derive(Clone, Debug)]
pub struct Token {
    pub tok: Tok,
    pub span: Span,
}

/// Splits `src` into tokens. The last token is always `Tok::Eof`, positioned at the end of input.
pub fn lex(src: &str) -> Result<Vec<Token>, Error> {
    let chars: Vec<char> = src.chars().collect();
    let mut out = Vec::new();
    let (mut i, mut line, mut col) = (0usize, 1u32, 1u32);

    while i < chars.len() {
        let c = chars[i];
        let span = Span { line, col };
        // Advances over `n` characters on the current line.
        let adv = |n: usize, i: &mut usize, col: &mut u32| {
            *i += n;
            *col += u32::try_from(n).expect("token length fits u32");
        };

        if c == '\n' {
            i += 1;
            line += 1;
            col = 1;
        } else if c.is_whitespace() {
            adv(1, &mut i, &mut col);
        } else if c == '/' && chars.get(i + 1) == Some(&'/') {
            while i < chars.len() && chars[i] != '\n' {
                adv(1, &mut i, &mut col);
            }
        } else if c.is_ascii_digit() {
            let start = i;
            while i < chars.len() && chars[i].is_ascii_digit() {
                i += 1;
            }
            let mut float = false;
            // A dot belongs to the literal only when a digit follows it.
            if chars.get(i) == Some(&'.') && chars.get(i + 1).is_some_and(|d| d.is_ascii_digit()) {
                float = true;
                i += 1;
                while i < chars.len() && chars[i].is_ascii_digit() {
                    i += 1;
                }
            }
            let text: String = chars[start..i].iter().collect();
            let sfx_start = i;
            while i < chars.len() && (chars[i].is_ascii_alphanumeric() || chars[i] == '_') {
                i += 1;
            }
            let suffix = (i > sfx_start).then(|| chars[sfx_start..i].iter().collect());
            col += u32::try_from(i - start).expect("token length fits u32");
            out.push(Token { tok: Tok::Num { text, float, suffix }, span });
        } else if c.is_ascii_alphabetic() || c == '_' {
            let start = i;
            while i < chars.len() && (chars[i].is_ascii_alphanumeric() || chars[i] == '_') {
                i += 1;
            }
            let word: String = chars[start..i].iter().collect();
            if word == "b" && chars.get(i) == Some(&'"') {
                return Err(Error::parse("byte strings are not supported yet", span));
            }
            col += u32::try_from(i - start).expect("token length fits u32");
            out.push(Token { tok: Tok::Ident(word), span });
        } else if c == '"' {
            let start = i;
            i += 1;
            let mut s = String::new();
            loop {
                match chars.get(i) {
                    None | Some('\n') => {
                        return Err(Error::parse("unterminated string literal", span));
                    }
                    Some('"') => {
                        i += 1;
                        break;
                    }
                    Some('\\') => {
                        let esc = match chars.get(i + 1) {
                            Some('n') => '\n',
                            Some('t') => '\t',
                            Some('"') => '"',
                            Some('\\') => '\\',
                            _ => {
                                return Err(Error::parse(
                                    "unknown escape in string literal",
                                    Span { line, col: col + u32::try_from(i - start).unwrap_or(0) },
                                ));
                            }
                        };
                        s.push(esc);
                        i += 2;
                    }
                    Some(&ch) => {
                        s.push(ch);
                        i += 1;
                    }
                }
            }
            col += u32::try_from(i - start).expect("token length fits u32");
            out.push(Token { tok: Tok::Str(s), span });
        } else {
            let next = chars.get(i + 1).copied();
            let (tok, n) = match (c, next) {
                ('-', Some('>')) => (Tok::Arrow, 2),
                ('=', Some('=')) => (Tok::EqEq, 2),
                ('!', Some('=')) => (Tok::Ne, 2),
                ('<', Some('=')) => (Tok::Le, 2),
                ('>', Some('=')) => (Tok::Ge, 2),
                ('(', _) => (Tok::LParen, 1),
                (')', _) => (Tok::RParen, 1),
                ('{', _) => (Tok::LBrace, 1),
                ('}', _) => (Tok::RBrace, 1),
                ('[', _) => (Tok::LBracket, 1),
                (']', _) => (Tok::RBracket, 1),
                (',', _) => (Tok::Comma, 1),
                (';', _) => (Tok::Semi, 1),
                (':', _) => (Tok::Colon, 1),
                ('|', _) => (Tok::Pipe, 1),
                ('=', _) => (Tok::Assign, 1),
                ('+', _) => (Tok::Plus, 1),
                ('-', _) => (Tok::Minus, 1),
                ('*', _) => (Tok::Star, 1),
                ('/', _) => (Tok::Slash, 1),
                ('<', _) => (Tok::Lt, 1),
                ('>', _) => (Tok::Gt, 1),
                _ => return Err(Error::parse(format!("unexpected character `{c}`"), span)),
            };
            adv(n, &mut i, &mut col);
            out.push(Token { tok, span });
        }
    }
    out.push(Token { tok: Tok::Eof, span: Span { line, col } });
    Ok(out)
}
