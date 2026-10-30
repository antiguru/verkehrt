use std::fmt;

/// A 1-based source location.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct Span {
    pub line: u32,
    pub col: u32,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum ErrorKind {
    Parse,
    Type,
    Runtime,
    Io,
    Usage,
}

impl ErrorKind {
    fn name(self) -> &'static str {
        match self {
            ErrorKind::Parse => "parse",
            ErrorKind::Type => "type",
            ErrorKind::Runtime => "runtime",
            ErrorKind::Io => "io",
            ErrorKind::Usage => "usage",
        }
    }
}

/// An interpreter error. Displays as `error[kind] at line:col: message`,
/// without the ` at ..` part when there is no span.
#[derive(Clone, Debug, PartialEq)]
pub struct Error {
    pub kind: ErrorKind,
    pub message: String,
    pub span: Option<Span>,
}

impl Error {
    pub fn parse(msg: impl Into<String>, span: Span) -> Error {
        Error { kind: ErrorKind::Parse, message: msg.into(), span: Some(span) }
    }

    pub fn ty(msg: impl Into<String>, span: Span) -> Error {
        Error { kind: ErrorKind::Type, message: msg.into(), span: Some(span) }
    }

    pub fn runtime(msg: impl Into<String>, span: Option<Span>) -> Error {
        Error { kind: ErrorKind::Runtime, message: msg.into(), span }
    }

    pub fn usage(msg: impl Into<String>) -> Error {
        Error { kind: ErrorKind::Usage, message: msg.into(), span: None }
    }

    pub fn io(err: &std::io::Error) -> Error {
        Error { kind: ErrorKind::Io, message: err.to_string(), span: None }
    }
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "error[{}]", self.kind.name())?;
        if let Some(s) = self.span {
            write!(f, " at {}:{}", s.line, s.col)?;
        }
        write!(f, ": {}", self.message)
    }
}

impl std::error::Error for Error {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn display_with_and_without_span() {
        let e = Error::ty("bad", Span { line: 3, col: 14 });
        assert_eq!(e.to_string(), "error[type] at 3:14: bad");
        let e = Error::runtime("oops", None);
        assert_eq!(e.to_string(), "error[runtime]: oops");
        assert_eq!(Error::parse("x", Span { line: 1, col: 1 }).kind, ErrorKind::Parse);
    }
}
