//! Surface syntax: lexer, AST, and recursive-descent parser.

pub mod ast;
mod lexer;
mod parser;

pub use parser::parse;

#[cfg(test)]
mod tests;
