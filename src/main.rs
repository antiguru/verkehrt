//! The `verkehrt` command-line tool: argument parsing, I/O wiring, and exit codes.

use std::io::{BufRead, IsTerminal, Write};
use std::process::ExitCode;

use clap::{Args, Parser, Subcommand};
use verkehrt::kernels::{Backend, Isa};
use verkehrt::{Interpreter, Options};

#[derive(Parser)]
#[command(name = "verkehrt", version, about = "An interpreted array language")]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand)]
enum Command {
    /// Run a program file
    Run {
        /// The program to run
        file: std::path::PathBuf,
        #[command(flatten)]
        flags: Flags,
    },
    /// Start an interactive session
    Repl {
        #[command(flatten)]
        flags: Flags,
    },
}

#[derive(Args)]
struct Flags {
    /// Instruction set to run kernels with (scalar, avx2, avx512, neon)
    #[arg(long, value_parser = parse_isa)]
    isa: Option<Isa>,
    /// How kernels are written: auto (compiler vectorized) or intrinsics (hand written, with
    /// auto and scalar as fallbacks)
    #[arg(long, value_parser = parse_backend, default_value = "intrinsics")]
    backend: Backend,
    /// Print the plan of each statement to stderr
    #[arg(long)]
    plan: bool,
    /// Print one line per kernel call to stderr
    #[arg(long)]
    trace: bool,
    /// Print per-kernel totals to stderr when done
    #[arg(long)]
    stats: bool,
}

impl Flags {
    fn options(&self) -> Options {
        Options {
            isa: self.isa,
            backend: self.backend,
            show_plan: self.plan,
            trace: self.trace,
            stats: self.stats,
        }
    }
}

fn parse_isa(name: &str) -> Result<Isa, String> {
    Isa::from_name(name).ok_or_else(|| format!("unknown instruction set '{name}'"))
}

fn parse_backend(name: &str) -> Result<Backend, String> {
    match name {
        "auto" => Ok(Backend::Auto),
        "intrinsics" => Ok(Backend::Intrinsics),
        _ => Err(format!(
            "unknown backend '{name}', expected auto or intrinsics"
        )),
    }
}

fn main() -> ExitCode {
    let cli = Cli::parse();
    let result = match cli.command {
        Command::Run { file, flags } => run(&file, &flags),
        Command::Repl { flags } => repl(&flags),
    };
    match result {
        Ok(()) => ExitCode::SUCCESS,
        Err(msg) => {
            eprintln!("{msg}");
            ExitCode::from(1)
        }
    }
}

fn run(file: &std::path::Path, flags: &Flags) -> Result<(), String> {
    let src = std::fs::read_to_string(file)
        .map_err(|e| format!("cannot read {}: {e}", file.display()))?;
    let mut interp = Interpreter::new(flags.options()).map_err(|e| e.to_string())?;
    let result = interp.run(
        &src,
        &mut std::io::stdout().lock(),
        &mut std::io::stderr().lock(),
    );
    // Stats are reported even when the program failed.
    if flags.stats {
        interp.write_stats(&mut std::io::stderr().lock());
    }
    result.map_err(|e| e.to_string())
}

fn repl(flags: &Flags) -> Result<(), String> {
    let mut interp = Interpreter::new(flags.options()).map_err(|e| e.to_string())?;
    if std::io::stdin().is_terminal() {
        repl_interactive(&mut interp)?;
    } else {
        for line in std::io::stdin().lock().lines() {
            let line = line.map_err(|e| format!("cannot read stdin: {e}"))?;
            if !repl_line(&mut interp, &line) {
                break;
            }
        }
    }
    if flags.stats {
        interp.write_stats(&mut std::io::stderr().lock());
    }
    Ok(())
}

fn repl_interactive(interp: &mut Interpreter) -> Result<(), String> {
    use rustyline::error::ReadlineError;
    let mut editor =
        rustyline::DefaultEditor::new().map_err(|e| format!("cannot start editor: {e}"))?;
    loop {
        match editor.readline("vk> ") {
            Ok(line) => {
                // A failed history write must not end the session.
                let _ = editor.add_history_entry(line.as_str());
                if !repl_line(interp, &line) {
                    return Ok(());
                }
            }
            // Ctrl-C discards the line, Ctrl-D ends the session.
            Err(ReadlineError::Interrupted) => {}
            Err(ReadlineError::Eof) => return Ok(()),
            Err(e) => return Err(format!("cannot read line: {e}")),
        }
    }
}

/// Handles one REPL line and returns false when the session should end.
/// `:stats` prints the totals, which are empty unless `--stats` was given.
fn repl_line(interp: &mut Interpreter, line: &str) -> bool {
    match line.trim() {
        ":quit" => return false,
        ":stats" => interp.write_stats(&mut std::io::stderr().lock()),
        "" => {}
        _ => {
            let mut out = std::io::stdout().lock();
            if let Err(e) = interp.eval_repl(line, &mut out, &mut std::io::stderr().lock()) {
                eprintln!("{e}");
            }
            // Flush so output appears before the next prompt.
            let _ = out.flush();
        }
    }
    true
}
