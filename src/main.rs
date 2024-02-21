#![warn(clippy::pedantic)]
use clap::Parser;
use std::path::PathBuf;

use crate::comet::Instruction;

mod comet;

#[derive(Debug, Parser)]
#[command(
    version,
    about,
    long_about = None,
    arg_required_else_help = true,
)]
struct Args {
    path: PathBuf,

    #[arg(short, long)]
    /// launch window with debug interface
    debug: bool,

    #[arg(short, long, value_name = "INT", default_value_t = 0)]
    /// halt after cycle count has been reached (will run forever if unset)
    max_cycles: usize,

    #[arg(short = 'M', long, value_name = "INT", default_value_t = 0)]
    /// use a custom address space size; the maximum addressable byte will be [int]-1
    /// if not provided, defaults to 2^26 (64 MiB)
    memory: usize,

    #[arg(short, long)]
    /// output benchmark info after execution is halted
    bench: bool,
}

fn main() { let args = Args::parse(); 
    let a: comet::B = dbg!(unsafe{Instruction { opcode: 0 }.b});
    let b = a.func();
    dbg!(b);
}
