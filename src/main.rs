#![warn(clippy::pedantic)]
#![allow(clippy::cast_possible_truncation)]
use clap::Parser;
use comet::{Emulator, Instruction, StFlag, CPU};
use ic::IC;
use mmu::MMU;
use std::{mem::size_of, path::PathBuf};

mod comet;
mod ic;
mod mmu;
mod io;

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

    #[arg(short = 'M', long, value_name = "INT", default_value_t = 1 << 26)]
    /// use a custom address space size; the maximum addressable byte will be [int]-1
    /// if not provided, defaults to 2^26 (64 MiB)
    memory: usize,

    #[arg(short, long)]
    /// output benchmark info after execution is halted
    bench: bool,
}

fn comet_main() -> anyhow::Result<()> {
    let args = Args::try_parse()?;
    let mut mmu = MMU::new(args.memory as u64)?;
    let ic = IC::new();
    mmu.load_image(&args.path)?;
    let mut cpu = CPU::new();
    cpu.set_flag(StFlag::EXT_F, true);

    let comet = Emulator::new(cpu, ic, mmu, args.debug, args.max_cycles);

    let result = comet.run();
    if args.bench {
        println!("\ttime      : {}s", result.elapsed);
        println!("\tcycles    : {}", result.cycle);
        println!("\tcycles/s  : {:.3}", result.cycle_per_sec());
    }

    Ok(())
}

fn main() {
    if size_of::<usize>() != size_of::<u64>() {
        println!("WARNING: Running on 32bit target");
    }
    match comet_main() {
        Ok(()) => {}
        Err(err) => println!("{err}"),
    }
}
