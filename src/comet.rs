#![warn(clippy::pedantic)]
#![allow(clippy::unused_unit)]

use bitfield_struct::bitfield;
use bitflags::bitflags;
use sa::static_assert;
use std::{
    fmt::Debug,
    ops::{Index, IndexMut},
    time::Instant,
};

use crate::{ic::IC, mmu::MMU};

#[rustfmt::skip]
#[bitfield(u32)]
#[derive(PartialEq, Eq)]
pub struct E {
    #[bits(8)]  pub opcode: u32,
    #[bits(8)]  pub imm:    u32,
    #[bits(4)]  pub func:   u32,
    #[bits(4)]  pub rs2:    u32,
    #[bits(4)]  pub rs1:    u32,
    #[bits(4)]  pub rde:    u32,
}
#[rustfmt::skip]
#[bitfield(u32)]
#[derive(PartialEq, Eq)]
pub struct R {
    #[bits(8)]  pub opcode: u32,
    #[bits(12)] pub imm:    u32,
    #[bits(4)]  pub rs2:    u32,
    #[bits(4)]  pub rs1:    u32,
    #[bits(4)]  pub rde:    u32,
}
#[rustfmt::skip]
#[bitfield(u32)]
#[derive(PartialEq, Eq)]
pub struct M {
    #[bits(8)]  pub opcode: u32,
    #[bits(16)] pub imm:    u32,
    #[bits(4)]  pub rs1:    u32,
    #[bits(4)]  pub rde:    u32,
}
#[rustfmt::skip]
#[bitfield(u32)]
#[derive(PartialEq, Eq)]
pub struct F {
    #[bits(8)]  pub opcode: u32,
    #[bits(16)] pub imm:    u32,
    #[bits(4)]  pub func:   u32,
    #[bits(4)]  pub rde:    u32,
}
#[rustfmt::skip]
#[bitfield(u32)]
#[derive(PartialEq, Eq)]
pub struct B {
    #[bits(8)]  pub opcode: u32,
    #[bits(20)] pub imm:    u32,
    #[bits(4)]  pub func:   u32,
}

#[derive(Clone, Copy)]
#[repr(C)]
pub union Instruction {
    pub opcode: u8,
    pub bits:   u32,
    pub e:      E,
    pub r:      R,
    pub m:      M,
    pub f:      F,
    pub b:      B,
}
impl Instruction {
    pub const fn zero() -> Self { Instruction { bits: 0 } }
    pub const fn opcode(self) -> u8 { unsafe { self.opcode } }
    pub const fn bits(self) -> u32 { unsafe { self.bits } }
}
impl Debug for Instruction {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{:b}", self.bits())
    }
}
static_assert!(core::mem::size_of::<Instruction>() == core::mem::size_of::<u32>());

#[rustfmt::skip]
#[derive(Debug, Clone, Copy)]
#[repr(u8)]
pub enum RegisterName {
    Rz,
    Ra, Rb, Rc, Rd,
    Re, Rf, Rg, Rh,
    Ri, Rj, Rk,
    Ip,
    Sp, Fp,
    St,
}

macro_rules! nth_bit {
    ($n: expr) => {
        1 << $n
    };
}
bitflags! {
    #[derive(Debug, Clone, Copy)]
    pub struct StFlag: u64 {
        const SIGN = nth_bit!(0);
        const ZERO = nth_bit!(1);
        const CARRY_BORROW = nth_bit!(2);
        const CARRY_BORROW_UNSIGNED = nth_bit!(3);
        const EQUAL = nth_bit!(4);
        const LESS = nth_bit!(5);
        const LESS_UNSIGNED = nth_bit!(6);
        const MODE = nth_bit!(7);

        const EXT_F = nth_bit!(31);
    }
}

#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct Registers(pub [u64; 16]);
impl Registers {
    pub const fn get_flag(&self, flag: StFlag) -> bool {
        self.0[RegisterName::St as usize] & flag.bits() == 1
    }
    pub fn set_flag(&mut self, flag: StFlag, value: bool) {
        if value {
            self[RegisterName::St] |= flag.bits();
        } else {
            self[RegisterName::St] &= !flag.bits();
        }
    }
}
impl Index<RegisterName> for Registers {
    type Output = u64;
    fn index(&self, index: RegisterName) -> &Self::Output { &self.0[index as usize] }
}
impl IndexMut<RegisterName> for Registers {
    fn index_mut(&mut self, index: RegisterName) -> &mut Self::Output {
        &mut self.0[index as usize]
    }
}
use RegisterName as RN;

#[allow(clippy::upper_case_acronyms)]
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct CPU {
    pub registers: Registers,
    pub cycle:     u64,
    pub instr:     Instruction,
    pub running:   bool,
}
impl CPU {
    pub const fn default() -> Self {
        Self {
            registers: Registers([0; 16]),
            cycle:     0,
            instr:     Instruction::zero(),
            running:   false,
        }
    }
    /// Same as `default()`, but running set to true
    pub const fn new() -> Self {
        Self {
            running: true,
            ..Self::default()
        }
    }
    pub const fn get_flag(&self, flag: StFlag) -> bool { self.registers.get_flag(flag) }
    pub fn set_flag(&mut self, flag: StFlag, value: bool) { self.registers.set_flag(flag, value); }
}

#[allow(clippy::upper_case_acronyms)]
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct IOC {
    in_pin:  bool,
    out_pin: bool,
    port:    u16,
}
impl IOC {
    pub const fn new() -> Self {
        Self {
            in_pin:  false,
            out_pin: false,
            port:    0,
        }
    }
}

#[derive(Debug, Clone)]
pub struct Emulator {
    pub cpu: CPU,
    pub ic:  IC,
    pub mmu: MMU,
    pub ioc: IOC,

    pub debug:       bool,
    pub no_color:    bool,
    pub cycle_limit: usize,
}
impl Emulator {
    pub const fn new(cpu: CPU, ic: IC, mmu: MMU, debug: bool, cycle_limit: usize) -> Self {
        Self {
            cpu,
            ic,
            mmu,
            ioc: IOC::new(),
            debug,
            no_color: false,
            cycle_limit,
        }
    }
    pub const fn current_instr(&self) -> Instruction {
        #[allow(clippy::cast_possible_truncation)]
        Instruction {
            bits: (self.cpu.registers.0[RN::St as usize] + 1) as u32,
        }
    }
    pub const fn regval(&self, reg: RegisterName) -> u64 { self.cpu.registers.0[reg as usize] }
    pub fn regval_mut(&mut self, reg: RegisterName) -> &mut u64 {
        &mut self.cpu.registers.0[reg as usize]
    }
    pub fn run_internal(&mut self) -> anyhow::Result<()> {
        self.cpu.cycle += 1;
        println!(
            "[at {:#016x} {:02x}]",
            self.regval(RN::Ip),
            self.current_instr().opcode()
        );

        todo!()
    }
    /// takes ownership of self
    pub fn run(mut self) -> anyhow::Result<RunStats> {
        let now = Instant::now();
        if self.cycle_limit == 0 {
            while self.cpu.running {
                self.run_internal()?;
            }
        } else {
            while self.cpu.running {
                if self.cycle_limit as u64 == self.cpu.cycle {
                    self.cpu.running = false;
                }
                self.run_internal()?;
            }
        }
        Ok(RunStats {
            elapsed: now.elapsed().as_secs_f64(),
            cycle:   self.cpu.cycle,
        })
    }
}

#[derive(Debug, Clone, Copy)]
pub struct RunStats {
    pub elapsed: f64,
    pub cycle:   u64,
}
impl RunStats {
    #[allow(clippy::cast_precision_loss)]
    pub fn cycle_per_sec(self) -> f64 { self.cycle as f64 / self.elapsed }
}
