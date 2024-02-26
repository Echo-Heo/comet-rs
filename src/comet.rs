#![warn(clippy::pedantic)]
#![allow(clippy::cast_possible_truncation)]
#![allow(clippy::cast_lossless)]
#![allow(clippy::cast_precision_loss)]
#![allow(clippy::cast_sign_loss)]
#![allow(clippy::cast_possible_wrap)]
#![deny(unsafe_code)]


use bitflags::bitflags;
use std::{
    fmt::Debug,
    ops::{Add, AddAssign, Div, Index, IndexMut, Mul, Neg, Sub},
    time::Instant,
};

use crate::{
    ic::{IntQueueEntry, IC},
    io::{Ports, IOC},
    mmu::{self, MMU},
    opcode::Opcode,
    safety::{BitAccess, FloatCastType, FloatPrecision, Instruction, Interrupt, LiType, Nibble, Register},
};
use half::f16;

macro_rules! sign_extend {
    ($val: expr, $bitsize: expr) => {
        (((($val as u64) << (64 - $bitsize)) as i64) >> (64 - $bitsize)) as u64
    };
}
/* macro_rules! zero_extend {
    ($val: expr, $bitsize: expr) => {
        (((($val as u64) << (64 - $bitsize)) as u64) >> (64 - $bitsize)) as u64
    };
} */

macro_rules! arithmetic {
    ($self: ident.$func: ident (*$r1: ident, *$r2: ident) -> $rd: ident) => {
        $self.$func($self.regval($r1), $self.regval($r2), $rd)
    };
    ($self: ident.$func: ident (*$r1: ident, $imm: ident) -> $rd: ident) => {
        $self.$func($self.regval($r1), sign_extend!($imm, 16), $rd)
    };
}
macro_rules! bitwise {
    ($self: ident.$func: ident (*$r1: ident, *$r2: ident) -> $rd: ident) => {
        $self.$func($self.regval($r1), $self.regval($r2), $rd)
    };
    ($self: ident.$func: ident (*$r1: ident, $imm: ident) -> $rd: ident) => {
        $self.$func($self.regval($r1), $imm as u64, $rd)
    };
}

trait Float:
    Sized
    + Copy
    + Neg<Output = Self>
    + Add<Output = Self>
    + AddAssign
    + Sub<Output = Self>
    + Mul<Output = Self>
    + Div<Output = Self>
    + PartialEq
    + PartialOrd {
    const ZERO: Self;
    type IntType: Copy + Into<u64>;
    fn to_bits(self) -> Self::IntType;
    fn from_bits(v: Self::IntType) -> Self;
    /// bit preserving cast
    fn from_bits_u64(v: u64) -> Self
    where
        u64: BitAccess<Self::IntType>, {
        Self::from_bits(<u64 as BitAccess<Self::IntType>>::access(v, 0))
    }
    // /// bit preserving cast
    // fn to_bits_u64(self) -> u64 { self.to_bits().into() }
    /* /// overriding bit preserving cast
    // probably doesnt need unsafe write
    fn to_bits_to_u64(self, v: &mut u64)
    where
        u64: BitAccess<Self::IntType>, {
        v.write(0, self.to_bits());
    } */

    /// arithmetic cast
    fn from_int(v: i64) -> Self;
    /// arithmetic cast
    fn to_int(self) -> i64;
    fn abs(self) -> Self;
    fn is_zero(self) -> bool { self == Self::ZERO }
    fn sqrt(self) -> Self;
    fn min(self, other: Self) -> Self;
    fn max(self, other: Self) -> Self;
    fn ceil(self) -> Self;
    fn is_nan(self) -> bool;
}

impl Float for f16 {
    const ZERO: Self = Self::ZERO;
    type IntType = u16;
    fn from_bits(v: Self::IntType) -> Self { f16::from_bits(v) }
    fn to_bits(self) -> Self::IntType { self.to_bits() }
    fn from_int(v: i64) -> Self { f16::from_f64(v as f64) }
    fn to_int(self) -> i64 { self.to_f64() as i64 }
    #[rustfmt::skip]
    fn abs(self) -> Self { if self.is_sign_negative() { -self } else { self } }
    fn sqrt(self) -> Self { f16::from_f64_const(self.to_f64_const().sqrt()) }
    fn min(self, other: Self) -> Self { self.min(other) }
    fn max(self, other: Self) -> Self { self.max(other) }
    // maybe fix?
    fn ceil(self) -> Self { Self::from_f64_const(self.to_f64_const().ceil()) }
    fn is_nan(self) -> bool { self.is_nan() }
}
impl Float for f32 {
    const ZERO: Self = 0.0;
    type IntType = u32;
    fn from_bits(v: Self::IntType) -> Self { Self::from_bits(v) }
    fn to_bits(self) -> Self::IntType { self.to_bits() }
    fn from_int(v: i64) -> Self { v as Self }
    fn to_int(self) -> i64 { self as i64 }
    fn abs(self) -> Self { self.abs() }
    fn sqrt(self) -> Self { self.sqrt() }
    fn min(self, other: Self) -> Self { self.min(other) }
    fn max(self, other: Self) -> Self { self.max(other) }
    fn ceil(self) -> Self { self.ceil() }
    fn is_nan(self) -> bool { self.is_nan() }
}
impl Float for f64 {
    const ZERO: Self = 0.0;
    type IntType = u64;
    fn from_bits(v: Self::IntType) -> Self { Self::from_bits(v) }
    fn to_bits(self) -> Self::IntType { self.to_bits() }
    fn from_int(v: i64) -> Self { v as Self }
    fn to_int(self) -> i64 { self as i64 }
    fn abs(self) -> Self { self.abs() }
    fn sqrt(self) -> Self { self.sqrt() }
    fn min(self, other: Self) -> Self { self.min(other) }
    fn max(self, other: Self) -> Self { self.max(other) }
    fn ceil(self) -> Self { self.ceil() }
    fn is_nan(self) -> bool { self.is_nan() }
}
trait FloatFrom<F: Float>: Float {
    fn from(v: F) -> Self;
}
impl FloatFrom<f32> for f16 {
    fn from(v: f32) -> Self { f16::from_f32(v) }
}
impl FloatFrom<f64> for f16 {
    fn from(v: f64) -> Self { f16::from_f64(v) }
}
impl FloatFrom<f16> for f32 {
    fn from(v: f16) -> Self { v.to_f32() }
}
impl FloatFrom<f64> for f32 {
    fn from(v: f64) -> Self { v as Self }
}
impl FloatFrom<f16> for f64 {
    fn from(v: f16) -> Self { v.to_f64() }
}
impl FloatFrom<f32> for f64 {
    fn from(v: f32) -> Self { v as Self }
}
impl<T: Float> FloatFrom<T> for T {
    fn from(v: T) -> Self { v }
}
/* trait FloatTo<F: Float>: Float {
    fn to(self) -> F;
} */
/* impl<From: Float, F: FloatFrom<From>> FloatTo<F> for From {
    fn to(self) -> F { F::from(self) }
} */

#[macro_export]
macro_rules! nth_bit {
    ($n: expr) => {
        1 << $n
    };
}
bitflags! {
    #[derive(Debug, Clone, Copy)]
    pub(crate) struct StFlag: u64 {
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

#[derive(Debug, Clone, Copy)]
struct ProcMode;
impl ProcMode {
    // const KERNEL: bool = false;
    const USER: bool = true;
}

#[repr(C)]
#[derive(Debug, Clone, Copy)]
struct Registers(pub(crate) [u64; 16]);
impl Registers {
    const fn index(&self, index: Register) -> u64 { self.0[index.0 .0 as usize] }
    const fn get_flag(&self, flag: StFlag) -> bool { self.index(RN::ST) & flag.bits() != 0 }
    fn set_flag(&mut self, flag: StFlag, value: bool) {
        if value {
            self[RN::ST] |= flag.bits();
        } else {
            self[RN::ST] &= !flag.bits();
        }
    }
    #[allow(clippy::fn_params_excessive_bools)]
    fn set_cmp(&mut self, equal: bool, less: bool, less_unsigned: bool, sign: bool, zero: bool) {
        self.set_flag(StFlag::EQUAL, equal);
        self.set_flag(StFlag::LESS, less);
        self.set_flag(StFlag::LESS_UNSIGNED, less_unsigned);
        self.set_flag(StFlag::SIGN, sign);
        self.set_flag(StFlag::ZERO, zero);
    }
    fn current_instr(&self) -> Instruction { Instruction(self[RN::ST].access(1)) }
    fn set_current_instr(&mut self, instruction: Instruction) { self[RN::ST].write(1, instruction.0); }
}
impl Index<Register> for Registers {
    type Output = u64;
    fn index(&self, index: Register) -> &Self::Output { &self.0[index.0 .0 as usize] }
}
impl IndexMut<Register> for Registers {
    fn index_mut(&mut self, index: Register) -> &mut Self::Output { &mut self.0[index.0 .0 as usize] }
}
use Register as RN;

#[allow(clippy::upper_case_acronyms)]
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub(crate) struct CPU {
    registers: Registers,
    cycle:     u64,
    instr:     Instruction,
    running:   bool,
}
impl CPU {
    pub(crate) const fn default() -> Self {
        Self {
            registers: Registers([0; 16]),
            cycle:     0,
            instr:     Instruction(0),
            running:   false,
        }
    }
    /// Same as `default()`, but running set to true
    pub(crate) const fn new() -> Self {
        Self {
            running: true,
            ..Self::default()
        }
    }
    const fn get_flag(&self, flag: StFlag) -> bool { self.registers.get_flag(flag) }
    pub(crate) fn set_flag(&mut self, flag: StFlag, value: bool) { self.registers.set_flag(flag, value); }
    #[allow(clippy::fn_params_excessive_bools)]
    fn set_cmp(&mut self, equal: bool, less: bool, less_unsigned: bool, sign: bool, zero: bool) {
        self.registers.set_cmp(equal, less, less_unsigned, sign, zero);
    }
    fn set_cmp_u64(&mut self, a: u64, b: u64) { self.set_cmp(a == b, (a as i64) < (b as i64), a < b, (a as i64) < 0, a == 0) }
    fn set_cmp_f<F: Float>(&mut self, a: F, b: F) { self.set_cmp(a == b, a < b, a < b, a < F::ZERO, a.is_zero()) }
    fn set_cmp_f_u64<F: Float>(&mut self, a: u64, b: u64)
    where
        u64: BitAccess<F::IntType>, {
        self.set_cmp_f(F::from_bits_u64(a), F::from_bits_u64(b));
    }
}

fn overflowing_add_unsigned(a: u64, b: u64, carry: bool) -> (u64, bool) {
    let (r1, c1) = a.overflowing_add(b);
    let (r2, c2) = r1.overflowing_add(carry as u64);
    (r2, c1 || c2)
}
fn overflowing_add_signed(a: i64, b: i64, carry: bool) -> (i64, bool) {
    let (r1, c1) = a.overflowing_add(b);
    let (r2, c2) = r1.overflowing_add(carry as i64);
    (r2, c1 || c2)
}
fn overflowing_sub_unsigned(a: u64, b: u64, carry: bool) -> (u64, bool) {
    let (r1, c1) = a.overflowing_sub(b);
    let (r2, c2) = r1.overflowing_sub(carry as u64);
    (r2, c1 || c2)
}
fn overflowing_sub_signed(a: i64, b: i64, carry: bool) -> (i64, bool) {
    let (r1, c1) = a.overflowing_sub(b);
    let (r2, c2) = r1.overflowing_sub(carry as i64);
    (r2, c1 || c2)
}

#[derive(Debug, Clone)]
pub(crate) struct Emulator {
    cpu: CPU,
    pub(crate) ic:  IC,
    mmu: MMU,
    pub(crate) ioc: IOC,

    debug:       bool,
    // no_color:    bool,
    cycle_limit: usize,
}
impl Emulator {
    pub(crate) const fn new(cpu: CPU, ic: IC, mmu: MMU, debug: bool, cycle_limit: usize) -> Self {
        Self {
            cpu,
            ic,
            mmu,
            ioc: IOC::new(),
            debug,
            // no_color: false,
            cycle_limit,
        }
    }
    fn current_instr(&self) -> Instruction { self.cpu.registers.current_instr() }
    fn set_current_instr(&mut self, instruction: Instruction) { self.cpu.registers.set_current_instr(instruction); }

    fn read_instruction(&self, addr: u64) -> Result<Instruction, mmu::Response> {
        let mut addr = addr;
        if self.cpu.get_flag(StFlag::MODE) == ProcMode::USER {
            addr = self.mmu.translate_address(addr, mmu::AccessMode::Execute)?;
        }
        self.mmu.phys_get_u32(addr).map(Instruction)
    }
    fn read_u8(&self, addr: u64) -> Result<u8, mmu::Response> {
        let mut addr = addr;
        if self.cpu.get_flag(StFlag::MODE) {
            addr = self.mmu.translate_address(addr, mmu::AccessMode::Read)?;
        }
        self.mmu.phys_get_u8(addr)
    }
    fn read_u16(&self, addr: u64) -> Result<u16, mmu::Response> {
        let mut addr = addr;
        if self.cpu.get_flag(StFlag::MODE) {
            addr = self.mmu.translate_address(addr, mmu::AccessMode::Read)?;
        }
        self.mmu.phys_get_u16(addr)
    }
    fn read_u32(&self, addr: u64) -> Result<u32, mmu::Response> {
        let mut addr = addr;
        if self.cpu.get_flag(StFlag::MODE) {
            addr = self.mmu.translate_address(addr, mmu::AccessMode::Read)?;
        }
        self.mmu.phys_get_u32(addr)
    }
    fn read_u64(&self, addr: u64) -> Result<u64, mmu::Response> {
        let mut addr = addr;
        if self.cpu.get_flag(StFlag::MODE) {
            addr = self.mmu.translate_address(addr, mmu::AccessMode::Read)?;
        }
        self.mmu.phys_get_u64(addr)
    }
    fn read_to_u8(&mut self, addr: u64, to: Register) -> Result<(), mmu::Response> {
        let v = self.read_u8(addr)?;
        self.regval_mut(to).write(0, v);
        Ok(())
    }
    fn read_to_u16(&mut self, addr: u64, to: Register) -> Result<(), mmu::Response> {
        let v = self.read_u16(addr)?;
        self.regval_mut(to).write(0, v);
        Ok(())
    }
    fn read_to_u32(&mut self, addr: u64, to: Register) -> Result<(), mmu::Response> {
        let v = self.read_u32(addr)?;
        self.regval_mut(to).write(0, v);
        Ok(())
    }
    fn read_to_u64(&mut self, addr: u64, to: Register) -> Result<(), mmu::Response> {
        let v = self.read_u64(addr)?;
        self.regval_mut(to).write(0, v);
        Ok(())
    }
    fn read_to_u8_signed(&mut self, addr: u64, to: Register) -> Result<(), mmu::Response> {
        let v = self.read_u8(addr)?;
        self.regval_mut(to).write(0, v);
        *self.regval_mut(to) = sign_extend!(self.regval(to), 8);
        Ok(())
    }
    fn read_to_u16_signed(&mut self, addr: u64, to: Register) -> Result<(), mmu::Response> {
        let v = self.read_u16(addr)?;
        self.regval_mut(to).write(0, v);
        *self.regval_mut(to) = sign_extend!(self.regval(to), 16);
        Ok(())
    }
    fn read_to_u32_signed(&mut self, addr: u64, to: Register) -> Result<(), mmu::Response> {
        let v = self.read_u32(addr)?;
        self.regval_mut(to).write(0, v);
        *self.regval_mut(to) = sign_extend!(self.regval(to), 32);
        Ok(())
    }
    fn write_u8(&mut self, addr: u64, value: u8) -> Result<(), mmu::Response> {
        let mut addr = addr;
        if self.cpu.get_flag(StFlag::MODE) {
            addr = self.mmu.translate_address(addr, mmu::AccessMode::Read)?;
        }
        self.mmu.phys_write_u8(addr, value)
    }
    fn write_u16(&mut self, addr: u64, value: u16) -> Result<(), mmu::Response> {
        let mut addr = addr;
        if self.cpu.get_flag(StFlag::MODE) {
            addr = self.mmu.translate_address(addr, mmu::AccessMode::Read)?;
        }
        self.mmu.phys_write_u16(addr, value)
    }
    fn write_u32(&mut self, addr: u64, value: u32) -> Result<(), mmu::Response> {
        let mut addr = addr;
        if self.cpu.get_flag(StFlag::MODE) {
            addr = self.mmu.translate_address(addr, mmu::AccessMode::Read)?;
        }
        self.mmu.phys_write_u32(addr, value)
    }
    fn write_u64(&mut self, addr: u64, value: u64) -> Result<(), mmu::Response> {
        let mut addr = addr;
        if self.cpu.get_flag(StFlag::MODE) {
            addr = self.mmu.translate_address(addr, mmu::AccessMode::Read)?;
        }
        self.mmu.phys_write_u64(addr, value)
    }

    fn push_interrupt(&mut self, err: Interrupt) {
        let mut err = err;
        if self.ic.queue.is_empty() {
            self.ic.ret_addr = self.cpu.registers[RN::IP];
            self.ic.ret_status = self.cpu.registers[RN::ST];
            self.cpu.set_flag(StFlag::MODE, ProcMode::USER);
        }
        if self.ic.queue.len() == self.ic.queue.capacity() {
            // interrupt queue overflow
            self.ic.queue.clear();
            err = Interrupt::InterruptOverflow;
        }
        // hijack instruction pointer
        match self
            .mmu
            .phys_read_u64(self.ic.ivt_base_address + 8 * err as u64, &mut self.cpu.registers[RN::IP])
        {
            Err(err) => {
                self.push_interrupt_from_mmu(err);
            }
            Ok(()) => {
                self.ic.queue.push(IntQueueEntry { code: err as u8 });
            }
        }
    }
    fn push_interrupt_from_mmu(&mut self, res: mmu::Response) { self.push_interrupt(res.to_interrupt()); }

    fn return_interrupt(&mut self) {
        if self.ic.queue.is_empty() {
            return;
        }

        let _ = self.ic.queue.remove(0);
        if self.ic.queue.is_empty() {
            self.cpu.registers[RN::IP] = self.ic.ret_addr;
            self.cpu.registers[RN::ST] = self.ic.ret_status;
        } else {
            // hijack instruction pointer
            let code = self.ic.queue[self.ic.queue.len() - 1].code;
            match self.mmu.phys_get_u64(self.ic.ivt_base_address + 8 * code as u64) {
                Err(err) => self.push_interrupt_from_mmu(err),
                Ok(res) => self.cpu.registers[RN::IP] = res,
            }
        }
    }
    const fn regval(&self, reg: Register) -> u64 { self.cpu.registers.index(reg) }
    fn regval_mut(&mut self, reg: Register) -> &mut u64 { &mut self.cpu.registers[reg] }
    /// !!!!! Order is reverse compared to assignment statement !!!!!
    fn regval_write(&mut self, from: Register, to: Register) { *self.regval_mut(to) = self.regval(from); }
    const fn proc_mode_is_user(&self) -> bool { self.cpu.get_flag(StFlag::MODE) == ProcMode::USER }
    const fn proc_mode_is_user_then_invalid(&self) -> Result<(), Interrupt> {
        if self.proc_mode_is_user() {
            Err(Interrupt::InvalidInstruction)
        } else {
            Ok(())
        }
    }
    fn run_internal(&mut self) {
        self.cpu.cycle += 1;
        
        // load instruction
        match self.read_instruction(self.regval(RN::IP)) {
            Ok(instr) => self.set_current_instr(instr),
            Err(err) => {
                self.push_interrupt_from_mmu(err);
                return;
            }
        }

        if self.debug {
            for i in 0x0..=0xF {
                if matches!(i, 4..=11 | 0) {
                    continue;
                }
                let register = Register(Nibble(i));
                print!("[{register}: {:#x}]", self.regval(register));
            }
            println!("[{:#010x}]", self.current_instr().0);
            println!("[{:?}]", Opcode::from_instruction(self.current_instr()));
        }

        *self.regval_mut(RN::IP) += 4;

        if let Err(err) = self.interpret_code() {
            self.push_interrupt(err);
        }

        if self.ioc.out_pin {
            self.dev_receive();
        }

        self.cpu.registers[RN::RZ] = 0;

        if self.regval(RN::SP) > self.regval(RN::FP) {
            self.push_interrupt(Interrupt::StackUnderflow);
        }
    }

    pub(crate) fn dev_receive(&mut self) {
        if let Some(port) = Ports::from_port(self.ioc.port) {
            port.run(self, self.ioc.port_data(self.ioc.port));
        }
        self.ioc.out_pin = false;
    }
    fn push_stack(&mut self, data: u64) {
        *self.regval_mut(RN::SP) -= 8;
        if let Err(err) = self.write_u64(self.regval(RN::SP), data) {
            self.push_interrupt_from_mmu(err);
        }
    }
    fn push_stack_from(&mut self, reg: Register) { self.push_stack(self.regval(reg)) }
    fn pop_stack_to(&mut self, reg: Register) {
        match self.read_u64(self.regval(RN::SP)) {
            Ok(val) => {
                *self.regval_mut(RN::SP) += 8;
                *self.regval_mut(reg) = val;
            }
            Err(err) => self.push_interrupt_from_mmu(err),
        }
    }
    fn get_load_address(&self, rs: Register, off: u8, rn: Register, sh: Nibble) -> u64 {
        // TODO: operator precedence?
        self.regval(rs) + sign_extend!(off, 8) + (self.regval(rn) << (sh.0 as u64))
    }
    // returned interrupt is ran through `push_interrupt`
    #[allow(clippy::too_many_lines)]
    fn interpret_code(&mut self) -> Result<(), Interrupt> {
        let ci = self.current_instr();
        let opcode = Opcode::from_instruction(self.current_instr())?;
        match opcode {
            Opcode::Int { imm } => Err(imm)?,
            opcode @ (Opcode::Iret | Opcode::Ires | Opcode::Usr { .. }) => {
                self.proc_mode_is_user_then_invalid()?;
                match opcode {
                    Opcode::Iret | Opcode::Ires => self.return_interrupt(),
                    Opcode::Usr { rd } => {
                        self.cpu.set_flag(StFlag::MODE, ProcMode::USER);
                        self.cpu.registers[RN::IP] = self.regval(rd);
                    }
                    _ => unreachable!(),
                }
            }
            opcode @ (Opcode::Outr { .. } | Opcode::Outi { .. } | Opcode::Inr { .. } | Opcode::Ini { .. }) => {
                self.proc_mode_is_user_then_invalid()?;
                match opcode {
                    Opcode::Outr { rd, rs } => self.ioc.send_out(self.regval(rd).into(), self.regval(rs)),
                    Opcode::Outi { imm, rs } => self.ioc.send_out(imm, self.regval(rs)),
                    Opcode::Inr { rd, rs } => *self.regval_mut(rd) = self.ioc.port_data(self.regval(rs).into()),
                    Opcode::Ini { rd, imm } => *self.regval_mut(rd) = self.ioc.port_data(imm),
                    _ => unreachable!(),
                }
            }
            Opcode::Jal { rs, imm } => {
                self.push_stack_from(RN::IP);
                *self.regval_mut(RN::IP) = self.regval(rs) + (sign_extend!(imm, 16) as i64 * 4) as u64;
            }
            Opcode::Jalr { rd, rs, imm } => {
                *self.regval_mut(rd) = self.regval(RN::IP);
                *self.regval_mut(RN::IP) = self.regval(rs) + (sign_extend!(imm, 16) as i64 * 4) as u64;
            }
            Opcode::Ret => self.pop_stack_to(RN::IP),
            Opcode::Retr { rs } => self.regval_write(rs, RN::IP),
            Opcode::B { cc, imm } => {
                if cc.cond(|flag| self.cpu.get_flag(flag)) {
                    *self.regval_mut(RN::IP) = self.regval(RN::IP).wrapping_add((sign_extend!(imm, 20) as i64 * 4) as u64);
                }
            }
            Opcode::Push { rs } => self.push_stack_from(rs),
            Opcode::Pop { rd } => self.pop_stack_to(rd),
            Opcode::Enter => {
                self.push_stack_from(RN::FP);
                self.regval_write(RN::SP, RN::FP);
            }
            Opcode::Leave => {
                self.regval_write(RN::FP, RN::SP);
                self.pop_stack_to(RN::FP);
            }
            Opcode::Li { rd, func, imm } => match func {
                LiType::Lli => self.regval_mut(rd).write(0, imm),
                LiType::Llis => *self.regval_mut(rd) = sign_extend!(imm, 16),
                LiType::Lui => self.regval_mut(rd).write(1, imm),
                LiType::Luis => *self.regval_mut(rd) = sign_extend!(imm, 16) << 16,
                LiType::Lti => self.regval_mut(rd).write(2, imm),
                LiType::Ltis => *self.regval_mut(rd) = sign_extend!(imm, 16) << 32,
                LiType::Ltui => self.regval_mut(rd).write(3, imm),
                LiType::Ltuis => *self.regval_mut(rd) = sign_extend!(imm, 16) << 48,
            },
            opcode @ (Opcode::Lw { rd, rs, rn, sh, off }
            | Opcode::Lh { rd, rs, rn, sh, off }
            | Opcode::Lhs { rd, rs, rn, sh, off }
            | Opcode::Lq { rd, rs, rn, sh, off }
            | Opcode::Lqs { rd, rs, rn, sh, off }
            | Opcode::Lb { rd, rs, rn, sh, off }
            | Opcode::Lbs { rd, rs, rn, sh, off }) => {
                let addr = self.get_load_address(rs, off, rn, sh);
                if let Err(err) = match opcode {
                    Opcode::Lw { .. } => self.read_to_u64(addr, rd),
                    Opcode::Lh { .. } => self.read_to_u32(addr, rd),
                    Opcode::Lhs { .. } => self.read_to_u32_signed(addr, rd),
                    Opcode::Lq { .. } => self.read_to_u16(addr, rd),
                    Opcode::Lqs { .. } => self.read_to_u16_signed(addr, rd),
                    Opcode::Lb { .. } => self.read_to_u8(addr, rd),
                    Opcode::Lbs { .. } => self.read_to_u8_signed(addr, rd),
                    _ => unreachable!(),
                } {
                    self.push_interrupt_from_mmu(err);
                }
            }
            opcode @ (Opcode::Sw { rd, rs, rn, sh, off }
            | Opcode::Sh { rd, rs, rn, sh, off }
            | Opcode::Sq { rd, rs, rn, sh, off }
            | Opcode::Sb { rd, rs, rn, sh, off }) => {
                let addr = self.get_load_address(rs, off, rn, sh);
                if let Err(err) = match opcode {
                    Opcode::Sw { .. } => self.write_u64(addr, self.regval(rd)),
                    Opcode::Sh { .. } => self.write_u32(addr, self.regval(rd) as u32),
                    Opcode::Sq { .. } => self.write_u16(addr, self.regval(rd) as u16),
                    Opcode::Sb { .. } => self.write_u8(addr, self.regval(rd) as u8),
                    _ => unreachable!(),
                } {
                    self.push_interrupt_from_mmu(err);
                }
            }
            Opcode::Cmpr { r1, r2 } => self.cpu.set_cmp_u64(self.regval(r1), self.regval(r2)),
            Opcode::Cmpi { r1, s, imm } => {
                let (a, b) = if s {
                    (sign_extend!(imm, 16), self.regval(r1))
                } else {
                    (self.regval(r1), sign_extend!(imm, 16))
                };
                self.cpu.set_cmp_u64(a, b);
            }

            Opcode::Addr { rd, r1, r2 } => arithmetic!(self.add(*r1, *r2) -> rd),
            Opcode::Addi { rd, r1, imm } => arithmetic!(self.add(*r1, imm) -> rd),
            Opcode::Subr { rd, r1, r2 } => arithmetic!(self.sub(*r1, *r2) -> rd),
            Opcode::Subi { rd, r1, imm } => arithmetic!(self.sub(*r1, imm) -> rd),
            Opcode::Imulr { rd, r1, r2 } => arithmetic!(self.imul(*r1, *r2) -> rd),
            Opcode::Imuli { rd, r1, imm } => arithmetic!(self.imul(*r1, imm) -> rd),
            Opcode::Idivr { rd, r1, r2 } => arithmetic!(self.idiv(*r1, *r2) -> rd),
            Opcode::Idivi { rd, r1, imm } => {
                if self.regval(Register(ci.nth_nibble(5))) == 0 {
                    Err(Interrupt::DivideByZero)?;
                }
                arithmetic!(self.idiv(*r1, imm) -> rd);
            }
            Opcode::Umulr { rd, r1, r2 } => arithmetic!(self.umul(*r1, *r2) -> rd),
            Opcode::Umuli { rd, r1, imm } => arithmetic!(self.umul(*r1, imm) -> rd),
            Opcode::Udivr { rd, r1, r2 } => arithmetic!(self.udiv(*r1, *r2) -> rd),
            Opcode::Udivi { rd, r1, imm } => arithmetic!(self.udiv(*r1, imm) -> rd),
            Opcode::Remr { rd, r1, r2 } => arithmetic!(self.rem(*r1, *r2) -> rd),
            Opcode::Remi { rd, r1, imm } => arithmetic!(self.rem(*r1, imm) -> rd),
            Opcode::Modr { rd, r1, r2 } => arithmetic!(self.r#mod(*r1, *r2) -> rd),
            Opcode::Modi { rd, r1, imm } => arithmetic!(self.r#mod(*r1, imm) -> rd),

            Opcode::Andr { rd, r1, r2 } => bitwise!(self.and(*r1, *r2) -> rd),
            Opcode::Andi { rd, r1, imm } => bitwise!(self.and(*r1, imm) -> rd),
            Opcode::Orr { rd, r1, r2 } => bitwise!(self.or(*r1, *r2) -> rd),
            Opcode::Ori { rd, r1, imm } => bitwise!(self.or(*r1, imm) -> rd),
            Opcode::Norr { rd, r1, r2 } => bitwise!(self.nor(*r1, *r2) -> rd),
            Opcode::Nori { rd, r1, imm } => bitwise!(self.nor(*r1, imm) -> rd),
            Opcode::Xorr { rd, r1, r2 } => bitwise!(self.xor(*r1, *r2) -> rd),
            Opcode::Xori { rd, r1, imm } => bitwise!(self.xor(*r1, imm) -> rd),
            Opcode::Shlr { rd, r1, r2 } => bitwise!(self.shl(*r1, *r2) -> rd),
            Opcode::Shli { rd, r1, imm } => bitwise!(self.shl(*r1, imm) -> rd),
            Opcode::Asrr { rd, r1, r2 } => bitwise!(self.sar(*r1, *r2) -> rd),
            Opcode::Asri { rd, r1, imm } => bitwise!(self.sar(*r1, imm) -> rd),
            Opcode::Lsrr { rd, r1, r2 } => bitwise!(self.shr(*r1, *r2) -> rd),
            Opcode::Lsri { rd, r1, imm } => bitwise!(self.shr(*r1, imm) -> rd),
            Opcode::Bitr { rd, r1, r2 } => bitwise!(self.bit(*r1, *r2) -> rd),
            Opcode::Biti { rd, r1, imm } => bitwise!(self.bit(*r1, imm) -> rd),

            Opcode::Fcmp { r1, r2, p } => match p {
                FloatPrecision::F16 => self.fcmp::<f16>(r1, r2),
                FloatPrecision::F32 => self.fcmp::<f32>(r1, r2),
                FloatPrecision::F64 => self.fcmp::<f64>(r1, r2),
            },
            Opcode::Fto { rd, rs, p } => match p {
                FloatPrecision::F16 => self.fto::<f16>(rd, rs),
                FloatPrecision::F32 => self.fto::<f32>(rd, rs),
                FloatPrecision::F64 => self.fto::<f64>(rd, rs),
            },
            Opcode::Ffrom { rd, rs, p } => match p {
                FloatPrecision::F16 => self.ffrom::<f16>(rd, rs),
                FloatPrecision::F32 => self.ffrom::<f32>(rd, rs),
                FloatPrecision::F64 => self.ffrom::<f64>(rd, rs),
            },
            Opcode::Fneg { rd, rs, p } => match p {
                FloatPrecision::F16 => self.fmonadic::<f16>(rd, rs, Neg::neg),
                FloatPrecision::F32 => self.fmonadic::<f32>(rd, rs, Neg::neg),
                FloatPrecision::F64 => self.fmonadic::<f64>(rd, rs, Neg::neg),
            },
            Opcode::Fabs { rd, rs, p } => match p {
                FloatPrecision::F16 => self.fmonadic::<f16>(rd, rs, Float::abs),
                FloatPrecision::F32 => self.fmonadic::<f32>(rd, rs, Float::abs),
                FloatPrecision::F64 => self.fmonadic::<f64>(rd, rs, Float::abs),
            },
            Opcode::Fadd { rd, r1, r2, p } => match p {
                FloatPrecision::F16 => self.fdyadic::<f16>(rd, r1, r2, Add::add),
                FloatPrecision::F32 => self.fdyadic::<f32>(rd, r1, r2, Add::add),
                FloatPrecision::F64 => self.fdyadic::<f64>(rd, r1, r2, Add::add),
            },
            Opcode::Fsub { rd, r1, r2, p } => match p {
                FloatPrecision::F16 => self.fdyadic::<f16>(rd, r1, r2, Sub::sub),
                FloatPrecision::F32 => self.fdyadic::<f32>(rd, r1, r2, Sub::sub),
                FloatPrecision::F64 => self.fdyadic::<f64>(rd, r1, r2, Sub::sub),
            },
            Opcode::Fmul { rd, r1, r2, p } => match p {
                FloatPrecision::F16 => self.fdyadic::<f16>(rd, r1, r2, Mul::mul),
                FloatPrecision::F32 => self.fdyadic::<f32>(rd, r1, r2, Mul::mul),
                FloatPrecision::F64 => self.fdyadic::<f64>(rd, r1, r2, Mul::mul),
            },
            Opcode::Fdiv { rd, r1, r2, p } => match p {
                FloatPrecision::F16 => self.fdyadic::<f16>(rd, r1, r2, Div::div),
                FloatPrecision::F32 => self.fdyadic::<f32>(rd, r1, r2, Div::div),
                FloatPrecision::F64 => self.fdyadic::<f64>(rd, r1, r2, Div::div),
            },
            Opcode::Fma { rd, r1, r2, p } => match p {
                FloatPrecision::F16 => self.fma::<f16>(rd, r1, r2),
                FloatPrecision::F32 => self.fma::<f32>(rd, r1, r2),
                FloatPrecision::F64 => self.fma::<f64>(rd, r1, r2),
            },
            Opcode::Fsqrt { rd, r1, p } => match p {
                FloatPrecision::F16 => self.fmonadic::<f16>(rd, r1, Float::sqrt),
                FloatPrecision::F32 => self.fmonadic::<f32>(rd, r1, Float::sqrt),
                FloatPrecision::F64 => self.fmonadic::<f64>(rd, r1, Float::sqrt),
            },
            Opcode::Fmin { rd, r1, r2, p } => match p {
                FloatPrecision::F16 => self.fdyadic::<f16>(rd, r1, r2, Float::min),
                FloatPrecision::F32 => self.fdyadic::<f32>(rd, r1, r2, Float::min),
                FloatPrecision::F64 => self.fdyadic::<f64>(rd, r1, r2, Float::min),
            },
            Opcode::Fmax { rd, r1, r2, p } => match p {
                FloatPrecision::F16 => self.fdyadic::<f16>(rd, r1, r2, Float::max),
                FloatPrecision::F32 => self.fdyadic::<f32>(rd, r1, r2, Float::max),
                FloatPrecision::F64 => self.fdyadic::<f64>(rd, r1, r2, Float::max),
            },
            Opcode::Fsat { rd, r1, p } => match p {
                FloatPrecision::F16 => self.fmonadic::<f16>(rd, r1, Float::ceil),
                FloatPrecision::F32 => self.fmonadic::<f32>(rd, r1, Float::ceil),
                FloatPrecision::F64 => self.fmonadic::<f64>(rd, r1, Float::ceil),
            },
            Opcode::Fcnv {
                rd,
                r1,
                p: FloatCastType { to, from },
            } => match (to, from) {
                (FloatPrecision::F16, FloatPrecision::F16) => self.fcnv::<f16, f16>(rd, r1),
                (FloatPrecision::F16, FloatPrecision::F32) => self.fcnv::<f16, f32>(rd, r1),
                (FloatPrecision::F16, FloatPrecision::F64) => self.fcnv::<f16, f64>(rd, r1),
                (FloatPrecision::F32, FloatPrecision::F16) => self.fcnv::<f32, f16>(rd, r1),
                (FloatPrecision::F32, FloatPrecision::F32) => self.fcnv::<f32, f32>(rd, r1),
                (FloatPrecision::F32, FloatPrecision::F64) => self.fcnv::<f32, f64>(rd, r1),
                (FloatPrecision::F64, FloatPrecision::F16) => self.fcnv::<f64, f16>(rd, r1),
                (FloatPrecision::F64, FloatPrecision::F32) => self.fcnv::<f64, f32>(rd, r1),
                (FloatPrecision::F64, FloatPrecision::F64) => self.fcnv::<f64, f64>(rd, r1),
            },
            Opcode::Fnan { rd, r1, p } => match p {
                FloatPrecision::F16 => self.fnan::<f16>(rd, r1),
                FloatPrecision::F32 => self.fnan::<f32>(rd, r1),
                FloatPrecision::F64 => self.fnan::<f64>(rd, r1),
            },
        }
        Ok(())
    }

    fn fcmp<F: Float>(&mut self, r1: Register, r2: Register)
    where
        u64: BitAccess<F::IntType>, {
        self.cpu.set_cmp_f_u64::<F>(self.regval(r1), self.regval(r2));
    }
    fn fto<F: Float>(&mut self, rd: Register, rs: Register)
    where
        u64: BitAccess<F::IntType>, {
        let v = self.regval(rs);
        self.regval_mut(rd).write(0, F::from_int(v as i64).to_bits());
    }
    fn ffrom<F: Float>(&mut self, rd: Register, rs: Register)
    where
        u64: BitAccess<F::IntType>, {
        *self.regval_mut(rd) = F::from_bits_u64(self.regval(rs)).to_int() as u64;
    }
    fn fmonadic<F: Float>(&mut self, rd: Register, rs: Register, op: impl Fn(F) -> F)
    where
        u64: BitAccess<F::IntType>, {
        let v = op(F::from_bits_u64(self.regval(rs)));
        self.regval_mut(rd).write(0, v.to_bits());
    }
    fn fdyadic<F: Float>(&mut self, rd: Register, r1: Register, r2: Register, op: impl Fn(F, F) -> F)
    where
        u64: BitAccess<F::IntType>, {
        let v = op(F::from_bits_u64(self.regval(r1)), F::from_bits_u64(self.regval(r2)));
        self.regval_mut(rd).write(0, v.to_bits());
    }
    fn fma<F: Float>(&mut self, rd: Register, r1: Register, r2: Register)
    where
        u64: BitAccess<F::IntType>, {
        let v = F::from_bits_u64(self.regval(rd)) + F::from_bits_u64(self.regval(r1)) * F::from_bits_u64(self.regval(r2));
        self.regval_mut(rd).write(0, v.to_bits());
    }
    fn fcnv<To: Float + FloatFrom<From>, From: Float>(&mut self, rd: Register, r1: Register)
    where
        u64: BitAccess<To::IntType> + BitAccess<From::IntType>, {
        let v = From::from_bits_u64(self.regval(r1));
        self.regval_mut(rd).write(0, To::from(v).to_bits());
    }
    fn fnan<F: Float>(&mut self, rd: Register, r1: Register)
    where
        u64: BitAccess<F::IntType>, {
        let v = F::from_bits_u64(self.regval(r1)).is_nan();
        *self.regval_mut(rd) = v as u64;
    }

    fn add(&mut self, a: u64, b: u64, reg: Register) {
        let (v, u_overflow) = overflowing_add_unsigned(a, b, self.cpu.get_flag(StFlag::CARRY_BORROW_UNSIGNED));
        *self.regval_mut(reg) = v;
        self.cpu.set_flag(StFlag::CARRY_BORROW_UNSIGNED, u_overflow);

        let (v, s_overflow) = overflowing_add_signed(a as i64, b as i64, self.cpu.get_flag(StFlag::CARRY_BORROW_UNSIGNED));
        *self.regval_mut(reg) = v as u64;
        self.cpu.set_flag(StFlag::CARRY_BORROW, s_overflow);
    }
    fn sub(&mut self, a: u64, b: u64, reg: Register) {
        let (v, u_overflow) = overflowing_sub_unsigned(a, b, self.cpu.get_flag(StFlag::CARRY_BORROW_UNSIGNED));
        *self.regval_mut(reg) = v;
        self.cpu.set_flag(StFlag::CARRY_BORROW_UNSIGNED, u_overflow);

        let (v, s_overflow) = overflowing_sub_signed(a as i64, b as i64, self.cpu.get_flag(StFlag::CARRY_BORROW_UNSIGNED));
        *self.regval_mut(reg) = v as u64;
        self.cpu.set_flag(StFlag::CARRY_BORROW, s_overflow);
    }
    fn imul(&mut self, a: u64, b: u64, reg: Register) { *self.regval_mut(reg) = ((a as i64) * (b as i64)) as u64; }
    fn idiv(&mut self, a: u64, b: u64, reg: Register) { *self.regval_mut(reg) = ((a as i64) / (b as i64)) as u64; }
    fn umul(&mut self, a: u64, b: u64, reg: Register) { *self.regval_mut(reg) = a * b; }
    fn udiv(&mut self, a: u64, b: u64, reg: Register) { *self.regval_mut(reg) = a / b; }
    fn rem(&mut self, a: u64, b: u64, reg: Register) { *self.regval_mut(reg) = ((a as i64) % (b as i64)) as u64; }
    fn r#mod(&mut self, a: u64, b: u64, reg: Register) {
        if b as i64 == -1 {
            *self.regval_mut(reg) = 0;
        } else {
            *self.regval_mut(reg) = ((a as i64).rem_euclid(b as i64)) as u64;
        }
    }
    fn and(&mut self, a: u64, b: u64, reg: Register) { *self.regval_mut(reg) = a & b; }
    fn or(&mut self, a: u64, b: u64, reg: Register) { *self.regval_mut(reg) = a | b; }
    fn nor(&mut self, a: u64, b: u64, reg: Register) { *self.regval_mut(reg) = !(a | b); }
    fn xor(&mut self, a: u64, b: u64, reg: Register) { *self.regval_mut(reg) = a ^ b; }
    fn shl(&mut self, a: u64, b: u64, reg: Register) { *self.regval_mut(reg) = a << b; }
    fn sar(&mut self, a: u64, b: u64, reg: Register) { *self.regval_mut(reg) = ((a as i64) >> b) as u64; }
    fn shr(&mut self, a: u64, b: u64, reg: Register) { *self.regval_mut(reg) = a >> b; }
    fn bit(&mut self, a: u64, b: u64, reg: Register) { *self.regval_mut(reg) = (a >> b) & 1; }

    /// takes ownership of self
    pub(crate) fn run(mut self) -> RunStats {
        let now = Instant::now();
        if self.cycle_limit == 0 {
            while self.cpu.running {
                self.run_internal();
            }
        } else {
            while self.cpu.running {
                if self.cycle_limit as u64 == self.cpu.cycle {
                    self.cpu.running = false;
                }
                self.run_internal();
            }
        }
        RunStats {
            elapsed: now.elapsed().as_secs_f64(),
            cycle:   self.cpu.cycle,
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub(crate) struct RunStats {
    pub(crate) elapsed: f64,
    pub(crate) cycle:   u64,
}
impl RunStats {
    #[allow(clippy::cast_precision_loss)]
    pub(crate) fn cycle_per_sec(self) -> f64 { self.cycle as f64 / self.elapsed }
}
