#![allow(clippy::cast_possible_truncation)]
#![allow(clippy::cast_lossless)]
#![allow(clippy::cast_precision_loss)]
#![allow(clippy::cast_sign_loss)]
#![allow(clippy::cast_possible_wrap)]
#![allow(clippy::unused_unit)]
#![allow(clippy::too_many_lines)]

use bitfield_struct::bitfield;
use bitflags::bitflags;
use sa::static_assert;
use std::{
    fmt::Debug,
    ops::{Add, AddAssign, Div, Index, IndexMut, Mul, Neg, Sub},
    time::Instant,
};

use crate::{
    ic::{IntQueueEntry, IC},
    io::IOC,
    mmu::{self, MMU},
    unsafe_read, unsafe_write,
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
macro_rules! get_usize {
    ($ident: ident.$($ident2: ident).*) => {
        $ident.$($ident2()).* as usize
    };
}
macro_rules! arithmetic {
    ($self: ident, $ci: ident, $func: ident, r) => {{
        let a = $self.regval_n(get_usize!($ci.r.rs1));
        let b = $self.regval_n(get_usize!($ci.r.rs2));
        let reg = get_usize!($ci.r.rde);
        $self.$func(a, b, reg);
    }};
    ($self: ident, $ci: ident, $func: ident, i) => {{
        let a = $self.regval_n(get_usize!($ci.m.rs1));
        let b = sign_extend!($ci.m().imm(), 16);
        let reg = get_usize!($ci.m.rde);
        $self.$func(a, b, reg);
    }};
}
macro_rules! bitwise {
    ($self: ident, $ci: ident, $func: ident, r) => {{
        let a = $self.regval_n(get_usize!($ci.r.rs1));
        let b = $self.regval_n(get_usize!($ci.r.rs2));
        let reg = get_usize!($ci.r.rde);
        $self.$func(a, b, reg);
    }};
    ($self: ident, $ci: ident, $func: ident, i) => {{
        let a = $self.regval_n(get_usize!($ci.m.rs1));
        let b = $ci.m().imm() as u64;
        let reg = get_usize!($ci.m.rde);
        $self.$func(a, b, reg);
    }};
}
macro_rules! make_comp {
    ($self: ident, $a: ident, $b: ident) => {{
        $self.cpu.set_flag(StFlag::EQUAL, $a == $b);
        $self.cpu.set_flag(StFlag::LESS, ($a as i64) < ($b as i64));
        $self.cpu.set_flag(StFlag::LESS_UNSIGNED, $a < $b);
        $self.cpu.set_flag(StFlag::SIGN, ($a as i64) < 0);
        $self.cpu.set_flag(StFlag::ZERO, $a == 0);
    }};
    ($self: ident, $a: ident, $b: ident, $T: ty) => {
        #[allow(clippy::float_cmp)]
        {
            $self.cpu.set_flag(StFlag::EQUAL, $a == $b);
            $self.cpu.set_flag(StFlag::LESS, $a < $b);
            $self.cpu.set_flag(StFlag::LESS_UNSIGNED, $a < $b);
            $self.cpu.set_flag(StFlag::SIGN, $a < <$T>::ZERO);
            $self.cpu.set_flag(StFlag::ZERO, $a == <$T>::ZERO);
        }
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
    type IntType: Into<u64> + Copy;
    fn to_bits(self) -> Self::IntType;
    /// bit preserving cast
    fn from_bits_u64(v: u64) -> Self { unsafe_read(&v, 0) }
    fn add_assign_bits(self, to: &mut u64) { *unsafe { std::ptr::from_mut::<u64>(to).cast::<Self>().as_mut().unwrap() } += self; }
    /// bit preserving cast
    fn to_bits_u64(self) -> u64 { self.to_bits().into() }
    /// overriding bit preserving cast
    // probably doesnt need unsafe write
    fn to_bits_to_u64(self, v: &mut u64) { unsafe_write(v, self.to_bits(), 0) }

    /// arithmetic cast
    fn from_int(v: u64) -> Self;
    /// arithmetic cast
    fn to_int(self) -> u64;
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
    fn to_bits(self) -> Self::IntType { self.to_bits() }
    fn from_int(v: u64) -> Self { f16::from_f64(v as f64) }
    fn to_int(self) -> u64 { self.to_f64() as u64 }
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
    fn to_bits(self) -> Self::IntType { self.to_bits() }
    fn from_int(v: u64) -> Self { v as Self }
    fn to_int(self) -> u64 { self as u64 }
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
    fn to_bits(self) -> Self::IntType { self.to_bits() }
    fn from_int(v: u64) -> Self { v as Self }
    fn to_int(self) -> u64 { self as u64 }
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
trait FloatTo<F: Float>: Float {
    fn to(self) -> F;
}
impl<From: Float, F: FloatFrom<From>> FloatTo<F> for From {
    fn to(self) -> F { F::from(self) }
}

#[rustfmt::skip]
#[bitfield(u32)]
#[derive(PartialEq, Eq)]
struct E {
    #[bits(8)]  opcode: u32,
    #[bits(8)]  imm:    u32,
    #[bits(4)]  func:   u32,
    #[bits(4)]  rs2:    u32,
    #[bits(4)]  rs1:    u32,
    #[bits(4)]  rde:    u32,
}
#[rustfmt::skip]
#[bitfield(u32)]
#[derive(PartialEq, Eq)]
struct R {
    #[bits(8)]  opcode: u32,
    #[bits(12)] imm:    u32,
    #[bits(4)]  rs2:    u32,
    #[bits(4)]  rs1:    u32,
    #[bits(4)]  rde:    u32,
}
#[rustfmt::skip]
#[bitfield(u32)]
#[derive(PartialEq, Eq)]
struct M {
    #[bits(8)]  opcode: u32,
    #[bits(16)] imm:    u32,
    #[bits(4)]  rs1:    u32,
    #[bits(4)]  rde:    u32,
}
#[rustfmt::skip]
#[bitfield(u32)]
#[derive(PartialEq, Eq)]
struct F {
    #[bits(8)]  opcode: u32,
    #[bits(16)] imm:    u32,
    #[bits(4)]  func:   u32,
    #[bits(4)]  rde:    u32,
}
#[rustfmt::skip]
#[bitfield(u32)]
#[derive(PartialEq, Eq)]
struct B {
    #[bits(8)]  opcode: u32,
    #[bits(20)] imm:    u32,
    #[bits(4)]  func:   u32,
}

#[derive(Clone, Copy)]
#[repr(C)]
union Instruction {
    opcode: u8,
    bits:   u32,
    e:      E,
    r:      R,
    m:      M,
    f:      F,
    b:      B,
}
impl Instruction {
    const fn zero() -> Self { Instruction { bits: 0 } }
    const fn opcode(self) -> u8 { unsafe { self.opcode } }
    const fn bits(self) -> u32 { unsafe { self.bits } }
    const fn from_bits(bits: u32) -> Self { Self { bits } }
    const fn e(self) -> E { unsafe { self.e } }
    const fn r(self) -> R { unsafe { self.r } }
    const fn m(self) -> M { unsafe { self.m } }
    const fn f(self) -> F { unsafe { self.f } }
    const fn b(self) -> B { unsafe { self.b } }
}
impl Debug for Instruction {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result { write!(f, "{:b}", self.bits()) }
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
impl RegisterName {
    const fn try_from_u8(val: u8) -> Option<Self> {
        const LIST: [RegisterName; 16] = [
            RN::Rz,
            RN::Ra,
            RN::Rb,
            RN::Rc,
            RN::Rd,
            RN::Re,
            RN::Rf,
            RN::Rg,
            RN::Rh,
            RN::Ri,
            RN::Rj,
            RN::Rk,
            RN::Ip,
            RN::Sp,
            RN::Fp,
            RN::St,
        ];
        if val >= 16 {
            None
        } else {
            Some(LIST[val as usize])
        }
    }
    /// # Panics
    ///
    /// Panics if value is not valid `RegisterName`
    const fn from_u8(val: u8) -> Self {
        const LIST: [RegisterName; 16] = [
            RN::Rz,
            RN::Ra,
            RN::Rb,
            RN::Rc,
            RN::Rd,
            RN::Re,
            RN::Rf,
            RN::Rg,
            RN::Rh,
            RN::Ri,
            RN::Rj,
            RN::Rk,
            RN::Ip,
            RN::Sp,
            RN::Fp,
            RN::St,
        ];
        if val >= 16 {
            panic!()
        } else {
            LIST[val as usize]
        }
    }
}
#[macro_export]
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

#[repr(u8)]
#[derive(Debug, Clone, Copy)]
enum ProcMode {
    Kernel,
    User,
}
impl ProcMode {
    const fn bool(self) -> bool { matches!(self, Self::User) }
}

#[repr(C)]
#[derive(Debug, Clone, Copy)]
struct Registers(pub [u64; 16]);
impl Registers {
    const fn get_flag(&self, flag: StFlag) -> bool { self.0[RN::St as usize] & flag.bits() == 1 }
    fn set_flag(&mut self, flag: StFlag, value: bool) {
        if value {
            self[RN::St] |= flag.bits();
        } else {
            self[RN::St] &= !flag.bits();
        }
    }
    const fn current_instr(&self) -> Instruction { unsafe_read(&self.0[RN::St as usize], 1) }
    fn set_current_instr(&mut self, instruction: Instruction) {
        unsafe_write(&mut self[RN::St], instruction, 1);
        /* self[RN::St] = ((self[RN::St] << 32) >> 32) + ((instruction.bits() as u64) << 32); */
    }
}
impl Index<RegisterName> for Registers {
    type Output = u64;
    fn index(&self, index: RegisterName) -> &Self::Output { &self.0[index as usize] }
}
impl IndexMut<RegisterName> for Registers {
    fn index_mut(&mut self, index: RegisterName) -> &mut Self::Output { &mut self.0[index as usize] }
}
use RegisterName as RN;

#[repr(u8)]
#[derive(Debug, Clone, Copy)]
pub enum InterruptErr {
    DivideByZero = 0,
    BreakPoint,
    InvalidInstruction,
    StackUnderflow,
    UnalignedAccess,
    AccessViolation,
    InterruptOverflow,
}
impl InterruptErr {
    pub const fn try_from_u8(val: u8) -> Option<Self> {
        match val {
            0 => Some(Self::DivideByZero),
            1 => Some(Self::BreakPoint),
            2 => Some(Self::InvalidInstruction),
            3 => Some(Self::StackUnderflow),
            4 => Some(Self::UnalignedAccess),
            5 => Some(Self::AccessViolation),
            6 => Some(Self::InterruptOverflow),
            _ => None,
        }
    }
    /// # Panics
    ///
    /// panics if value is not a valid `InterruptErr`
    pub const fn from_u8(val: u8) -> Self {
        match val {
            0 => Self::DivideByZero,
            1 => Self::BreakPoint,
            2 => Self::InvalidInstruction,
            3 => Self::StackUnderflow,
            4 => Self::UnalignedAccess,
            5 => Self::AccessViolation,
            6 => Self::InterruptOverflow,
            _ => panic!(),
        }
    }
}

#[allow(clippy::upper_case_acronyms)]
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct CPU {
    registers: Registers,
    cycle:     u64,
    instr:     Instruction,
    running:   bool,
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
    const fn get_flag(&self, flag: StFlag) -> bool { self.registers.get_flag(flag) }
    pub fn set_flag(&mut self, flag: StFlag, value: bool) { self.registers.set_flag(flag, value); }
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
pub struct Emulator {
    cpu: CPU,
    ic:  IC,
    mmu: MMU,
    ioc: IOC,

    debug:       bool,
    no_color:    bool,
    cycle_limit: usize,
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
    const fn current_instr(&self) -> Instruction { self.cpu.registers.current_instr() }
    fn set_current_instr(&mut self, instruction: Instruction) { self.cpu.registers.set_current_instr(instruction); }

    fn read_instruction(&self, addr: u64) -> Result<Instruction, mmu::Response> {
        let mut addr = addr;
        if self.cpu.get_flag(StFlag::MODE) == ProcMode::User.bool() {
            addr = self.mmu.translate_address(addr, mmu::AccessMode::Execute)?;
        }
        self.mmu.phys_get_u32(addr).map(Instruction::from_bits)
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

    fn push_interrupt(&mut self, err: InterruptErr) {
        let mut err = err;
        if self.ic.queue.is_empty() {
            self.ic.ret_addr = self.cpu.registers[RN::Ip];
            self.ic.ret_status = self.cpu.registers[RN::St];
            self.cpu.set_flag(StFlag::MODE, ProcMode::User.bool());
        }
        if self.ic.queue.len() == self.ic.queue.capacity() {
            // interrupt queue overflow
            self.ic.queue.clear();
            err = InterruptErr::InterruptOverflow;
        }
        // hijack instruction pointer
        match self
            .mmu
            .phys_read_u64(self.ic.ivt_base_address + 8 * err as u64, &mut self.cpu.registers[RN::Ip])
        {
            Err(err) => {
                self.push_interrupt_from_mmu(err);
            }
            Ok(()) => {
                self.ic.queue.push(IntQueueEntry { code: err as u8 });
            }
        }
    }
    fn push_interrupt_from_mmu(&mut self, res: mmu::Response) { self.push_interrupt(res.to_interrupt_err()); }

    fn return_interrupt(&mut self) {
        if self.ic.queue.is_empty() {
            return;
        }

        let _ = self.ic.queue.remove(0);
        if self.ic.queue.is_empty() {
            self.cpu.registers[RN::Ip] = self.ic.ret_addr;
            self.cpu.registers[RN::St] = self.ic.ret_status;
        } else {
            // hijack instruction pointer
            let code = self.ic.queue[self.ic.queue.len() - 1].code;
            match self.mmu.phys_get_u64(self.ic.ivt_base_address + 8 * code as u64) {
                Err(err) => self.push_interrupt_from_mmu(err),
                Ok(res) => self.cpu.registers[RN::Ip] = res,
            }
        }
    }
    const fn regval(&self, reg: RegisterName) -> u64 { self.cpu.registers.0[reg as usize] }
    const fn regval_n(&self, reg: usize) -> u64 { self.cpu.registers.0[reg] }
    fn regval_mut(&mut self, reg: RegisterName) -> &mut u64 { &mut self.cpu.registers.0[reg as usize] }
    fn regval_mut_n(&mut self, reg: usize) -> &mut u64 { &mut self.cpu.registers.0[reg] }
    const fn proc_mode_is_user(&self) -> bool { self.cpu.get_flag(StFlag::MODE) == ProcMode::User.bool() }
    const fn proc_mode_is_user_then_invalid(&self) -> Result<(), InvalidInstructionError> {
        if self.proc_mode_is_user() {
            Err(InvalidInstructionError)
        } else {
            Ok(())
        }
    }
    fn run_internal(&mut self) {
        self.cpu.cycle += 1;
        println!("[at {:#016x} {:02x}]", self.regval(RN::Ip), self.current_instr().opcode());

        // load instruction
        match self.read_instruction(self.regval(RN::Ip)) {
            Ok(instr) => self.set_current_instr(instr),
            Err(err) => {
                self.push_interrupt_from_mmu(err);
                return;
            }
        }

        *self.regval_mut(RN::Ip) += 4;

        if let Err(InvalidInstructionError) = self.interpret_code() {
            self.push_interrupt(InterruptErr::InvalidInstruction);
        }

        if self.ioc.out_pin {
            self.ioc.dev_receive();
        }

        self.cpu.registers[RN::Rz] = 0;

        if self.regval(RN::Sp) > self.regval(RN::Fp) {
            self.push_interrupt(InterruptErr::StackUnderflow);
        }
    }

    fn push_stack(&mut self, data: u64) {
        *self.regval_mut(RN::Sp) -= 8;
        if let Err(err) = self.write_u64(self.regval(RN::Sp), data) {
            self.push_interrupt_from_mmu(err);
        }
    }
    fn pop_stack(&mut self, value: &mut u64) {
        match self.read_u64(self.regval(RN::Sp)) {
            Ok(val) => {
                *self.regval_mut(RN::Sp) += 8;
                *value = val;
            }
            Err(err) => self.push_interrupt_from_mmu(err),
        }
    }
    fn pop_stack_to(&mut self, reg: RegisterName) {
        match self.read_u64(self.regval(RN::Sp)) {
            Ok(val) => {
                *self.regval_mut(RN::Sp) += 8;
                *self.regval_mut(reg) = val;
            }
            Err(err) => self.push_interrupt_from_mmu(err),
        }
    }
    fn if_cond_then_do_that_weird_thang(&mut self, ci: Instruction, cond: bool) {
        if cond {
            *self.regval_mut(RN::Ip) += (4 * sign_extend!(ci.m().imm(), 20) as i64) as u64;
        }
    }
    fn interpret_code(&mut self) -> Result<(), InvalidInstructionError> {
        let ci = self.current_instr();
        match ci.opcode() {
            // System control
            0x01 => match ci.f().func() {
                // int
                0x00 => self.push_interrupt(InterruptErr::from_u8(ci.f().imm() as u8)),
                // iret | ires
                0x01 | 0x02 => {
                    self.proc_mode_is_user_then_invalid()?;
                    self.return_interrupt();
                }
                // usr
                0x03 => {
                    self.proc_mode_is_user_then_invalid()?;
                    self.cpu.set_flag(StFlag::MODE, ProcMode::User.bool());
                    self.cpu.registers[RN::Ip] = self.regval_n(get_usize!(ci.f.rde));
                }
                _ => Err(InvalidInstructionError)?,
            },
            // outr
            0x02 => {
                self.proc_mode_is_user_then_invalid()?;
                self.ioc
                    .send_out(self.regval_n(get_usize!(ci.m.rde)) as u16, self.regval_n(get_usize!(ci.m.rs1)));
            }
            // outi
            0x03 => {
                self.proc_mode_is_user_then_invalid()?;
                self.ioc.send_out(ci.m().imm() as u16, self.regval_n(get_usize!(ci.m.rs1)));
            }
            // inr
            0x04 => {
                self.proc_mode_is_user_then_invalid()?;
                *self.regval_mut_n(get_usize!(ci.m.rde)) = self.ioc.port_data(self.regval_n(get_usize!(ci.m.rs1)) as u16);
            }
            // ini
            0x05 => {
                self.proc_mode_is_user_then_invalid()?;
                *self.regval_mut_n(get_usize!(ci.m.rde)) = self.ioc.port_data(ci.m().imm() as u16);
            }
            // jal
            0x06 => {
                self.push_stack(self.regval(RN::Ip));
                *self.regval_mut(RN::Ip) = self.regval_n(get_usize!(ci.m.rs1)) + (4 * sign_extend!(ci.m().imm(), 16) as i64) as u64;
            }
            // jalr
            0x07 => {
                *self.regval_mut_n(ci.m().rde() as usize) = self.regval(RN::Ip);
                *self.regval_mut(RN::Ip) = self.regval_n(get_usize!(ci.m.rs1)) + (4 * sign_extend!(ci.m().imm(), 16) as i64) as u64;
            }
            // ret
            0x08 => {
                self.pop_stack_to(RN::Ip);
            }
            // retr
            0x09 => {
                *self.regval_mut(RN::Ip) = self.regval_n(get_usize!(ci.m.rs1));
            }
            // branch instructions
            0x0a => match ci.b().imm() {
                // bra
                0x0 => {
                    self.if_cond_then_do_that_weird_thang(ci, true);
                }
                // beq
                0x1 => {
                    self.if_cond_then_do_that_weird_thang(ci, self.cpu.get_flag(StFlag::EQUAL));
                }
                // bez
                0x2 => {
                    self.if_cond_then_do_that_weird_thang(ci, self.cpu.get_flag(StFlag::ZERO));
                }
                // blt
                0x3 => {
                    self.if_cond_then_do_that_weird_thang(ci, self.cpu.get_flag(StFlag::LESS));
                }
                // ble
                0x4 => {
                    self.if_cond_then_do_that_weird_thang(ci, self.cpu.get_flag(StFlag::LESS) || self.cpu.get_flag(StFlag::EQUAL));
                }
                // bltu
                0x5 => {
                    self.if_cond_then_do_that_weird_thang(ci, self.cpu.get_flag(StFlag::LESS_UNSIGNED));
                }
                // bleu
                0x6 => {
                    self.if_cond_then_do_that_weird_thang(ci, self.cpu.get_flag(StFlag::LESS_UNSIGNED) || self.cpu.get_flag(StFlag::EQUAL));
                }
                // bne
                0x9 => {
                    self.if_cond_then_do_that_weird_thang(ci, !self.cpu.get_flag(StFlag::EQUAL));
                }
                // bnz
                0xa => {
                    self.if_cond_then_do_that_weird_thang(ci, !self.cpu.get_flag(StFlag::ZERO));
                }
                // bge
                0xb => {
                    self.if_cond_then_do_that_weird_thang(ci, !self.cpu.get_flag(StFlag::LESS));
                }
                // bgt
                0xc => {
                    self.if_cond_then_do_that_weird_thang(ci, !self.cpu.get_flag(StFlag::LESS) && !self.cpu.get_flag(StFlag::EQUAL));
                }
                // bgeu
                0xd => {
                    self.if_cond_then_do_that_weird_thang(ci, !self.cpu.get_flag(StFlag::LESS_UNSIGNED));
                }
                // bteu
                0xe => {
                    self.if_cond_then_do_that_weird_thang(ci, !self.cpu.get_flag(StFlag::LESS_UNSIGNED) && !self.cpu.get_flag(StFlag::EQUAL));
                }
                _ => Err(InvalidInstructionError)?,
            },
            // push
            0x0b => {
                self.push_stack(self.regval_n(get_usize!(ci.m.rs1)));
            }
            // pop
            0x0c => {
                self.pop_stack_to(RN::from_u8(ci.m().rde() as u8));
            }
            // enter
            0x0d => {
                self.push_stack(self.regval(RN::Fp));
                *self.regval_mut(RN::Fp) = self.regval(RN::Sp);
            }
            // leave
            0x0e => {
                *self.regval_mut(RN::Sp) = self.regval(RN::Fp);
                self.pop_stack_to(RN::Fp);
            }
            // load immediate
            0x10 => match ci.f().func() {
                // lli
                0 => unsafe_write(self.regval_mut_n(get_usize!(ci.f.rde)), ci.f().imm() as u16, 0),
                // llis
                1 => *self.regval_mut_n(get_usize!(ci.f.rde)) = sign_extend!(ci.f().imm(), 16),
                // lui
                2 => unsafe_write(self.regval_mut_n(get_usize!(ci.f.rde)), ci.f().imm() as u16, 1),
                // luis
                3 => *self.regval_mut_n(get_usize!(ci.f.rde)) = sign_extend!(ci.f().imm(), 16) << 16,
                // lti
                4 => unsafe_write(self.regval_mut_n(get_usize!(ci.f.rde)), ci.f().imm() as u16, 2),
                // ltis
                5 => *self.regval_mut_n(get_usize!(ci.f.rde)) = sign_extend!(ci.f().imm(), 16) << 32,
                // ltui
                6 => unsafe_write(self.regval_mut_n(get_usize!(ci.f.rde)), ci.f().imm() as u16, 3),
                // ltuis
                7 => *self.regval_mut_n(get_usize!(ci.f.rde)) = sign_extend!(ci.f().imm(), 16) << 48,
                _ => Err(InvalidInstructionError)?,
            },
            // lw
            0x11 => match self.read_u64(
                self.regval_n(get_usize!(ci.e.rs1)) + sign_extend!(ci.e().imm(), 8) + ((self.regval_n(get_usize!(ci.e.rs1))) << ci.e().func() as u64),
            ) {
                Ok(val) => *self.regval_mut_n(get_usize!(ci.e.rde)) = val,
                Err(err) => self.push_interrupt_from_mmu(err),
            },
            // lh
            0x12 => match self.read_u32(
                self.regval_n(get_usize!(ci.e.rs1)) + sign_extend!(ci.e().imm(), 8) + ((self.regval_n(get_usize!(ci.e.rs1))) << ci.e().func() as u64),
            ) {
                Ok(val) => unsafe_write(self.regval_mut_n(get_usize!(ci.e.rde)), val, 0),
                Err(err) => self.push_interrupt_from_mmu(err),
            },
            // lhs
            0x13 => {
                match self.read_u32(
                    self.regval_n(get_usize!(ci.e.rs1))
                        + sign_extend!(ci.e().imm(), 8)
                        + ((self.regval_n(get_usize!(ci.e.rs1))) << ci.e().func() as u64),
                ) {
                    Ok(val) => unsafe_write(self.regval_mut_n(get_usize!(ci.e.rde)), val, 0),
                    Err(err) => self.push_interrupt_from_mmu(err),
                }
                *self.regval_mut_n(get_usize!(ci.e.rde)) = sign_extend!(self.regval_n(get_usize!(ci.e.rde)), 32);
            }
            // lq
            0x14 => match self.read_u16(
                self.regval_n(get_usize!(ci.e.rs1)) + sign_extend!(ci.e().imm(), 8) + ((self.regval_n(get_usize!(ci.e.rs1))) << ci.e().func() as u64),
            ) {
                Ok(val) => unsafe_write(self.regval_mut_n(get_usize!(ci.e.rde)), val, 0),
                Err(err) => self.push_interrupt_from_mmu(err),
            },
            // lqs
            0x15 => {
                match self.read_u16(
                    self.regval_n(get_usize!(ci.e.rs1))
                        + sign_extend!(ci.e().imm(), 8)
                        + ((self.regval_n(get_usize!(ci.e.rs1))) << ci.e().func() as u64),
                ) {
                    Ok(val) => unsafe_write(self.regval_mut_n(get_usize!(ci.e.rde)), val, 0),
                    Err(err) => self.push_interrupt_from_mmu(err),
                }
                *self.regval_mut_n(get_usize!(ci.e.rde)) = sign_extend!(self.regval_n(get_usize!(ci.e.rde)), 32);
            }
            // lb
            0x16 => {
                match self.read_u8(
                    self.regval_n(get_usize!(ci.e.rs1))
                        + sign_extend!(ci.e().imm(), 8)
                        + ((self.regval_n(get_usize!(ci.e.rs1))) << ci.e().func() as u64),
                ) {
                    Ok(val) => unsafe_write(self.regval_mut_n(get_usize!(ci.e.rde)), val, 0),
                    Err(err) => self.push_interrupt_from_mmu(err),
                }
            }
            // lbs
            0x17 => {
                match self.read_u8(
                    self.regval_n(get_usize!(ci.e.rs1))
                        + sign_extend!(ci.e().imm(), 8)
                        + ((self.regval_n(get_usize!(ci.e.rs1))) << ci.e().func() as u64),
                ) {
                    Ok(val) => unsafe_write(self.regval_mut_n(get_usize!(ci.e.rde)), val, 0),
                    Err(err) => self.push_interrupt_from_mmu(err),
                }
                *self.regval_mut_n(get_usize!(ci.e.rde)) = sign_extend!(self.regval_n(get_usize!(ci.e.rde)), 32);
            }
            // sw
            0x18 => {
                if let Err(err) = self.write_u64(
                    self.regval_n(get_usize!(ci.e.rs1))
                        + sign_extend!(ci.e().imm(), 8)
                        + ((self.regval_n(get_usize!(ci.e.rs1))) << ci.e().func() as u64),
                    self.regval_n(get_usize!(ci.e.rde)),
                ) {
                    self.push_interrupt_from_mmu(err);
                }
            }
            // sh
            0x19 => {
                if let Err(err) = self.write_u32(
                    self.regval_n(get_usize!(ci.e.rs1))
                        + sign_extend!(ci.e().imm(), 8)
                        + ((self.regval_n(get_usize!(ci.e.rs1))) << ci.e().func() as u64),
                    self.regval_n(get_usize!(ci.e.rde)) as u32,
                ) {
                    self.push_interrupt_from_mmu(err);
                }
            }
            // sq
            0x1a => {
                if let Err(err) = self.write_u16(
                    self.regval_n(get_usize!(ci.e.rs1))
                        + sign_extend!(ci.e().imm(), 8)
                        + ((self.regval_n(get_usize!(ci.e.rs1))) << ci.e().func() as u64),
                    self.regval_n(get_usize!(ci.e.rde)) as u16,
                ) {
                    self.push_interrupt_from_mmu(err);
                }
            }
            // sb
            0x1b => {
                if let Err(err) = self.write_u8(
                    self.regval_n(get_usize!(ci.e.rs1))
                        + sign_extend!(ci.e().imm(), 8)
                        + ((self.regval_n(get_usize!(ci.e.rs1))) << ci.e().func() as u64),
                    self.regval_n(get_usize!(ci.e.rde)) as u8,
                ) {
                    self.push_interrupt_from_mmu(err);
                }
            }
            // cmpr
            0x1e => {
                let a = self.regval_n(get_usize!(ci.m.rde));
                let b = self.regval_n(get_usize!(ci.m.rs1));
                self.cmp(a, b);
            }
            // cmpi
            0x1f => {
                let (a, b) = if ci.f().func() == 0 {
                    (self.regval_n(get_usize!(ci.f.rde)), sign_extend!(ci.f().imm(), 16))
                } else if ci.f().func() == 1 {
                    (sign_extend!(ci.f().imm(), 16), self.regval_n(get_usize!(ci.f.rde)))
                } else {
                    return Err(InvalidInstructionError);
                };
                self.cmp(a, b);
            }
            // addr
            0x20 => arithmetic!(self, ci, add, r),
            // addi
            0x21 => arithmetic!(self, ci, add, i),
            // subr
            0x22 => arithmetic!(self, ci, sub, r),
            // subi
            0x23 => arithmetic!(self, ci, sub, i),
            // imulr
            0x24 => arithmetic!(self, ci, imul, r),
            // imuli
            0x25 => arithmetic!(self, ci, imul, i),
            // idivr
            0x26 => arithmetic!(self, ci, idiv, r),
            // idivi
            0x27 => {
                if self.regval_n(get_usize!(ci.e.rs2)) == 0 {
                    self.push_interrupt(InterruptErr::DivideByZero);
                } else {
                    arithmetic!(self, ci, idiv, i);
                }
            }
            // umulr
            0x28 => arithmetic!(self, ci, umul, r),
            // umuli
            0x29 => arithmetic!(self, ci, umul, i),
            // udivr
            0x2a => arithmetic!(self, ci, udiv, r),
            // udivi
            0x2b => arithmetic!(self, ci, udiv, i),
            // remr
            0x2c => arithmetic!(self, ci, rem, r),
            // remi
            0x2d => arithmetic!(self, ci, rem, i),
            // modr
            0x2e => arithmetic!(self, ci, r#mod, r),
            // modi
            0x2f => arithmetic!(self, ci, r#mod, i),

            // andr
            0x30 => bitwise!(self, ci, and, r),
            // andi
            0x31 => bitwise!(self, ci, and, i),
            // orr
            0x32 => bitwise!(self, ci, or, r),
            // ori
            0x33 => bitwise!(self, ci, or, i),
            // norr
            0x34 => bitwise!(self, ci, nor, r),
            // nori
            0x35 => bitwise!(self, ci, nor, i),
            // xorr
            0x36 => bitwise!(self, ci, xor, r),
            // xori
            0x37 => bitwise!(self, ci, xor, i),
            // shlr
            0x38 => bitwise!(self, ci, shl, r),
            // shli
            0x39 => bitwise!(self, ci, shl, i),
            // asrr
            0x3a => bitwise!(self, ci, sar, r),
            // asri
            0x3b => bitwise!(self, ci, sar, i),
            // lsrr
            0x3c => bitwise!(self, ci, shr, r),
            // lsri
            0x3d => bitwise!(self, ci, shr, i),
            // bitr
            0x3e => bitwise!(self, ci, bit, r),
            // biti
            0x3f => bitwise!(self, ci, bit, i),

            /* Extension F- Floating-Point Operations */
            0x40 => match ci.e().func() {
                0 => self.fcomp::<f16>(ci),
                1 => self.fcomp::<f32>(ci),
                2 => self.fcomp::<f64>(ci),
                _ => Err(InvalidInstructionError)?,
            },
            // fto
            0x41 => match ci.e().func() {
                0 => self.fto::<f16>(ci),
                1 => self.fto::<f32>(ci),
                2 => self.fto::<f64>(ci),
                _ => Err(InvalidInstructionError)?,
            },
            // ffrom
            0x42 => match ci.e().func() {
                0 => self.ffrom::<f16>(ci),
                1 => self.ffrom::<f32>(ci),
                2 => self.ffrom::<f64>(ci),
                _ => Err(InvalidInstructionError)?,
            },
            // fneg
            0x43 => match ci.e().func() {
                0 => self.fneg::<f16>(ci),
                1 => self.fneg::<f32>(ci),
                2 => self.fneg::<f64>(ci),
                _ => Err(InvalidInstructionError)?,
            },
            // fabs
            0x44 => match ci.e().func() {
                0 => self.fabs::<f16>(ci),
                1 => self.fabs::<f32>(ci),
                2 => self.fabs::<f64>(ci),
                _ => Err(InvalidInstructionError)?,
            },
            // fadd
            0x45 => match ci.e().func() {
                0 => self.fadd::<f16>(ci),
                1 => self.fadd::<f32>(ci),
                2 => self.fadd::<f64>(ci),
                _ => Err(InvalidInstructionError)?,
            },
            // fsub
            0x46 => match ci.e().func() {
                0 => self.fsub::<f16>(ci),
                1 => self.fsub::<f32>(ci),
                2 => self.fsub::<f64>(ci),
                _ => Err(InvalidInstructionError)?,
            },
            // fmul
            0x47 => match ci.e().func() {
                0 => self.fmul::<f16>(ci),
                1 => self.fmul::<f32>(ci),
                2 => self.fmul::<f64>(ci),
                _ => Err(InvalidInstructionError)?,
            },
            // fdiv
            0x48 => match ci.e().func() {
                0 => self.fdiv::<f16>(ci),
                1 => self.fdiv::<f32>(ci),
                2 => self.fdiv::<f64>(ci),
                _ => Err(InvalidInstructionError)?,
            },
            // fma
            0x49 => match ci.e().func() {
                0 => self.fma::<f16>(ci),
                1 => self.fma::<f32>(ci),
                2 => self.fma::<f64>(ci),
                _ => Err(InvalidInstructionError)?,
            },
            // fsqrt
            0x4a => match ci.e().func() {
                0 => self.fsqrt::<f16>(ci),
                1 => self.fsqrt::<f32>(ci),
                2 => self.fsqrt::<f64>(ci),
                _ => Err(InvalidInstructionError)?,
            },
            // fmin
            0x4b => match ci.e().func() {
                0 => self.fmin::<f16>(ci),
                1 => self.fmin::<f32>(ci),
                2 => self.fmin::<f64>(ci),
                _ => Err(InvalidInstructionError)?,
            },
            // fmax
            0x4c => match ci.e().func() {
                0 => self.fmax::<f16>(ci),
                1 => self.fmax::<f32>(ci),
                2 => self.fmax::<f64>(ci),
                _ => Err(InvalidInstructionError)?,
            },
            // fsat
            0x4d => match ci.e().func() {
                0 => self.fsat::<f16>(ci),
                1 => self.fsat::<f32>(ci),
                2 => self.fsat::<f64>(ci),
                _ => Err(InvalidInstructionError)?,
            },
            // fcnv
            0x4e => match ci.e().func() {
                0b_00_00 => self.fconvert::<f16, f16>(ci),
                0b_00_01 => self.fconvert::<f16, f32>(ci),
                0b_00_10 => self.fconvert::<f16, f64>(ci),
                0b_01_00 => self.fconvert::<f32, f16>(ci),
                0b_01_01 => self.fconvert::<f32, f32>(ci),
                0b_01_10 => self.fconvert::<f32, f64>(ci),
                0b_10_00 => self.fconvert::<f64, f16>(ci),
                0b_10_01 => self.fconvert::<f64, f32>(ci),
                0b_10_10 => self.fconvert::<f64, f64>(ci),
                _ => Err(InvalidInstructionError)?,
            },
            // fnan
            0x4f => match ci.e().func() {
                0 => self.fnan::<f16>(ci),
                1 => self.fnan::<f32>(ci),
                2 => self.fnan::<f64>(ci),
                _ => Err(InvalidInstructionError)?,
            },
            _ => Err(InvalidInstructionError)?,
        }
        Ok(())
    }
    fn cmp(&mut self, a: u64, b: u64) {
        make_comp!(self, a, b);
    }

    fn add(&mut self, a: u64, b: u64, reg: usize) {
        let (v, u_overflow) = overflowing_add_unsigned(a, b, self.cpu.get_flag(StFlag::CARRY_BORROW_UNSIGNED));
        *self.regval_mut_n(reg) = v;
        self.cpu.set_flag(StFlag::CARRY_BORROW_UNSIGNED, u_overflow);

        let (v, s_overflow) = overflowing_add_signed(a as i64, b as i64, self.cpu.get_flag(StFlag::CARRY_BORROW_UNSIGNED));
        *self.regval_mut_n(reg) = v as u64;
        self.cpu.set_flag(StFlag::CARRY_BORROW, s_overflow);
    }
    fn sub(&mut self, a: u64, b: u64, reg: usize) {
        let (v, u_overflow) = overflowing_sub_unsigned(a, b, self.cpu.get_flag(StFlag::CARRY_BORROW_UNSIGNED));
        *self.regval_mut_n(reg) = v;
        self.cpu.set_flag(StFlag::CARRY_BORROW_UNSIGNED, u_overflow);

        let (v, s_overflow) = overflowing_sub_signed(a as i64, b as i64, self.cpu.get_flag(StFlag::CARRY_BORROW_UNSIGNED));
        *self.regval_mut_n(reg) = v as u64;
        self.cpu.set_flag(StFlag::CARRY_BORROW, s_overflow);
    }
    fn imul(&mut self, a: u64, b: u64, reg: usize) { *self.regval_mut_n(reg) = ((a as i64) * (b as i64)) as u64; }
    fn idiv(&mut self, a: u64, b: u64, reg: usize) { *self.regval_mut_n(reg) = ((a as i64) / (b as i64)) as u64; }
    fn umul(&mut self, a: u64, b: u64, reg: usize) { *self.regval_mut_n(reg) = a * b; }
    fn udiv(&mut self, a: u64, b: u64, reg: usize) { *self.regval_mut_n(reg) = a / b; }
    fn rem(&mut self, a: u64, b: u64, reg: usize) { *self.regval_mut_n(reg) = ((a as i64) % (b as i64)) as u64; }
    fn r#mod(&mut self, a: u64, b: u64, reg: usize) {
        if b as i64 == -1 {
            *self.regval_mut_n(reg) = 0;
        } else {
            *self.regval_mut_n(reg) = ((a as i64).rem_euclid(b as i64)) as u64;
        }
    }
    fn and(&mut self, a: u64, b: u64, reg: usize) { *self.regval_mut_n(reg) = a & b; }
    fn or(&mut self, a: u64, b: u64, reg: usize) { *self.regval_mut_n(reg) = a | b; }
    fn nor(&mut self, a: u64, b: u64, reg: usize) { *self.regval_mut_n(reg) = !(a | b); }
    fn xor(&mut self, a: u64, b: u64, reg: usize) { *self.regval_mut_n(reg) = a ^ b; }
    fn shl(&mut self, a: u64, b: u64, reg: usize) { *self.regval_mut_n(reg) = a << b; }
    fn sar(&mut self, a: u64, b: u64, reg: usize) { *self.regval_mut_n(reg) = ((a as i64) >> b) as u64; }
    fn shr(&mut self, a: u64, b: u64, reg: usize) { *self.regval_mut_n(reg) = a >> b; }
    fn bit(&mut self, a: u64, b: u64, reg: usize) { *self.regval_mut_n(reg) = (a >> b) & 1; }
    fn fcomp<F: Float>(&mut self, ci: Instruction) {
        let a = F::from_bits_u64(self.regval_n(get_usize!(ci.m.rde)));
        let b = F::from_bits_u64(self.regval_n(get_usize!(ci.m.rs1)));
        make_comp!(self, a, b, F);
    }
    // int -> float cast
    fn fto<F: Float>(&mut self, ci: Instruction) {
        let from = self.regval_n(get_usize!(ci.e.rs1));
        let to = self.regval_mut_n(get_usize!(ci.e.rde));
        F::from_int(from).to_bits_to_u64(to);
    }
    // float -> int cast
    fn ffrom<F: Float>(&mut self, ci: Instruction) {
        let from = self.regval_n(get_usize!(ci.e.rs1));
        let to = self.regval_mut_n(get_usize!(ci.e.rde));
        *to = F::from_bits_u64(from).to_int();
    }
    fn fneg<F: Float>(&mut self, ci: Instruction) {
        let from = self.regval_n(get_usize!(ci.e.rs1));
        let to = self.regval_mut_n(get_usize!(ci.e.rde));
        *to = F::from_bits_u64(from).neg().to_int();
    }
    fn fabs<F: Float>(&mut self, ci: Instruction) {
        let from = self.regval_n(get_usize!(ci.e.rs1));
        let to = self.regval_mut_n(get_usize!(ci.e.rde));
        *to = F::from_bits_u64(from).abs().to_int();
    }
    fn fadd<F: Float>(&mut self, ci: Instruction) {
        let a = self.regval_n(get_usize!(ci.e.rs1));
        let b = self.regval_n(get_usize!(ci.e.rs2));
        let to = self.regval_mut_n(get_usize!(ci.e.rde));
        *to = F::from_bits_u64(a).add(F::from_bits_u64(b)).to_int();
    }
    fn fsub<F: Float>(&mut self, ci: Instruction) {
        let a = self.regval_n(get_usize!(ci.e.rs1));
        let b = self.regval_n(get_usize!(ci.e.rs2));
        let to = self.regval_mut_n(get_usize!(ci.e.rde));
        *to = F::from_bits_u64(a).sub(F::from_bits_u64(b)).to_int();
    }
    fn fmul<F: Float>(&mut self, ci: Instruction) {
        let a = self.regval_n(get_usize!(ci.e.rs1));
        let b = self.regval_n(get_usize!(ci.e.rs2));
        let to = self.regval_mut_n(get_usize!(ci.e.rde));
        *to = F::from_bits_u64(a).mul(F::from_bits_u64(b)).to_int();
    }
    fn fdiv<F: Float>(&mut self, ci: Instruction) {
        let a = self.regval_n(get_usize!(ci.e.rs1));
        let b = self.regval_n(get_usize!(ci.e.rs2));
        let to = self.regval_mut_n(get_usize!(ci.e.rde));
        if F::from_bits_u64(b).is_zero() {
            self.push_interrupt(InterruptErr::DivideByZero);
        } else {
            *to = F::from_bits_u64(a).div(F::from_bits_u64(b)).to_int();
        }
    }
    fn fma<F: Float>(&mut self, ci: Instruction) {
        let a = self.regval_n(get_usize!(ci.e.rs1));
        let b = self.regval_n(get_usize!(ci.e.rs2));
        let to = self.regval_mut_n(get_usize!(ci.e.rde));
        (F::from_bits_u64(a) * F::from_bits_u64(b)).add_assign_bits(to);
    }
    fn fsqrt<F: Float>(&mut self, ci: Instruction) {
        let from = self.regval_n(get_usize!(ci.e.rs1));
        let to = self.regval_mut_n(get_usize!(ci.e.rde));
        *to = F::from_bits_u64(from).sqrt().to_int();
    }
    fn fmin<F: Float>(&mut self, ci: Instruction) {
        let a = self.regval_n(get_usize!(ci.e.rs1));
        let b = self.regval_n(get_usize!(ci.e.rs2));
        let to = self.regval_mut_n(get_usize!(ci.e.rde));
        *to = F::from_bits_u64(a).min(F::from_bits_u64(b)).to_int();
    }
    fn fmax<F: Float>(&mut self, ci: Instruction) {
        let a = self.regval_n(get_usize!(ci.e.rs1));
        let b = self.regval_n(get_usize!(ci.e.rs2));
        let to = self.regval_mut_n(get_usize!(ci.e.rde));
        *to = F::from_bits_u64(a).max(F::from_bits_u64(b)).to_int();
    }
    fn fsat<F: Float>(&mut self, ci: Instruction) {
        let from = self.regval_n(get_usize!(ci.e.rs1));
        let to = self.regval_mut_n(get_usize!(ci.e.rde));
        *to = F::from_bits_u64(from).ceil().to_int();
    }
    fn fconvert<From: Float, To: FloatFrom<From>>(&mut self, ci: Instruction) {
        let from = self.regval_n(get_usize!(ci.e.rs1));
        let to = self.regval_mut_n(get_usize!(ci.e.rde));
        To::from(From::from_bits_u64(from)).to_bits_to_u64(to);
    }
    fn fnan<F: Float>(&mut self, ci: Instruction) {
        let from = self.regval_n(get_usize!(ci.e.rs1));
        let to = self.regval_mut_n(get_usize!(ci.e.rde));
        *to = F::from_bits_u64(from).is_nan() as u64;
    }
    /// takes ownership of self
    pub fn run(mut self) -> RunStats {
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
struct InvalidInstructionError;

#[derive(Debug, Clone, Copy)]
pub struct RunStats {
    pub elapsed: f64,
    pub cycle:   u64,
}
impl RunStats {
    #[allow(clippy::cast_precision_loss)]
    pub fn cycle_per_sec(self) -> f64 { self.cycle as f64 / self.elapsed }
}
