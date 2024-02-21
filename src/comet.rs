#![warn(clippy::pedantic)]
#![allow(clippy::cast_possible_truncation)]
#![allow(clippy::unused_unit)]
#![allow(clippy::too_many_lines)]

use bitfield_struct::bitfield;
use bitflags::bitflags;
use sa::static_assert;
use std::{
    fmt::Debug,
    ops::{Index, IndexMut},
    time::Instant,
};

use crate::{
    ic::{IntQueueEntry, IC},
    io::IOC,
    mmu::{self, MMU},
};

macro_rules! sign_extend {
    ($val: expr, $bitsize: expr) => {
        (((($val as u64) << (64 - $bitsize)) as i64) >> (64 - $bitsize)) as u64
    };
}
macro_rules! zero_extend {
    ($val: expr, $bitsize: expr) => {
        (((($val as u64) << (64 - $bitsize)) as u64) >> (64 - $bitsize)) as u64
    };
}

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
    pub const fn from_bits(bits: u32) -> Self { Self { bits } }
    pub const fn e(self) -> E { unsafe { self.e } }
    pub const fn r(self) -> R { unsafe { self.r } }
    pub const fn m(self) -> M { unsafe { self.m } }
    pub const fn f(self) -> F { unsafe { self.f } }
    pub const fn b(self) -> B { unsafe { self.b } }
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
impl RegisterName {
    pub const fn try_from_u8(val: u8) -> Option<Self> {
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
    pub const fn from_u8(val: u8) -> Self {
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
pub enum ProcMode {
    Kernel,
    User,
}
impl ProcMode {
    pub const fn bool(self) -> bool { matches!(self, Self::User) }
}

#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct Registers(pub [u64; 16]);
impl Registers {
    pub const fn get_flag(&self, flag: StFlag) -> bool {
        self.0[RN::St as usize] & flag.bits() == 1
    }
    pub fn set_flag(&mut self, flag: StFlag, value: bool) {
        if value {
            self[RN::St] |= flag.bits();
        } else {
            self[RN::St] &= !flag.bits();
        }
    }
    pub const fn current_instr(&self) -> Instruction {
        Instruction::from_bits((self.0[RN::St as usize] >> 32) as u32)
    }
    pub fn set_current_instr(&mut self, instruction: Instruction) {
        unsafe {
            (self.0[RN::St as usize] as *mut u64)
                .cast::<Instruction>()
                .offset(1)
                .write(instruction);
        }
        /* self[RN::St] = ((self[RN::St] << 32) >> 32) + ((instruction.bits() as u64) << 32); */
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
    pub const fn current_instr(&self) -> Instruction { self.cpu.registers.current_instr() }
    pub fn set_current_instr(&mut self, instruction: Instruction) {
        self.cpu.registers.set_current_instr(instruction);
    }

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
    fn read_u128(&self, addr: u64) -> Result<u128, mmu::Response> {
        let mut addr = addr;
        if self.cpu.get_flag(StFlag::MODE) {
            addr = self.mmu.translate_address(addr, mmu::AccessMode::Read)?;
        }
        self.mmu.phys_get_u128(addr)
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
    fn write_u128(&mut self, addr: u64, value: u128) -> Result<(), mmu::Response> {
        let mut addr = addr;
        if self.cpu.get_flag(StFlag::MODE) {
            addr = self.mmu.translate_address(addr, mmu::AccessMode::Read)?;
        }
        self.mmu.phys_write_u128(addr, value)
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
        match self.mmu.phys_read_u64(
            self.ic.ivt_base_address + 8 * err as u64,
            &mut self.cpu.registers[RN::Ip],
        ) {
            Err(err) => {
                self.push_interrupt_from_mmu(err);
            }
            Ok(()) => {
                self.ic.queue.push(IntQueueEntry { code: err as u8 });
            }
        }
    }
    fn push_interrupt_from_mmu(&mut self, res: mmu::Response) {
        self.push_interrupt(res.to_interrupt_err());
    }

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
            match self
                .mmu
                .phys_get_u64(self.ic.ivt_base_address + 8 * code as u64)
            {
                Err(err) => self.push_interrupt_from_mmu(err),
                Ok(res) => self.cpu.registers[RN::Ip] = res,
            }
        }
    }
    pub const fn regval(&self, reg: RegisterName) -> u64 { self.cpu.registers.0[reg as usize] }
    pub const fn regval_n(&self, reg: usize) -> u64 { self.cpu.registers.0[reg] }
    pub fn regval_mut(&mut self, reg: RegisterName) -> &mut u64 {
        &mut self.cpu.registers.0[reg as usize]
    }
    pub fn regval_mut_n(&mut self, reg: usize) -> &mut u64 { &mut self.cpu.registers.0[reg] }
    pub const fn proc_mode_is_user(&self) -> bool {
        self.cpu.get_flag(StFlag::MODE) == ProcMode::User.bool()
    }
    pub const fn proc_mode_is_user_then_invalid(&self) -> Result<(), InvalidInstructionError> {
        if self.proc_mode_is_user() {
            Err(InvalidInstructionError)
        } else {
            Ok(())
        }
    }
    pub fn run_internal(&mut self) {
        self.cpu.cycle += 1;
        println!(
            "[at {:#016x} {:02x}]",
            self.regval(RN::Ip),
            self.current_instr().opcode()
        );

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
            0x01 => match unsafe { ci.f }.func() {
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
                    self.cpu.registers[RN::Ip] = self.regval_n(ci.f().rde() as usize);
                }
                _ => Err(InvalidInstructionError)?,
            },
            // outr
            0x02 => {
                self.proc_mode_is_user_then_invalid()?;
                self.ioc.send_out(
                    self.regval_n(ci.m().rde() as usize) as u16,
                    self.regval_n(ci.m().rs1() as usize),
                );
            }
            // outi
            0x03 => {
                self.proc_mode_is_user_then_invalid()?;
                self.ioc
                    .send_out(ci.m().imm() as u16, self.regval_n(ci.m().rs1() as usize));
            }
            // inr
            0x04 => {
                self.proc_mode_is_user_then_invalid()?;
                *self.regval_mut_n(ci.m().rde() as usize) = self
                    .ioc
                    .port_data(self.regval_n(ci.m().rs1() as usize) as u16);
            }
            // ini
            0x05 => {
                self.proc_mode_is_user_then_invalid()?;
                *self.regval_mut_n(ci.m().rde() as usize) = self.ioc.port_data(ci.m().imm() as u16);
            }
            // jal
            0x06 => {
                self.push_stack(self.regval(RN::Ip));
                *self.regval_mut(RN::Ip) = self.regval_n(ci.m().rs1() as usize)
                    + (4 * sign_extend!(ci.m().imm(), 16) as i64) as u64;
            }
            // jalr
            0x07 => {
                *self.regval_mut_n(ci.m().rde() as usize) = self.regval(RN::Ip);
                *self.regval_mut(RN::Ip) = self.regval_n(ci.m().rs1() as usize)
                    + (4 * sign_extend!(ci.m().imm(), 16) as i64) as u64;
            }
            // ret
            0x08 => {
                self.pop_stack_to(RN::Ip);
            }
            // retr
            0x09 => {
                *self.regval_mut(RN::Ip) = self.regval_n(ci.m().rs1() as usize);
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
                    self.if_cond_then_do_that_weird_thang(
                        ci,
                        self.cpu.get_flag(StFlag::LESS) || self.cpu.get_flag(StFlag::EQUAL),
                    );
                }
                // bltu
                0x5 => {
                    self.if_cond_then_do_that_weird_thang(
                        ci,
                        self.cpu.get_flag(StFlag::LESS_UNSIGNED),
                    );
                }
                // bleu
                0x6 => {
                    self.if_cond_then_do_that_weird_thang(
                        ci,
                        self.cpu.get_flag(StFlag::LESS_UNSIGNED)
                            || self.cpu.get_flag(StFlag::EQUAL),
                    );
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
                    self.if_cond_then_do_that_weird_thang(
                        ci,
                        !self.cpu.get_flag(StFlag::LESS) && !self.cpu.get_flag(StFlag::EQUAL),
                    );
                }
                // bgeu
                0xd => {
                    self.if_cond_then_do_that_weird_thang(
                        ci,
                        !self.cpu.get_flag(StFlag::LESS_UNSIGNED),
                    );
                }
                // bteu
                0xe => {
                    self.if_cond_then_do_that_weird_thang(
                        ci,
                        !self.cpu.get_flag(StFlag::LESS_UNSIGNED)
                            && !self.cpu.get_flag(StFlag::EQUAL),
                    );
                }
                _ => Err(InvalidInstructionError)?,
            },
            _ => todo!(),
            // push
            0x0b => {
                self.push_stack(self.regval_n(ci.m().rs1() as usize));
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
                0 => {
                    todo!()
                }
                _ => todo!(),
            },
        }

        todo!()
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
pub struct InvalidInstructionError;

#[derive(Debug, Clone, Copy)]
pub struct RunStats {
    pub elapsed: f64,
    pub cycle:   u64,
}
impl RunStats {
    #[allow(clippy::cast_precision_loss)]
    pub fn cycle_per_sec(self) -> f64 { self.cycle as f64 / self.elapsed }
}
