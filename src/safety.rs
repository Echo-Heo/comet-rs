#![warn(clippy::pedantic)]
#![deny(unsafe_code)]

#![allow(clippy::cast_possible_truncation)]
#![allow(clippy::cast_lossless)]
#![allow(clippy::cast_precision_loss)]
#![allow(clippy::cast_sign_loss)]
#![allow(clippy::cast_possible_wrap)]

use std::{
    fmt::Display,
    ops::{BitAnd, BitOr, BitXor, Not},
};

use crate::comet::StFlag;

#[allow(unused)]
pub(crate) trait BitFrom<T: Copy>: Copy {
    fn from(value: T) -> Self;
}
#[allow(unused)]
pub(crate) trait BitTo<T: Copy>: Copy {
    fn to(self) -> T;
}
pub(crate) trait BitAccess<T: Copy>: Copy {
    /// # Panics
    ///
    /// May panic if index out of bounds
    fn access(self, index: u8) -> T;

    /// # Panics
    ///
    /// May panic if index out of bounds
    fn write(&mut self, index: u8, value: T);
}
impl<T: Copy> BitFrom<T> for T {
    fn from(value: T) -> Self { value }
}
impl<T: Copy> BitTo<T> for T {
    fn to(self) -> T { self }
}
impl<T: Copy> BitAccess<T> for T {
    fn access(self, index: u8) -> T {
        assert_eq!(index, 0);
        self
    }
    fn write(&mut self, index: u8, value: T) {
        assert_eq!(index, 0);
        *self = value;
    }
}

macro_rules! impl_as {
    ($ty1: ty, $ty2: ty) => {
        impl BitFrom<$ty1> for $ty2 {
            #[inline(always)]
            fn from(value: $ty1) -> Self { value as Self }
        }
        impl BitTo<$ty1> for $ty2 {
            #[inline(always)]
            fn to(self) -> $ty1 { self as $ty1 }
        }
    };
}
impl_as! {u8, u16}
impl_as! {u8, u32}
impl_as! {u8, u64}
impl_as! {u8, usize}
impl_as! {u16, u8}
impl_as! {u16, u32}
impl_as! {u16, u64}
impl_as! {u16, usize}
impl_as! {u32, u8}
impl_as! {u32, u16}
impl_as! {u32, u64}
impl_as! {u32, usize}
impl_as! {u64, u8}
impl_as! {u64, u16}
impl_as! {u64, u32}
impl_as! {u64, usize}
impl_as! {usize, u8}
impl_as! {usize, u16}
impl_as! {usize, u32}
impl_as! {usize, u64}

// Bit access; safe equivalent of pointer hacks. they will all compile to the same thing so it doesn't matter

macro_rules! impl_bit_access {
    (bool, $ty: ident, 1) => {
        impl BitAccess<bool> for $ty {
            #[inline(always)]
            fn access(self, index: u8) -> bool { (self >> (index as Self)) % 2 == 1 }
            #[inline(always)]
            fn write(&mut self, index: u8, value: bool) {
                if value {
                    *self |= (1 << (index as Self))
                } else {
                    *self &= !(1 << (index as Self))
                }
            }
        }
    };
    (Nibble, $ty: ident, 4) => {
        impl BitAccess<Nibble> for $ty {
            #[inline(always)]
            fn access(self, index: u8) -> Nibble { Nibble::from_u8((self >> (index as Self * 4)) as u8) }
            #[inline(always)]
            fn write(&mut self, index: u8, value: Nibble) {
                *self = (*self & !((0x0F as Self) << (index as Self * 4))) | ((value.0 as Self) << (index as Self * 4))
            }
        }
    };
    ($ty1: ident, $ty2: ident, $ty1_bitsize: expr) => {
        impl BitAccess<$ty1> for $ty2 {
            #[inline(always)]
            fn access(self, index: u8) -> $ty1 { (self >> (index as Self * $ty1_bitsize)) as $ty1 }
            #[inline(always)]
            fn write(&mut self, index: u8, value: $ty1) {
                *self = (*self & !(($ty1::MAX as Self) << (index as Self * $ty1_bitsize))) | ((value as Self) << (index as Self * $ty1_bitsize))
            }
        }
    };
}
impl_bit_access! {u8, u16, 8}
impl_bit_access! {u8, u32, 8}
impl_bit_access! {u8, u64, 8}
impl_bit_access! {u8, usize, 8}
impl_bit_access! {u16, u32, 16}
impl_bit_access! {u16, u64, 16}
impl_bit_access! {u16, usize, 16}
impl_bit_access! {u32, u64, 32}

/// Type guaranteed to have no upper bits;
/// Do not use this to store things, only access and write nibbles
/// since it is a u8 underneath.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub(crate) struct Nibble(pub(crate) u8);
impl Nibble {
    // pub(crate) const MAX: Self = Nibble(0x0F);
    // pub(crate) const MIN: Self = Nibble(0);
    pub(crate) const fn from_u8(value: u8) -> Self { Self(value & 0x0F) }
    /// Only use this if you're sure the value doesn't have upper bytes.
    pub(crate) const fn from_u8_unchecked(value: u8) -> Self { Self(value) }
    // pub(crate) const fn as_u8(self) -> u8 { self.0 }
    pub(crate) const fn register(self) -> Register { Register(self) }
    pub(crate) const fn try_into_bool(self) -> Option<bool> {
        match self.0 {
            1 => Some(true),
            0 => Some(false),
            _ => None,
        }
    }
}
impl Display for Nibble {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result { self.0.fmt(f) }
}
impl BitAnd for Nibble {
    type Output = Self;
    fn bitand(self, rhs: Self) -> Self::Output { Self(self.0 & rhs.0) }
}
impl BitOr for Nibble {
    type Output = Self;
    fn bitor(self, rhs: Self) -> Self::Output { Self(self.0 | rhs.0) }
}
impl BitXor for Nibble {
    type Output = Self;
    fn bitxor(self, rhs: Self) -> Self::Output { Self(self.0 ^ rhs.0) }
}
impl Not for Nibble {
    type Output = Self;
    fn not(self) -> Self::Output { Self(self.0 ^ 0x0F) }
}

macro_rules! impl_from_to_nibble {
    ($ty: ident) => {
        impl From<$ty> for Nibble {
            fn from(value: $ty) -> Self { Nibble::from_u8(value as u8) }
        }
        impl From<Nibble> for $ty {
            fn from(value: Nibble) -> Self { value.0 as Self }
        }
    };
}
impl_from_to_nibble! {u8}
impl_from_to_nibble! {u16}
impl_from_to_nibble! {u32}
impl_from_to_nibble! {u64}
impl_from_to_nibble! {usize}
impl_bit_access! {Nibble, u8, 4}
impl_bit_access! {Nibble, u16, 4}
impl_bit_access! {Nibble, u32, 4}
impl_bit_access! {Nibble, u64, 4}
impl_bit_access! {Nibble, usize, 4}
impl_bit_access! {bool, u8, 1}
impl_bit_access! {bool, u16, 1}
impl_bit_access! {bool, u32, 1}
impl_bit_access! {bool, u64, 1}
impl_bit_access! {bool, usize, 1}
impl BitAccess<bool> for Nibble {
    fn access(self, index: u8) -> bool { self.0.access(index) }
    fn write(&mut self, index: u8, value: bool) {
        if index < 4 {
            self.0.write(index, value);
        }
    }
}


/// ```plain
///     31..28 │ 27..24 │ 23..20 │ 19..16 │           15..8 │            7..0 │
/// E │    rde │    rs1 │    rs2 │   func │            imm8 │          opcode │
/// R │    rde │    rs1 │    rs2 │                    imm12 │          opcode │
/// M │    rde │    rs1 │                             imm16 │          opcode │
/// F │    rde │   func │                             imm16 │          opcode │
/// B │   func │                                      imm20 │          opcode │
/// ```
#[derive(Debug, Clone, Copy)]
pub(crate) struct Instruction(pub(crate) u32);

impl Instruction {
    pub(crate) const fn nth_nibble(self, index: u32) -> Nibble { Nibble::from_u8((self.0 >> (index * 4)) as u8) }
    pub(crate) const fn opcode(self) -> u8 { self.0 as u8 }
    pub(crate) const fn e(self) -> E { E::new(self) }
    pub(crate) const fn r(self) -> R { R::new(self) }
    pub(crate) const fn m(self) -> M { M::new(self) }
    pub(crate) const fn f(self) -> F { F::new(self) }
    pub(crate) const fn b(self) -> B { B::new(self) }
}

/// Struct for destructuring. Do not use to store data.
#[derive(Debug, Clone, Copy)]
pub(crate) struct E {
    pub(crate) imm:  u8,
    pub(crate) func: Nibble,
    pub(crate) rs2:  Register,
    pub(crate) rs1:  Register,
    pub(crate) rd:   Register,
}
impl E {
    pub(crate) const fn new(i: Instruction) -> Self {
        Self {
            imm:  (i.0 >> 8) as u8,
            func: Nibble::from_u8((i.0 >> 16) as u8),
            rs2:  Nibble::from_u8((i.0 >> 20) as u8).register(),
            rs1:  Nibble::from_u8((i.0 >> 24) as u8).register(),
            rd:   Nibble::from_u8_unchecked((i.0 >> 28) as u8).register(),
        }
    }
}
/// Struct for destructuring. Do not use to store data.
#[allow(unused)]
#[derive(Debug, Clone, Copy)]
pub(crate) struct R {
    pub(crate) imm: u16,
    pub(crate) rs2: Register,
    pub(crate) rs1: Register,
    pub(crate) rd:  Register,
}
impl R {
    pub(crate) const fn new(i: Instruction) -> Self {
        Self {
            imm: ((i.0 >> 8) & 0x0FFF) as u16,
            rs2: Nibble::from_u8((i.0 >> 20) as u8).register(),
            rs1: Nibble::from_u8((i.0 >> 24) as u8).register(),
            rd:  Nibble::from_u8_unchecked((i.0 >> 28) as u8).register(),
        }
    }
}
/// Struct for destructuring. Do not use to store data.
#[derive(Debug, Clone, Copy)]
pub(crate) struct M {
    pub(crate) imm: u16,
    pub(crate) rs:  Register,
    pub(crate) rd:  Register,
}
impl M {
    pub(crate) const fn new(i: Instruction) -> Self {
        Self {
            imm: (i.0 >> 8) as u16,
            rs:  Nibble::from_u8((i.0 >> 24) as u8).register(),
            rd:  Nibble::from_u8_unchecked((i.0 >> 28) as u8).register(),
        }
    }
}
/// Struct for destructuring. Do not use to store data.
#[derive(Debug, Clone, Copy)]
pub(crate) struct F {
    pub(crate) imm:  u16,
    pub(crate) func: Nibble,
    pub(crate) rd:   Register,
}
impl F {
    pub(crate) const fn new(i: Instruction) -> Self {
        Self {
            imm:  (i.0 >> 8) as u16,
            func: Nibble::from_u8((i.0 >> 24) as u8),
            rd:   Nibble::from_u8_unchecked((i.0 >> 28) as u8).register(),
        }
    }
}
/// Struct for destructuring. Do not use to store data.
#[derive(Debug, Clone, Copy)]
pub(crate) struct B {
    pub(crate) imm:  u32,
    pub(crate) func: Nibble,
}
impl B {
    pub(crate) const fn new(i: Instruction) -> Self {
        Self {
            imm:  (i.0 >> 8) & 0x000F_FFFF,
            func: Nibble::from_u8_unchecked((i.0 >> 28) as u8),
        }
    }
}

#[allow(clippy::enum_variant_names)]
#[derive(Debug, Clone, Copy)]
pub(crate) enum Interrupt {
    DivideByZero    = 0x00,
    BreakPoint      = 0x01,
    InvalidInstruction = 0x02,
    StackUnderflow  = 0x03,
    UnalignedAccess = 0x04,
    AccessViolation = 0x05,
    InterruptOverflow = 0x06,
}
impl Interrupt {
    /* pub(crate) const fn from_u8(value: u8) -> Result<Self, Interrupt> {
        match value {
            0 => Ok(Self::DivideByZero),
            1 => Ok(Self::BreakPoint),
            2 => Ok(Self::InvalidInstruction),
            3 => Ok(Self::StackUnderflow),
            4 => Ok(Self::UnalignedAccess),
            5 => Ok(Self::AccessViolation),
            6 => Ok(Self::InterruptOverflow),
            _ => Err(Interrupt::InvalidInstruction),
        }
    } */
    pub(crate) const fn from_u16(value: u16) -> Result<Self, Interrupt> {
        match value {
            0 => Ok(Self::DivideByZero),
            1 => Ok(Self::BreakPoint),
            2 => Ok(Self::InvalidInstruction),
            3 => Ok(Self::StackUnderflow),
            4 => Ok(Self::UnalignedAccess),
            5 => Ok(Self::AccessViolation),
            6 => Ok(Self::InterruptOverflow),
            _ => Err(Interrupt::InvalidInstruction),
        }
    }
}

macro_rules! impl_register {
    (impl Register {
        $($NAME: ident = $value: expr;)*
    }) => {
        #[allow(unused)]
        impl Register {
            $(pub(crate) const $NAME: Self = Self(Nibble($value));)*
        }
    };
}

#[derive(Debug, Clone, Copy)]
pub(crate) struct Register(pub(crate) Nibble);
impl_register! {
    impl Register {
        RZ = 0x0;
        RA = 0x1;
        RB = 0x2;
        RC = 0x3;
        RD = 0x4;
        RE = 0x5;
        RF = 0x6;
        RG = 0x7;
        RH = 0x8;
        RI = 0x9;
        RJ = 0xA;
        RK = 0xB;
        IP = 0xC;
        SP = 0xD;
        FP = 0xE;
        ST = 0xF;
    }
}
impl Display for Register {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let str = match self.0.0 {
            0x0 => "RZ",
            0x1 => "RA",
            0x2 => "RB",
            0x3 => "RC", 
            0x4 => "RD",
            0x5 => "RE",
            0x6 => "RF",
            0x7 => "RG",
            0x8 => "RH",
            0x9 => "RI",
            0xA => "RJ",
            0xB => "RK",
            0xC => "IP",
            0xD => "SP",
            0xE => "FP",
            0xF => "ST",
            _ => unreachable!()
        };
        write!(f, "{str}")
    }
}

#[derive(Debug, Clone, Copy)]
pub(crate) struct Port(pub(crate) u16);


#[derive(Debug, Clone, Copy)]
pub(crate) enum BranchCond {
    Ra  = 0x0,
    Eq  = 0x1,
    Ez  = 0x2,
    Lt  = 0x3,
    Le  = 0x4,
    Ltu = 0x5,
    Leu = 0x6,
    Ne  = 0x9,
    Nz  = 0xA,
    Ge  = 0xB,
    Gt  = 0xC,
    Geu = 0xD,
    Gtu = 0xE,
}
impl BranchCond {
    const RA: Nibble = Nibble(Self::Ra as u8);
    const EQ: Nibble = Nibble(Self::Eq as u8);
    const EZ: Nibble = Nibble(Self::Ez as u8);
    const LT: Nibble = Nibble(Self::Lt as u8);
    const LE: Nibble = Nibble(Self::Le as u8);
    const LTU: Nibble = Nibble(Self::Ltu as u8);
    const LEU: Nibble = Nibble(Self::Leu as u8);
    const NE: Nibble = Nibble(Self::Ne as u8);
    const NZ: Nibble = Nibble(Self::Nz as u8);
    const GE: Nibble = Nibble(Self::Ge as u8);
    const GT: Nibble = Nibble(Self::Gt as u8);
    const GEU: Nibble = Nibble(Self::Geu as u8);
    const GTU: Nibble = Nibble(Self::Gtu as u8);
    pub(crate) const fn new(value: Nibble) -> Result<Self, Interrupt> {
        match value {
            Self::RA => Ok(Self::Ra),
            Self::EQ => Ok(Self::Eq),
            Self::EZ => Ok(Self::Ez),
            Self::LT => Ok(Self::Lt),
            Self::LE => Ok(Self::Le),
            Self::LTU => Ok(Self::Ltu),
            Self::LEU => Ok(Self::Leu),
            Self::NE => Ok(Self::Ne),
            Self::NZ => Ok(Self::Nz),
            Self::GE => Ok(Self::Ge),
            Self::GT => Ok(Self::Gt),
            Self::GEU => Ok(Self::Geu),
            Self::GTU => Ok(Self::Gtu),
            _ => Err(Interrupt::InvalidInstruction),
        }
    }
    pub(crate) fn cond(self, get_flag: impl Fn(StFlag) -> bool) -> bool {
        match self {
            Self::Ra => true,
            Self::Eq => get_flag(StFlag::EQUAL),
            Self::Ez => get_flag(StFlag::ZERO),
            Self::Lt => get_flag(StFlag::LESS),
            Self::Le => get_flag(StFlag::LESS) || get_flag(StFlag::EQUAL),
            Self::Ltu => get_flag(StFlag::LESS_UNSIGNED),
            Self::Leu => get_flag(StFlag::LESS_UNSIGNED) || get_flag(StFlag::EQUAL),
            Self::Ne => !get_flag(StFlag::EQUAL),
            Self::Nz => !get_flag(StFlag::ZERO),
            Self::Ge => !get_flag(StFlag::LESS),
            Self::Gt => !get_flag(StFlag::LESS) && !get_flag(StFlag::EQUAL),
            Self::Geu => !get_flag(StFlag::LESS_UNSIGNED),
            Self::Gtu => !get_flag(StFlag::LESS_UNSIGNED) && !get_flag(StFlag::EQUAL),
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub(crate) enum LiType {
    Lli   = 0,
    Llis  = 1,
    Lui   = 2,
    Luis  = 3,
    Lti   = 4,
    Ltis  = 5,
    Ltui  = 6,
    Ltuis = 7,
}
impl LiType {
    pub(crate) const fn new(value: Nibble) -> Result<Self, Interrupt> {
        match value {
            Nibble(0) => Ok(Self::Lli),
            Nibble(1) => Ok(Self::Llis),
            Nibble(2) => Ok(Self::Lui),
            Nibble(3) => Ok(Self::Luis),
            Nibble(4) => Ok(Self::Lti),
            Nibble(5) => Ok(Self::Ltis),
            Nibble(6) => Ok(Self::Ltui),
            Nibble(7) => Ok(Self::Ltuis),
            _ => Err(Interrupt::InvalidInstruction),
        }
    }
}
#[derive(Debug, Clone, Copy)]
pub(crate) enum FloatPrecision {
    F16,
    F32,
    F64,
}
impl FloatPrecision {
    pub(crate) const fn new(value: Nibble) -> Result<Self, Interrupt> {
        match value {
            Nibble(0) => Ok(Self::F16),
            Nibble(1) => Ok(Self::F32),
            Nibble(2) => Ok(Self::F64),
            _ => Err(Interrupt::InvalidInstruction),
        }
    }
}
#[derive(Debug, Clone, Copy)]
pub(crate) struct FloatCastType {
    pub(crate) to:   FloatPrecision,
    pub(crate) from: FloatPrecision,
}
impl FloatCastType {
    pub(crate) fn new(value: Nibble) -> Result<Self, Interrupt> {
        Ok(Self {
            to:   FloatPrecision::new(Nibble(value.0 & 0x11))?,
            from: FloatPrecision::new(Nibble(value.0 >> 2))?,
        })
    }
}

impl From<u64> for Port {
    fn from(value: u64) -> Self { Port(value as u16) }
}

