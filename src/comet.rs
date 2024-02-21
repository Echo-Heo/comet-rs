#![warn(clippy::pedantic)]
#![allow(clippy::unused_unit)]

use bitfield_struct::bitfield;
use sa::static_assert;

#[bitfield(u32)]
#[derive(PartialEq, Eq)]
pub struct E {
    #[bits(8)]
    pub opcode: u32,
    #[bits(8)]
    pub imm:    u32,
    #[bits(4)]
    pub func:   u32,
    #[bits(4)]
    pub rs2:    u32,
    #[bits(4)]
    pub rs1:    u32,
    #[bits(4)]
    pub rde:    u32,
}
#[bitfield(u32)]
#[derive(PartialEq, Eq)]
pub struct R {
    #[bits(8)]
    pub opcode: u32,
    #[bits(12)]
    pub imm:    u32,
    #[bits(4)]
    pub rs2:    u32,
    #[bits(4)]
    pub rs1:    u32,
    #[bits(4)]
    pub rde:    u32,
}
#[bitfield(u32)]
#[derive(PartialEq, Eq)]
pub struct M {
    #[bits(8)]
    pub opcode: u32,
    #[bits(16)]
    pub imm:    u32,
    #[bits(4)]
    pub rs1:    u32,
    #[bits(4)]
    pub rde:    u32,
}
#[bitfield(u32)]
#[derive(PartialEq, Eq)]
pub struct F {
    #[bits(8)]
    pub opcode: u32,
    #[bits(16)]
    pub imm:    u32,
    #[bits(4)]
    pub func:   u32,
    #[bits(4)]
    pub rde:    u32,
}
#[bitfield(u32)]
#[derive(PartialEq, Eq)]
pub struct B {
    #[bits(8)]
    pub opcode: u32,
    #[bits(20)]
    pub imm:    u32,
    #[bits(4)]
    pub func:   u32,
}

pub union Instruction {
    pub opcode: u8,
    pub e:      E,
    pub r:      R,
    pub m:      M,
    pub f:      F,
    pub b:      B,
}

static_assert!(core::mem::size_of::<Instruction>() == core::mem::size_of::<u32>());
