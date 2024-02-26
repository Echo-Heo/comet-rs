#![warn(clippy::pedantic)]
#![allow(clippy::cast_possible_truncation)]
#![deny(unsafe_code)]


use crate::{nth_bit, safety::Interrupt};
use std::{
    fs::File,
    io,
    mem::size_of,
    path::{Path, PathBuf},
};

use thiserror::Error;

const MEM_PAGE_SIZE: u64 = 0x4000;
const MEM_DEFAULT_SIZE: u64 = 4096 * MEM_PAGE_SIZE;

#[allow(unused)]
#[derive(Debug, Clone, Copy)]
pub(crate) enum AccessMode {
    Translate,
    Read,
    Write,
    Execute,
}
#[derive(Debug, Clone, Copy)]
pub(crate) enum Response {
    AccViolation,
    NoPerms,
    OutOfBounds,
    Unaligned,
}
impl Response {
    pub(crate) const fn to_interrupt(self) -> Interrupt {
        match self {
            Self::AccViolation | Self::NoPerms | Self::OutOfBounds => Interrupt::AccessViolation,
            Self::Unaligned => Interrupt::UnalignedAccess,
        }
    }
}

#[allow(clippy::upper_case_acronyms)]
#[repr(C)]
#[derive(Debug, Clone)]
pub(crate) struct MMU {
    pub(crate) memory:          Box<[u8]>,
    pub(crate) page_table_base: u64,
}
#[derive(Debug, Clone, Copy, Error)]
#[error("")]
pub(crate) struct MMUInitError;
impl MMU {
    pub(crate) const fn mem_max(&self) -> u64 { (self.memory.len() - 1) as u64 }
    // pub(crate) const fn mem_len(&self) -> u64 { self.memory.len() as u64 }
    fn new_option(mem_cap: u64) -> Option<Self> {
        let mem_cap = if mem_cap == 0 { MEM_DEFAULT_SIZE } else { mem_cap };
        let capacity = mem_cap.try_into().ok()?;
        let memory = vec![0; capacity].into_boxed_slice();
        Some(Self { memory, page_table_base: 0 })
    }
    pub(crate) fn new(mem_cap: u64) -> Result<Self, MMUInitError> { Self::new_option(mem_cap).ok_or(MMUInitError) }
    pub(crate) fn load_image(&mut self, path: &Path) -> anyhow::Result<()> {
        let mut bin_file = File::open(path)?;
        let mut memory = &mut *self.memory;
        io::copy(&mut bin_file, &mut memory)?;
        Ok(())
    }

    pub(crate) fn phys_get_sized<const SIZE: usize>(&self, addr: u64) -> Result<&'_ [u8; SIZE], Response> {
        if addr > self.mem_max() {
            Err(Response::OutOfBounds)
        } else if addr as usize % SIZE != 0 {
            Err(Response::Unaligned)
        } else {
            Ok(self.memory[addr as usize..].first_chunk::<SIZE>().unwrap())
        }
    }
    pub(crate) fn phys_write_sized<const SIZE: usize>(&mut self, addr: u64, what: [u8; SIZE]) -> Result<(), Response> {
        if addr > self.mem_max() {
            Err(Response::OutOfBounds)
        } else if addr as usize % SIZE != 0 {
            Err(Response::Unaligned)
        } else {
            *self.memory[addr as usize..].first_chunk_mut::<SIZE>().unwrap() = what;
            Ok(())
        }
    }
    pub(crate) fn phys_get_u8(&self, addr: u64) -> Result<u8, Response> {
        let [res] = self.phys_get_sized::<{ size_of::<u8>() }>(addr)?;
        Ok(*res)
    }
    pub(crate) fn phys_get_u16(&self, addr: u64) -> Result<u16, Response> {
        let bytes = self.phys_get_sized::<{ size_of::<u16>() }>(addr)?;
        Ok(u16::from_le_bytes(*bytes))
    }
    pub(crate) fn phys_get_u32(&self, addr: u64) -> Result<u32, Response> {
        let bytes = self.phys_get_sized::<{ size_of::<u32>() }>(addr)?;
        Ok(u32::from_le_bytes(*bytes))
    }
    pub(crate) fn phys_get_u64(&self, addr: u64) -> Result<u64, Response> {
        let bytes = self.phys_get_sized::<{ size_of::<u64>() }>(addr)?;
        Ok(u64::from_le_bytes(*bytes))
    }

    // physical read/write
    #[allow(unused)]
    pub(crate) fn phys_read_u8(&self, addr: u64, var: &mut u8) -> Result<(), Response> {
        *var = self.phys_get_u8(addr)?;
        Ok(())
    }
    #[allow(unused)]
    pub(crate) fn phys_read_u16(&self, addr: u64, var: &mut u16) -> Result<(), Response> {
        *var = self.phys_get_u16(addr)?;
        Ok(())
    }
    #[allow(unused)]
    pub(crate) fn phys_read_u32(&self, addr: u64, var: &mut u32) -> Result<(), Response> {
        *var = self.phys_get_u32(addr)?;
        Ok(())
    }
    pub(crate) fn phys_read_u64(&self, addr: u64, var: &mut u64) -> Result<(), Response> {
        *var = self.phys_get_u64(addr)?;
        Ok(())
    }

    pub(crate) fn phys_write_u8(&mut self, addr: u64, value: u8) -> Result<(), Response> { self.phys_write_sized(addr, value.to_le_bytes()) }
    pub(crate) fn phys_write_u16(&mut self, addr: u64, value: u16) -> Result<(), Response> { self.phys_write_sized(addr, value.to_le_bytes()) }
    pub(crate) fn phys_write_u32(&mut self, addr: u64, value: u32) -> Result<(), Response> { self.phys_write_sized(addr, value.to_le_bytes()) }
    pub(crate) fn phys_write_u64(&mut self, addr: u64, value: u64) -> Result<(), Response> { self.phys_write_sized(addr, value.to_le_bytes()) }

    pub(crate) fn translate_address(&self, r#virtual: u64, mode: AccessMode) -> Result<u64, Response> {
        let level_1_index = ((0b11_1111u64 << 58) & r#virtual) >> 58;
        let level_2_index = ((0b111_1111_1111u64 << 47) & r#virtual) >> 47;
        let level_3_index = ((0b111_1111_1111u64 << 36) & r#virtual) >> 36;
        let level_4_index = ((0b111_1111_1111u64 << 25) & r#virtual) >> 25;
        let level_5_index = ((0b111_1111_1111u64 << 14) & r#virtual) >> 14;
        let level_6_index = 0b11_1111_1111_1111_u64 & r#virtual;

        let next = self.translate_address_level(self.page_table_base, level_1_index, mode)?;

        // level two
        let next = self.translate_address_level(next, level_2_index, mode)?;

        // level three
        let next = self.translate_address_level(next, level_3_index, mode)?;

        // level four
        let next = self.translate_address_level(next, level_4_index, mode)?;

        // level five
        let next = self.translate_address_level(next, level_5_index, mode)?;

        Ok(next + level_6_index)
    }

    fn translate_address_level(&self, next: u64, level_index: u64, mode: AccessMode) -> Result<u64, Response> {
        let pde = match self.phys_get_u64(next + level_index * 8) {
            Ok(pde) if pde & 1 != 0 => pde,
            _ => return Err(Response::AccViolation),
        };
        let auth_pde = if pde & nth_bit!(1) != 0 { pde } else { 0 };
        // get PDE
        // set authoritative perms
        // check perms
        if !has_perm(if auth_pde == 0 { pde } else { auth_pde }, mode) {
            return Err(Response::NoPerms);
        }
        Ok(0xFFFF_FFFF_FFFF_C000 & pde)
    }
}

fn has_perm(pde: u64, mode: AccessMode) -> bool {
    !(pde & nth_bit!(2) == 0 && matches!(mode, AccessMode::Read)
        || pde & nth_bit!(3) == 0 && matches!(mode, AccessMode::Write)
        || pde & nth_bit!(4) == 0 && matches!(mode, AccessMode::Execute))
}

#[derive(Debug, Clone, Error)]
#[error("crash: accessed but could not load file \"{0}\" into memory (ask sandwichman about this)")]
pub(crate) struct LoadImageError(pub(crate) PathBuf);
