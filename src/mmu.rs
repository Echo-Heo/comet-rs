#![warn(clippy::pedantic)]
#![allow(clippy::cast_possible_truncation)]

use crate::{comet::Instruction, nth_bit};
use std::{
    fs::File,
    io::{Read, Seek},
    mem::size_of,
    path::{Path, PathBuf},
};

use thiserror::Error;

use crate::comet::InterruptErr;

const MEM_PAGE_SIZE: u64 = 0x4000;
const MEM_DEFAULT_SIZE: u64 = 4096 * MEM_PAGE_SIZE;

#[derive(Debug, Clone, Copy)]
pub enum AccessMode {
    Translate,
    Read,
    Write,
    Execute,
}
#[derive(Debug, Clone, Copy)]
pub enum Response {
    AccViolation,
    NoPerms,
    OutOfBounds,
    Unaligned,
}
impl Response {
    pub const fn to_interrupt_err(self) -> InterruptErr {
        match self {
            Self::AccViolation | Self::NoPerms | Self::OutOfBounds => InterruptErr::AccessViolation,
            Self::Unaligned => InterruptErr::UnalignedAccess,
        }
    }
}

#[allow(clippy::upper_case_acronyms)]
#[repr(C)]
#[derive(Debug, Clone)]
pub struct MMU {
    pub memory:          Box<[u8]>,
    pub page_table_base: u64,
}
#[derive(Debug, Clone, Copy, Error)]
#[error("")]
pub struct MMUInitError;
impl MMU {
    pub const fn mem_max(&self) -> u64 { (self.memory.len() - 1) as u64 }
    pub const fn mem_max_usize(&self) -> usize { self.memory.len() - 1 }
    pub const fn mem_len(&self) -> u64 { self.memory.len() as u64 }
    pub const fn mem_len_usize(&self) -> usize { self.memory.len() }
    fn new_option(mem_cap: u64) -> Option<Self> {
        let mem_cap = if mem_cap == 0 {
            MEM_DEFAULT_SIZE
        } else {
            mem_cap
        };
        let capacity = mem_cap.try_into().ok()?;
        let memory = vec![0; capacity].into_boxed_slice();
        Some(Self {
            memory,
            page_table_base: 0,
        })
    }
    pub fn new(mem_cap: u64) -> Result<Self, MMUInitError> {
        Self::new_option(mem_cap).ok_or(MMUInitError)
    }
    pub fn load_image(&mut self, path: &Path) -> anyhow::Result<()> {
        let mut bin = File::open(path)?;
        let bin_size = bin.seek(std::io::SeekFrom::End(0))?;
        bin.seek(std::io::SeekFrom::Start(0))?;
        let ret_code = bin.read(&mut self.memory)?;
        if bin_size != ret_code as u64 {
            Err(LoadImageError(path.to_owned()))?;
        }
        Ok(())
    }
    
    pub fn mem_get_sized<const SIZE: usize>(
        &self, addr: usize,
    ) -> Result<&'_ [u8; SIZE], Response> {
        if addr as u64 > self.mem_max() {
            Err(Response::OutOfBounds)
        } else if addr % SIZE != 0 {
            Err(Response::Unaligned)
        } else {
            Ok(self.memory[addr..].first_chunk::<SIZE>().unwrap())
        }
    }
    pub fn mem_get_u8(&self, addr: u64) -> Result<u8, Response> {
        let [res] = self.mem_get_sized::<{ size_of::<u8>() }>(addr as usize)?;
        Ok(*res)
    }
    pub fn mem_get_u16(&self, addr: u64) -> Result<u16, Response> {
        let bytes = self.mem_get_sized::<{ size_of::<u16>() }>(addr as usize)?;
        Ok(u16::from_ne_bytes(*bytes))
    }
    pub fn mem_get_u32(&self, addr: u64) -> Result<u32, Response> {
        let bytes = self.mem_get_sized::<{ size_of::<u32>() }>(addr as usize)?;
        Ok(u32::from_ne_bytes(*bytes))
    }
    pub fn mem_get_u64(&self, addr: u64) -> Result<u64, Response> {
        let bytes = self.mem_get_sized::<{ size_of::<u64>() }>(addr as usize)?;
        Ok(u64::from_ne_bytes(*bytes))
    }
    pub fn mem_get_u128(&self, addr: u64) -> Result<u128, Response> {
        let bytes = self.mem_get_sized::<{ size_of::<u128>() }>(addr as usize)?;
        Ok(u128::from_ne_bytes(*bytes))
    }
    // physical read/write

    pub fn phys_read_u8(&self, addr: u64, var: &mut u8) -> Result<(), Response> {
        *var = self.mem_get_u8(addr)?;
        Ok(())
    }
    pub fn phys_read_u16(&self, addr: u64, var: &mut u16) -> Result<(), Response> {
        *var = self.mem_get_u16(addr)?;
        Ok(())
    }
    pub fn phys_read_u32(&self, addr: u64, var: &mut u32) -> Result<(), Response> {
        *var = self.mem_get_u32(addr)?;
        Ok(())
    }
    pub fn phys_read_u64(&self, addr: u64, var: &mut u64) -> Result<(), Response> {
        *var = self.mem_get_u64(addr)?;
        Ok(())
    }
    pub fn phys_read_u128(&self, addr: u64, var: &mut u128) -> Result<(), Response> {
        *var = self.mem_get_u128(addr)?;
        Ok(())
    }

    pub fn translate_address(
        &self, r#virtual: u64, mode: AccessMode,
    ) -> Result<u64, Response> {
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

    fn translate_address_level(
        &self, next: u64, level_index: u64, mode: AccessMode,
    ) -> Result<u64, Response> {
        let pde = match self.mem_get_u64(next + level_index * 8) {
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
pub struct LoadImageError(pub PathBuf);
