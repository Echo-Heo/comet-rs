#![warn(clippy::pedantic)]

use std::{
    fs::File,
    io::{Read, Seek},
    path::{Path, PathBuf},
};

use thiserror::Error;

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
    Success,
    AccViolation,
    NoPerms,
    OutOfBounds,
    Unaligned,
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
}
#[derive(Debug, Clone, Error)]
#[error("crash: accessed but could not load file \"{0}\" into memory (ask sandwichman about this)")]
pub struct LoadImageError(pub PathBuf);
