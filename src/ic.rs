#![warn(clippy::pedantic)]

#[derive(Debug, Clone, Copy)]
pub struct IntQueueEntry {
    pub code: u8,
}
impl IntQueueEntry {
    pub const NEW: Self = IntQueueEntry { code: 0 };
}

#[allow(clippy::upper_case_acronyms)]
#[repr(C)]
#[derive(Debug, Clone)]
pub struct IC {
    pub ivt_base_address: u64,
    pub ret_addr:         u64,
    pub ret_status:       u64,
    pub queue:            Vec<IntQueueEntry>,
}
impl IC {
    pub fn new() -> Self {
        Self {
            ivt_base_address: 0,
            ret_addr:         0,
            ret_status:       0,
            queue:            vec![IntQueueEntry::NEW; 32],
        }
    }
}
